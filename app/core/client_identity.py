"""
Unified client identification and risk assessment.

This module consolidates all fingerprinting, session tracking, and risk
scoring into a single coherent system for abuse prevention.

Public API:
- get_client_ip(request) -> str
- get_server_fingerprint(request) -> str  
- get_vote_identifier(request, symbol) -> str
- check_vote_allowed(request, symbol) -> VoteCheck
- record_vote(request, symbol) -> None
- get_suspicious_log(limit) -> list[dict]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any

from fastapi import Request

from app.cache.client import get_valkey_client
from app.core.logging import get_logger

logger = get_logger("core.client_identity")

# =============================================================================
# Configuration
# =============================================================================

CACHE_PREFIX = "stonkmarket:v2:identity"
VOTE_COOLDOWN_DAYS = 7
SESSION_TTL_DAYS = 365
MAX_IPS_PER_FINGERPRINT = 5
MAX_VOTES_PER_HOUR = 20
SUSPICIOUS_STOCK_VOTES_THRESHOLD = 10


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"           # 0-25: Normal user
    MEDIUM = "medium"     # 26-50: Some anomalies
    HIGH = "high"         # 51-75: Likely manipulation
    CRITICAL = "critical" # 76-100: Definite abuse


# =============================================================================
# Core Utilities (Single Source of Truth)
# =============================================================================

def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request, respecting proxy headers.
    
    Priority: Cloudflare > X-Forwarded-For > X-Real-IP > Direct
    """
    # Cloudflare
    if cf_ip := request.headers.get("CF-Connecting-IP"):
        return cf_ip.strip()
    
    # X-Forwarded-For (take first = original client)
    if forwarded := request.headers.get("X-Forwarded-For"):
        return forwarded.split(",")[0].strip()
    
    # Nginx
    if real_ip := request.headers.get("X-Real-IP"):
        return real_ip.strip()
    
    # Direct connection
    if request.client:
        return request.client.host
    
    return "unknown"


def _hash(value: str, length: int = 16) -> str:
    """Create a short hash of a value."""
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def get_server_fingerprint(request: Request) -> str:
    """
    Generate server-side browser fingerprint from headers.
    
    This is harder to spoof than client-provided fingerprints.
    Used for risk assessment, not tracking.
    """
    parts = [
        request.headers.get("User-Agent", ""),
        request.headers.get("Accept-Language", ""),
        request.headers.get("Accept-Encoding", ""),
        request.headers.get("Accept", ""),
        request.headers.get("Sec-CH-UA", ""),
        request.headers.get("Sec-CH-UA-Platform", ""),
        request.headers.get("Sec-CH-UA-Mobile", ""),
        request.headers.get("DNT", ""),
        request.headers.get("Sec-Fetch-Site", ""),
        request.headers.get("Sec-Fetch-Mode", ""),
        request.headers.get("Sec-Fetch-Dest", ""),
    ]
    return _hash("|".join(parts), 32)


def get_client_fingerprint(request: Request) -> Optional[str]:
    """Get client-provided fingerprint from header (untrusted)."""
    return request.headers.get("X-Client-Fingerprint")


def get_vote_identifier(request: Request, symbol: str) -> str:
    """
    Create unique identifier for a vote on a specific symbol.
    
    Combines IP + server fingerprint + symbol for deduplication.
    """
    ip = get_client_ip(request)
    server_fp = get_server_fingerprint(request)
    vote_str = f"{ip}|{server_fp}|vote:{symbol.upper()}"
    return _hash(vote_str, 40)


def get_request_fingerprint(
    request: Request,
    include_ip: bool = True,
    include_headers: bool = True,
) -> str:
    """
    Create a composite fingerprint for the request.
    
    Used for rate limiting. Combines IP + server fingerprint.
    Returns a hash for privacy.
    """
    parts = []
    
    if include_ip:
        parts.append(f"ip:{get_client_ip(request)}")
    
    if include_headers:
        parts.append(f"browser:{get_server_fingerprint(request)}")
    
    if not parts:
        parts.append(f"ip:{get_client_ip(request)}")
    
    return _hash("|".join(parts), 40)


def get_suggestion_identifier(request: Request) -> str:
    """
    Create an identifier for suggestion rate limiting.
    
    Uses the full fingerprint to rate limit how often someone can
    suggest new stocks.
    """
    return get_request_fingerprint(request)


# =============================================================================
# Vote Permission Checking
# =============================================================================

@dataclass
class VoteCheck:
    """Result of checking if a vote is allowed."""
    allowed: bool
    reason: str
    risk_score: int
    risk_level: RiskLevel
    flags: list[str] = field(default_factory=list)
    reduce_weight: bool = False
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "flags": self.flags,
        }


async def check_vote_allowed(
    request: Request,
    symbol: str,
) -> VoteCheck:
    """
    Check if this request should be allowed to vote.
    
    Performs:
    1. Cooldown check (7 days per IP+fingerprint per stock)
    2. Velocity check (max votes per hour)
    3. Risk scoring (coordinated voting detection)
    
    Returns VoteCheck with decision and risk info.
    """
    client = await get_valkey_client()
    symbol = symbol.upper()
    
    ip = get_client_ip(request)
    ip_hash = _hash(ip)
    server_fp = get_server_fingerprint(request)
    client_fp = get_client_fingerprint(request)
    
    flags = []
    risk_score = 0
    
    # === Cooldown Check ===
    vote_id = get_vote_identifier(request, symbol)
    cooldown_key = f"{CACHE_PREFIX}:vote:{vote_id}"
    
    existing_vote = await client.get(cooldown_key)
    if existing_vote:
        try:
            vote_data = json.loads(existing_vote)
            vote_time = datetime.fromisoformat(vote_data["voted_at"])
            cooldown_end = vote_time + timedelta(days=VOTE_COOLDOWN_DAYS)
            if datetime.utcnow() < cooldown_end:
                days_left = (cooldown_end - datetime.utcnow()).days + 1
                return VoteCheck(
                    allowed=False,
                    reason=f"Already voted. Try again in {days_left} days.",
                    risk_score=0,
                    risk_level=RiskLevel.LOW,
                )
        except (json.JSONDecodeError, KeyError):
            pass
    
    # === IP-based Cooldown (backup check) ===
    ip_vote_key = f"{CACHE_PREFIX}:ip_vote:{ip_hash}:{symbol}"
    ip_voted = await client.get(ip_vote_key)
    if ip_voted:
        try:
            vote_data = json.loads(ip_voted)
            vote_time = datetime.fromisoformat(vote_data["voted_at"])
            cooldown_end = vote_time + timedelta(days=VOTE_COOLDOWN_DAYS)
            if datetime.utcnow() < cooldown_end:
                days_left = (cooldown_end - datetime.utcnow()).days + 1
                return VoteCheck(
                    allowed=False,
                    reason=f"Vote recorded from your network. Try again in {days_left} days.",
                    risk_score=0,
                    risk_level=RiskLevel.LOW,
                )
        except (json.JSONDecodeError, KeyError):
            pass
    
    # === Velocity Check ===
    velocity_key = f"{CACHE_PREFIX}:velocity:{ip_hash}"
    velocity = await client.get(velocity_key)
    vote_velocity = int(velocity) if velocity else 0
    
    if vote_velocity >= MAX_VOTES_PER_HOUR:
        risk_score += 30
        flags.append(f"velocity_exceeded:{vote_velocity}")
    elif vote_velocity >= MAX_VOTES_PER_HOUR // 2:
        risk_score += 15
        flags.append(f"high_velocity:{vote_velocity}")
    
    # === Coordinated Voting Detection ===
    # Check how many votes for this stock from similar IPs
    stock_ip_key = f"{CACHE_PREFIX}:stock_ip:{symbol}:{ip_hash[:8]}"
    stock_votes = await client.get(stock_ip_key)
    stock_votes_count = int(stock_votes) if stock_votes else 0
    
    if stock_votes_count >= SUSPICIOUS_STOCK_VOTES_THRESHOLD:
        risk_score += 35
        flags.append(f"coordinated_voting:{stock_votes_count}")
    elif stock_votes_count >= 5:
        risk_score += 15
        flags.append(f"ip_cluster_voting:{stock_votes_count}")
    
    # === Client Fingerprint Checks ===
    if client_fp:
        # Check if this fingerprint has been used from many IPs
        fp_ips_key = f"{CACHE_PREFIX}:fp_ips:{_hash(client_fp)}"
        fp_ips = await client.smembers(fp_ips_key)
        if len(fp_ips) > MAX_IPS_PER_FINGERPRINT:
            risk_score += 15
            flags.append(f"fingerprint_ip_hopping:{len(fp_ips)}")
    else:
        # No client fingerprint (may be blocking scripts)
        risk_score += 5
        flags.append("no_client_fingerprint")
    
    # === Browser Signal Checks ===
    if not request.headers.get("Accept-Language"):
        risk_score += 10
        flags.append("missing_accept_language")
    
    user_agent = request.headers.get("User-Agent", "")
    if not user_agent or len(user_agent) < 20:
        risk_score += 5
        flags.append("suspicious_user_agent")
    
    # === Determine Risk Level ===
    risk_score = min(risk_score, 100)
    
    if risk_score >= 76:
        risk_level = RiskLevel.CRITICAL
    elif risk_score >= 51:
        risk_level = RiskLevel.HIGH
    elif risk_score >= 26:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW
    
    # === Decision ===
    if risk_level == RiskLevel.CRITICAL:
        # Log and block
        await _log_suspicious(request, symbol, risk_score, flags, "blocked")
        return VoteCheck(
            allowed=False,
            reason="Vote could not be processed. Please try again later.",
            risk_score=risk_score,
            risk_level=risk_level,
            flags=flags,
        )
    
    # Allow but maybe with reduced weight
    reduce_weight = risk_level == RiskLevel.HIGH
    if reduce_weight:
        await _log_suspicious(request, symbol, risk_score, flags, "reduced_weight")
    
    return VoteCheck(
        allowed=True,
        reason="OK",
        risk_score=risk_score,
        risk_level=risk_level,
        flags=flags,
        reduce_weight=reduce_weight,
    )


# =============================================================================
# Vote Recording
# =============================================================================

async def record_vote(
    request: Request,
    symbol: str,
) -> None:
    """
    Record that a vote was cast.
    
    Updates all tracking keys for future checks.
    """
    client = await get_valkey_client()
    symbol = symbol.upper()
    
    ip = get_client_ip(request)
    ip_hash = _hash(ip)
    client_fp = get_client_fingerprint(request)
    
    now = datetime.utcnow().isoformat()
    vote_data = json.dumps({"voted_at": now, "symbol": symbol})
    cooldown_seconds = VOTE_COOLDOWN_DAYS * 86400
    
    # Record vote by composite identifier
    vote_id = get_vote_identifier(request, symbol)
    cooldown_key = f"{CACHE_PREFIX}:vote:{vote_id}"
    await client.set(cooldown_key, vote_data, ex=cooldown_seconds)
    
    # Record vote by IP (backup)
    ip_vote_key = f"{CACHE_PREFIX}:ip_vote:{ip_hash}:{symbol}"
    await client.set(ip_vote_key, vote_data, ex=cooldown_seconds)
    
    # Increment velocity counter (1 hour TTL)
    velocity_key = f"{CACHE_PREFIX}:velocity:{ip_hash}"
    await client.incr(velocity_key)
    await client.expire(velocity_key, 3600)
    
    # Track stock-IP correlation (24 hour TTL)
    stock_ip_key = f"{CACHE_PREFIX}:stock_ip:{symbol}:{ip_hash[:8]}"
    await client.incr(stock_ip_key)
    await client.expire(stock_ip_key, 86400)
    
    # Track IPs per client fingerprint
    if client_fp:
        fp_ips_key = f"{CACHE_PREFIX}:fp_ips:{_hash(client_fp)}"
        await client.sadd(fp_ips_key, ip_hash)
        await client.expire(fp_ips_key, SESSION_TTL_DAYS * 86400)
    
    logger.debug(f"Recorded vote: {symbol} from IP {ip_hash[:8]}...")


# =============================================================================
# Suggestion Rate Limiting
# =============================================================================

async def check_can_suggest(request: Request) -> tuple[bool, str]:
    """
    Check if this request can submit a new suggestion.
    
    Limit: 3 suggestions per day per IP.
    """
    client = await get_valkey_client()
    ip_hash = _hash(get_client_ip(request))
    
    suggest_key = f"{CACHE_PREFIX}:suggest_count:{ip_hash}"
    count = await client.get(suggest_key)
    current = int(count) if count else 0
    
    if current >= 3:
        return False, "You can only suggest 3 stocks per day. Try again tomorrow."
    
    return True, "OK"


async def record_suggestion(request: Request) -> None:
    """Record that a suggestion was made."""
    client = await get_valkey_client()
    ip_hash = _hash(get_client_ip(request))
    
    suggest_key = f"{CACHE_PREFIX}:suggest_count:{ip_hash}"
    await client.incr(suggest_key)
    await client.expire(suggest_key, 86400)


# =============================================================================
# Suspicious Activity Logging
# =============================================================================

async def _log_suspicious(
    request: Request,
    symbol: str,
    score: int,
    flags: list[str],
    action: str,
) -> None:
    """Log suspicious vote attempt."""
    client = await get_valkey_client()
    
    ip_hash = _hash(get_client_ip(request))
    server_fp = get_server_fingerprint(request)
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "ip_hash": ip_hash,
        "server_fp": server_fp[:16],
        "score": score,
        "flags": flags,
        "action": action,
    }
    
    log_key = f"{CACHE_PREFIX}:suspicious_log"
    await client.lpush(log_key, json.dumps(entry))
    await client.ltrim(log_key, 0, 999)
    
    logger.warning(
        f"Suspicious vote: score={score} symbol={symbol} "
        f"ip={ip_hash[:8]}... action={action} flags={flags}"
    )


async def get_suspicious_log(limit: int = 100) -> list[dict]:
    """Get recent suspicious vote log entries."""
    client = await get_valkey_client()
    log_key = f"{CACHE_PREFIX}:suspicious_log"
    entries = await client.lrange(log_key, 0, limit - 1)
    return [json.loads(e) for e in entries]


# =============================================================================
# Voted Symbols Tracking (for UI filtering)
# =============================================================================

async def get_voted_symbols(request: Request) -> set[str]:
    """Get all symbols this user has voted on recently."""
    client = await get_valkey_client()
    ip_hash = _hash(get_client_ip(request))
    
    # Scan for IP-based vote keys
    pattern = f"{CACHE_PREFIX}:ip_vote:{ip_hash}:*"
    voted = set()
    
    async for key in client.scan_iter(match=pattern):
        # Extract symbol from key
        parts = key.split(":")
        if len(parts) >= 5:
            voted.add(parts[-1])
    
    return voted
