"""
Anonymous session management for abuse prevention.

Uses Valkey (Redis) to track anonymous user sessions and their votes.
Sessions are identified by a client-generated device fingerprint that's
stored across multiple browser storage mechanisms (cookies, localStorage,
sessionStorage, IndexedDB).

The session tracks:
- Device fingerprint (client-generated, stored in browser)
- IP addresses seen (for detecting fingerprint sharing)
- Votes cast (to prevent duplicate voting)
- Created/last seen timestamps
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, Any

from fastapi import Request

from app.cache.client import get_valkey_client
from app.core.logging import get_logger

logger = get_logger("core.anon_session")

# Session configuration
ANON_SESSION_PREFIX = "stonkmarket:v1:anon_session"
VOTE_COOLDOWN_DAYS = 7
SESSION_TTL_DAYS = 365  # Sessions expire after 1 year of inactivity
MAX_IPS_PER_SESSION = 5  # Flag suspicious if more than 5 IPs use same fingerprint


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, respecting proxy headers."""
    # Cloudflare
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # Generic forwarded header
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Nginx real IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Direct connection
    if request.client:
        return request.client.host

    return "unknown"


def _hash_ip(ip: str) -> str:
    """Hash IP for privacy - we don't need to store raw IPs."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _session_key(device_id: str) -> str:
    """Generate Valkey key for a session."""
    # Hash the device ID for consistent key length
    hashed = hashlib.sha256(device_id.encode()).hexdigest()[:32]
    return f"{ANON_SESSION_PREFIX}:{hashed}"


def _vote_key(device_id: str, symbol: str) -> str:
    """Generate Valkey key for a vote record."""
    hashed = hashlib.sha256(device_id.encode()).hexdigest()[:32]
    return f"{ANON_SESSION_PREFIX}:{hashed}:vote:{symbol.upper()}"


def _ip_vote_key(ip_hash: str, symbol: str) -> str:
    """Generate Valkey key for IP-based vote tracking."""
    return f"{ANON_SESSION_PREFIX}:ip:{ip_hash}:vote:{symbol.upper()}"


async def get_or_create_session(
    device_id: str,
    request: Request,
) -> Dict[str, Any]:
    """
    Get existing session or create new one.
    
    Returns session data with:
    - device_id: The client fingerprint
    - ip_hashes: Set of IP hashes that have used this session
    - created_at: When session was first created
    - last_seen: Last activity timestamp
    - suspicious: Whether session has suspicious activity
    """
    client = await get_valkey_client()
    key = _session_key(device_id)
    ip_hash = _hash_ip(_get_client_ip(request))
    
    # Try to get existing session
    session_data = await client.get(key)
    
    if session_data:
        session = json.loads(session_data)
        
        # Update IP hashes
        ip_hashes = set(session.get("ip_hashes", []))
        ip_hashes.add(ip_hash)
        session["ip_hashes"] = list(ip_hashes)
        
        # Check for suspicious activity (too many IPs)
        if len(ip_hashes) > MAX_IPS_PER_SESSION:
            session["suspicious"] = True
            logger.warning(
                f"Suspicious session: device {device_id[:8]}... used from {len(ip_hashes)} IPs"
            )
        
        # Update last seen
        session["last_seen"] = datetime.utcnow().isoformat()
        
        # Save updated session
        await client.set(
            key,
            json.dumps(session),
            ex=SESSION_TTL_DAYS * 86400,
        )
        
        return session
    
    # Create new session
    session = {
        "device_id": device_id,
        "ip_hashes": [ip_hash],
        "created_at": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat(),
        "suspicious": False,
    }
    
    await client.set(
        key,
        json.dumps(session),
        ex=SESSION_TTL_DAYS * 86400,
    )
    
    logger.debug(f"Created anon session for device {device_id[:8]}...")
    return session


async def check_can_vote(
    device_id: str,
    symbol: str,
    request: Request,
) -> tuple[bool, str]:
    """
    Check if this device/IP can vote on a symbol.
    
    Returns (can_vote, reason).
    
    Checks:
    1. Device fingerprint hasn't voted in cooldown period
    2. IP address hasn't voted in cooldown period
    3. Session isn't flagged as suspicious
    """
    client = await get_valkey_client()
    symbol = symbol.upper()
    ip_hash = _hash_ip(_get_client_ip(request))
    
    # Check device-based vote
    device_vote_key = _vote_key(device_id, symbol)
    device_vote = await client.get(device_vote_key)
    if device_vote:
        vote_data = json.loads(device_vote)
        vote_time = datetime.fromisoformat(vote_data["voted_at"])
        cooldown_end = vote_time + timedelta(days=VOTE_COOLDOWN_DAYS)
        if datetime.utcnow() < cooldown_end:
            days_left = (cooldown_end - datetime.utcnow()).days + 1
            return False, f"You already voted for this stock. Try again in {days_left} days."
    
    # Check IP-based vote
    ip_vote_key = _ip_vote_key(ip_hash, symbol)
    ip_vote = await client.get(ip_vote_key)
    if ip_vote:
        vote_data = json.loads(ip_vote)
        vote_time = datetime.fromisoformat(vote_data["voted_at"])
        cooldown_end = vote_time + timedelta(days=VOTE_COOLDOWN_DAYS)
        if datetime.utcnow() < cooldown_end:
            days_left = (cooldown_end - datetime.utcnow()).days + 1
            return False, f"A vote was already recorded from your network. Try again in {days_left} days."
    
    # Check session suspicion level
    session = await get_or_create_session(device_id, request)
    if session.get("suspicious"):
        # Don't block, but log for manual review
        logger.warning(
            f"Vote from suspicious session: device {device_id[:8]}... on {symbol}"
        )
    
    return True, "OK"


async def record_vote(
    device_id: str,
    symbol: str,
    request: Request,
) -> None:
    """
    Record that this device/IP voted on a symbol.
    
    Sets vote records in Valkey with cooldown TTL.
    """
    client = await get_valkey_client()
    symbol = symbol.upper()
    ip_hash = _hash_ip(_get_client_ip(request))
    now = datetime.utcnow().isoformat()
    
    vote_data = json.dumps({
        "voted_at": now,
        "symbol": symbol,
    })
    
    # Record device vote
    device_vote_key = _vote_key(device_id, symbol)
    await client.set(
        device_vote_key,
        vote_data,
        ex=VOTE_COOLDOWN_DAYS * 86400,
    )
    
    # Record IP vote
    ip_vote_key = _ip_vote_key(ip_hash, symbol)
    await client.set(
        ip_vote_key,
        vote_data,
        ex=VOTE_COOLDOWN_DAYS * 86400,
    )
    
    # Update session
    await get_or_create_session(device_id, request)
    
    logger.debug(f"Recorded vote: device {device_id[:8]}... on {symbol}")


async def check_can_suggest(
    device_id: str,
    request: Request,
) -> tuple[bool, str]:
    """
    Check if this device/IP can suggest a new stock.
    
    Rate limit: 3 suggestions per day per device/IP.
    """
    client = await get_valkey_client()
    ip_hash = _hash_ip(_get_client_ip(request))
    
    # Check device suggestion count
    device_key = f"{ANON_SESSION_PREFIX}:{hashlib.sha256(device_id.encode()).hexdigest()[:32]}:suggest_count"
    device_count = await client.get(device_key)
    if device_count and int(device_count) >= 3:
        return False, "You can only suggest 3 stocks per day. Try again tomorrow."
    
    # Check IP suggestion count
    ip_key = f"{ANON_SESSION_PREFIX}:ip:{ip_hash}:suggest_count"
    ip_count = await client.get(ip_key)
    if ip_count and int(ip_count) >= 3:
        return False, "Too many suggestions from your network. Try again tomorrow."
    
    return True, "OK"


async def record_suggestion(
    device_id: str,
    request: Request,
) -> None:
    """
    Record that this device/IP made a suggestion.
    
    Increments daily counter with 24h TTL.
    """
    client = await get_valkey_client()
    ip_hash = _hash_ip(_get_client_ip(request))
    
    # Increment device suggestion count
    device_key = f"{ANON_SESSION_PREFIX}:{hashlib.sha256(device_id.encode()).hexdigest()[:32]}:suggest_count"
    await client.incr(device_key)
    await client.expire(device_key, 86400)  # 24h TTL
    
    # Increment IP suggestion count
    ip_key = f"{ANON_SESSION_PREFIX}:ip:{ip_hash}:suggest_count"
    await client.incr(ip_key)
    await client.expire(ip_key, 86400)  # 24h TTL
    
    logger.debug(f"Recorded suggestion: device {device_id[:8]}...")


async def get_voted_symbols(device_id: str, request: Request) -> Set[str]:
    """
    Get all symbols this device has voted on (within cooldown period).
    
    Used to filter out already-voted suggestions in the UI.
    """
    client = await get_valkey_client()
    ip_hash = _hash_ip(_get_client_ip(request))
    
    # Get all vote keys for this device
    device_prefix = f"{ANON_SESSION_PREFIX}:{hashlib.sha256(device_id.encode()).hexdigest()[:32]}:vote:*"
    ip_prefix = f"{ANON_SESSION_PREFIX}:ip:{ip_hash}:vote:*"
    
    voted = set()
    
    # Scan device votes
    async for key in client.scan_iter(match=device_prefix):
        # Extract symbol from key
        parts = key.split(":vote:")
        if len(parts) == 2:
            voted.add(parts[1])
    
    # Scan IP votes
    async for key in client.scan_iter(match=ip_prefix):
        parts = key.split(":vote:")
        if len(parts) == 2:
            voted.add(parts[1])
    
    return voted


def get_device_fingerprint(request: Request) -> Optional[str]:
    """
    Get device fingerprint from request header.
    
    The client sends this via X-Client-Fingerprint header.
    Returns None if not provided.
    """
    return request.headers.get("X-Client-Fingerprint")
