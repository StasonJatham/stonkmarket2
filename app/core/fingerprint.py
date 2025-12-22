"""
Request fingerprinting for abuse prevention.

Combines multiple signals to create a robust client identifier that's
harder to spoof than IP alone. Used for rate limiting and vote deduplication.
"""

from __future__ import annotations

import hashlib

from fastapi import Request

from app.core.logging import get_logger

logger = get_logger("core.fingerprint")


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request, respecting proxy headers.

    Checks headers in order:
    1. CF-Connecting-IP (Cloudflare)
    2. X-Forwarded-For (generic proxy)
    3. X-Real-IP (nginx)
    4. Direct connection
    """
    # Cloudflare
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # Generic forwarded header
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP (original client)
        return forwarded.split(",")[0].strip()

    # Nginx real IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Direct connection
    if request.client:
        return request.client.host

    return "unknown"


def get_browser_fingerprint(request: Request) -> str:
    """
    Create a fingerprint hash from browser-specific headers.

    Combines multiple headers that together create a somewhat unique
    browser signature. Not perfect, but adds a layer of difficulty
    for users trying to game the voting system.
    """
    # Headers to include in fingerprint
    # These are relatively stable for a browser session
    fingerprint_headers = [
        "User-Agent",
        "Accept-Language",
        "Accept-Encoding",
        "Accept",
        "Sec-CH-UA",  # Client hints (Chrome)
        "Sec-CH-UA-Mobile",
        "Sec-CH-UA-Platform",
        "DNT",  # Do Not Track
        "Sec-Fetch-Site",
        "Sec-Fetch-Mode",
    ]

    # Collect header values
    parts = []
    for header in fingerprint_headers:
        value = request.headers.get(header, "")
        parts.append(f"{header}:{value}")

    # Join and hash
    fingerprint_string = "|".join(parts)
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:32]


def get_request_fingerprint(
    request: Request,
    include_ip: bool = True,
    include_headers: bool = True,
) -> str:
    """
    Create a composite fingerprint for the request.

    Combines:
    - Client IP address (most reliable)
    - Browser fingerprint from headers (adds difficulty to spoof)

    Returns a hash that can be used for rate limiting and vote deduplication.

    Note: This is not foolproof. VPNs, different browsers, incognito mode,
    etc. can all bypass this. The goal is to make casual abuse harder,
    not to prevent determined attackers.
    """
    parts = []

    if include_ip:
        parts.append(f"ip:{get_client_ip(request)}")

    if include_headers:
        parts.append(f"browser:{get_browser_fingerprint(request)}")

    # If we have nothing, fall back to something
    if not parts:
        parts.append(f"ip:{get_client_ip(request)}")

    fingerprint_string = "|".join(parts)

    # Return a hash for privacy (don't store raw IPs in votes table)
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:40]


def get_vote_identifier(request: Request, symbol: str) -> str:
    """
    Create a unique identifier for a vote on a specific symbol.

    This ties the fingerprint to a specific symbol so we can track
    unique votes per symbol, not just global rate limits.
    """
    fingerprint = get_request_fingerprint(request)
    # Include symbol in hash so same user can vote for different stocks
    vote_string = f"{fingerprint}:vote:{symbol.upper()}"
    return hashlib.sha256(vote_string.encode()).hexdigest()[:40]


def get_suggestion_identifier(request: Request) -> str:
    """
    Create an identifier for suggestion rate limiting.

    Uses the full fingerprint to rate limit how often someone can
    suggest new stocks.
    """
    return get_request_fingerprint(request)
