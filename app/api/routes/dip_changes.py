"""Dip changes API endpoint with rate limiting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from hashlib import sha256

from fastapi import APIRouter, HTTPException, Query, Request, Header
from pydantic import BaseModel

from app.repositories import dip_history
from app.repositories import user_api_keys
from app.cache.cache import cache_manager
from app.core.logging import get_logger

logger = get_logger("dip_changes_api")
router = APIRouter(prefix="/dips", tags=["dip-changes"])


# ============================================================================
# Schemas
# ============================================================================


class DipChange(BaseModel):
    """A single dip change record."""

    symbol: str
    action: str  # 'added', 'removed', 'updated'
    current_price: Optional[float] = None
    ath_price: Optional[float] = None
    dip_percentage: Optional[float] = None
    recorded_at: Optional[str] = None


class DipChangesResponse(BaseModel):
    """Response for dip changes endpoint."""

    changes: list[DipChange]
    summary: dict
    rate_limit: dict


class DipChangesSummary(BaseModel):
    """Summary of dip changes."""

    hours: int
    since: str
    added: int
    removed: int
    updated: int
    unique_symbols: int


# ============================================================================
# Rate Limiting
# ============================================================================

FREE_RATE_LIMIT_SECONDS = 3600  # 1 hour for free users
RATE_LIMIT_KEY_PREFIX = "rate:dip_changes:"


def get_client_identifier(request: Request) -> str:
    """Get a unique identifier for the client (IP-based)."""
    # Get real IP from headers (behind proxy) or fallback to client host
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"

    # Hash the IP for privacy
    return sha256(ip.encode()).hexdigest()[:16]


async def check_rate_limit(
    request: Request,
    api_key: Optional[str] = None,
) -> dict:
    """
    Check rate limit for the request.

    Returns:
        Dict with rate limit info: {allowed, remaining, reset_at, is_premium}
    """
    # If API key provided, validate and check if it bypasses rate limits
    if api_key:
        key_data = await user_api_keys.validate_api_key(api_key)
        if key_data and key_data.get("rate_limit_bypass"):
            return {
                "allowed": True,
                "remaining": -1,  # Unlimited
                "reset_at": None,
                "is_premium": True,
                "vote_weight": key_data.get("vote_weight", 10),
            }

    # Free user rate limiting
    client_id = get_client_identifier(request)
    rate_key = f"{RATE_LIMIT_KEY_PREFIX}{client_id}"

    try:
        # Check if rate limited
        last_request = await cache_manager.get(rate_key)

        if last_request:
            last_time = datetime.fromisoformat(last_request)
            elapsed = (datetime.now(timezone.utc) - last_time.replace(tzinfo=timezone.utc)).total_seconds()
            remaining = FREE_RATE_LIMIT_SECONDS - elapsed

            if remaining > 0:
                reset_at = last_time.timestamp() + FREE_RATE_LIMIT_SECONDS
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_at": datetime.fromtimestamp(reset_at).isoformat(),
                    "is_premium": False,
                    "retry_after_seconds": int(remaining),
                }

        # Set rate limit
        await cache_manager.set(
            rate_key, datetime.now(timezone.utc).isoformat(), ttl=FREE_RATE_LIMIT_SECONDS
        )

        return {
            "allowed": True,
            "remaining": 0,
            "reset_at": (datetime.now(timezone.utc).timestamp() + FREE_RATE_LIMIT_SECONDS),
            "is_premium": False,
        }

    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")
        # Allow request on cache failure
        return {"allowed": True, "remaining": 1, "reset_at": None, "is_premium": False}


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/changes", response_model=DipChangesResponse)
async def get_dip_changes(
    request: Request,
    hours: int = Query(
        default=24, ge=1, le=168, description="Hours to look back (max 168 = 1 week)"
    ),
    action: Optional[str] = Query(default=None, pattern="^(added|removed|updated)$"),
    limit: int = Query(default=100, ge=1, le=500),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    Get dip changes in the last X hours.

    Rate limited to 1 request per hour for free users.
    Premium API key holders bypass rate limits.

    Use X-API-Key header for authenticated requests.
    """
    # Check rate limit
    rate_info = await check_rate_limit(request, x_api_key)

    if not rate_info["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "retry_after_seconds": rate_info.get("retry_after_seconds", 3600),
                "reset_at": rate_info.get("reset_at"),
                "message": "Free tier allows 1 request per hour. Get an API key for unlimited access.",
            },
            headers={"Retry-After": str(rate_info.get("retry_after_seconds", 3600))},
        )

    # Get changes
    changes = await dip_history.get_dip_changes(
        hours=hours,
        action=action,
        limit=limit,
    )

    # Get summary
    summary = await dip_history.get_dip_changes_summary(hours=hours)

    return DipChangesResponse(
        changes=[DipChange(**c) for c in changes],
        summary=summary,
        rate_limit={
            "is_premium": rate_info.get("is_premium", False),
            "remaining": rate_info.get("remaining"),
            "reset_at": rate_info.get("reset_at"),
        },
    )


@router.get("/changes/summary")
async def get_changes_summary(
    request: Request,
    hours: int = Query(default=24, ge=1, le=168),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    Get a summary of dip changes without the full list.

    Same rate limiting as /changes endpoint.
    """
    # Check rate limit
    rate_info = await check_rate_limit(request, x_api_key)

    if not rate_info["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "retry_after_seconds": rate_info.get("retry_after_seconds", 3600),
            },
        )

    summary = await dip_history.get_dip_changes_summary(hours=hours)

    return {
        "summary": summary,
        "rate_limit": {
            "is_premium": rate_info.get("is_premium", False),
        },
    }


@router.get("/changes/{symbol}/history")
async def get_symbol_change_history(
    symbol: str,
    request: Request,
    days: int = Query(default=30, ge=1, le=90),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    Get the change history for a specific symbol.

    Shows when the symbol was added/removed from the dip list.
    """
    # Check rate limit
    rate_info = await check_rate_limit(request, x_api_key)

    if not rate_info["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={"error": "Rate limit exceeded"},
        )

    history = await dip_history.get_symbol_history(symbol, days=days)

    return {
        "symbol": symbol.upper(),
        "days": days,
        "history": history,
        "count": len(history),
    }
