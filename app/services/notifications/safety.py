"""Safety checks for notification sending.

Implements:
- Rate limiting (50 notifications/hour/user)
- Staleness protection (skip if data too old)
- Content deduplication (avoid identical messages)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from app.cache.client import get_valkey_client
from app.core.logging import get_logger
from app.repositories.notifications_orm import check_recent_hash


logger = get_logger("notifications.safety")

# Rate limit settings
RATE_LIMIT_KEY_PREFIX = "notify:rate"
RATE_LIMIT_MAX = 50  # Max notifications per hour
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds

# Staleness settings
MAX_DATA_AGE_HOURS = 24  # Don't alert on data older than this


def _get_rate_key(user_id: int) -> str:
    """Build Redis key for user rate limit."""
    return f"{RATE_LIMIT_KEY_PREFIX}:{user_id}"


async def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Check if user has exceeded notification rate limit.
    
    Args:
        user_id: The user ID
        
    Returns:
        Tuple of (is_limited, current_count)
        is_limited=True means should NOT send
    """
    client = await get_valkey_client()
    if not client:
        # No cache = no rate limiting (fail open)
        return False, 0
    
    key = _get_rate_key(user_id)
    count = await client.get(key)
    
    if count is None:
        return False, 0
    
    current = int(count)
    return current >= RATE_LIMIT_MAX, current


async def increment_rate_limit(user_id: int) -> int:
    """Increment the notification count for rate limiting.
    
    Args:
        user_id: The user ID
        
    Returns:
        New count after increment
    """
    client = await get_valkey_client()
    if not client:
        return 0
    
    key = _get_rate_key(user_id)
    
    # Use pipeline for atomic incr + expire
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, RATE_LIMIT_WINDOW)
    results = await pipe.execute()
    
    new_count = results[0] if results else 0
    
    if new_count >= RATE_LIMIT_MAX:
        logger.warning(
            "User rate limit reached",
            extra={
                "user_id": user_id,
                "count": new_count,
                "limit": RATE_LIMIT_MAX,
            }
        )
    
    return new_count


async def get_rate_limit_status(user_id: int) -> dict:
    """Get current rate limit status for a user.
    
    Args:
        user_id: The user ID
        
    Returns:
        Dict with count, limit, remaining, and reset_in
    """
    client = await get_valkey_client()
    if not client:
        return {
            "count": 0,
            "limit": RATE_LIMIT_MAX,
            "remaining": RATE_LIMIT_MAX,
            "reset_in": None,
        }
    
    key = _get_rate_key(user_id)
    
    pipe = client.pipeline()
    pipe.get(key)
    pipe.ttl(key)
    results = await pipe.execute()
    
    count = int(results[0]) if results[0] else 0
    ttl = results[1] if results[1] and results[1] > 0 else None
    
    return {
        "count": count,
        "limit": RATE_LIMIT_MAX,
        "remaining": max(0, RATE_LIMIT_MAX - count),
        "reset_in": ttl,
    }


def check_staleness(
    data_timestamp: datetime | None,
    max_age_hours: int = MAX_DATA_AGE_HOURS,
) -> tuple[bool, float | None]:
    """Check if data is too stale to trigger notifications.
    
    Args:
        data_timestamp: When the data was last updated
        max_age_hours: Maximum acceptable age
        
    Returns:
        Tuple of (is_stale, age_in_hours)
        is_stale=True means should NOT alert
    """
    if data_timestamp is None:
        # No timestamp = stale (be conservative)
        return True, None
    
    # Ensure timezone-aware comparison
    if data_timestamp.tzinfo is None:
        data_timestamp = data_timestamp.replace(tzinfo=UTC)
    
    now = datetime.now(UTC)
    age = now - data_timestamp
    age_hours = age.total_seconds() / 3600
    
    return age_hours > max_age_hours, age_hours


async def check_duplicate(
    title: str,
    body: str,
    user_id: int,
    hours: int = 24,
) -> tuple[bool, str]:
    """Check if this notification was recently sent (deduplication).
    
    Args:
        title: Notification title
        body: Notification body
        user_id: The user ID
        hours: How far back to check
        
    Returns:
        Tuple of (is_duplicate, content_hash)
        is_duplicate=True means should NOT send
    """
    # Generate content hash
    content = f"{title}:{body}".encode()
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Check if recently sent
    is_dup = await check_recent_hash(content_hash, user_id, hours)
    
    if is_dup:
        logger.debug(
            "Duplicate notification detected",
            extra={
                "user_id": user_id,
                "hash": content_hash[:16],
            }
        )
    
    return is_dup, content_hash


def compute_content_hash(title: str, body: str) -> str:
    """Compute SHA-256 hash of notification content.
    
    Args:
        title: Notification title
        body: Notification body
        
    Returns:
        Hex digest of the hash
    """
    content = f"{title}:{body}".encode()
    return hashlib.sha256(content).hexdigest()
