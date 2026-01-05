"""Cooldown management for notification rules.

Uses Redis/Valkey to track cooldown periods and prevent notification spam.

Key structure:
    notify:cooldown:{user_id}:{rule_id} -> timestamp when cooldown expires
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.cache.client import get_valkey_client
from app.core.logging import get_logger


logger = get_logger("notifications.cooldown")

COOLDOWN_KEY_PREFIX = "notify:cooldown"


def _get_cooldown_key(user_id: int, rule_id: int) -> str:
    """Build Redis key for a rule's cooldown."""
    return f"{COOLDOWN_KEY_PREFIX}:{user_id}:{rule_id}"


async def check_cooldown(user_id: int, rule_id: int) -> bool:
    """Check if a rule is currently in cooldown.
    
    Args:
        user_id: The user ID
        rule_id: The rule ID
        
    Returns:
        True if still in cooldown (should NOT send), False if ready to send
    """
    client = await get_valkey_client()
    if not client:
        # No cache = no cooldown enforcement (fail open for notifications)
        return False
    
    key = _get_cooldown_key(user_id, rule_id)
    value = await client.get(key)
    
    if value is None:
        return False
    
    # Check if cooldown has expired (shouldn't happen with TTL, but be safe)
    try:
        expires_at = datetime.fromisoformat(value.decode())
        if expires_at <= datetime.now(UTC):
            await client.delete(key)
            return False
        return True
    except (ValueError, AttributeError):
        # Invalid value, delete and allow
        await client.delete(key)
        return False


async def set_cooldown(user_id: int, rule_id: int, cooldown_minutes: int) -> None:
    """Set cooldown for a rule after sending a notification.
    
    Args:
        user_id: The user ID
        rule_id: The rule ID
        cooldown_minutes: How long to wait before next notification
    """
    if cooldown_minutes <= 0:
        return
    
    client = await get_valkey_client()
    if not client:
        logger.warning("No cache client, skipping cooldown set")
        return
    
    key = _get_cooldown_key(user_id, rule_id)
    expires_at = datetime.now(UTC) + timedelta(minutes=cooldown_minutes)
    
    # Set with TTL so it auto-expires
    await client.setex(
        key,
        cooldown_minutes * 60,  # TTL in seconds
        expires_at.isoformat(),
    )
    
    logger.debug(
        "Set cooldown",
        extra={
            "user_id": user_id,
            "rule_id": rule_id,
            "cooldown_minutes": cooldown_minutes,
            "expires_at": expires_at.isoformat(),
        }
    )


async def clear_cooldown(user_id: int, rule_id: int) -> bool:
    """Clear cooldown for a rule (manual reset).
    
    Args:
        user_id: The user ID
        rule_id: The rule ID
        
    Returns:
        True if a cooldown was cleared, False if none existed
    """
    client = await get_valkey_client()
    if not client:
        return False
    
    key = _get_cooldown_key(user_id, rule_id)
    deleted = await client.delete(key)
    return deleted > 0


async def get_remaining_cooldown(user_id: int, rule_id: int) -> int | None:
    """Get remaining cooldown time in seconds.
    
    Args:
        user_id: The user ID
        rule_id: The rule ID
        
    Returns:
        Seconds remaining, or None if not in cooldown
    """
    client = await get_valkey_client()
    if not client:
        return None
    
    key = _get_cooldown_key(user_id, rule_id)
    value = await client.get(key)
    
    if value is None:
        return None
    
    try:
        expires_at = datetime.fromisoformat(value.decode())
        remaining = (expires_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            await client.delete(key)
            return None
        return int(remaining)
    except (ValueError, AttributeError):
        await client.delete(key)
        return None


async def clear_all_cooldowns_for_user(user_id: int) -> int:
    """Clear all cooldowns for a user (e.g., when testing).
    
    Args:
        user_id: The user ID
        
    Returns:
        Number of cooldowns cleared
    """
    client = await get_valkey_client()
    if not client:
        return 0
    
    # Use SCAN to find all keys for this user
    pattern = f"{COOLDOWN_KEY_PREFIX}:{user_id}:*"
    cursor = 0
    deleted = 0
    
    while True:
        cursor, keys = await client.scan(cursor, match=pattern)
        if keys:
            deleted += await client.delete(*keys)
        if cursor == 0:
            break
    
    return deleted
