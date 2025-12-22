"""Token blacklist for JWT revocation using Valkey.

This module provides a mechanism to revoke JWT tokens before their natural expiration.
Tokens are added to a blacklist when:
- User explicitly logs out
- Admin revokes a user's sessions
- Security incident requires immediate token invalidation

The blacklist uses Valkey with automatic expiration matching the token's TTL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

from .client import get_valkey_client

logger = get_logger("cache.token_blacklist")

BLACKLIST_PREFIX = "stonkmarket:token_blacklist"


async def blacklist_token(jti: str, exp: datetime) -> bool:
    """
    Add a token to the blacklist.

    Args:
        jti: The unique token identifier (JWT ID)
        exp: The token's expiration datetime

    Returns:
        True if successfully blacklisted
    """
    try:
        client = await get_valkey_client()
        key = f"{BLACKLIST_PREFIX}:{jti}"

        # Calculate TTL - only need to keep until token expires naturally
        now = datetime.now(timezone.utc)
        ttl_seconds = int((exp - now).total_seconds())

        if ttl_seconds <= 0:
            # Token already expired, no need to blacklist
            return True

        # Store with expiration
        await client.set(key, "revoked", ex=ttl_seconds)
        logger.info(f"Token blacklisted: {jti[:8]}... (expires in {ttl_seconds}s)")
        return True

    except Exception as e:
        logger.error(f"Failed to blacklist token: {e}")
        return False


async def is_token_blacklisted(jti: str) -> bool:
    """
    Check if a token is blacklisted.

    Args:
        jti: The unique token identifier (JWT ID)

    Returns:
        True if token is blacklisted (revoked)
    """
    try:
        client = await get_valkey_client()
        key = f"{BLACKLIST_PREFIX}:{jti}"
        result = await client.exists(key)
        return bool(result)

    except Exception as e:
        logger.error(f"Failed to check token blacklist: {e}")
        # Fail open - if we can't check, assume not blacklisted
        # This is a tradeoff; fail closed would be more secure but could lock out users
        return False


async def blacklist_user_tokens(
    username: str, before: Optional[datetime] = None
) -> int:
    """
    Blacklist all tokens for a user issued before a given time.

    This uses a different mechanism - storing the "tokens invalid before" timestamp.
    When validating, we check if the token's iat is before this timestamp.

    Args:
        username: The username whose tokens to invalidate
        before: Invalidate tokens issued before this time (default: now)

    Returns:
        1 if successful, 0 if failed
    """
    try:
        client = await get_valkey_client()
        key = f"{BLACKLIST_PREFIX}:user:{username}"

        timestamp = before or datetime.now(timezone.utc)

        # Store for the maximum token lifetime
        ttl = settings.access_token_expire_minutes * 60
        await client.set(key, timestamp.isoformat(), ex=ttl)

        logger.info(
            f"All tokens for user '{username}' issued before {timestamp.isoformat()} are now invalid"
        )
        return 1

    except Exception as e:
        logger.error(f"Failed to blacklist user tokens: {e}")
        return 0


async def get_user_token_invalidation_time(username: str) -> Optional[datetime]:
    """
    Get the timestamp before which all tokens for a user are invalid.

    Args:
        username: The username to check

    Returns:
        Datetime before which tokens are invalid, or None if no restriction
    """
    try:
        client = await get_valkey_client()
        key = f"{BLACKLIST_PREFIX}:user:{username}"
        result = await client.get(key)

        if result:
            return datetime.fromisoformat(result)
        return None

    except Exception as e:
        logger.error(f"Failed to get user token invalidation time: {e}")
        return None
