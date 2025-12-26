"""User API key repository using SQLAlchemy ORM.

Handles public API access keys for external integrations.
"""

from __future__ import annotations

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, func, delete

from app.database.connection import get_session
from app.database.orm import UserApiKey
from app.core.logging import get_logger

logger = get_logger("repositories.user_api_keys")


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, key_prefix)
        The full_key should be shown to the user once and never stored.
    """
    # Generate a secure random key with prefix for identification
    prefix = "sm_"  # stonkmarket prefix
    random_part = secrets.token_urlsafe(32)
    full_key = f"{prefix}{random_part}"

    # Hash the key for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    # Store first 8 chars as prefix for easy identification
    key_prefix = full_key[:11]  # sm_ + first 8 chars

    return full_key, key_hash, key_prefix


def hash_api_key(key: str) -> str:
    """Hash an API key for lookup."""
    return hashlib.sha256(key.encode()).hexdigest()


async def create_user_api_key(
    name: str,
    description: Optional[str] = None,
    user_id: Optional[int] = None,
    vote_weight: int = 10,
    rate_limit_bypass: bool = True,
    expires_days: Optional[int] = None,
) -> tuple[str, dict]:
    """
    Create a new user API key.

    Returns:
        Tuple of (full_key, key_record)
        The full_key is only returned once and should be given to the user.
    """
    full_key, key_hash, key_prefix = generate_api_key()
    now = datetime.now(timezone.utc)
    expires_at = None
    if expires_days:
        expires_at = now + timedelta(days=expires_days)

    async with get_session() as session:
        api_key = UserApiKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            description=description,
            user_id=user_id,
            vote_weight=vote_weight,
            rate_limit_bypass=rate_limit_bypass,
            is_active=True,
            created_at=now,
            expires_at=expires_at,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)
        
        key_record = {
            "id": api_key.id,
            "key_prefix": api_key.key_prefix,
            "name": api_key.name,
            "description": api_key.description,
            "vote_weight": api_key.vote_weight,
            "rate_limit_bypass": api_key.rate_limit_bypass,
            "is_active": api_key.is_active,
            "created_at": api_key.created_at,
            "expires_at": api_key.expires_at,
        }

    logger.info(f"Created user API key: {key_prefix}*** for '{name}'")
    return full_key, key_record


async def validate_api_key(key: str) -> Optional[dict]:
    """
    Validate an API key and return its details if valid.

    Returns:
        Key details dict if valid, None if invalid/expired/inactive.
    """
    key_hash = hash_api_key(key)
    now = datetime.now(timezone.utc)

    async with get_session() as session:
        result = await session.execute(
            select(UserApiKey)
            .where(
                UserApiKey.key_hash == key_hash,
                UserApiKey.is_active == True,
            )
            .with_for_update()
        )
        api_key = result.scalar_one_or_none()

        if api_key:
            # Check expiration
            if api_key.expires_at and api_key.expires_at <= now:
                return None
            
            # Update last used and usage count atomically
            api_key.last_used_at = now
            api_key.usage_count = (api_key.usage_count or 0) + 1
            await session.commit()

            return {
                "id": api_key.id,
                "key_prefix": api_key.key_prefix,
                "name": api_key.name,
                "user_id": api_key.user_id,
                "vote_weight": api_key.vote_weight,
                "rate_limit_bypass": api_key.rate_limit_bypass,
                "is_active": api_key.is_active,
                "usage_count": api_key.usage_count,
                "created_at": api_key.created_at,
                "expires_at": api_key.expires_at,
            }

    return None


async def get_api_key_by_id(key_id: int) -> Optional[dict]:
    """Get API key details by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(UserApiKey).where(UserApiKey.id == key_id)
        )
        api_key = result.scalar_one_or_none()

        if api_key:
            return {
                "id": api_key.id,
                "key_prefix": api_key.key_prefix,
                "name": api_key.name,
                "description": api_key.description,
                "user_id": api_key.user_id,
                "vote_weight": api_key.vote_weight,
                "rate_limit_bypass": api_key.rate_limit_bypass,
                "is_active": api_key.is_active,
                "usage_count": api_key.usage_count,
                "last_used_at": api_key.last_used_at,
                "created_at": api_key.created_at,
                "expires_at": api_key.expires_at,
            }
    return None


async def list_user_api_keys(
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List all user API keys."""
    async with get_session() as session:
        stmt = select(UserApiKey)
        
        if active_only:
            stmt = stmt.where(UserApiKey.is_active == True)
        
        stmt = stmt.order_by(UserApiKey.created_at.desc()).limit(limit).offset(offset)
        
        result = await session.execute(stmt)
        keys = result.scalars().all()

        return [
            {
                "id": k.id,
                "key_prefix": k.key_prefix,
                "name": k.name,
                "description": k.description,
                "user_id": k.user_id,
                "vote_weight": k.vote_weight,
                "rate_limit_bypass": k.rate_limit_bypass,
                "is_active": k.is_active,
                "usage_count": k.usage_count,
                "last_used_at": k.last_used_at,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
            }
            for k in keys
        ]


async def deactivate_api_key(key_id: int) -> bool:
    """Deactivate an API key."""
    async with get_session() as session:
        result = await session.execute(
            select(UserApiKey).where(UserApiKey.id == key_id)
        )
        api_key = result.scalar_one_or_none()

        if api_key:
            api_key.is_active = False
            await session.commit()
            logger.info(f"Deactivated user API key ID: {key_id}")
            return True
    return False


async def reactivate_api_key(key_id: int) -> bool:
    """Reactivate an API key."""
    async with get_session() as session:
        result = await session.execute(
            select(UserApiKey).where(UserApiKey.id == key_id)
        )
        api_key = result.scalar_one_or_none()

        if api_key:
            api_key.is_active = True
            await session.commit()
            logger.info(f"Reactivated user API key ID: {key_id}")
            return True
    return False


async def delete_api_key(key_id: int) -> bool:
    """Permanently delete an API key."""
    async with get_session() as session:
        result = await session.execute(
            delete(UserApiKey).where(UserApiKey.id == key_id)
        )
        await session.commit()
        
        if result.rowcount > 0:
            logger.info(f"Deleted user API key ID: {key_id}")
            return True
    return False


async def update_api_key(
    key_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    vote_weight: Optional[int] = None,
    rate_limit_bypass: Optional[bool] = None,
    expires_at: Optional[datetime] = None,
) -> Optional[dict]:
    """Update an API key's settings."""
    async with get_session() as session:
        result = await session.execute(
            select(UserApiKey).where(UserApiKey.id == key_id)
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return None

        if name is not None:
            api_key.name = name
        if description is not None:
            api_key.description = description
        if vote_weight is not None:
            api_key.vote_weight = vote_weight
        if rate_limit_bypass is not None:
            api_key.rate_limit_bypass = rate_limit_bypass
        if expires_at is not None:
            api_key.expires_at = expires_at

        await session.commit()
        await session.refresh(api_key)

        return {
            "id": api_key.id,
            "key_prefix": api_key.key_prefix,
            "name": api_key.name,
            "description": api_key.description,
            "vote_weight": api_key.vote_weight,
            "rate_limit_bypass": api_key.rate_limit_bypass,
            "is_active": api_key.is_active,
            "usage_count": api_key.usage_count,
            "created_at": api_key.created_at,
            "expires_at": api_key.expires_at,
        }


async def get_key_stats() -> dict:
    """Get statistics about user API keys."""
    async with get_session() as session:
        now = datetime.now(timezone.utc)
        
        result = await session.execute(
            select(
                func.count().label("total_keys"),
                func.count().filter(UserApiKey.is_active == True).label("active_keys"),
                func.count().filter(UserApiKey.is_active == False).label("inactive_keys"),
                func.count().filter(
                    UserApiKey.expires_at.isnot(None),
                    UserApiKey.expires_at < now
                ).label("expired_keys"),
                func.sum(UserApiKey.usage_count).label("total_usage"),
                func.max(UserApiKey.last_used_at).label("last_used"),
            )
        )
        row = result.one()

        return {
            "total_keys": row.total_keys or 0,
            "active_keys": row.active_keys or 0,
            "inactive_keys": row.inactive_keys or 0,
            "expired_keys": row.expired_keys or 0,
            "total_usage": int(row.total_usage or 0),
            "last_used": row.last_used,
        }
