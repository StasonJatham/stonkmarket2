"""User API key repository for public API access."""

from __future__ import annotations

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from asyncpg import Connection

from app.database.connection import get_db, fetch_one, fetch_all, execute, fetch_val
from app.core.logging import get_logger

logger = get_logger("user_api_keys")


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
    now = datetime.utcnow()
    expires_at = None
    if expires_days:
        expires_at = now + timedelta(days=expires_days)
    
    async with get_db() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO user_api_keys (
                key_hash, key_prefix, name, description, user_id,
                vote_weight, rate_limit_bypass, is_active, created_at, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE, $8, $9)
            RETURNING id, key_prefix, name, description, vote_weight, 
                      rate_limit_bypass, is_active, created_at, expires_at
            """,
            key_hash, key_prefix, name, description, user_id,
            vote_weight, rate_limit_bypass, now, expires_at
        )
    
    logger.info(f"Created user API key: {key_prefix}*** for '{name}'")
    
    return full_key, dict(row)


async def validate_api_key(key: str) -> Optional[dict]:
    """
    Validate an API key and return its details if valid.
    
    Returns:
        Key details dict if valid, None if invalid/expired/inactive.
    """
    key_hash = hash_api_key(key)
    now = datetime.utcnow()
    
    async with get_db() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, key_prefix, name, user_id, vote_weight, rate_limit_bypass,
                   is_active, usage_count, created_at, expires_at
            FROM user_api_keys
            WHERE key_hash = $1
              AND is_active = TRUE
              AND (expires_at IS NULL OR expires_at > $2)
            """,
            key_hash, now
        )
        
        if row:
            # Update last used and usage count
            await conn.execute(
                """
                UPDATE user_api_keys
                SET last_used_at = $1, usage_count = usage_count + 1
                WHERE id = $2
                """,
                now, row['id']
            )
            
            return dict(row)
    
    return None


async def get_api_key_by_id(key_id: int) -> Optional[dict]:
    """Get API key details by ID."""
    row = await fetch_one(
        """
        SELECT id, key_prefix, name, description, user_id, vote_weight,
               rate_limit_bypass, is_active, usage_count, last_used_at,
               created_at, expires_at
        FROM user_api_keys
        WHERE id = $1
        """,
        key_id
    )
    return dict(row) if row else None


async def list_user_api_keys(
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List all user API keys."""
    query = """
        SELECT id, key_prefix, name, description, user_id, vote_weight,
               rate_limit_bypass, is_active, usage_count, last_used_at,
               created_at, expires_at
        FROM user_api_keys
    """
    
    if active_only:
        query += " WHERE is_active = TRUE"
    
    query += " ORDER BY created_at DESC LIMIT $1 OFFSET $2"
    
    rows = await fetch_all(query, limit, offset)
    return [dict(r) for r in rows]


async def deactivate_api_key(key_id: int) -> bool:
    """Deactivate an API key."""
    result = await execute(
        "UPDATE user_api_keys SET is_active = FALSE WHERE id = $1",
        key_id
    )
    
    if "UPDATE 1" in result:
        logger.info(f"Deactivated user API key ID: {key_id}")
        return True
    return False


async def reactivate_api_key(key_id: int) -> bool:
    """Reactivate an API key."""
    result = await execute(
        "UPDATE user_api_keys SET is_active = TRUE WHERE id = $1",
        key_id
    )
    
    if "UPDATE 1" in result:
        logger.info(f"Reactivated user API key ID: {key_id}")
        return True
    return False


async def delete_api_key(key_id: int) -> bool:
    """Permanently delete an API key."""
    result = await execute(
        "DELETE FROM user_api_keys WHERE id = $1",
        key_id
    )
    
    if "DELETE 1" in result:
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
    updates = []
    params = []
    param_idx = 1
    
    if name is not None:
        updates.append(f"name = ${param_idx}")
        params.append(name)
        param_idx += 1
    
    if description is not None:
        updates.append(f"description = ${param_idx}")
        params.append(description)
        param_idx += 1
    
    if vote_weight is not None:
        updates.append(f"vote_weight = ${param_idx}")
        params.append(vote_weight)
        param_idx += 1
    
    if rate_limit_bypass is not None:
        updates.append(f"rate_limit_bypass = ${param_idx}")
        params.append(rate_limit_bypass)
        param_idx += 1
    
    if expires_at is not None:
        updates.append(f"expires_at = ${param_idx}")
        params.append(expires_at)
        param_idx += 1
    
    if not updates:
        return await get_api_key_by_id(key_id)
    
    params.append(key_id)
    
    query = f"""
        UPDATE user_api_keys
        SET {', '.join(updates)}
        WHERE id = ${param_idx}
        RETURNING id, key_prefix, name, description, vote_weight,
                  rate_limit_bypass, is_active, usage_count, created_at, expires_at
    """
    
    row = await fetch_one(query, *params)
    return dict(row) if row else None


async def get_key_stats() -> dict:
    """Get statistics about user API keys."""
    async with get_db() as conn:
        stats = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) as total_keys,
                COUNT(*) FILTER (WHERE is_active) as active_keys,
                COUNT(*) FILTER (WHERE NOT is_active) as inactive_keys,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at < NOW()) as expired_keys,
                SUM(usage_count) as total_usage,
                MAX(last_used_at) as last_used
            FROM user_api_keys
            """
        )
    
    return dict(stats) if stats else {}
