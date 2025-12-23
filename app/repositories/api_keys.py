"""Secure API keys repository - PostgreSQL async."""

from __future__ import annotations

from typing import Optional, Any

from app.core.encryption import encrypt_api_key, decrypt_api_key, get_key_hint
from app.database.connection import fetch_one, fetch_all, execute


async def get_key(service_name: str) -> Optional[dict[str, Any]]:
    """Get an API key by service name."""
    row = await fetch_one(
        """
        SELECT id, service_name, encrypted_key, key_hint, created_at, updated_at, created_by
        FROM secure_api_keys WHERE service_name = $1
        """,
        service_name,
    )
    if row:
        return dict(row)
    return None


async def get_decrypted_key(service_name: str) -> Optional[str]:
    """Get and decrypt an API key by service name."""
    key_record = await get_key(service_name)
    if key_record and key_record.get("encrypted_key"):
        return decrypt_api_key(key_record["encrypted_key"])
    return None


async def list_keys() -> list[dict]:
    """
    List all API keys (without decrypted values).

    Returns list of dicts with service_name, key_hint, created_at, updated_at, created_by
    """
    rows = await fetch_all(
        """
        SELECT id, service_name, key_hint, created_at, updated_at, created_by
        FROM secure_api_keys
        ORDER BY service_name
        """
    )
    return [
        {
            "id": row["id"],
            "key_name": row["service_name"],  # Map to key_name for API compatibility
            "key_hint": row["key_hint"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "created_by": row["created_by"],
        }
        for row in rows
    ]


async def upsert_key(
    service_name: str,
    api_key: str,
    created_by_id: Optional[int] = None,
) -> dict[str, Any]:
    """Create or update an API key."""
    encrypted = encrypt_api_key(api_key)
    hint = get_key_hint(api_key)

    await execute(
        """
        INSERT INTO secure_api_keys(service_name, encrypted_key, key_hint, created_by, created_at, updated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW())
        ON CONFLICT(service_name) DO UPDATE SET
            encrypted_key = excluded.encrypted_key,
            key_hint = excluded.key_hint,
            updated_at = NOW()
        """,
        service_name, encrypted, hint, created_by_id,
    )
    return await get_key(service_name)  # type: ignore


async def delete_key(service_name: str) -> bool:
    """Delete an API key."""
    result = await execute("DELETE FROM secure_api_keys WHERE service_name = $1", service_name)
    # result is like "DELETE 1"
    try:
        count = int(result.split()[-1])
        return count > 0
    except (ValueError, IndexError):
        return False


# Convenience constants for well-known keys
OPENAI_API_KEY = "openai_api_key"
LOGO_DEV_PUBLIC_KEY = "logo_dev_public_key"
LOGO_DEV_SECRET_KEY = "logo_dev_secret_key"


async def seed_api_keys_from_env() -> None:
    """
    Seed API keys from environment variables into the database.
    
    Only seeds if the key is set in env AND not already in database.
    This allows env vars to provide initial values that are then managed via UI.
    """
    from app.core.config import settings
    from app.core.logging import get_logger
    
    logger = get_logger("api_keys.seed")
    
    # Seed Logo.dev public key if set in env and not in db
    if settings.logo_dev_public_key:
        existing = await get_key(LOGO_DEV_PUBLIC_KEY)
        if not existing:
            await upsert_key(LOGO_DEV_PUBLIC_KEY, settings.logo_dev_public_key)
            logger.info("Seeded Logo.dev public key from environment")
    
    # Seed Logo.dev secret key if set in env and not in db
    if settings.logo_dev_secret_key:
        existing = await get_key(LOGO_DEV_SECRET_KEY)
        if not existing:
            await upsert_key(LOGO_DEV_SECRET_KEY, settings.logo_dev_secret_key)
            logger.info("Seeded Logo.dev secret key from environment")
