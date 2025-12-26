"""Secure API keys repository using SQLAlchemy ORM.

This is the modern ORM-based implementation replacing raw SQL in api_keys.py.

Usage:
    from app.repositories.api_keys_orm import (
        get_key, get_decrypted_key, list_keys, upsert_key, delete_key
    )
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.encryption import decrypt_api_key, encrypt_api_key, get_key_hint
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import SecureApiKey


logger = get_logger("repositories.api_keys_orm")


async def get_key(service_name: str) -> dict[str, Any] | None:
    """Get an API key by service name."""
    async with get_session() as session:
        result = await session.execute(
            select(SecureApiKey).where(SecureApiKey.service_name == service_name)
        )
        key = result.scalar_one_or_none()

        if key:
            return {
                "id": key.id,
                "service_name": key.service_name,
                "encrypted_key": key.encrypted_key,
                "key_hint": key.key_hint,
                "created_at": key.created_at,
                "updated_at": key.updated_at,
                "created_by": key.created_by_id,
            }
        return None


async def get_decrypted_key(service_name: str) -> str | None:
    """Get and decrypt an API key by service name."""
    key_record = await get_key(service_name)
    if key_record and key_record.get("encrypted_key"):
        return decrypt_api_key(key_record["encrypted_key"])
    return None


async def list_keys() -> list[dict[str, Any]]:
    """
    List all API keys (without decrypted values).

    Returns list of dicts with service_name, key_hint, created_at, updated_at, created_by
    """
    async with get_session() as session:
        result = await session.execute(
            select(SecureApiKey).order_by(SecureApiKey.service_name)
        )
        keys = result.scalars().all()

        return [
            {
                "id": key.id,
                "key_name": key.service_name,  # Map to key_name for API compatibility
                "key_hint": key.key_hint,
                "created_at": key.created_at,
                "updated_at": key.updated_at,
                "created_by": key.created_by_id,
            }
            for key in keys
        ]


async def upsert_key(
    service_name: str,
    api_key: str,
    created_by_id: int | None = None,
) -> dict[str, Any]:
    """Create or update an API key."""
    encrypted = encrypt_api_key(api_key)
    hint = get_key_hint(api_key)
    now = datetime.now(UTC)

    async with get_session() as session:
        stmt = insert(SecureApiKey).values(
            service_name=service_name,
            encrypted_key=encrypted,
            key_hint=hint,
            created_by_id=created_by_id,
            created_at=now,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["service_name"],
            set_={
                "encrypted_key": encrypted,
                "key_hint": hint,
                "updated_at": now,
            }
        )
        await session.execute(stmt)
        await session.commit()

    return await get_key(service_name)  # type: ignore


async def delete_key(service_name: str) -> bool:
    """Delete an API key."""
    async with get_session() as session:
        result = await session.execute(
            select(SecureApiKey).where(SecureApiKey.service_name == service_name)
        )
        key = result.scalar_one_or_none()

        if key:
            await session.delete(key)
            await session.commit()
            return True
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

    # Seed OpenAI API key if set in env and not in db
    if settings.openai_api_key:
        existing = await get_key(OPENAI_API_KEY)
        if not existing:
            await upsert_key(OPENAI_API_KEY, settings.openai_api_key)
            logger.info("Seeded OpenAI API key from environment")

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
