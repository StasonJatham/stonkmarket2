"""Auth user repository - PostgreSQL async."""

from __future__ import annotations

from typing import Optional

from app.database.connection import fetch_one, execute
from app.database.models import AuthUser


async def get_user(username: str) -> Optional[AuthUser]:
    """Get a user by username."""
    row = await fetch_one(
        """
        SELECT id, username, password_hash, is_admin, mfa_secret, mfa_enabled, mfa_backup_codes, updated_at 
        FROM auth_user WHERE username = $1
        """,
        username.lower(),
    )
    if row:
        return AuthUser.from_row(row)
    return None


async def upsert_user(username: str, password_hash: str) -> Optional[AuthUser]:
    """Create or update a user."""
    await execute(
        """
        INSERT INTO auth_user(username, password_hash, created_at, updated_at) 
        VALUES ($1, $2, NOW(), NOW())
        ON CONFLICT(username) DO UPDATE SET 
            password_hash=excluded.password_hash, 
            updated_at=NOW()
        """,
        username.lower(),
        password_hash,
    )
    return await get_user(username)


async def set_mfa_secret(username: str, mfa_secret: str) -> bool:
    """Set MFA secret for a user (not yet enabled)."""
    result = await execute(
        """
        UPDATE auth_user SET mfa_secret = $1, updated_at = NOW()
        WHERE username = $2
        """,
        mfa_secret,
        username.lower(),
    )
    return result > 0


async def enable_mfa(username: str, backup_codes: str) -> bool:
    """Enable MFA for a user after verification."""
    result = await execute(
        """
        UPDATE auth_user 
        SET mfa_enabled = TRUE, mfa_backup_codes = $1, updated_at = NOW()
        WHERE username = $2
        """,
        backup_codes,
        username.lower(),
    )
    return result > 0


async def disable_mfa(username: str) -> bool:
    """Disable MFA for a user."""
    result = await execute(
        """
        UPDATE auth_user 
        SET mfa_enabled = FALSE, mfa_secret = NULL, mfa_backup_codes = NULL, updated_at = NOW()
        WHERE username = $1
        """,
        username.lower(),
    )
    return result > 0


async def verify_and_consume_backup_code(username: str, code_hash: str) -> bool:
    """Verify a backup code and remove it from the list."""
    user = await get_user(username)
    if not user or not user.mfa_backup_codes:
        return False

    import json

    try:
        backup_codes = json.loads(user.mfa_backup_codes)
        if code_hash in backup_codes:
            backup_codes.remove(code_hash)
            await execute(
                """
                UPDATE auth_user 
                SET mfa_backup_codes = $1, updated_at = NOW()
                WHERE username = $2
                """,
                json.dumps(backup_codes),
                username.lower(),
            )
            return True
    except (json.JSONDecodeError, TypeError):
        pass

    return False


async def get_single_user() -> Optional[AuthUser]:
    """Get the single user for single-user mode."""
    row = await fetch_one(
        "SELECT id, username, password_hash, is_admin, mfa_secret, mfa_enabled, mfa_backup_codes, updated_at FROM auth_user LIMIT 1"
    )
    if row:
        return AuthUser.from_row(row)
    return None


async def user_exists() -> bool:
    """Check if any user exists."""
    row = await fetch_one("SELECT 1 FROM auth_user LIMIT 1")
    return row is not None
