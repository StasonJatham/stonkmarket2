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
    # execute returns string like "UPDATE 1" - extract the count
    try:
        count = int(result.split()[-1])
        return count > 0
    except (ValueError, IndexError):
        return False


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
    # execute returns string like "UPDATE 1" - extract the count
    try:
        count = int(result.split()[-1])
        return count > 0
    except (ValueError, IndexError):
        return False


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
    # execute returns string like "UPDATE 1" - extract the count
    try:
        count = int(result.split()[-1])
        return count > 0
    except (ValueError, IndexError):
        return False


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


async def seed_admin_from_env() -> None:
    """
    Seed admin user from environment variables into the database.
    
    Creates the admin user if it doesn't exist, using ADMIN_USER and ADMIN_PASS
    from the environment. Also updates password if it changed.
    """
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.core.security import hash_password
    
    logger = get_logger("auth_user.seed")
    
    username = settings.admin_user
    password = settings.admin_pass
    
    if not username or not password:
        logger.warning("ADMIN_USER or ADMIN_PASS not set in environment")
        return
    
    if password == "changeme":
        logger.warning("ADMIN_PASS is set to default 'changeme' - please set a secure password!")
    
    # Check if user exists
    existing = await get_user(username)
    
    if existing:
        # User exists - update password hash if needed
        password_hash = hash_password(password)
        from app.core.security import verify_password
        if not verify_password(password, existing.password_hash):
            # Password changed, update it
            await execute(
                """
                UPDATE auth_user SET password_hash = $1, updated_at = NOW()
                WHERE username = $2
                """,
                password_hash,
                username.lower(),
            )
            logger.info(f"Updated password for admin user '{username}'")
        else:
            logger.debug(f"Admin user '{username}' already exists with correct password")
    else:
        # Create admin user
        password_hash = hash_password(password)
        await execute(
            """
            INSERT INTO auth_user(username, password_hash, is_admin, created_at, updated_at)
            VALUES ($1, $2, TRUE, NOW(), NOW())
            """,
            username.lower(),
            password_hash,
        )
        logger.info(f"Created admin user '{username}' from environment")
