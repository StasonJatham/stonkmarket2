"""Auth user repository using SQLAlchemy ORM.

This is the modern ORM-based implementation replacing raw SQL in auth_user.py.

Usage:
    from app.repositories.auth_user_orm import get_user, upsert_user, enable_mfa
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import AuthUser as AuthUserORM


logger = get_logger("repositories.auth_user_orm")


@dataclass
class AuthUser:
    """Authentication user with MFA support."""

    id: int
    username: str
    password_hash: str
    is_admin: bool = False
    mfa_secret: str | None = None
    mfa_enabled: bool = False
    mfa_backup_codes: str | None = None  # JSON list of hashed backup codes
    updated_at: datetime | None = None

    @classmethod
    def from_orm(cls, user: AuthUserORM) -> AuthUser:
        """Create from ORM model."""
        return cls(
            id=user.id,
            username=user.username,
            password_hash=user.password_hash,
            is_admin=user.is_admin or False,
            mfa_secret=user.mfa_secret,
            mfa_enabled=user.mfa_enabled or False,
            mfa_backup_codes=user.mfa_backup_codes,
            updated_at=user.updated_at,
        )


async def get_user(username: str) -> AuthUser | None:
    """Get a user by username."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == username.lower())
        )
        user = result.scalar_one_or_none()

        if user:
            return AuthUser.from_orm(user)
        return None


async def get_user_by_id(user_id: int) -> AuthUser | None:
    """Get a user by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.id == user_id)
        )
        user = result.scalar_one_or_none()

        if user:
            return AuthUser.from_orm(user)
        return None


async def upsert_user(username: str, password_hash: str) -> AuthUser | None:
    """Create or update a user."""
    now = datetime.now(UTC)

    async with get_session() as session:
        stmt = insert(AuthUserORM).values(
            username=username.lower(),
            password_hash=password_hash,
            created_at=now,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["username"],
            set_={
                "password_hash": password_hash,
                "updated_at": now,
            }
        )
        await session.execute(stmt)
        await session.commit()

    return await get_user(username)


async def set_mfa_secret(username: str, mfa_secret: str) -> bool:
    """Set MFA secret for a user (not yet enabled)."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == username.lower())
        )
        user = result.scalar_one_or_none()

        if user:
            user.mfa_secret = mfa_secret
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return True
        return False


async def enable_mfa(username: str, backup_codes: str) -> bool:
    """Enable MFA for a user after verification."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == username.lower())
        )
        user = result.scalar_one_or_none()

        if user:
            user.mfa_enabled = True
            user.mfa_backup_codes = backup_codes
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return True
        return False


async def disable_mfa(username: str) -> bool:
    """Disable MFA for a user."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == username.lower())
        )
        user = result.scalar_one_or_none()

        if user:
            user.mfa_enabled = False
            user.mfa_secret = None
            user.mfa_backup_codes = None
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return True
        return False


async def verify_and_consume_backup_code(username: str, code_hash: str) -> bool:
    """Verify a backup code and remove it from the list."""
    import json

    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == username.lower())
        )
        user = result.scalar_one_or_none()

        if not user or not user.mfa_backup_codes:
            return False

        try:
            backup_codes = json.loads(user.mfa_backup_codes)
            if code_hash in backup_codes:
                backup_codes.remove(code_hash)
                user.mfa_backup_codes = json.dumps(backup_codes)
                user.updated_at = datetime.now(UTC)
                await session.commit()
                return True
        except (json.JSONDecodeError, TypeError):
            pass

    return False


async def get_single_user() -> AuthUser | None:
    """Get the single user for single-user mode."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).limit(1)
        )
        user = result.scalar_one_or_none()

        if user:
            return AuthUser.from_orm(user)
        return None


async def user_exists() -> bool:
    """Check if any user exists."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM.id).limit(1)
        )
        return result.scalar_one_or_none() is not None


async def update_password(username: str, password_hash: str) -> bool:
    """Update a user's password."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == username.lower())
        )
        user = result.scalar_one_or_none()

        if user:
            user.password_hash = password_hash
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return True
        return False


async def migrate_username(
    old_username: str,
    new_username: str,
    new_password_hash: str,
) -> bool:
    """
    Migrate a user to a new username, preserving all settings.
    
    Creates new user with all MFA settings, then deletes old user.
    """
    async with get_session() as session:
        # Get old user
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.username == old_username.lower())
        )
        old_user = result.scalar_one_or_none()

        if not old_user:
            return False

        now = datetime.now(UTC)

        # Create new user with all settings from old user
        new_user = AuthUserORM(
            username=new_username.lower(),
            password_hash=new_password_hash,
            is_admin=old_user.is_admin,
            mfa_secret=old_user.mfa_secret,
            mfa_enabled=old_user.mfa_enabled,
            mfa_backup_codes=old_user.mfa_backup_codes,
            created_at=now,
            updated_at=now,
        )
        session.add(new_user)

        # Delete old user
        await session.delete(old_user)
        await session.commit()
        return True


async def seed_admin_from_env() -> None:
    """
    Seed admin user from environment variables into the database.
    
    Creates the admin user if it doesn't exist, using ADMIN_USER and ADMIN_PASS
    from the environment. Also updates password if it changed.
    """
    from app.core.config import settings
    from app.core.security import hash_password, verify_password

    username = settings.admin_user
    password = settings.admin_pass

    if not username or not password:
        logger.warning("ADMIN_USER or ADMIN_PASS not set in environment")
        return

    if password == "changeme":
        logger.warning("ADMIN_PASS is set to default 'changeme' - please set a secure password!")

    # Check if user exists
    existing = await get_user(username)
    now = datetime.now(UTC)

    if existing:
        # User exists - update password hash if needed
        if not verify_password(password, existing.password_hash):
            # Password changed, update it
            password_hash = hash_password(password)
            async with get_session() as session:
                result = await session.execute(
                    select(AuthUserORM).where(AuthUserORM.username == username.lower())
                )
                user = result.scalar_one_or_none()
                if user:
                    user.password_hash = password_hash
                    user.updated_at = now
                    await session.commit()
            logger.info(f"Updated password for admin user '{username}'")
        else:
            logger.debug(f"Admin user '{username}' already exists with correct password")
    else:
        # Create admin user (use ON CONFLICT for race condition safety with multiple workers)
        password_hash = hash_password(password)
        async with get_session() as session:
            stmt = insert(AuthUserORM).values(
                username=username.lower(),
                password_hash=password_hash,
                is_admin=True,
                created_at=now,
                updated_at=now,
            ).on_conflict_do_update(
                index_elements=["username"],
                set_={
                    "password_hash": password_hash,
                    "is_admin": True,
                    "updated_at": now,
                }
            )
            await session.execute(stmt)
            await session.commit()
        logger.info(f"Created admin user '{username}' from environment")
