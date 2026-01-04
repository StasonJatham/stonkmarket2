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


# Default preferences for new users
DEFAULT_USER_PREFERENCES = {
    "theme": "system",
    "chart_period": "1Y",
    "currency": "USD",
    "notifications": True,
    "onboarding_complete": False,
}


@dataclass
class AuthUser:
    """Authentication user with multi-tenant and MFA support."""

    id: int
    username: str
    password_hash: str | None
    is_admin: bool = False
    email: str | None = None
    email_verified: bool = False
    avatar_url: str | None = None
    preferences: dict | None = None
    auth_provider: str = "local"
    provider_id: str | None = None
    mfa_secret: str | None = None
    mfa_enabled: bool = False
    mfa_backup_codes: str | None = None  # JSON list of hashed backup codes
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login_at: datetime | None = None

    @classmethod
    def from_orm(cls, user: AuthUserORM) -> AuthUser:
        """Create from ORM model."""
        return cls(
            id=user.id,
            username=user.username,
            password_hash=user.password_hash,
            is_admin=user.is_admin or False,
            email=user.email,
            email_verified=user.email_verified or False,
            avatar_url=user.avatar_url,
            preferences=user.preferences or {},
            auth_provider=user.auth_provider or "local",
            provider_id=user.provider_id,
            mfa_secret=user.mfa_secret,
            mfa_enabled=user.mfa_enabled or False,
            mfa_backup_codes=user.mfa_backup_codes,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
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


# =============================================================================
# OAUTH & SOCIAL AUTH FUNCTIONS
# =============================================================================


async def get_user_by_email(email: str) -> AuthUser | None:
    """Get a user by email address."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.email == email.lower())
        )
        user = result.scalar_one_or_none()

        if user:
            return AuthUser.from_orm(user)
        return None


async def get_user_by_provider(provider: str, provider_id: str) -> AuthUser | None:
    """Get a user by OAuth provider and provider ID."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(
                AuthUserORM.auth_provider == provider,
                AuthUserORM.provider_id == provider_id,
            )
        )
        user = result.scalar_one_or_none()

        if user:
            return AuthUser.from_orm(user)
        return None


async def create_oauth_user(
    username: str,
    email: str | None,
    avatar_url: str | None,
    auth_provider: str,
    provider_id: str,
) -> AuthUser:
    """Create a new user from OAuth provider."""
    from app.repositories import portfolios_orm as portfolios_repo
    
    now = datetime.now(UTC)
    
    async with get_session() as session:
        # Check if username exists, if so append a random suffix
        base_username = username.lower()
        final_username = base_username
        counter = 1
        
        while True:
            result = await session.execute(
                select(AuthUserORM.id).where(AuthUserORM.username == final_username)
            )
            if result.scalar_one_or_none() is None:
                break
            final_username = f"{base_username}{counter}"
            counter += 1
        
        new_user = AuthUserORM(
            username=final_username,
            email=email.lower() if email else None,
            email_verified=True if email else False,  # OAuth emails are verified
            avatar_url=avatar_url,
            auth_provider=auth_provider,
            provider_id=provider_id,
            password_hash=None,  # No password for OAuth users
            preferences=DEFAULT_USER_PREFERENCES.copy(),
            created_at=now,
            updated_at=now,
        )
        session.add(new_user)
        await session.flush()  # Get user.id
        
        user_id = new_user.id
        await session.commit()
    
    # Create default portfolio for new user (outside the auth session)
    await portfolios_repo.create_portfolio(
        user_id=user_id,
        name="My Portfolio",
        description="Your first portfolio - add holdings to get started!",
        base_currency="USD",
    )
    
    return await get_user_by_id(user_id)  # type: ignore


async def link_provider(user_id: int, provider: str, provider_id: str) -> bool:
    """Link an OAuth provider to an existing user account."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.auth_provider = provider
            user.provider_id = provider_id
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return True
        return False


async def update_last_login(user_id: int) -> None:
    """Update last login timestamp for a user."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            user.last_login_at = datetime.now(UTC)
            await session.commit()


async def update_user_preferences(user_id: int, preferences: dict) -> dict:
    """Update user preferences (merge with existing)."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            existing_prefs = user.preferences or {}
            merged_prefs = {**existing_prefs, **preferences}
            user.preferences = merged_prefs
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return merged_prefs
        return {}


async def get_user_preferences(user_id: int) -> dict:
    """Get user preferences."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM.preferences).where(AuthUserORM.id == user_id)
        )
        prefs = result.scalar_one_or_none()
        return prefs or DEFAULT_USER_PREFERENCES.copy()


async def update_user_profile(
    user_id: int,
    email: str | None = None,
    avatar_url: str | None = None,
) -> AuthUser | None:
    """Update user profile fields."""
    async with get_session() as session:
        result = await session.execute(
            select(AuthUserORM).where(AuthUserORM.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            if email is not None:
                user.email = email.lower()
                user.email_verified = False  # Needs re-verification
            if avatar_url is not None:
                user.avatar_url = avatar_url
            user.updated_at = datetime.now(UTC)
            await session.commit()
            return AuthUser.from_orm(user)
        return None


async def create_user_with_defaults(
    username: str,
    password_hash: str,
    email: str | None = None,
) -> AuthUser:
    """
    Create a new user with default portfolio and preferences.
    
    Used for local registration (not OAuth).
    """
    from app.repositories import portfolios_orm as portfolios_repo
    
    now = datetime.now(UTC)
    
    async with get_session() as session:
        new_user = AuthUserORM(
            username=username.lower(),
            email=email.lower() if email else None,
            password_hash=password_hash,
            auth_provider="local",
            preferences=DEFAULT_USER_PREFERENCES.copy(),
            created_at=now,
            updated_at=now,
        )
        session.add(new_user)
        await session.flush()
        
        user_id = new_user.id
        await session.commit()
    
    # Create default portfolio
    await portfolios_repo.create_portfolio(
        user_id=user_id,
        name="My Portfolio",
        description="Your first portfolio - add holdings to get started!",
        base_currency="USD",
    )
    
    return await get_user_by_id(user_id)  # type: ignore
