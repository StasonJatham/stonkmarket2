"""API dependencies for authentication, rate limiting, and database access."""

from __future__ import annotations

from typing import Optional

from fastapi import Cookie, Depends, Header, Request

from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import TokenData, decode_access_token, validate_token_not_revoked
from app.core.client_identity import (
    get_client_ip,
    get_request_fingerprint,
    get_vote_identifier,
    get_suggestion_identifier,
)
from app.cache.rate_limit import check_rate_limit


# Re-export identity functions
__all__ = [
    "get_client_ip",
    "get_request_fingerprint",
    "get_vote_identifier",
    "get_suggestion_identifier",
    "get_current_user",
    "require_user",
    "require_admin",
    "rate_limit_auth",
    "rate_limit_api",
    "get_db",
]


def _extract_token(
    authorization: Optional[str] = Header(default=None),
    session: Optional[str] = Cookie(default=None),
) -> Optional[str]:
    """Extract JWT token from Authorization header or session cookie."""
    # Prefer Authorization header (for API clients)
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token

    # Fall back to session cookie (for browser clients)
    if session:
        return session

    return None


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session: Optional[str] = Cookie(default=None),
) -> Optional[TokenData]:
    """
    Get current authenticated user (optional).

    Returns None if not authenticated or token is revoked.
    """
    token = _extract_token(authorization, session)
    if not token:
        return None

    try:
        token_data = decode_access_token(token)

        # Check if token has been revoked
        if not await validate_token_not_revoked(token_data):
            return None

        # Verify user still exists in database (async)
        from app.repositories import auth_user_orm as auth_repo

        user = await auth_repo.get_user(token_data.sub)
        if user is None:
            return None

        return token_data
    except AuthenticationError:
        return None


async def require_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session: Optional[str] = Cookie(default=None),
) -> TokenData:
    """
    Require authenticated user.

    Raises AuthenticationError if not authenticated or token is revoked.
    """
    token = _extract_token(authorization, session)
    if not token:
        raise AuthenticationError(
            message="Authentication required",
            error_code="MISSING_CREDENTIALS",
        )

    try:
        token_data = decode_access_token(token)
    except AuthenticationError:
        raise

    # Check if token has been revoked
    if not await validate_token_not_revoked(token_data):
        raise AuthenticationError(
            message="Token has been revoked",
            error_code="TOKEN_REVOKED",
        )

    # Verify user still exists in database (async)
    from app.repositories import auth_user_orm as auth_repo

    user = await auth_repo.get_user(token_data.sub)
    if user is None:
        raise AuthenticationError(
            message="User no longer exists",
            error_code="USER_NOT_FOUND",
        )

    return token_data


async def require_admin(
    user: TokenData = Depends(require_user),
) -> TokenData:
    """
    Require admin user.

    Raises AuthorizationError if not admin.
    """
    if not user.is_admin:
        raise AuthorizationError(
            message="Admin privileges required",
            error_code="ADMIN_REQUIRED",
        )
    return user


async def rate_limit_auth(
    request: Request,
) -> None:
    """Apply rate limiting for auth endpoints."""
    if not settings.rate_limit_enabled:
        return

    client_ip = get_client_ip(request)
    await check_rate_limit(client_ip, key_prefix="auth")


async def rate_limit_api(
    request: Request,
    user: Optional[TokenData] = Depends(get_current_user),
) -> None:
    """Apply rate limiting for API endpoints.

    Rate limits are designed to prevent abuse, not interfere with normal usage:
    - Admin users: No rate limiting
    - Authenticated users: 600 requests/minute (very generous)
    - Anonymous users: 60 requests/minute
    """
    if not settings.rate_limit_enabled:
        return

    # Admins bypass rate limiting entirely
    if user and user.is_admin:
        return

    # Use user ID if authenticated, otherwise IP
    identifier = user.sub if user else get_client_ip(request)
    is_authenticated = user is not None

    # Import here to avoid circular imports
    from app.cache.rate_limit import get_api_rate_limiter

    limiter = get_api_rate_limiter(authenticated=is_authenticated)
    await check_rate_limit(identifier, limiter=limiter)


# =============================================================================
# DATABASE SESSION DEPENDENCY
# =============================================================================

from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency for database sessions (request-scoped).
    
    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    
    The session is automatically committed on success and rolled back on error.
    """
    from app.database.connection import get_session
    
    async with get_session() as session:
        yield session
