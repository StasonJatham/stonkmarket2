"""API dependencies for authentication, rate limiting, and database access."""

from __future__ import annotations

from fastapi import Cookie, Depends, Header, Request

from app.cache.rate_limit import check_rate_limit
from app.core.client_identity import (
    get_client_ip,
    get_request_fingerprint,
    get_suggestion_identifier,
    get_vote_identifier,
)
from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import TokenData, decode_access_token, validate_token_not_revoked


# Re-export identity functions
__all__ = [
    "get_client_ip",
    "get_current_user",
    "get_db",
    "get_request_fingerprint",
    "get_suggestion_identifier",
    "get_vote_identifier",
    "rate_limit_api",
    "rate_limit_auth",
    "require_admin",
    "require_user",
]


def _extract_token(
    authorization: str | None = Header(default=None),
    session: str | None = Cookie(default=None),
) -> str | None:
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
    authorization: str | None = Header(default=None),
    session: str | None = Cookie(default=None),
) -> TokenData | None:
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
    authorization: str | None = Header(default=None),
    session: str | None = Cookie(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TokenData:
    """
    Require authenticated user.

    Supports authentication via:
    1. JWT Bearer token (Authorization header)
    2. Session cookie
    3. API Key (X-API-Key header)

    Raises AuthenticationError if not authenticated or token is revoked.
    """
    # First, try API key authentication
    if x_api_key:
        from app.repositories import user_api_keys_orm as api_keys_repo
        from app.repositories import auth_user_orm as auth_repo

        key_data = await api_keys_repo.validate_api_key(x_api_key)
        if key_data:
            # Get user info from the API key's user_id
            user_id = key_data.get("user_id")
            if user_id:
                user = await auth_repo.get_user_by_id(user_id)
                if user:
                    # Create a TokenData-like object for API key auth
                    return TokenData(
                        sub=user.username,
                        is_admin=user.is_admin,
                        exp=0,  # API keys don't expire via JWT
                        iat=0,
                        iss="stonkmarket",
                        aud="stonkmarket-api",
                        jti=f"apikey:{key_data['id']}",
                    )
        # API key provided but invalid
        raise AuthenticationError(
            message="Invalid API key",
            error_code="INVALID_API_KEY",
        )

    # Fall back to JWT token authentication
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
    user: TokenData | None = Depends(get_current_user),
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

from collections.abc import AsyncIterator

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
