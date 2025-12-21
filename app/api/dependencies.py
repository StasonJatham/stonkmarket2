"""API dependencies for authentication, rate limiting, and database access."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import Cookie, Depends, Header, Request

from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import TokenData, decode_access_token, validate_token_not_revoked
from app.core.fingerprint import (
    get_client_ip,
    get_request_fingerprint,
    get_vote_identifier,
    get_suggestion_identifier,
)
from app.database import get_db_connection
from app.cache.rate_limit import check_rate_limit


def get_db():
    """
    Dependency for getting a database connection.

    Usage:
        @router.get("/items")
        async def get_items(conn = Depends(get_db)):
            ...
    """
    with get_db_connection() as conn:
        yield conn


# Re-export fingerprint functions for backwards compatibility
__all__ = [
    "get_db",
    "get_client_ip",
    "get_request_fingerprint",
    "get_vote_identifier",
    "get_suggestion_identifier",
    "get_current_user",
    "require_user",
    "require_admin",
    "rate_limit_auth",
    "rate_limit_api",
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
    conn: sqlite3.Connection = Depends(get_db),
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

        # Verify user still exists in database
        from app.repositories.auth_user import get_user

        user = get_user(conn, token_data.sub)
        if user is None:
            return None

        return token_data
    except AuthenticationError:
        return None


async def require_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session: Optional[str] = Cookie(default=None),
    conn: sqlite3.Connection = Depends(get_db),
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

    # Verify user still exists in database
    from app.repositories.auth_user import get_user

    user = get_user(conn, token_data.sub)
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
