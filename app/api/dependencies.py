"""API dependencies for authentication, rate limiting, and database access."""

from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import Cookie, Depends, Header, Request

from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import TokenData, decode_access_token
from app.core.fingerprint import (
    get_client_ip,
    get_request_fingerprint,
    get_vote_identifier,
    get_suggestion_identifier,
)
from app.database import get_db_connection
from app.cache.rate_limit import (
    check_rate_limit,
    get_suggest_rate_limiter,
    get_vote_rate_limiter,
)


def get_db():
    """
    Dependency for getting a database connection.

    Usage:
        @router.get("/items")
        async def get_items(conn = Depends(get_db)):
            ...
    """
    for conn in get_db_connection():
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
    "rate_limit_suggest",
    "rate_limit_vote",
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

    Returns None if not authenticated.
    """
    token = _extract_token(authorization, session)
    if not token:
        return None

    try:
        token_data = decode_access_token(token)

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

    Raises AuthenticationError if not authenticated.
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
    """Apply rate limiting for API endpoints."""
    if not settings.rate_limit_enabled:
        return

    # Use user ID if authenticated, otherwise IP
    identifier = user.sub if user else get_client_ip(request)
    await check_rate_limit(identifier, key_prefix="api")


async def rate_limit_suggest(request: Request) -> str:
    """
    Apply rate limiting for stock suggestion endpoint.
    
    Uses fingerprint-based identification to make abuse harder.
    Returns the fingerprint identifier for vote deduplication.
    """
    if not settings.rate_limit_enabled:
        return get_suggestion_identifier(request)
    
    identifier = get_suggestion_identifier(request)
    limiter = get_suggest_rate_limiter()
    await check_rate_limit(identifier, limiter=limiter)
    return identifier


async def rate_limit_vote(request: Request) -> str:
    """
    Apply rate limiting for voting endpoint.
    
    Uses fingerprint-based identification to prevent vote manipulation.
    Returns the fingerprint identifier for vote deduplication.
    """
    if not settings.rate_limit_enabled:
        return get_request_fingerprint(request)
    
    identifier = get_request_fingerprint(request)
    limiter = get_vote_rate_limiter()
    await check_rate_limit(identifier, limiter=limiter)
    return identifier
