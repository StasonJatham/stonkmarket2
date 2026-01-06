"""API dependencies for authentication, rate limiting, and database access."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from fastapi import Cookie, Depends, Header, Path, Request

from app.cache.rate_limit import check_rate_limit
from app.core.client_identity import (
    get_client_ip,
    get_request_fingerprint,
    get_suggestion_identifier,
    get_vote_identifier,
)
from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError, NotFoundError
from app.core.security import TokenData, decode_access_token, validate_token_not_revoked


# Re-export identity functions
__all__ = [
    "CurrentAdmin",
    "DeletePortfolio",
    "EditPortfolio",
    "Permission",
    "ResourceAccess",
    "ViewPortfolio",
    "check_portfolio_access",
    "get_client_ip",
    "get_current_user",
    "get_db",
    "get_request_fingerprint",
    "get_suggestion_identifier",
    "get_vote_identifier",
    "normalize_symbol",
    "rate_limit_api",
    "rate_limit_auth",
    "require_admin",
    "require_user",
]


# =============================================================================
# SYMBOL NORMALIZATION
# =============================================================================


def normalize_symbol(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize a stock symbol from path parameter."""
    return symbol.strip().upper()


# =============================================================================
# RESOURCE-BASED AUTHORIZATION
# =============================================================================


class Permission(str, Enum):
    """Resource permissions for authorization checks."""
    VIEW = "view"
    EDIT = "edit"
    DELETE = "delete"
    SHARE = "share"


@dataclass
class ResourceAccess:
    """Result of an authorization check."""
    allowed: bool
    reason: str
    is_owner: bool = False
    is_admin: bool = False
    is_public: bool = False


async def _get_user_id_from_token(user: TokenData) -> int | None:
    """Get user ID from token data."""
    from app.repositories import auth_user_orm as auth_repo
    record = await auth_repo.get_user(user.sub)
    return record.id if record else None


async def check_portfolio_access(
    portfolio_id: int,
    user: TokenData | None,
    permission: Permission = Permission.VIEW,
) -> ResourceAccess:
    """
    Check if user can access a portfolio.
    
    Authorization Logic:
    1. Owner -> Full access (view, edit, delete, share)
    2. Admin -> Full access (for support/moderation)
    3. Public visibility -> View only
    4. Shared link -> View only (checked separately)
    5. Else -> Denied
    
    Returns:
        ResourceAccess with allowed=True/False and reason
    """
    from app.repositories import portfolios_orm as portfolios_repo
    
    # Fetch portfolio with visibility info
    portfolio = await portfolios_repo.get_portfolio_with_visibility(portfolio_id)
    
    if not portfolio:
        return ResourceAccess(allowed=False, reason="Portfolio not found")
    
    # Check 1: Owner
    if user:
        user_id = await _get_user_id_from_token(user)
        if user_id and portfolio["user_id"] == user_id:
            return ResourceAccess(
                allowed=True, 
                reason="Owner access",
                is_owner=True,
            )
    
    # Check 2: Admin
    if user and user.is_admin:
        return ResourceAccess(
            allowed=True,
            reason="Admin access",
            is_admin=True,
        )
    
    # Check 3: Public visibility (view only)
    visibility = portfolio.get("visibility", "private")
    if visibility == "public" and permission == Permission.VIEW:
        return ResourceAccess(
            allowed=True,
            reason="Public portfolio",
            is_public=True,
        )
    
    # Check 4: Non-view permission on public resource
    if visibility == "public" and permission != Permission.VIEW:
        return ResourceAccess(
            allowed=False,
            reason="Public portfolios are read-only",
        )
    
    # Default: Denied
    return ResourceAccess(
        allowed=False,
        reason="Access denied",
    )


async def _require_portfolio_access(
    portfolio_id: int,
    permission: Permission,
    user: TokenData | None,
) -> int:
    """
    Validate portfolio access and return portfolio_id if allowed.
    
    Raises appropriate HTTP exceptions if access denied.
    """
    access = await check_portfolio_access(portfolio_id, user, permission)
    
    if not access.allowed:
        if access.reason == "Portfolio not found":
            raise NotFoundError(message="Portfolio not found")
        raise AuthorizationError(
            message=access.reason,
            error_code="PORTFOLIO_ACCESS_DENIED",
        )
    
    return portfolio_id


def PortfolioAccess(permission: Permission):
    """Factory to create a dependency for specific permission level."""
    async def dependency(
        portfolio_id: int = Path(...),
        user: TokenData = Depends(require_user),
    ) -> int:
        return await _require_portfolio_access(portfolio_id, permission, user)
    return dependency


# NOTE: PortfolioAccessOptionalAuth and typed aliases are defined after get_current_user
# See "PORTFOLIO ACCESS DEPENDENCIES" section below


# =============================================================================
# TOKEN EXTRACTION
# =============================================================================


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


# =============================================================================
# PORTFOLIO ACCESS DEPENDENCIES (defined after get_current_user/require_user)
# =============================================================================


def PortfolioAccessOptionalAuth(permission: Permission):
    """Factory for routes that allow public access (optional auth)."""
    async def dependency(
        portfolio_id: int = Path(...),
        user: TokenData | None = Depends(get_current_user),
    ) -> int:
        return await _require_portfolio_access(portfolio_id, permission, user)
    return dependency


# Typed aliases for common portfolio access patterns
ViewPortfolio = Annotated[int, Depends(PortfolioAccessOptionalAuth(Permission.VIEW))]
EditPortfolio = Annotated[int, Depends(PortfolioAccess(Permission.EDIT))]
DeletePortfolio = Annotated[int, Depends(PortfolioAccess(Permission.DELETE))]
SharePortfolio = Annotated[int, Depends(PortfolioAccess(Permission.SHARE))]

# Typed alias for admin-only endpoints
CurrentAdmin = Annotated[TokenData, Depends(require_admin)]


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
