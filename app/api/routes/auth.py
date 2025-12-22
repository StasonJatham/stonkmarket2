"""Authentication routes - PostgreSQL async version."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import get_client_ip, require_user, require_admin
from app.cache.rate_limit import check_rate_limit
from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.security import (
    TokenData,
    create_access_token,
    hash_password,
    verify_password,
)
from app.database.connection import execute
from app.repositories import auth_user as auth_repo
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    UserResponse,
)

router = APIRouter()


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate user",
    description="Login with username and password to receive an access token.",
    responses={
        401: {"description": "Invalid credentials"},
        429: {"description": "Too many login attempts"},
    },
)
async def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
) -> LoginResponse:
    """
    Authenticate user and return access token.

    Rate limited to prevent brute force attacks.
    """
    # Apply rate limiting
    client_ip = get_client_ip(request)
    if settings.rate_limit_enabled:
        await check_rate_limit(client_ip, key_prefix="auth")

    # Lookup user (async)
    user = await auth_repo.get_user(payload.username)
    if user is None:
        raise AuthenticationError(
            message="Invalid credentials",
            error_code="INVALID_CREDENTIALS",
        )

    # Verify password
    if not verify_password(payload.password, user.password_hash):
        raise AuthenticationError(
            message="Invalid credentials",
            error_code="INVALID_CREDENTIALS",
        )

    # Get admin status from database (not just username comparison)
    is_admin = user.is_admin

    # Generate JWT token
    access_token = create_access_token(
        username=user.username,
        is_admin=is_admin,
    )

    # Set cookie for browser clients
    response.set_cookie(
        key="session",
        value=access_token,
        httponly=True,
        secure=settings.https_enabled,
        samesite="lax",
        domain=settings.domain,
        max_age=settings.access_token_expire_minutes * 60,
    )

    return LoginResponse(
        username=user.username,
        is_admin=is_admin,
        access_token=access_token,
        token_type="bearer",
    )


@router.post(
    "/logout",
    status_code=204,
    summary="Logout user",
    description="Revoke the current token and clear the session cookie.",
)
async def logout(
    response: Response,
    user: TokenData = Depends(require_user),
) -> None:
    """Revoke current token and clear session cookie."""
    from app.cache.token_blacklist import blacklist_token

    # Blacklist the current token so it can't be reused
    await blacklist_token(user.jti, user.exp)

    # Clear the session cookie
    response.delete_cookie(
        key="session",
        domain=settings.domain,
        path="/",
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user.",
    responses={
        401: {"description": "Not authenticated"},
    },
)
async def get_me(
    user: TokenData = Depends(require_user),
) -> UserResponse:
    """Get current authenticated user information."""
    return UserResponse(
        username=user.sub,
        is_admin=user.is_admin,
    )


@router.put(
    "/credentials",
    response_model=UserResponse,
    summary="Update credentials",
    description="Change username and/or password for current user.",
    responses={
        401: {"description": "Invalid current password"},
    },
)
async def update_credentials(
    payload: PasswordChangeRequest,
    response: Response,
    user: TokenData = Depends(require_user),
) -> UserResponse:
    """
    Update user credentials (username/password).

    Requires current password for verification.
    """
    # Get current user record
    current_user = await auth_repo.get_user(user.sub)
    if current_user is None:
        raise AuthenticationError(
            message="User not found",
            error_code="USER_NOT_FOUND",
        )

    # Verify current password
    if not verify_password(payload.current_password, current_user.password_hash):
        raise AuthenticationError(
            message="Invalid current password",
            error_code="INVALID_PASSWORD",
        )

    # Determine new username
    new_username = payload.new_username or user.sub

    # Hash new password
    new_password_hash = hash_password(payload.new_password)

    # Update or create new user record
    await auth_repo.upsert_user(new_username, new_password_hash)

    # If username changed, delete old record
    if new_username != user.sub:
        await execute(
            "DELETE FROM auth_user WHERE username = $1", user.sub.lower()
        )

    # Get admin status from database
    updated_user = await auth_repo.get_user(new_username)
    is_admin = updated_user.is_admin if updated_user else False

    # Generate new token with updated info
    new_token = create_access_token(
        username=new_username,
        is_admin=is_admin,
    )

    # Update session cookie
    response.set_cookie(
        key="session",
        value=new_token,
        httponly=True,
        secure=settings.https_enabled,
        samesite="lax",
        domain=settings.domain,
        max_age=settings.access_token_expire_minutes * 60,
    )

    return UserResponse(
        username=new_username,
        is_admin=is_admin,
    )


@router.post(
    "/logout-all",
    status_code=204,
    summary="Logout from all devices",
    description="Revoke all tokens for the current user across all devices.",
)
async def logout_all(
    response: Response,
    user: TokenData = Depends(require_user),
) -> None:
    """Revoke all tokens for this user (logout from all devices)."""
    from app.cache.token_blacklist import blacklist_user_tokens

    # Invalidate all tokens issued before now
    await blacklist_user_tokens(user.sub)

    # Clear the session cookie
    response.delete_cookie(
        key="session",
        domain=settings.domain,
        path="/",
    )


@router.post(
    "/revoke-user/{username}",
    status_code=204,
    summary="Revoke user sessions (admin)",
    description="Admin endpoint to revoke all sessions for a specific user.",
    dependencies=[Depends(require_admin)],
)
async def revoke_user_sessions(
    username: str,
    admin: TokenData = Depends(require_admin),
) -> None:
    """Admin: Revoke all tokens for a specific user."""
    from app.cache.token_blacklist import blacklist_user_tokens

    await blacklist_user_tokens(username.lower())
