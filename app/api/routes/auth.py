"""Authentication routes - strict REST endpoints."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Request, Response

from app.api.dependencies import get_db, get_client_ip, require_user, rate_limit_auth
from app.cache.rate_limit import check_rate_limit
from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.security import (
    TokenData,
    create_access_token,
    hash_password,
    verify_password,
)
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
    conn: sqlite3.Connection = Depends(get_db),
) -> LoginResponse:
    """
    Authenticate user and return access token.

    Rate limited to prevent brute force attacks.
    """
    # Apply rate limiting
    client_ip = get_client_ip(request)
    if settings.rate_limit_enabled:
        await check_rate_limit(client_ip, key_prefix="auth")

    # Lookup user
    user = auth_repo.get_user(conn, payload.username)
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

    # Check if admin
    is_admin = user.username == settings.default_admin_user

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
    description="Clear the session cookie.",
)
async def logout(response: Response) -> None:
    """Clear session cookie to log out."""
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
    conn: sqlite3.Connection = Depends(get_db),
) -> UserResponse:
    """
    Update user credentials (username/password).

    Requires current password for verification.
    """
    # Get current user record
    current_user = auth_repo.get_user(conn, user.sub)
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
    auth_repo.upsert_user(conn, new_username, new_password_hash)

    # If username changed, delete old record
    if new_username != user.sub:
        conn.execute("DELETE FROM auth_user WHERE username = ?", (user.sub,))
        conn.commit()

    # Check if still admin
    is_admin = new_username == settings.default_admin_user

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
