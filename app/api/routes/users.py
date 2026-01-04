"""User profile and preferences API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError
from app.core.security import TokenData
from app.repositories import auth_user_orm as auth_repo


router = APIRouter(prefix="/users", tags=["Users"])


class UserProfileResponse(BaseModel):
    """Current user profile response."""
    
    id: int
    username: str
    email: str | None = None
    email_verified: bool = False
    avatar_url: str | None = None
    auth_provider: str = "local"
    is_admin: bool = False
    created_at: datetime


class UserPreferencesResponse(BaseModel):
    """User preferences response."""
    
    theme: str = Field(default="system", description="system, light, or dark")
    default_currency: str = Field(default="USD")
    email_notifications: bool = Field(default=True)
    show_welcome_guide: bool = Field(default=True)


class UserPreferencesUpdate(BaseModel):
    """Update user preferences request (partial update)."""
    
    theme: str | None = Field(default=None)
    default_currency: str | None = Field(default=None)
    email_notifications: bool | None = Field(default=None)
    show_welcome_guide: bool | None = Field(default=None)


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user(
    user: TokenData = Depends(require_user),
) -> UserProfileResponse:
    """Get current user profile."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    
    return UserProfileResponse(
        id=record.id,
        username=record.username,
        email=record.email,
        email_verified=record.email_verified,
        avatar_url=record.avatar_url,
        auth_provider=record.auth_provider,
        is_admin=record.is_admin,
        created_at=record.created_at,
    )


@router.get("/me/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    user: TokenData = Depends(require_user),
) -> UserPreferencesResponse:
    """Get current user preferences."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    
    prefs = await auth_repo.get_user_preferences(record.id)
    return UserPreferencesResponse(**prefs)


@router.patch("/me/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    payload: UserPreferencesUpdate,
    user: TokenData = Depends(require_user),
) -> UserPreferencesResponse:
    """Update current user preferences (partial update)."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    
    # Build update dict from non-None values
    updates: dict[str, Any] = {}
    if payload.theme is not None:
        updates["theme"] = payload.theme
    if payload.default_currency is not None:
        updates["default_currency"] = payload.default_currency
    if payload.email_notifications is not None:
        updates["email_notifications"] = payload.email_notifications
    if payload.show_welcome_guide is not None:
        updates["show_welcome_guide"] = payload.show_welcome_guide
    
    if updates:
        prefs = await auth_repo.update_user_preferences(record.id, updates)
    else:
        prefs = await auth_repo.get_user_preferences(record.id)
    
    return UserPreferencesResponse(**prefs)
