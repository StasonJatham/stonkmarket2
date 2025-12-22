"""User API key management endpoints (admin only)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.dependencies import require_admin
from app.repositories import user_api_keys
from app.core.logging import get_logger

logger = get_logger("user_api_keys_api")
router = APIRouter(prefix="/admin/user-keys", tags=["admin-user-keys"])


# ============================================================================
# Schemas
# ============================================================================


class CreateUserKeyRequest(BaseModel):
    """Request to create a new user API key."""

    name: str = Field(
        ..., min_length=1, max_length=100, description="Name/label for the key"
    )
    description: Optional[str] = Field(None, max_length=500)
    vote_weight: int = Field(
        default=10, ge=1, le=100, description="Vote weight multiplier"
    )
    rate_limit_bypass: bool = Field(default=True, description="Bypass rate limits")
    expires_days: Optional[int] = Field(
        None, ge=1, le=365, description="Days until expiration (null = never)"
    )


class CreateUserKeyResponse(BaseModel):
    """Response after creating a user API key."""

    key: str = Field(..., description="The full API key (show only once!)")
    id: int
    key_prefix: str
    name: str
    vote_weight: int
    rate_limit_bypass: bool
    expires_at: Optional[str] = None
    warning: str = "This key will only be shown once. Please save it securely."


class UserKeyInfo(BaseModel):
    """Public info about a user API key (no secret)."""

    id: int
    key_prefix: str
    name: str
    description: Optional[str] = None
    vote_weight: int
    rate_limit_bypass: bool
    is_active: bool
    usage_count: int
    last_used_at: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None


class UpdateUserKeyRequest(BaseModel):
    """Request to update a user API key."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    vote_weight: Optional[int] = Field(None, ge=1, le=100)
    rate_limit_bypass: Optional[bool] = None


class UserKeyStatsResponse(BaseModel):
    """Statistics about user API keys."""

    total_keys: int
    active_keys: int
    inactive_keys: int
    expired_keys: int
    total_usage: int
    last_used: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "", response_model=CreateUserKeyResponse, dependencies=[Depends(require_admin)]
)
async def create_user_key(request: CreateUserKeyRequest):
    """
    Create a new user API key.

    The full key is only returned once - save it securely!
    Keys provide:
    - Rate limit bypass for /api/dips/changes
    - Weighted votes (default 10x)

    Requires admin authentication.
    """
    full_key, key_data = await user_api_keys.create_user_api_key(
        name=request.name,
        description=request.description,
        vote_weight=request.vote_weight,
        rate_limit_bypass=request.rate_limit_bypass,
        expires_days=request.expires_days,
    )

    logger.info(
        f"Admin created user API key: {key_data.get('key_prefix')}*** for '{request.name}'"
    )

    return CreateUserKeyResponse(
        key=full_key,
        id=key_data["id"],
        key_prefix=key_data["key_prefix"],
        name=key_data["name"],
        vote_weight=key_data["vote_weight"],
        rate_limit_bypass=key_data["rate_limit_bypass"],
        expires_at=key_data["expires_at"].isoformat()
        if key_data.get("expires_at")
        else None,
    )


@router.get("", response_model=list[UserKeyInfo], dependencies=[Depends(require_admin)])
async def list_user_keys(
    active_only: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    List all user API keys.

    Does not expose the actual key values (only prefix).
    """
    keys = await user_api_keys.list_user_api_keys(
        active_only=active_only,
        limit=limit,
        offset=offset,
    )

    return [
        UserKeyInfo(
            id=k["id"],
            key_prefix=k["key_prefix"],
            name=k["name"],
            description=k.get("description"),
            vote_weight=k["vote_weight"],
            rate_limit_bypass=k["rate_limit_bypass"],
            is_active=k["is_active"],
            usage_count=k.get("usage_count", 0),
            last_used_at=k["last_used_at"].isoformat()
            if k.get("last_used_at")
            else None,
            created_at=k["created_at"].isoformat() if k.get("created_at") else "",
            expires_at=k["expires_at"].isoformat() if k.get("expires_at") else None,
        )
        for k in keys
    ]


@router.get(
    "/stats", response_model=UserKeyStatsResponse, dependencies=[Depends(require_admin)]
)
async def get_key_stats():
    """Get statistics about user API keys."""
    stats = await user_api_keys.get_key_stats()

    return UserKeyStatsResponse(
        total_keys=stats.get("total_keys", 0),
        active_keys=stats.get("active_keys", 0),
        inactive_keys=stats.get("inactive_keys", 0),
        expired_keys=stats.get("expired_keys", 0),
        total_usage=stats.get("total_usage", 0) or 0,
        last_used=stats["last_used"].isoformat() if stats.get("last_used") else None,
    )


@router.get(
    "/{key_id}", response_model=UserKeyInfo, dependencies=[Depends(require_admin)]
)
async def get_user_key(key_id: int):
    """Get details of a specific user API key."""
    key_data = await user_api_keys.get_api_key_by_id(key_id)

    if not key_data:
        raise HTTPException(status_code=404, detail="API key not found")

    return UserKeyInfo(
        id=key_data["id"],
        key_prefix=key_data["key_prefix"],
        name=key_data["name"],
        description=key_data.get("description"),
        vote_weight=key_data["vote_weight"],
        rate_limit_bypass=key_data["rate_limit_bypass"],
        is_active=key_data["is_active"],
        usage_count=key_data.get("usage_count", 0),
        last_used_at=key_data["last_used_at"].isoformat()
        if key_data.get("last_used_at")
        else None,
        created_at=key_data["created_at"].isoformat()
        if key_data.get("created_at")
        else "",
        expires_at=key_data["expires_at"].isoformat()
        if key_data.get("expires_at")
        else None,
    )


@router.patch(
    "/{key_id}", response_model=UserKeyInfo, dependencies=[Depends(require_admin)]
)
async def update_user_key(key_id: int, request: UpdateUserKeyRequest):
    """Update a user API key's settings."""
    key_data = await user_api_keys.update_api_key(
        key_id=key_id,
        name=request.name,
        description=request.description,
        vote_weight=request.vote_weight,
        rate_limit_bypass=request.rate_limit_bypass,
    )

    if not key_data:
        raise HTTPException(status_code=404, detail="API key not found")

    logger.info(f"Admin updated user API key ID: {key_id}")

    return UserKeyInfo(
        id=key_data["id"],
        key_prefix=key_data["key_prefix"],
        name=key_data["name"],
        description=key_data.get("description"),
        vote_weight=key_data["vote_weight"],
        rate_limit_bypass=key_data["rate_limit_bypass"],
        is_active=key_data["is_active"],
        usage_count=key_data.get("usage_count", 0),
        created_at=key_data["created_at"].isoformat()
        if key_data.get("created_at")
        else "",
        expires_at=key_data["expires_at"].isoformat()
        if key_data.get("expires_at")
        else None,
    )


@router.post("/{key_id}/deactivate", dependencies=[Depends(require_admin)])
async def deactivate_user_key(key_id: int):
    """Deactivate a user API key (can be reactivated later)."""
    success = await user_api_keys.deactivate_api_key(key_id)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    logger.info(f"Admin deactivated user API key ID: {key_id}")

    return {"message": "API key deactivated", "key_id": key_id}


@router.post("/{key_id}/reactivate", dependencies=[Depends(require_admin)])
async def reactivate_user_key(key_id: int):
    """Reactivate a deactivated user API key."""
    success = await user_api_keys.reactivate_api_key(key_id)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    logger.info(f"Admin reactivated user API key ID: {key_id}")

    return {"message": "API key reactivated", "key_id": key_id}


@router.delete("/{key_id}", dependencies=[Depends(require_admin)])
async def delete_user_key(key_id: int):
    """Permanently delete a user API key."""
    success = await user_api_keys.delete_api_key(key_id)

    if not success:
        raise HTTPException(status_code=404, detail="API key not found")

    logger.info(f"Admin deleted user API key ID: {key_id}")

    return {"message": "API key deleted permanently", "key_id": key_id}
