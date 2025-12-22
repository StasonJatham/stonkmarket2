"""Secure API key management routes (MFA-protected) - PostgreSQL async."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import require_admin
from app.core.encryption import decrypt_api_key
from app.core.exceptions import AuthenticationError, NotFoundError, ValidationError
from app.core.mfa import verify_totp, verify_backup_code
from app.core.security import TokenData
from app.repositories import auth_user as auth_repo
from app.repositories import api_keys as api_keys_repo
from app.schemas.api_keys import (
    ApiKeyCreate,
    ApiKeyResponse,
    ApiKeyList,
    ApiKeyDelete,
    ApiKeyReveal,
    ApiKeyRevealResponse,
)

router = APIRouter()


async def _verify_mfa_for_request(
    user: TokenData,
    mfa_code: str,
) -> int:
    """
    Verify MFA code for an API key operation.
    
    Returns the user ID on success.
    Raises AuthenticationError if MFA verification fails.
    """
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise AuthenticationError(message="User not found")

    if not db_user.mfa_enabled:
        raise ValidationError("MFA must be enabled to manage API keys")

    # Try TOTP first
    if db_user.mfa_secret and verify_totp(db_user.mfa_secret, mfa_code):
        return db_user.id

    # Try backup code
    if db_user.mfa_backup_codes:
        valid, updated = verify_backup_code(mfa_code, db_user.mfa_backup_codes)
        if valid:
            # Update backup codes (one was used)
            if updated:
                await auth_repo.update_backup_codes(user.sub, updated)
            return db_user.id

    raise AuthenticationError(message="Invalid MFA code", error_code="INVALID_MFA_CODE")


@router.get(
    "",
    response_model=ApiKeyList,
    summary="List API keys",
    description="List all stored API keys (without revealing values). Requires admin.",
)
async def list_api_keys(
    user: TokenData = Depends(require_admin),
) -> ApiKeyList:
    """List all API keys (hints only, not actual values)."""
    keys = await api_keys_repo.list_keys()

    return ApiKeyList(
        keys=[
            ApiKeyResponse(
                id=k["id"],
                key_name=k["key_name"],
                key_hint=k["key_hint"],
                created_at=str(k["created_at"]) if k["created_at"] else "",
                updated_at=str(k["updated_at"]) if k["updated_at"] else "",
                created_by=str(k["created_by"]) if k["created_by"] else None,
            )
            for k in keys
        ]
    )


@router.post(
    "",
    response_model=ApiKeyResponse,
    summary="Create or update API key",
    description="Store a new API key or update existing. Requires MFA verification.",
)
async def create_or_update_api_key(
    payload: ApiKeyCreate,
    user: TokenData = Depends(require_admin),
) -> ApiKeyResponse:
    """Create or update an API key with MFA verification."""
    # Verify MFA and get user ID
    user_id = await _verify_mfa_for_request(user, payload.mfa_code)

    # Store the key
    key = await api_keys_repo.upsert_key(
        service_name=payload.key_name,
        api_key=payload.api_key,
        created_by_id=user_id,
    )

    return ApiKeyResponse(
        id=key["id"],
        key_name=key["service_name"],
        key_hint=key["key_hint"],
        created_at=str(key["created_at"]) if key["created_at"] else "",
        updated_at=str(key["updated_at"]) if key["updated_at"] else "",
        created_by=str(key["created_by"]) if key["created_by"] else None,
    )


@router.post(
    "/{key_name}/reveal",
    response_model=ApiKeyRevealResponse,
    summary="Reveal API key value",
    description="Get the decrypted API key value. Requires MFA verification.",
)
async def reveal_api_key(
    key_name: str,
    payload: ApiKeyReveal,
    user: TokenData = Depends(require_admin),
) -> ApiKeyRevealResponse:
    """Reveal an API key with MFA verification."""
    # Verify MFA
    await _verify_mfa_for_request(user, payload.mfa_code)

    # Get and decrypt the key
    key = await api_keys_repo.get_key(key_name)
    if not key:
        raise NotFoundError(f"API key '{key_name}' not found")

    decrypted = decrypt_api_key(key["encrypted_key"])
    if not decrypted:
        raise ValidationError("Failed to decrypt API key")

    return ApiKeyRevealResponse(
        key_name=key_name,
        api_key=decrypted,
    )


@router.delete(
    "/{key_name}",
    summary="Delete API key",
    description="Delete an API key. Requires MFA verification.",
)
async def delete_api_key(
    key_name: str,
    payload: ApiKeyDelete,
    user: TokenData = Depends(require_admin),
) -> dict:
    """Delete an API key with MFA verification."""
    # Verify MFA
    await _verify_mfa_for_request(user, payload.mfa_code)

    # Delete the key
    if not await api_keys_repo.delete_key(key_name):
        raise NotFoundError(f"API key '{key_name}' not found")

    return {"message": f"API key '{key_name}' deleted successfully"}


@router.get(
    "/check/{key_name}",
    summary="Check if API key exists and is valid",
    description="Check if a specific API key is configured. Admin only.",
)
async def check_api_key(
    key_name: str,
    user: TokenData = Depends(require_admin),
) -> dict:
    """Check if an API key exists (no MFA required, doesn't reveal value)."""
    key = await api_keys_repo.get_key(key_name)

    return {
        "key_name": key_name,
        "exists": key is not None,
        "key_hint": key["key_hint"] if key else None,
    }
