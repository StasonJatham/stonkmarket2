"""MFA (Multi-Factor Authentication) routes."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends

from app.api.dependencies import require_admin, require_user
from app.core.exceptions import AuthenticationError, ValidationError
from app.core.mfa import (
    generate_backup_codes,
    generate_mfa_secret,
    generate_provisioning_uri,
    generate_qr_code_base64,
    hash_backup_codes,
    verify_backup_code,
    verify_totp,
)
from app.core.security import TokenData
from app.repositories import auth_user_orm as auth_repo
from app.schemas.mfa import (
    MFADisableRequest,
    MFASetupResponse,
    MFAStatusResponse,
    MFAValidateRequest,
    MFAValidateResponse,
    MFAVerifyRequest,
    MFAVerifyResponse,
)


router = APIRouter()


@router.get(
    "/status",
    response_model=MFAStatusResponse,
    summary="Get MFA status",
    description="Check if MFA is enabled for the current user.",
)
async def get_mfa_status(
    user: TokenData = Depends(require_user),
) -> MFAStatusResponse:
    """Get MFA status for current user."""
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise AuthenticationError(message="User not found")

    backup_count = None
    if db_user.mfa_enabled and db_user.mfa_backup_codes:
        try:
            codes = json.loads(db_user.mfa_backup_codes)
            backup_count = len(codes)
        except json.JSONDecodeError:
            backup_count = 0

    return MFAStatusResponse(
        enabled=db_user.mfa_enabled,
        has_backup_codes=bool(db_user.mfa_backup_codes),
        backup_codes_remaining=backup_count,
    )


@router.post(
    "/setup",
    response_model=MFASetupResponse,
    summary="Initiate MFA setup",
    description="Generate a new MFA secret and QR code. Must be verified to enable.",
)
async def setup_mfa(
    user: TokenData = Depends(require_admin),
) -> MFASetupResponse:
    """
    Initiate MFA setup for admin user.

    This generates a new secret but does not enable MFA until verified.
    """
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise AuthenticationError(message="User not found")

    if db_user.mfa_enabled:
        raise ValidationError("MFA is already enabled. Disable it first to re-setup.")

    # Generate new secret
    secret = generate_mfa_secret()

    # Save secret (not yet enabled)
    await auth_repo.set_mfa_secret(user.sub, secret)

    # Generate provisioning URI and QR code
    provisioning_uri = generate_provisioning_uri(secret, user.sub)
    qr_code = generate_qr_code_base64(provisioning_uri)

    return MFASetupResponse(
        secret=secret,
        provisioning_uri=provisioning_uri,
        qr_code_base64=qr_code,
    )


@router.post(
    "/verify",
    response_model=MFAVerifyResponse,
    summary="Verify and enable MFA",
    description="Verify the first MFA code to enable MFA. Returns backup codes.",
)
async def verify_and_enable_mfa(
    payload: MFAVerifyRequest,
    user: TokenData = Depends(require_admin),
) -> MFAVerifyResponse:
    """
    Verify MFA code and enable MFA.

    This must be called after /setup with a valid TOTP code from the authenticator app.
    Returns one-time backup codes that the user must save.
    """
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise AuthenticationError(message="User not found")

    if db_user.mfa_enabled:
        raise ValidationError("MFA is already enabled")

    if not db_user.mfa_secret:
        raise ValidationError("MFA setup not initiated. Call /setup first.")

    # Verify the code
    if not verify_totp(db_user.mfa_secret, payload.code):
        raise AuthenticationError(
            message="Invalid MFA code", error_code="INVALID_MFA_CODE"
        )

    # Generate backup codes
    plain_codes, hashed_codes = generate_backup_codes(10)
    backup_codes_json = hash_backup_codes(hashed_codes)

    # Enable MFA
    await auth_repo.enable_mfa(user.sub, backup_codes_json)

    return MFAVerifyResponse(
        enabled=True,
        backup_codes=plain_codes,
    )


@router.post(
    "/disable",
    summary="Disable MFA",
    description="Disable MFA for the current user (requires valid MFA code).",
)
async def disable_mfa(
    payload: MFADisableRequest,
    user: TokenData = Depends(require_admin),
) -> dict:
    """Disable MFA for admin user."""
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise AuthenticationError(message="User not found")

    if not db_user.mfa_enabled:
        raise ValidationError("MFA is not enabled")

    # Verify the code
    code_valid = False

    # Try TOTP first
    if db_user.mfa_secret and verify_totp(db_user.mfa_secret, payload.code):
        code_valid = True

    # Try backup code
    if not code_valid and db_user.mfa_backup_codes:
        valid, _ = verify_backup_code(payload.code, db_user.mfa_backup_codes)
        code_valid = valid

    if not code_valid:
        raise AuthenticationError(
            message="Invalid MFA code", error_code="INVALID_MFA_CODE"
        )

    # Disable MFA
    await auth_repo.disable_mfa(user.sub)

    return {"message": "MFA disabled successfully"}


@router.post(
    "/validate",
    response_model=MFAValidateResponse,
    summary="Validate MFA for sensitive operation",
    description="Validate MFA code and get a short-lived token for MFA-protected operations.",
)
async def validate_mfa(
    payload: MFAValidateRequest,
    user: TokenData = Depends(require_admin),
) -> MFAValidateResponse:
    """
    Validate MFA code for sensitive operations.

    Returns a short-lived MFA token that can be used for subsequent
    MFA-protected operations (like managing API keys) without re-entering MFA.
    """
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise AuthenticationError(message="User not found")

    if not db_user.mfa_enabled:
        raise ValidationError("MFA is not enabled for this user")

    code_valid = False
    updated_backup_codes = None

    # Try TOTP first
    if db_user.mfa_secret and verify_totp(db_user.mfa_secret, payload.code):
        code_valid = True

    # Try backup code (single-use)
    if not code_valid and db_user.mfa_backup_codes:
        valid, updated = verify_backup_code(payload.code, db_user.mfa_backup_codes)
        if valid:
            code_valid = True
            updated_backup_codes = updated

    if not code_valid:
        return MFAValidateResponse(valid=False, mfa_token=None)

    # Update backup codes if one was used
    if updated_backup_codes:
        await auth_repo.update_backup_codes(user.sub, updated_backup_codes)

    # Generate short-lived MFA token (5 minutes)
    mfa_token = f"{user.sub}:{secrets.token_urlsafe(32)}:{int((datetime.now(UTC) + timedelta(minutes=5)).timestamp())}"

    return MFAValidateResponse(valid=True, mfa_token=mfa_token)


def verify_mfa_token(token: str, username: str) -> bool:
    """
    Verify an MFA token for subsequent operations.

    Args:
        token: The MFA token from /validate
        username: Expected username

    Returns:
        True if token is valid and not expired
    """
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False

        token_user, _, expiry_str = parts
        if token_user != username:
            return False

        expiry = int(expiry_str)
        if datetime.now(UTC).timestamp() > expiry:
            return False

        return True
    except Exception:
        return False
