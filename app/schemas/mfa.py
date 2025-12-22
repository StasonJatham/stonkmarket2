"""MFA-related Pydantic schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MFASetupResponse(BaseModel):
    """Response when initiating MFA setup."""

    secret: str = Field(..., description="Base32 encoded TOTP secret (show to user)")
    provisioning_uri: str = Field(
        ..., description="otpauth:// URI for authenticator apps"
    )
    qr_code_base64: str = Field(..., description="QR code as base64 PNG image")

    model_config = {
        "json_schema_extra": {
            "example": {
                "secret": "JBSWY3DPEHPK3PXP",
                "provisioning_uri": "otpauth://totp/Stonkmarket:admin?secret=JBSWY3DPEHPK3PXP&issuer=Stonkmarket",
                "qr_code_base64": "iVBORw0KGgo...",
            }
        }
    }


class MFAVerifyRequest(BaseModel):
    """Request to verify MFA code and enable MFA."""

    code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")

    model_config = {"json_schema_extra": {"example": {"code": "123456"}}}


class MFAVerifyResponse(BaseModel):
    """Response after successfully enabling MFA."""

    enabled: bool
    backup_codes: List[str] = Field(
        ..., description="One-time backup codes (show once, user must save)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "enabled": True,
                "backup_codes": ["A1B2C3D4", "E5F6G7H8", "I9J0K1L2"],
            }
        }
    }


class MFAStatusResponse(BaseModel):
    """MFA status for current user."""

    enabled: bool
    has_backup_codes: bool
    backup_codes_remaining: Optional[int] = None


class MFADisableRequest(BaseModel):
    """Request to disable MFA (requires current MFA code)."""

    code: str = Field(..., description="Current MFA code or backup code")


class MFAValidateRequest(BaseModel):
    """Request to validate MFA for sensitive operations."""

    code: str = Field(..., description="MFA code or backup code")


class MFAValidateResponse(BaseModel):
    """Response after MFA validation."""

    valid: bool
    mfa_token: Optional[str] = Field(
        None, description="Short-lived token for subsequent MFA-protected operations"
    )
