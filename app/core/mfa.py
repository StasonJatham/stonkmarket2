"""MFA (Multi-Factor Authentication) with TOTP support."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import secrets

import pyotp
import qrcode

from app.core.config import settings


def generate_mfa_secret() -> str:
    """Generate a new TOTP secret."""
    return pyotp.random_base32()


def get_totp(secret: str) -> pyotp.TOTP:
    """Get TOTP instance for a secret."""
    return pyotp.TOTP(secret)


def verify_totp(secret: str, code: str) -> bool:
    """
    Verify a TOTP code.

    Args:
        secret: The user's MFA secret
        code: The 6-digit code to verify

    Returns:
        True if the code is valid
    """
    totp = get_totp(secret)
    # Allow 1 window of drift (30 seconds before/after)
    return totp.verify(code, valid_window=1)


def generate_provisioning_uri(secret: str, username: str) -> str:
    """
    Generate the provisioning URI for authenticator apps.

    Args:
        secret: The MFA secret
        username: The username for the account

    Returns:
        otpauth:// URI for QR code
    """
    totp = get_totp(secret)
    return totp.provisioning_uri(
        name=username,
        issuer_name=settings.app_name,
    )


def generate_qr_code_base64(provisioning_uri: str) -> str:
    """
    Generate a QR code as base64 PNG.

    Args:
        provisioning_uri: The otpauth:// URI

    Returns:
        Base64-encoded PNG image
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(provisioning_uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_backup_codes(count: int = 10) -> tuple[list[str], list[str]]:
    """
    Generate backup codes for MFA recovery.

    Args:
        count: Number of backup codes to generate

    Returns:
        Tuple of (plain_codes, hashed_codes)
        - plain_codes: Show to user once, don't store
        - hashed_codes: Store in database
    """
    plain_codes = []
    hashed_codes = []

    for _ in range(count):
        # Generate 8-character alphanumeric code
        code = secrets.token_hex(4).upper()
        plain_codes.append(code)

        # Hash for storage
        hashed = hashlib.sha256(code.encode()).hexdigest()
        hashed_codes.append(hashed)

    return plain_codes, hashed_codes


def verify_backup_code(
    code: str, stored_hashes_json: str
) -> tuple[bool, str | None]:
    """
    Verify a backup code and return updated hashes (with used code removed).

    Args:
        code: The backup code to verify
        stored_hashes_json: JSON string of hashed backup codes

    Returns:
        Tuple of (is_valid, updated_hashes_json)
        - is_valid: True if code was valid
        - updated_hashes_json: New JSON with used code removed (None if invalid)
    """
    try:
        hashes = json.loads(stored_hashes_json)
    except (json.JSONDecodeError, TypeError):
        return False, None

    code_hash = hashlib.sha256(code.upper().encode()).hexdigest()

    if code_hash in hashes:
        hashes.remove(code_hash)
        return True, json.dumps(hashes)

    return False, None


def hash_backup_codes(codes: list[str]) -> str:
    """Convert list of hashed codes to JSON for storage."""
    return json.dumps(codes)
