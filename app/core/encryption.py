"""Encryption utilities for secure API key storage."""

from __future__ import annotations

import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.core.config import settings


def _get_encryption_key() -> bytes:
    """
    Derive encryption key from auth_secret.

    Uses PBKDF2 to derive a proper encryption key from the auth secret.
    """
    # Use a fixed salt derived from app name (stable across restarts)
    salt = settings.app_name.encode()[:16].ljust(16, b"\0")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )

    key = base64.urlsafe_b64encode(kdf.derive(settings.auth_secret.encode()))
    return key


def _get_fernet() -> Fernet:
    """Get Fernet instance for encryption/decryption."""
    return Fernet(_get_encryption_key())


def encrypt_api_key(plaintext_key: str) -> str:
    """
    Encrypt an API key for storage.

    Args:
        plaintext_key: The API key to encrypt

    Returns:
        Encrypted key as base64 string
    """
    fernet = _get_fernet()
    encrypted = fernet.encrypt(plaintext_key.encode())
    return encrypted.decode()


def decrypt_api_key(encrypted_key: str) -> str | None:
    """
    Decrypt an API key from storage.

    Args:
        encrypted_key: The encrypted key from database

    Returns:
        Decrypted API key, or None if decryption fails
    """
    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted_key.encode())
        return decrypted.decode()
    except Exception:
        return None


def get_key_hint(api_key: str) -> str:
    """
    Generate a hint for an API key (shows first and last 4 chars).

    Args:
        api_key: The full API key

    Returns:
        Hint like "sk-...abc1"
    """
    if len(api_key) <= 8:
        return "*" * len(api_key)

    return f"{api_key[:4]}...{api_key[-4:]}"
