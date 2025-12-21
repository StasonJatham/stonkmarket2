"""Security utilities: password hashing, JWT tokens, CSRF."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from pydantic import BaseModel

from .config import settings
from .exceptions import AuthenticationError


# JWT Configuration
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "stonkmarket"
JWT_AUDIENCE = "stonkmarket-api"


class TokenData(BaseModel):
    """Decoded JWT token data."""

    sub: str  # username
    exp: datetime
    iat: datetime
    iss: str
    aud: str
    jti: str  # unique token ID for revocation
    is_admin: bool = False


def hash_password(password: str) -> str:
    """Hash password using bcrypt with salt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_access_token(
    username: str,
    is_admin: bool = False,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))

    payload = {
        "sub": username,
        "exp": expires,
        "iat": now,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "jti": secrets.token_urlsafe(16),
        "is_admin": is_admin,
    }

    return jwt.encode(payload, settings.auth_secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Decode and validate JWT access token."""
    try:
        payload = jwt.decode(
            token,
            settings.auth_secret,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "require": ["exp", "iat", "sub", "iss", "aud", "jti"],
            },
        )
        return TokenData(
            sub=payload["sub"],
            exp=datetime.fromisoformat(payload["exp"].isoformat()) if isinstance(payload["exp"], datetime) else datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromisoformat(payload["iat"].isoformat()) if isinstance(payload["iat"], datetime) else datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            iss=payload["iss"],
            aud=payload["aud"],
            jti=payload["jti"],
            is_admin=payload.get("is_admin", False),
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationError(message="Token has expired", error_code="TOKEN_EXPIRED")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(message="Invalid token", error_code="INVALID_TOKEN")


def generate_csrf_token() -> str:
    """Generate a CSRF token."""
    return secrets.token_urlsafe(32)


def validate_csrf_token(session_token: str, request_token: str) -> bool:
    """Validate CSRF token using constant-time comparison."""
    return secrets.compare_digest(session_token, request_token)


async def validate_token_not_revoked(token_data: TokenData) -> bool:
    """
    Check if a token has been revoked.
    
    Args:
        token_data: Decoded token data
        
    Returns:
        True if token is valid (not revoked), False if revoked
    """
    from app.cache.token_blacklist import is_token_blacklisted, get_user_token_invalidation_time
    
    # Check if specific token is blacklisted
    if await is_token_blacklisted(token_data.jti):
        return False
    
    # Check if all user tokens before a certain time are invalidated
    invalidation_time = await get_user_token_invalidation_time(token_data.sub)
    if invalidation_time and token_data.iat < invalidation_time:
        return False
    
    return True
