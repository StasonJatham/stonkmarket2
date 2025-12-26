"""Auth-related schemas."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    """Login request schema."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username",
        examples=["admin"],
    )
    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Password",
    )
    mfa_code: str | None = Field(
        default=None,
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
        description="6-digit MFA code (required if MFA is enabled)",
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate and sanitize username."""
        v = v.strip().lower()
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        return v


class LoginResponse(BaseModel):
    """Login response schema."""

    username: str = Field(..., description="Authenticated username")
    is_admin: bool = Field(..., description="Whether user is admin")
    access_token: str = Field(default="", description="JWT access token (empty if mfa_required)")
    token_type: str = Field(default="bearer", description="Token type")
    mfa_required: bool = Field(default=False, description="Whether MFA code is required")


class UserResponse(BaseModel):
    """Current user response schema."""

    username: str = Field(..., description="Username")
    is_admin: bool = Field(..., description="Whether user is admin")


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""

    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Current password",
    )
    new_username: str | None = Field(
        default=None,
        min_length=3,
        max_length=50,
        description="New username (optional)",
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password (min 8 chars)",
    )

    @field_validator("new_username")
    @classmethod
    def validate_new_username(cls, v: str | None) -> str | None:
        """Validate and sanitize new username."""
        if v is None:
            return v
        v = v.strip().lower()
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        return v

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v
