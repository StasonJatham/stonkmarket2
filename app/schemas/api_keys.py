"""API key management schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApiKeyCreate(BaseModel):
    """Request to create/update an API key."""

    key_name: str = Field(
        ..., min_length=1, max_length=100, description="Unique key identifier"
    )
    api_key: str = Field(..., min_length=10, description="The API key value")
    mfa_code: str = Field(
        default="", max_length=8, description="MFA code for verification (optional if session active)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "key_name": "openai_api_key",
                "api_key": "sk-...",
                "mfa_code": "123456",
            }
        }
    }


class ApiKeyResponse(BaseModel):
    """API key info (without the actual key value)."""

    id: int
    key_name: str
    key_hint: str | None = Field(
        None, description="Partial key hint like 'sk-...abc1'"
    )
    created_at: str
    updated_at: str
    created_by: str | None = Field(
        None, description="Username who created the key (None for system-seeded keys)"
    )


class ApiKeyList(BaseModel):
    """List of API keys."""

    keys: list[ApiKeyResponse]


class ApiKeyDelete(BaseModel):
    """Request to delete an API key."""

    mfa_code: str = Field(default="", max_length=8, description="MFA code (optional if session active)")


class ApiKeyReveal(BaseModel):
    """Request to reveal an API key (requires MFA)."""

    mfa_code: str = Field(default="", max_length=8, description="MFA code (optional if session active)")


class ApiKeyRevealResponse(BaseModel):
    """Response with revealed API key."""

    key_name: str
    api_key: str
