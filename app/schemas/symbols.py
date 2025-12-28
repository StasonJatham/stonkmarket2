"""Symbol-related schemas."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class SymbolBase(BaseModel):
    """Base symbol schema."""

    min_dip_pct: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Minimum dip percentage threshold (0-1)",
        examples=[0.10],
    )
    min_days: int = Field(
        ...,
        ge=0,
        le=365,
        description="Minimum days below threshold",
        examples=[2],
    )


class SymbolCreate(SymbolBase):
    """Create symbol request schema."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol",
        examples=["AAPL"],
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize stock symbol."""
        v = v.strip().upper()
        # Allow only alphanumeric and dots (for symbols like BRK.A)
        if not re.match(r"^[A-Z0-9.]{1,10}$", v):
            raise ValueError("Symbol must be 1-10 alphanumeric characters")
        return v


class SymbolUpdate(SymbolBase):
    """Update symbol request schema."""

    pass


class SymbolResponse(BaseModel):
    """Symbol response schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    min_dip_pct: float = Field(..., description="Minimum dip percentage threshold")
    min_days: int = Field(..., description="Minimum days below threshold")
    name: str | None = Field(None, description="Company name")
    fetch_status: str | None = Field(None, description="Data fetch status: pending, fetching, fetched, error")
    fetch_error: str | None = Field(None, description="Error message if fetch failed")
    task_id: str | None = Field(None, description="Celery task id")

    model_config = {"from_attributes": True}


class SymbolListResponse(BaseModel):
    """Paginated symbol list response."""

    items: list[SymbolResponse]
    total: int
    limit: int
    offset: int
