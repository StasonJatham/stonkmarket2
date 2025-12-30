"""Portfolio Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PortfolioCreateRequest(BaseModel):
    """Create portfolio request."""

    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    base_currency: str = Field(default="USD", max_length=10)


class PortfolioUpdateRequest(BaseModel):
    """Update portfolio request."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    base_currency: str | None = Field(default=None, max_length=10)
    is_active: bool | None = None


class PortfolioResponse(BaseModel):
    """Portfolio response."""

    id: int
    user_id: int
    name: str
    description: str | None = None
    base_currency: str
    is_active: bool
    # AI analysis
    ai_analysis_summary: str | None = None
    ai_analysis_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None


class HoldingInput(BaseModel):
    """Holding input."""

    symbol: str = Field(..., min_length=1, max_length=20)
    quantity: float = Field(..., gt=0)
    avg_cost: float | None = Field(default=None, ge=0)
    target_weight: float | None = Field(default=None, ge=0, le=1)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.upper().strip()


class HoldingResponse(BaseModel):
    """Holding response."""

    id: int
    portfolio_id: int
    symbol: str
    quantity: float
    avg_cost: float | None = None
    target_weight: float | None = None
    created_at: datetime
    updated_at: datetime | None = None


class TransactionInput(BaseModel):
    """Transaction input."""

    symbol: str = Field(..., min_length=1, max_length=20)
    side: str = Field(..., description="buy, sell, dividend, split, deposit, withdrawal")
    quantity: float | None = Field(default=None, ge=0)
    price: float | None = Field(default=None, ge=0)
    fees: float | None = Field(default=None, ge=0)
    trade_date: date
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.upper().strip()


class TransactionResponse(BaseModel):
    """Transaction response."""

    id: int
    portfolio_id: int
    symbol: str
    side: str
    quantity: float | None = None
    price: float | None = None
    fees: float | None = None
    trade_date: date
    notes: str | None = None
    created_at: datetime


class PortfolioDetailResponse(PortfolioResponse):
    """Portfolio with holdings and recent transactions."""

    holdings: list[HoldingResponse] = Field(default_factory=list)
    transactions: list[TransactionResponse] = Field(default_factory=list)


class PortfolioAnalyticsRequest(BaseModel):
    """Analytics request."""

    tools: list[str] = Field(default_factory=list, description="Tool names to run")
    window: str | None = Field(default=None, description="Return window like 1y, 3y, 90d")
    start_date: date | None = None
    end_date: date | None = None
    benchmark: str | None = None
    params: dict[str, Any] | None = None
    force_refresh: bool = False


class ToolResult(BaseModel):
    """Generic tool result."""

    tool: str
    status: str
    data: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    source: str | None = Field(default=None, description="computed, cache, or db")
    generated_at: datetime | None = None


class PortfolioAnalyticsJobResponse(BaseModel):
    """Analytics job status."""

    job_id: str
    portfolio_id: int
    status: str
    tools: list[str]
    results_count: int
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class PortfolioAnalyticsResponse(BaseModel):
    """Analytics response."""

    portfolio_id: int
    as_of_date: date
    results: list[ToolResult]
    job_id: str | None = None
    job_status: str | None = None
    scheduled_tools: list[str] = Field(default_factory=list)
