"""Portfolio Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class PortfolioCreateRequest(BaseModel):
    """Create portfolio request."""

    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)
    base_currency: str = Field(default="USD", max_length=10)
    cash_balance: Optional[float] = Field(default=0.0, ge=0.0)


class PortfolioUpdateRequest(BaseModel):
    """Update portfolio request."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)
    base_currency: Optional[str] = Field(default=None, max_length=10)
    cash_balance: Optional[float] = Field(default=None, ge=0.0)
    is_active: Optional[bool] = None


class PortfolioResponse(BaseModel):
    """Portfolio response."""

    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    base_currency: str
    cash_balance: Optional[float] = None
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None


class HoldingInput(BaseModel):
    """Holding input."""

    symbol: str = Field(..., min_length=1, max_length=20)
    quantity: float = Field(..., gt=0)
    avg_cost: Optional[float] = Field(default=None, ge=0)
    target_weight: Optional[float] = Field(default=None, ge=0, le=1)

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
    avg_cost: Optional[float] = None
    target_weight: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class TransactionInput(BaseModel):
    """Transaction input."""

    symbol: str = Field(..., min_length=1, max_length=20)
    side: str = Field(..., description="buy, sell, dividend, split, deposit, withdrawal")
    quantity: Optional[float] = Field(default=None, ge=0)
    price: Optional[float] = Field(default=None, ge=0)
    fees: Optional[float] = Field(default=None, ge=0)
    trade_date: date
    notes: Optional[str] = Field(default=None, max_length=2000)

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
    quantity: Optional[float] = None
    price: Optional[float] = None
    fees: Optional[float] = None
    trade_date: date
    notes: Optional[str] = None
    created_at: datetime


class PortfolioDetailResponse(PortfolioResponse):
    """Portfolio with holdings and recent transactions."""

    holdings: list[HoldingResponse] = Field(default_factory=list)
    transactions: list[TransactionResponse] = Field(default_factory=list)


class PortfolioAnalyticsRequest(BaseModel):
    """Analytics request."""

    tools: list[str] = Field(default_factory=list, description="Tool names to run")
    window: Optional[str] = Field(default=None, description="Return window like 1y, 3y, 90d")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    benchmark: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    force_refresh: bool = False


class ToolResult(BaseModel):
    """Generic tool result."""

    tool: str
    status: str
    data: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    source: Optional[str] = Field(default=None, description="computed, cache, or db")
    generated_at: Optional[datetime] = None


class PortfolioAnalyticsJobResponse(BaseModel):
    """Analytics job status."""

    job_id: str
    portfolio_id: int
    status: str
    tools: list[str]
    results_count: int
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class PortfolioAnalyticsResponse(BaseModel):
    """Analytics response."""

    portfolio_id: int
    as_of_date: date
    results: list[ToolResult]
    job_id: Optional[str] = None
    job_status: Optional[str] = None
    scheduled_tools: list[str] = Field(default_factory=list)
