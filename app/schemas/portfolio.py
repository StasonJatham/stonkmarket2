"""Portfolio Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PortfolioVisibility(str, Enum):
    """Portfolio visibility options."""
    
    private = "private"
    public = "public"
    shared_link = "shared_link"


class PortfolioCreateRequest(BaseModel):
    """Create portfolio request."""

    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    base_currency: str = Field(default="USD", max_length=10)
    cash_balance: float | None = Field(default=None, ge=0)
    visibility: PortfolioVisibility = Field(default=PortfolioVisibility.private)


class PortfolioUpdateRequest(BaseModel):
    """Update portfolio request."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    base_currency: str | None = Field(default=None, max_length=10)
    is_active: bool | None = None
    visibility: PortfolioVisibility | None = None


class PortfolioResponse(BaseModel):
    """Portfolio response."""

    id: int
    user_id: int
    name: str
    description: str | None = None
    base_currency: str
    is_active: bool
    # Visibility settings
    visibility: PortfolioVisibility = Field(default=PortfolioVisibility.private)
    share_token: str | None = None
    shared_at: datetime | None = None
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


# =============================================================================
# AI Portfolio Analysis - Structured Output Schema
# =============================================================================


class PortfolioHealthEnum(str, Enum):
    """Portfolio health status."""
    
    STRONG = "strong"
    GOOD = "good"
    FAIR = "fair"
    WEAK = "weak"


class AIInsight(BaseModel):
    """Single AI insight with type indicator."""
    
    type: str = Field(..., description="positive, warning, or neutral")
    text: str = Field(..., max_length=200)


class AIActionItem(BaseModel):
    """Single actionable recommendation."""
    
    priority: int = Field(..., ge=1, le=3, description="1=high, 2=medium, 3=low")
    action: str = Field(..., max_length=200)


class AIRiskAlert(BaseModel):
    """Single risk alert."""
    
    severity: str = Field(..., description="high, medium, or low")
    alert: str = Field(..., max_length=200)


class AIPortfolioAnalysis(BaseModel):
    """Structured AI portfolio analysis output."""
    
    health: PortfolioHealthEnum = Field(
        ..., 
        description="Overall portfolio health rating"
    )
    headline: str = Field(
        ..., 
        max_length=120,
        description="One-sentence summary with key metric"
    )
    insights: list[AIInsight] = Field(
        ..., 
        min_length=1,
        max_length=4,
        description="2-4 key observations"
    )
    actions: list[AIActionItem] = Field(
        default_factory=list,
        max_length=3,
        description="0-3 specific recommendations"
    )
    risks: list[AIRiskAlert] = Field(
        default_factory=list,
        max_length=3,
        description="0-3 risk alerts (empty if none)"
    )


class VisibilityUpdateRequest(BaseModel):
    """Update portfolio visibility."""
    
    visibility: PortfolioVisibility


class ShareLinkResponse(BaseModel):
    """Response with share link details."""
    
    share_token: str
    share_url: str
    shared_at: datetime


class PublicPortfolioSummary(BaseModel):
    """Public portfolio for discovery listings."""
    
    id: int
    name: str
    description: str | None = None
    holdings_count: int
    owner_username: str
    created_at: datetime
