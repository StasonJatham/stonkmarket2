"""DipFinder Pydantic schemas for API requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DipClassEnum(str, Enum):
    """Dip classification."""

    MARKET_DIP = "MARKET_DIP"
    STOCK_SPECIFIC = "STOCK_SPECIFIC"
    MIXED = "MIXED"


class AlertLevelEnum(str, Enum):
    """Alert level."""

    NONE = "NONE"
    GOOD = "GOOD"
    STRONG = "STRONG"


# === Request Schemas ===


class DipFinderSignalRequest(BaseModel):
    """Request for computing dip signals."""

    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of ticker symbols to analyze",
    )
    benchmark: Optional[str] = Field(
        default=None,
        description="Benchmark ticker (default: SPY)",
    )
    window: Optional[int] = Field(
        default=None,
        ge=7,
        le=365,
        description="Window for dip calculation in days (default: 30)",
    )
    force_refresh: bool = Field(
        default=False,
        description="If true, bypass cache and recompute",
    )


class DipFinderRunRequest(BaseModel):
    """Request to run DipFinder computation."""

    tickers: Optional[List[str]] = Field(
        default=None,
        max_length=100,
        description="Specific tickers to analyze (uses user's universe if None)",
    )
    benchmark: Optional[str] = Field(
        default=None,
        description="Benchmark ticker (default: SPY)",
    )
    windows: Optional[List[int]] = Field(
        default=None,
        description="Windows to compute (default: [7, 30, 100, 365])",
    )


# === Response Schemas ===


class QualityFactorsResponse(BaseModel):
    """Quality score contributing factors."""

    ticker: str
    score: float = Field(..., ge=0, le=100)
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_to_market_cap: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    market_cap: Optional[float] = None
    avg_volume: Optional[float] = None
    profitability_score: float = 50.0
    balance_sheet_score: float = 50.0
    cash_generation_score: float = 50.0
    growth_score: float = 50.0
    liquidity_score: float = 50.0
    fields_available: int = 0
    fields_total: int = 10


class StabilityFactorsResponse(BaseModel):
    """Stability score contributing factors."""

    ticker: str
    score: float = Field(..., ge=0, le=100)
    beta: Optional[float] = None
    volatility_252d: Optional[float] = None
    max_drawdown_5y: Optional[float] = None
    typical_dip_365: Optional[float] = None
    beta_score: float = 50.0
    volatility_score: float = 50.0
    drawdown_score: float = 50.0
    typical_dip_score: float = 50.0
    fundamental_stability_score: float = 50.0
    has_price_data: bool = False
    price_data_days: int = 0


class DipSignalResponse(BaseModel):
    """Complete dip signal response."""

    ticker: str = Field(..., description="Stock ticker symbol")
    window: int = Field(..., description="Window in days")
    benchmark: str = Field(..., description="Benchmark ticker")
    as_of_date: str = Field(..., description="Date of analysis (ISO format)")

    # Dip metrics
    dip_stock: float = Field(..., description="Stock dip fraction")
    peak_stock: float = Field(..., description="Peak price in window")
    current_price: Optional[float] = Field(None, description="Current price")
    dip_pctl: float = Field(..., description="Dip percentile (0-100)")
    dip_vs_typical: float = Field(..., description="Ratio vs typical dip")
    persist_days: int = Field(..., description="Days dip has persisted")

    # Market context
    dip_mkt: float = Field(..., description="Benchmark dip fraction")
    excess_dip: float = Field(..., description="Stock dip - benchmark dip")
    dip_class: str = Field(..., description="Dip classification")

    # Scores
    quality_score: float = Field(..., ge=0, le=100)
    stability_score: float = Field(..., ge=0, le=100)
    dip_score: float = Field(..., ge=0, le=100)
    final_score: float = Field(..., ge=0, le=100)

    # Alert
    alert_level: str = Field(..., description="Alert level")
    should_alert: bool = Field(..., description="Whether to alert")
    reason: str = Field(..., description="Human-readable explanation")

    # Detailed factors (optional, included on request)
    quality_factors: Optional[QualityFactorsResponse] = Field(
        None,
        description="Quality score factors",
    )
    stability_factors: Optional[StabilityFactorsResponse] = Field(
        None,
        description="Stability score factors",
    )

    model_config = {
        "from_attributes": True,
    }


class DipSignalListResponse(BaseModel):
    """Response containing multiple dip signals."""

    signals: List[DipSignalResponse] = Field(..., description="List of signals")
    count: int = Field(..., description="Number of signals")
    benchmark: str = Field(..., description="Benchmark used")
    window: int = Field(..., description="Window used")
    as_of_date: str = Field(..., description="Computation date")


class DipHistoryEntry(BaseModel):
    """Single dip history entry."""

    id: int
    ticker: str
    event_type: str = Field(
        ..., description="entered_dip, exited_dip, deepened, recovered, alert_triggered"
    )
    window_days: int
    dip_pct: Optional[float] = None
    final_score: Optional[float] = None
    dip_class: Optional[str] = None
    recorded_at: str = Field(..., description="Timestamp (ISO format)")


class DipHistoryResponse(BaseModel):
    """Response containing dip history for a ticker."""

    ticker: str
    history: List[DipHistoryEntry]
    count: int


class DipFinderRunResponse(BaseModel):
    """Response for run request."""

    status: str = Field(..., description="Status: started, completed, failed")
    message: str
    tickers_processed: int = 0
    signals_generated: int = 0
    alerts_triggered: int = 0
    errors: List[str] = Field(default_factory=list)


class DipFinderConfigResponse(BaseModel):
    """Current DipFinder configuration."""

    windows: List[int]
    min_dip_abs: float
    min_persist_days: int
    dip_percentile_threshold: float
    dip_vs_typical_threshold: float
    market_dip_threshold: float
    excess_dip_stock_specific: float
    excess_dip_market: float
    quality_gate: float
    stability_gate: float
    alert_good: float
    alert_strong: float
    weight_dip: float
    weight_quality: float
    weight_stability: float
    default_benchmark: str
