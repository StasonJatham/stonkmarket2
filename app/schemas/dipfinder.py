"""DipFinder Pydantic schemas for API requests and responses."""

from __future__ import annotations

from enum import Enum

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

    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of ticker symbols to analyze",
    )
    benchmark: str | None = Field(
        default=None,
        description="Benchmark ticker (default: SPY)",
    )
    window: int | None = Field(
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

    tickers: list[str] | None = Field(
        default=None,
        max_length=100,
        description="Specific tickers to analyze (uses user's universe if None)",
    )
    benchmark: str | None = Field(
        default=None,
        description="Benchmark ticker (default: SPY)",
    )
    windows: list[int] | None = Field(
        default=None,
        description="Windows to compute (default: [7, 30, 100, 365])",
    )


# === Response Schemas ===


class QualityFactorsResponse(BaseModel):
    """Quality score contributing factors."""

    ticker: str
    score: float = Field(..., ge=0, le=100)
    profit_margin: float | None = None
    operating_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    free_cash_flow: float | None = None
    fcf_to_market_cap: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    market_cap: float | None = None
    avg_volume: float | None = None
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
    beta: float | None = None
    volatility_252d: float | None = None
    max_drawdown_5y: float | None = None
    typical_dip_365: float | None = None
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
    current_price: float | None = Field(None, description="Current price")
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
    opportunity_type: str = Field(
        default="NONE",
        description="Opportunity type: OUTLIER (conservative), BOUNCE (aggressive), BOTH, or NONE",
    )
    reason: str = Field(..., description="Human-readable explanation")

    # Detailed factors (optional, included on request)
    quality_factors: QualityFactorsResponse | None = Field(
        None,
        description="Quality score factors",
    )
    stability_factors: StabilityFactorsResponse | None = Field(
        None,
        description="Stability score factors",
    )

    model_config = {
        "from_attributes": True,
    }


class DipSignalListResponse(BaseModel):
    """Response containing multiple dip signals."""

    signals: list[DipSignalResponse] = Field(..., description="List of signals")
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
    dip_pct: float | None = None
    final_score: float | None = None
    dip_class: str | None = None
    recorded_at: str = Field(..., description="Timestamp (ISO format)")


class DipHistoryResponse(BaseModel):
    """Response containing dip history for a ticker."""

    ticker: str
    history: list[DipHistoryEntry]
    count: int


class DipFinderRunResponse(BaseModel):
    """Response for run request."""

    status: str = Field(..., description="Status: started, completed, failed")
    message: str
    task_id: str | None = Field(None, description="Celery task id")
    tickers_processed: int = 0
    signals_generated: int = 0
    alerts_triggered: int = 0
    errors: list[str] = Field(default_factory=list)


class DipFinderConfigResponse(BaseModel):
    """Current DipFinder configuration."""

    windows: list[int]
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
