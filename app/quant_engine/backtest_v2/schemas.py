"""
Pydantic V2 Schemas for Backtest V2 API.

These schemas define the API response structure with full validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TradeMarkerSchema(BaseModel):
    """A single trade marker for frontend charting."""

    timestamp: str = Field(..., description="ISO format timestamp")
    price: float = Field(..., gt=0, description="Trade price")
    type: Literal["buy", "sell"] = Field(..., description="Trade direction")
    shares: float | None = Field(None, ge=0, description="Number of shares")
    value: float | None = Field(None, description="Trade value in dollars")
    reason: str | None = Field(None, description="Reason for trade")
    pnl_pct: float | None = Field(None, description="P&L % (for sells only)")
    regime: str | None = Field(None, description="Market regime at time of trade")

    model_config = {"extra": "ignore"}


class EquityPointSchema(BaseModel):
    """A single point on the equity curve."""

    date: str
    portfolio_value: float
    cash: float
    shares: float
    cumulative_invested: float
    cumulative_return_pct: float


class ScenarioResultSchema(BaseModel):
    """Result of a simulation scenario."""

    scenario: str = Field(..., description="Scenario name")
    
    # Final metrics
    final_value: float = Field(..., description="Final portfolio value")
    total_invested: float = Field(..., description="Total capital deployed")
    total_return_pct: float = Field(..., description="Total return percentage")
    annualized_return_pct: float = Field(..., description="Annualized return")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio (return/drawdown)")
    
    # Trade statistics
    n_trades: int = Field(0, description="Number of trades")
    win_rate: float = Field(0.0, description="Win rate percentage")
    avg_trade_return: float = Field(0.0, description="Average return per trade")
    
    # Trade markers for charting
    markers: list[TradeMarkerSchema] = Field(default_factory=list)
    
    # Equity curve (optional, can be large)
    equity_curve: list[EquityPointSchema] | None = None

    model_config = {"extra": "ignore"}


class RegimeStateSchema(BaseModel):
    """Current market regime state."""

    regime: str = Field(..., description="BULL, BEAR, CRASH, RECOVERY")
    strategy_mode: str = Field(..., description="Strategy mode for this regime")
    spy_price: float = Field(..., description="Current SPY price")
    spy_sma200: float = Field(..., description="SPY 200-day SMA")
    drawdown_pct: float = Field(..., description="SPY drawdown from 52-week high")
    above_sma200: bool = Field(..., description="SPY above SMA200")
    description: str = Field(..., description="Human-readable regime description")

    model_config = {"extra": "ignore"}


class FundamentalCheckSchema(BaseModel):
    """Result of a single fundamental check."""

    name: str
    passed: bool
    value: float | None
    threshold: float | None
    reason: str


class GuardrailResultSchema(BaseModel):
    """Result of fundamental guardrail evaluation."""

    passed: bool
    recommendation: str  # STRONG_BUY, ACCUMULATE, HOLD, AVOID, SELL
    confidence: float
    quality_score: float
    risk_level: str
    summary: str
    checks: list[FundamentalCheckSchema] = Field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    critical_failures: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class WFOResultSchema(BaseModel):
    """Walk-Forward Optimization result schema."""

    result: str  # PASSED, FAILED_*
    message: str
    n_folds: int
    avg_oos_return_pct: float
    std_oos_return_pct: float
    avg_oos_sharpe: float
    avg_oos_win_rate: float
    total_oos_trades: int
    pct_folds_profitable: float
    kill_switch_triggered: bool = False
    kill_switch_reasons: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class AccumulationMetricsSchema(BaseModel):
    """Bear market accumulation efficiency metrics."""

    # Shares acquired
    shares_acquired_strategy: float
    shares_acquired_dca: float
    accumulation_score: float = Field(..., description=">1.0 means strategy acquired more shares")
    
    # Cost basis
    avg_cost_strategy: float
    avg_cost_dca: float
    cost_improvement_pct: float
    
    # Recovery
    recovery_days_strategy: int | None
    recovery_days_dca: int | None
    recovery_improvement_days: int | None

    model_config = {"extra": "ignore"}


class CrashTestResultSchema(BaseModel):
    """Result of stress testing on a crash period."""

    crash_name: str  # e.g., "2022 Tech Crash"
    start_date: str
    end_date: str
    spy_drawdown_pct: float
    stock_drawdown_pct: float
    
    # How the strategy performed
    strategy_return_pct: float
    buyhold_return_pct: float
    outperformance_pct: float
    
    # Accumulation metrics
    accumulation: AccumulationMetricsSchema | None = None
    
    # Did the strategy help?
    strategy_helped: bool

    model_config = {"extra": "ignore"}


class BacktestV2Response(BaseModel):
    """Complete backtest V2 response with regime-adaptive strategies."""

    # Identifiers
    symbol: str
    analysis_date: str
    period_years: float

    # =========================================================================
    # REGIME INFORMATION
    # =========================================================================
    regime: RegimeStateSchema
    regime_history_summary: dict[str, int] | None = None  # Count of days in each regime

    # =========================================================================
    # FUNDAMENTAL GUARDRAILS (for bear mode)
    # =========================================================================
    fundamental_check: GuardrailResultSchema | None = None
    fundamental_data_date: str | None = None

    # =========================================================================
    # RECOMMENDATION
    # =========================================================================
    recommendation: str = Field(
        ..., 
        description="CUSTOM_STRATEGY, BUY_AND_HOLD, ACCUMULATE, or ALLOCATE_TO_SPY"
    )
    recommendation_reason: str
    confidence: float = Field(..., ge=0, le=100)

    # Winner's metrics
    winner_return_pct: float
    winner_sharpe: float
    winner_drawdown_pct: float
    winner_calmar: float

    # =========================================================================
    # COMPARISONS
    # =========================================================================
    strategy_vs_bh: float = Field(..., description="Strategy return - Buy&Hold return")
    strategy_vs_spy: float = Field(..., description="Strategy return - SPY return")
    stock_vs_spy: float = Field(..., description="Stock B&H return - SPY return")

    # =========================================================================
    # KILL SWITCH
    # =========================================================================
    kill_switch_triggered: bool = False
    kill_switch_reason: str | None = None

    # =========================================================================
    # SCENARIO RESULTS
    # =========================================================================
    scenarios: dict[str, ScenarioResultSchema] = Field(default_factory=dict)

    # =========================================================================
    # TRADE MARKERS (for the recommended strategy)
    # =========================================================================
    trade_markers: list[TradeMarkerSchema] = Field(default_factory=list)

    # =========================================================================
    # CRASH TESTS (if applicable)
    # =========================================================================
    crash_tests: list[CrashTestResultSchema] = Field(default_factory=list)

    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    @field_validator("trade_markers")
    @classmethod
    def validate_markers_order(cls, markers: list[TradeMarkerSchema]) -> list[TradeMarkerSchema]:
        """Ensure trade markers are in chronological order."""
        if len(markers) <= 1:
            return markers
        for i in range(1, len(markers)):
            if markers[i].timestamp < markers[i - 1].timestamp:
                raise ValueError("Trade markers must be in chronological order")
        return markers

    @field_validator("trade_markers")
    @classmethod
    def validate_buy_before_sell(cls, markers: list[TradeMarkerSchema]) -> list[TradeMarkerSchema]:
        """Ensure sells have corresponding buys before them."""
        open_positions = 0
        for marker in markers:
            if marker.type == "buy":
                open_positions += 1
            elif marker.type == "sell":
                if open_positions <= 0:
                    raise ValueError(f"Sell at {marker.timestamp} has no corresponding buy")
                open_positions -= 1
        return markers

    @model_validator(mode="after")
    def validate_recommendation_consistency(self) -> "BacktestV2Response":
        """Ensure recommendation is consistent with metrics."""
        if self.kill_switch_triggered and self.recommendation not in ["ALLOCATE_TO_SPY", "AVOID"]:
            raise ValueError("Kill switch triggered but recommendation is not defensive")
        return self

    model_config = {"extra": "ignore"}


class QuickAnalysisResponse(BaseModel):
    """Quick analysis for real-time decisions."""

    symbol: str
    regime: str
    regime_allows_buy: bool
    fundamental_passed: bool | None
    recommendation: str
    confidence: float
    summary: str

    model_config = {"extra": "ignore"}
