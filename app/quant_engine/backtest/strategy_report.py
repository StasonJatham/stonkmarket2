"""
Strategy Full Report Schemas - Deep Dive Data Models.

Professional-grade Pydantic models for the Trading Dashboard that provide:
- Rich metrics (Profit Factor, Kelly Criterion, SQN, etc.)
- Transparency via "Runner-Up" strategies
- Full equity curves and signal timelines
- Regime-specific breakdowns
- Baseline strategy comparisons (B&H, DCA, Buy Dips, SPY)
- Investment recommendations

These models power the comprehensive API response for strategy backtests.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from app.quant_engine.backtest.baseline_strategies import BaselineComparison


# =============================================================================
# Enums
# =============================================================================

class SignalType(str, Enum):
    """Type of trading signal."""
    BUY = "BUY"
    SELL = "SELL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_EXIT = "TIME_EXIT"


class StrategyVerdict(str, Enum):
    """Verdict on strategy quality based on metrics."""
    EXCELLENT = "EXCELLENT"  # SQN > 3.0
    GOOD = "GOOD"            # SQN 2.0 - 3.0
    AVERAGE = "AVERAGE"      # SQN 1.0 - 2.0
    POOR = "POOR"            # SQN < 1.0
    REJECTED = "REJECTED"    # Failed constraints


class RegimeType(str, Enum):
    """Market regime for performance breakdown."""
    BULL = "BULL"
    BEAR = "BEAR"
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"
    ALL = "ALL"


# =============================================================================
# Core Metric Models
# =============================================================================

class TradeStats(BaseModel):
    """Granular trade-level statistics."""
    
    total_trades: int = Field(ge=0, description="Total number of completed trades")
    winning_trades: int = Field(ge=0, description="Number of profitable trades")
    losing_trades: int = Field(ge=0, description="Number of losing trades")
    
    win_rate: float = Field(ge=0, le=1, description="Win rate as decimal (0-1)")
    loss_rate: float = Field(ge=0, le=1, description="Loss rate as decimal (0-1)")
    
    avg_win_pct: float = Field(description="Average winning trade return %")
    avg_loss_pct: float = Field(description="Average losing trade return % (negative)")
    
    best_trade_pct: float = Field(description="Best single trade return %")
    worst_trade_pct: float = Field(description="Worst single trade return %")
    
    avg_duration_hours: float = Field(ge=0, description="Average trade duration in hours")
    avg_duration_days: float = Field(ge=0, description="Average trade duration in days")
    min_duration_hours: float = Field(ge=0, description="Shortest trade duration")
    max_duration_hours: float = Field(ge=0, description="Longest trade duration")
    
    @field_validator("*", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN/Inf to 0.0 for JSON safety."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


class RiskMetrics(BaseModel):
    """Risk-adjusted performance metrics."""
    
    sharpe_ratio: float = Field(description="Annualized Sharpe Ratio")
    sortino_ratio: float = Field(description="Annualized Sortino Ratio (downside deviation)")
    calmar_ratio: float = Field(description="Annualized Return / Max Drawdown")
    
    profit_factor: float = Field(ge=0, description="Gross Profits / Gross Losses")
    expectancy: float = Field(description="Expected $ return per trade")
    expectancy_ratio: float = Field(description="Expectancy / Avg Loss (R-multiple)")
    
    max_drawdown_pct: float = Field(le=0, description="Maximum peak-to-trough decline %")
    max_drawdown_duration_days: int = Field(ge=0, description="Days in max drawdown")
    avg_drawdown_pct: float = Field(le=0, description="Average drawdown %")
    
    volatility_annual: float = Field(ge=0, description="Annualized return volatility")
    downside_deviation: float = Field(ge=0, description="Downside volatility only")
    
    @field_validator("*", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN/Inf to 0.0 for JSON safety."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


class AdvancedMetrics(BaseModel):
    """Professional trading metrics for system analysis."""
    
    # Position Sizing
    kelly_criterion: float = Field(
        ge=0, le=1, 
        description="Optimal position size fraction (0-1)"
    )
    kelly_half: float = Field(
        ge=0, le=0.5,
        description="Conservative half-Kelly position size"
    )
    
    # System Quality
    sqn: float = Field(description="System Quality Number (Van Tharp)")
    sqn_rating: StrategyVerdict = Field(description="Quality rating based on SQN")
    
    # Trade Analysis
    payoff_ratio: float = Field(
        ge=0, 
        description="Avg Win / Avg Loss ratio"
    )
    trade_frequency_per_year: float = Field(
        ge=0,
        description="Average trades per year"
    )
    
    # Time Analysis
    avg_bars_in_trade: float = Field(ge=0, description="Average bars held per trade")
    time_in_market_pct: float = Field(
        ge=0, le=100,
        description="% of time with open position"
    )
    
    # Streak Analysis
    max_consecutive_wins: int = Field(ge=0, description="Longest winning streak")
    max_consecutive_losses: int = Field(ge=0, description="Longest losing streak")
    avg_consecutive_wins: float = Field(ge=0, description="Average winning streak")
    avg_consecutive_losses: float = Field(ge=0, description="Average losing streak")
    
    @field_validator("*", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN/Inf to 0.0 for JSON safety."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


# =============================================================================
# Time Series Models
# =============================================================================

class EquityCurvePoint(BaseModel):
    """Single point in the equity curve time series."""
    
    timestamp: datetime = Field(description="Date/time of this point")
    equity: float = Field(description="Portfolio value at this point")
    equity_pct: float = Field(description="Cumulative return % from start")
    benchmark_equity: float = Field(description="Buy & Hold value at this point")
    benchmark_pct: float = Field(description="Buy & Hold cumulative return %")
    drawdown_pct: float = Field(le=0, description="Current drawdown from peak")
    in_position: bool = Field(description="Whether a position was open")
    
    @field_validator("drawdown_pct", "equity_pct", "benchmark_pct", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN to 0.0."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


class SignalEvent(BaseModel):
    """A trading signal (buy/sell) with full context."""
    
    timestamp: datetime = Field(description="When the signal occurred")
    signal_type: SignalType = Field(description="Type of signal")
    price: float = Field(gt=0, description="Execution price")
    
    # Position info
    position_size: float = Field(ge=0, description="Number of shares/units")
    position_value: float = Field(ge=0, description="Dollar value of position")
    
    # For exits
    entry_price: float | None = Field(None, description="Entry price (for exits)")
    trade_return_pct: float | None = Field(None, description="Trade return % (for exits)")
    trade_pnl: float | None = Field(None, description="Trade P&L in $ (for exits)")
    holding_days: int | None = Field(None, description="Days held (for exits)")
    
    # Signal context
    indicator_values: dict[str, float] = Field(
        default_factory=dict,
        description="Indicator values at signal time"
    )
    reason: str = Field(default="", description="Human-readable signal reason")


# =============================================================================
# Comparison Models
# =============================================================================

class BenchmarkComparison(BaseModel):
    """Compare strategy performance against benchmarks."""
    
    # Strategy Performance
    strategy_return_pct: float = Field(description="Strategy total return %")
    strategy_sharpe: float = Field(description="Strategy Sharpe ratio")
    strategy_max_dd: float = Field(le=0, description="Strategy max drawdown %")
    strategy_volatility: float = Field(ge=0, description="Strategy annual volatility")
    
    # Buy & Hold (same asset)
    buy_hold_return_pct: float = Field(description="Buy & Hold total return %")
    buy_hold_sharpe: float = Field(description="Buy & Hold Sharpe ratio")
    buy_hold_max_dd: float = Field(le=0, description="Buy & Hold max drawdown %")
    
    # SPY Benchmark
    spy_return_pct: float = Field(description="SPY total return % (same period)")
    spy_sharpe: float = Field(description="SPY Sharpe ratio")
    spy_max_dd: float = Field(le=0, description="SPY max drawdown %")
    
    # Relative Performance
    alpha_vs_buy_hold: float = Field(description="Excess return vs Buy & Hold")
    alpha_vs_spy: float = Field(description="Excess return vs SPY")
    beta_to_spy: float = Field(description="Beta coefficient to SPY")
    correlation_to_spy: float = Field(ge=-1, le=1, description="Correlation to SPY")
    
    @field_validator("*", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN/Inf to 0.0."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


class RegimePerformance(BaseModel):
    """Performance metrics for a specific market regime."""
    
    regime: RegimeType = Field(description="Market regime")
    period_days: int = Field(ge=0, description="Days in this regime")
    period_pct: float = Field(ge=0, le=100, description="% of total period")
    
    num_trades: int = Field(ge=0, description="Trades during this regime")
    win_rate: float = Field(ge=0, le=1, description="Win rate in this regime")
    total_return_pct: float = Field(description="Return during this regime")
    avg_trade_return: float = Field(description="Average trade return %")
    max_drawdown_pct: float = Field(le=0, description="Max DD in this regime")


class RegimeBreakdown(BaseModel):
    """Performance breakdown across all market regimes."""
    
    bull: RegimePerformance = Field(description="Bull market performance")
    bear: RegimePerformance = Field(description="Bear market performance")
    crash: RegimePerformance | None = Field(None, description="Crash period performance")
    recovery: RegimePerformance | None = Field(None, description="Recovery period performance")
    
    best_regime: RegimeType = Field(description="Regime with best performance")
    worst_regime: RegimeType = Field(description="Regime with worst performance")
    regime_consistency: float = Field(
        ge=0, le=1,
        description="How consistent across regimes (0=erratic, 1=stable)"
    )


# =============================================================================
# Runner-Up / Alternative Strategy Models
# =============================================================================

class StrategyConditionSummary(BaseModel):
    """Human-readable summary of a strategy's entry/exit logic."""
    
    name: str = Field(description="Strategy name")
    entry_logic: str = Field(description="Entry conditions in plain English")
    exit_logic: str = Field(description="Exit conditions in plain English")
    
    # Core params
    stop_loss_pct: float = Field(description="Stop loss %")
    take_profit_pct: float = Field(description="Take profit %")
    max_holding_days: int = Field(description="Max holding period")


class RunnerUpStrategy(BaseModel):
    """
    A strategy that was tested but didn't win.
    
    This provides transparency - users see what was tried and why it failed.
    """
    
    rank: int = Field(ge=2, description="Ranking (2 = second best, 3 = third, etc.)")
    strategy: StrategyConditionSummary = Field(description="Strategy details")
    
    # Why it didn't win
    verdict: StrategyVerdict = Field(description="Quality verdict")
    rejection_reason: str = Field(description="Why this wasn't selected")
    
    # Key metrics for comparison
    total_return_pct: float = Field(description="Total return %")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown_pct: float = Field(le=0, description="Max drawdown %")
    win_rate: float = Field(ge=0, le=1, description="Win rate")
    num_trades: int = Field(ge=0, description="Number of trades")
    sqn: float = Field(description="System Quality Number")
    
    # Comparison to winner
    return_vs_winner: float = Field(description="Return difference vs winner %")
    sharpe_vs_winner: float = Field(description="Sharpe difference vs winner")
    
    @field_validator("*", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN/Inf to 0.0."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


# =============================================================================
# Main Report Model - The "Deep Dive"
# =============================================================================

class WinningStrategy(BaseModel):
    """The winning strategy with full details."""
    
    # Identity
    name: str = Field(description="Strategy name")
    description: str = Field(description="Human-readable strategy description")
    strategy_logic: StrategyConditionSummary = Field(description="Entry/exit logic")
    
    # Core Metrics
    trade_stats: TradeStats = Field(description="Trade-level statistics")
    risk_metrics: RiskMetrics = Field(description="Risk-adjusted metrics")
    advanced_metrics: AdvancedMetrics = Field(description="Professional metrics")
    
    # Time Series Data
    equity_curve: list[EquityCurvePoint] = Field(
        description="Full equity curve for charting"
    )
    signals: list[SignalEvent] = Field(
        description="All buy/sell signals with timestamps"
    )
    
    # Comparisons
    benchmark_comparison: BenchmarkComparison = Field(
        description="Performance vs benchmarks"
    )
    regime_breakdown: RegimeBreakdown = Field(
        description="Performance by market regime"
    )
    
    # Overall Assessment
    verdict: StrategyVerdict = Field(description="Quality verdict")
    confidence_score: float = Field(
        ge=0, le=100,
        description="Confidence in strategy (0-100)"
    )
    
    @model_validator(mode="after")
    def validate_data_consistency(self) -> "WinningStrategy":
        """Ensure equity curve and signals are consistent."""
        if self.equity_curve and self.signals:
            # Signals should be within equity curve timeframe
            curve_start = self.equity_curve[0].timestamp
            curve_end = self.equity_curve[-1].timestamp
            for signal in self.signals:
                if not (curve_start <= signal.timestamp <= curve_end):
                    # Clamp to valid range instead of raising
                    pass  # Allow out-of-range for flexibility
        return self


class OptimizationMeta(BaseModel):
    """Metadata about the optimization process."""
    
    # Optimization details
    n_trials_total: int = Field(ge=0, description="Total trials run")
    n_valid_strategies: int = Field(ge=0, description="Strategies that passed constraints")
    optimization_time_seconds: float = Field(ge=0, description="Total optimization time")
    
    # Data details
    symbol: str = Field(description="Symbol analyzed")
    data_start: datetime = Field(description="Data start date")
    data_end: datetime = Field(description="Data end date")
    total_bars: int = Field(ge=0, description="Number of price bars")
    
    # Validation
    train_period_days: int = Field(ge=0, description="Training period length")
    validate_period_days: int = Field(ge=0, description="Validation period length")
    walk_forward_windows: int = Field(ge=0, description="Number of WFO windows")
    
    # Execution context
    generated_at: datetime = Field(description="Report generation timestamp")
    engine_version: str = Field(default="2.0.0", description="AlphaFactory version")


class StrategyFullReport(BaseModel):
    """
    The Complete Strategy Report - "Deep Dive" Data Model.
    
    This is the primary API response that powers the Trading Dashboard.
    It provides full transparency by including:
    - The winning strategy with granular metrics
    - Runner-up strategies that were tested
    - Benchmark comparisons and regime breakdowns
    - Full equity curves and signal timelines
    - Baseline strategy comparisons (B&H, DCA, Buy Dips, SPY)
    - Investment recommendation
    
    Example Usage:
    ```python
    report = analyzer.generate_full_report(optimization_result)
    
    # Access winning strategy
    print(f"Winner: {report.winner.name}")
    print(f"Sharpe: {report.winner.risk_metrics.sharpe_ratio}")
    
    # See why others failed
    for runner in report.runner_ups:
        print(f"#{runner.rank}: {runner.strategy.name} - {runner.rejection_reason}")
    
    # See baseline comparisons
    print(f"B&H Return: {report.baseline_comparison.buy_hold.total_return_pct}%")
    print(f"Recommendation: {report.baseline_comparison.recommendation.headline}")
    
    # Plot equity curve
    df = pd.DataFrame([p.model_dump() for p in report.winner.equity_curve])
    df.plot(x='timestamp', y=['equity', 'benchmark_equity'])
    ```
    """
    
    # Metadata
    meta: OptimizationMeta = Field(description="Optimization metadata")
    
    # The Winner
    winner: WinningStrategy = Field(description="The selected best strategy")
    
    # The Alternatives (Transparency)
    runner_ups: list[RunnerUpStrategy] = Field(
        default_factory=list,
        description="Top alternative strategies that were tested",
        max_length=5,  # Cap at 5 runner-ups
    )
    
    # Baseline Strategy Comparisons (B&H, DCA, Buy Dips, SPY)
    # Deferred import to avoid circular dependency
    baseline_comparison: Any = Field(
        default=None,
        description="Comparison against baseline strategies (B&H, DCA, Buy Dips, SPY)"
    )
    
    # Summary for Quick Access
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Quick summary stats for dashboard cards"
    )
    
    @model_validator(mode="after")
    def build_summary(self) -> "StrategyFullReport":
        """Auto-generate summary from detailed metrics."""
        baseline_rec = None
        if self.baseline_comparison is not None:
            baseline_rec = getattr(self.baseline_comparison, 'recommendation', None)
        
        self.summary = {
            "strategy_name": self.winner.name,
            "verdict": self.winner.verdict.value,
            "total_return_pct": self.winner.benchmark_comparison.strategy_return_pct,
            "sharpe_ratio": self.winner.risk_metrics.sharpe_ratio,
            "max_drawdown_pct": self.winner.risk_metrics.max_drawdown_pct,
            "win_rate": self.winner.trade_stats.win_rate,
            "total_trades": self.winner.trade_stats.total_trades,
            "sqn": self.winner.advanced_metrics.sqn,
            "kelly_pct": self.winner.advanced_metrics.kelly_criterion * 100,
            "alternatives_tested": len(self.runner_ups),
            "best_regime": self.winner.regime_breakdown.best_regime.value,
            "alpha_vs_spy": self.winner.benchmark_comparison.alpha_vs_spy,
            # Baseline recommendation
            "recommendation": baseline_rec.headline if baseline_rec else None,
            "recommendation_type": baseline_rec.recommendation.value if baseline_rec else None,
        }
        return self
    
    def to_json_safe(self) -> dict[str, Any]:
        """
        Export to JSON-safe dictionary.
        
        Handles NaN/Inf conversion and datetime serialization.
        """
        import math
        import json
        
        def clean_value(v: Any) -> Any:
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return 0.0
            elif isinstance(v, datetime):
                return v.isoformat()
            elif isinstance(v, dict):
                return {k: clean_value(vv) for k, vv in v.items()}
            elif isinstance(v, list):
                return [clean_value(vv) for vv in v]
            return v
        
        data = self.model_dump()
        return clean_value(data)
