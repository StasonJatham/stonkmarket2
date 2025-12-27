"""
Pydantic schemas for the Quantitative Portfolio Engine V2 API.

Non-predictive, risk-based portfolio optimization.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Signal Scanner Schemas
# ============================================================================


class SignalResultResponse(BaseModel):
    """A single technical signal result with optimized parameters."""
    
    model_config = ConfigDict(from_attributes=True)
    
    name: str = Field(..., description="Signal name (e.g., 'RSI Oversold')")
    description: str = Field("", description="Human-readable description")
    value: float = Field(..., description="Current signal value")
    is_buy_signal: bool = Field(..., description="Is this currently a buy signal?")
    strength: float = Field(..., description="Signal strength 0-1")
    
    # Optimized parameters
    optimal_threshold: float = Field(..., description="Best threshold for this stock")
    optimal_holding_days: int = Field(..., description="Best holding period after signal triggers")
    
    # Backtest results at optimal params
    win_rate: float = Field(..., description="Historical win rate (0-1)")
    avg_return_pct: float = Field(..., description="Average return when signal triggered")
    max_return_pct: float = Field(0.0, description="Best historical return")
    min_return_pct: float = Field(0.0, description="Worst historical return")
    n_signals: int = Field(..., description="Number of historical signals backtested")
    
    # Improvement from optimization
    improvement_pct: float = Field(0.0, description="Improvement vs default parameters")


class StockSignalResponse(BaseModel):
    """Signal scan results for a single stock."""
    
    model_config = ConfigDict(from_attributes=True)
    
    symbol: str = Field(..., description="Stock ticker")
    name: str = Field(..., description="Company name")
    buy_score: float = Field(..., description="Overall buy score 0-100")
    opportunity_type: str = Field(..., description="STRONG_BUY, BUY, WEAK_BUY, NEUTRAL")
    opportunity_reason: str = Field(..., description="Explanation of opportunity")
    
    # Current metrics
    current_price: float = Field(..., description="Current price")
    price_vs_52w_high_pct: float = Field(..., description="% from 52-week high (negative = below)")
    price_vs_52w_low_pct: float = Field(..., description="% from 52-week low (positive = above)")
    zscore_20d: float = Field(..., description="Z-score vs 20-day mean")
    zscore_60d: float = Field(..., description="Z-score vs 60-day mean")
    rsi_14: float = Field(50.0, description="Current RSI (14-day)")
    
    # Best recommendation
    best_signal_name: str = Field("", description="Best signal for this stock")
    best_holding_days: int = Field(0, description="Recommended holding period after buy")
    best_expected_return: float = Field(0.0, description="Expected return (win_rate * avg_return)")
    
    # Top signals
    signals: list[SignalResultResponse] = Field(
        default_factory=list, description="Top technical signals for this stock"
    )
    active_buy_signals: list[SignalResultResponse] = Field(
        default_factory=list, description="Currently active buy signals"
    )


class SignalScanResponse(BaseModel):
    """Response from the signal scanner."""
    
    scanned_at: datetime = Field(..., description="Scan timestamp")
    holding_days_tested: list[int] = Field(
        default_factory=lambda: [5, 10, 20, 40, 60],
        description="Holding periods tested during optimization"
    )
    stocks: list[StockSignalResponse] = Field(
        ..., description="Stocks ranked by buy opportunity"
    )
    top_opportunities: list[str] = Field(
        ..., description="Top 3 stock symbols by buy score"
    )
    n_active_signals: int = Field(0, description="Total active buy signals across all stocks")


# ============================================================================
# Portfolio Analytics Schemas
# ============================================================================


class RiskMetricsResponse(BaseModel):
    """Risk metrics for a portfolio."""
    
    portfolio_volatility: float = Field(..., description="Annualized portfolio volatility")
    var_95: float = Field(..., description="Daily Value at Risk (95%)")
    cvar_95: float = Field(..., description="Daily Expected Shortfall (95%)")
    max_drawdown: float = Field(..., description="Maximum historical drawdown")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, VERY_HIGH")


class DiversificationResponse(BaseModel):
    """Diversification metrics."""
    
    effective_n: float = Field(..., description="Effective number of independent positions")
    diversification_ratio: float = Field(..., description="Weighted avg vol / portfolio vol")
    hhi: float = Field(..., description="Herfindahl-Hirschman Index")
    is_well_diversified: bool = Field(..., description="Whether portfolio is well diversified")


class RegimeResponse(BaseModel):
    """Current market regime."""
    
    regime: str = Field(..., description="Combined regime label (e.g., 'bull_low')")
    trend: str = Field(..., description="Trend direction: bull, bear, neutral")
    volatility: str = Field(..., description="Volatility level: low, medium, high")
    description: str = Field(..., description="Human-readable regime description")
    recommendation: str = Field(..., description="Risk budget recommendation")


class PortfolioAnalyticsResponse(BaseModel):
    """Complete portfolio analytics response."""
    
    portfolio_id: int = Field(..., description="Portfolio ID")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
    total_value_eur: float = Field(..., description="Total portfolio value")
    n_positions: int = Field(..., description="Number of positions")
    risk_score: int = Field(..., ge=1, le=10, description="Overall risk score 1-10")
    
    # User-friendly summary
    summary: dict = Field(..., description="Risk summary for display")
    risk: dict = Field(..., description="Risk explanation in plain language")
    diversification: dict = Field(..., description="Diversification status")
    market: dict = Field(..., description="Market regime info")
    
    insights: list[str] = Field(..., description="Key insights")
    action_items: list[str] = Field(..., description="Recommended actions")


# ============================================================================
# Allocation Recommendation Schemas
# ============================================================================


class TradeRecommendation(BaseModel):
    """Single trade recommendation."""
    
    symbol: str = Field(..., description="Asset ticker")
    action: str = Field(..., description="BUY or SELL")
    amount_eur: float = Field(..., description="Trade amount in EUR")
    current_weight_pct: float = Field(..., description="Current weight percentage")
    target_weight_pct: float = Field(..., description="Target weight percentage")
    reason: str = Field(..., description="Reason for trade")


class AllocationResponse(BaseModel):
    """Allocation recommendation response."""
    
    portfolio_id: int = Field(..., description="Portfolio ID")
    method: str = Field(..., description="Optimization method used")
    inflow_eur: float = Field(..., description="Investment amount")
    portfolio_value_eur: float = Field(..., description="Current portfolio value")
    
    confidence: str = Field(..., description="Confidence level: LOW, MEDIUM, HIGH")
    explanation: str = Field(..., description="Method explanation")
    risk_improvement: str = Field(..., description="Summary of risk change")
    
    current_risk: dict = Field(..., description="Current risk metrics")
    optimal_risk: dict = Field(..., description="Optimal risk metrics")
    
    trades: list[TradeRecommendation] = Field(..., description="Recommended trades")
    warnings: list[str] = Field(default_factory=list, description="Warnings")

