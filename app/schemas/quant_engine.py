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


# ============================================================================
# Global Recommendations Schemas (for Landing/Dashboard - no auth)
# ============================================================================


class QuantRecommendation(BaseModel):
    """Individual stock recommendation for the global endpoint.
    
    Combines signal scanner data with dip finder state for a unified view.
    """
    
    model_config = ConfigDict(from_attributes=True)
    
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str | None = Field(None, description="Company name")
    action: str = Field("HOLD", description="BUY, SELL, or HOLD")
    notional_eur: float = Field(0.0, description="Suggested notional in EUR")
    delta_weight: float = Field(0.0, description="Change in weight (-1 to 1)")
    target_weight: float = Field(0.0, description="Target weight in model portfolio")
    
    # Price data
    last_price: float | None = Field(None, description="Latest stock price")
    change_percent: float | None = Field(None, description="Daily price change percentage")
    market_cap: float | None = Field(None, description="Market capitalization")
    
    # Sector info
    sector: str | None = Field(None, description="Stock sector")
    sector_etf: str | None = Field(None, description="Sector benchmark ETF symbol")
    
    # Signal-based metrics
    mu_hat: float = Field(0.0, description="Expected return estimate from signals")
    uncertainty: float = Field(1.0, description="Uncertainty in return estimate")
    risk_contribution: float = Field(0.0, description="Contribution to overall risk")
    
    # Top technical signal
    top_signal_name: str | None = Field(None, description="Name of top technical signal")
    top_signal_is_buy: bool = Field(False, description="Whether top signal is a buy signal")
    top_signal_strength: float = Field(0.0, description="Strength of top signal (0-1)")
    top_signal_description: str | None = Field(None, description="Description of top signal")
    
    # Opportunity score
    opportunity_score: float | None = Field(None, description="Composite opportunity score 0-100")
    opportunity_rating: str | None = Field(None, description="Rating: strong_buy, buy, hold, avoid")
    
    # Dip-based metrics
    dip_score: float | None = Field(None, description="Dip score (z-score or buy_score)")
    dip_bucket: str | None = Field(None, description="Dip bucket classification")
    marginal_utility: float = Field(0.0, description="Marginal utility for ranking")
    
    # Legacy compatibility fields
    legacy_dip_pct: float | None = Field(None, description="Percentage below ATH")
    legacy_days_in_dip: int | None = Field(None, description="Days in current dip")
    legacy_domain_score: float | None = Field(None, description="Domain-specific score")
    
    # AI Analysis (if available)
    ai_summary: str | None = Field(None, description="AI-generated analysis snippet")
    ai_rating: str | None = Field(None, description="AI rating: strong_buy, buy, hold, sell, strong_sell")


class QuantAuditBlock(BaseModel):
    """Audit block for transparency in recommendations."""
    
    model_config = ConfigDict(from_attributes=True)
    
    timestamp: str = Field(..., description="Generation timestamp ISO format")
    config_hash: int = Field(0, description="Hash of configuration used")
    mu_hat_summary: dict = Field(default_factory=dict, description="Summary of return estimates")
    risk_model_summary: dict = Field(default_factory=dict, description="Risk model parameters")
    optimizer_status: str = Field("success", description="Optimizer status")
    constraint_binding: list[str] = Field(default_factory=list, description="Active constraints")
    turnover_realized: float = Field(0.0, description="Turnover fraction")
    regime_state: str = Field("unknown", description="Current market regime")
    dip_stats: dict | None = Field(None, description="Dip statistics")
    error_message: str | None = Field(None, description="Error if any")


class QuantEngineResponse(BaseModel):
    """Response for the global /recommendations endpoint.
    
    Used by landing page and dashboard to display ranked stocks.
    """
    
    model_config = ConfigDict(from_attributes=True)
    
    recommendations: list[QuantRecommendation] = Field(
        ..., description="Ranked stock recommendations"
    )
    as_of_date: str = Field(..., description="Data as-of date")
    portfolio_value_eur: float = Field(0.0, description="Model portfolio value")
    inflow_eur: float = Field(1000.0, description="Inflow amount used")
    total_trades: int = Field(0, description="Number of trade recommendations")
    total_transaction_cost_eur: float = Field(0.0, description="Estimated transaction costs")
    expected_portfolio_return: float = Field(0.0, description="Expected annual return")
    expected_portfolio_risk: float = Field(0.0, description="Expected annual volatility")
    audit: QuantAuditBlock = Field(..., description="Audit trail for transparency")


