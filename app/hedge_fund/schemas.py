"""
Pydantic schemas for the hedge fund analysis module.

All data structures used across agents, orchestrator, and LLM gateway.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class Signal(str, Enum):
    """Trading signal strength."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class LLMMode(str, Enum):
    """LLM execution mode."""

    REALTIME = "realtime"
    BATCH = "batch"


class AgentType(str, Enum):
    """Types of agents in the system."""

    FUNDAMENTALS = "fundamentals"
    TECHNICALS = "technicals"
    VALUATION = "valuation"
    SENTIMENT = "sentiment"
    RISK = "risk"
    PERSONA = "persona"
    PORTFOLIO = "portfolio"
    EXTERNAL = "external"


# =============================================================================
# Input Schemas
# =============================================================================


class TickerInput(BaseModel):
    """Input for analysis - single ticker with optional date range."""

    symbol: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    start_date: Optional[date] = Field(
        None, description="Start date for historical data"
    )
    end_date: Optional[date] = Field(None, description="End date for historical data")

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()


class AnalysisRequest(BaseModel):
    """Full analysis request with multiple tickers."""

    tickers: list[TickerInput] = Field(..., min_length=1)
    run_id: Optional[str] = Field(
        None, description="Optional run ID for batch tracking"
    )
    mode: LLMMode = Field(
        LLMMode.REALTIME, description="LLM execution mode"
    )
    agents: Optional[list[str]] = Field(
        None, description="Specific agents to run (None = all)"
    )
    personas: Optional[list[str]] = Field(
        None, description="Specific investor personas to use"
    )


# =============================================================================
# Market Data Schemas
# =============================================================================


class PricePoint(BaseModel):
    """Single OHLCV price point."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


class PriceSeries(BaseModel):
    """Historical price series for a ticker."""

    symbol: str
    prices: list[PricePoint]
    currency: str = "USD"

    @property
    def latest(self) -> Optional[PricePoint]:
        return self.prices[-1] if self.prices else None

    @property
    def returns(self) -> list[float]:
        """Calculate daily returns."""
        if len(self.prices) < 2:
            return []
        return [
            (self.prices[i].close - self.prices[i - 1].close) / self.prices[i - 1].close
            for i in range(1, len(self.prices))
        ]


class Fundamentals(BaseModel):
    """Fundamental data for a company."""

    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None

    # Valuation ratios
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None

    # Growth
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None

    # Financial health
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    interest_coverage: Optional[float] = None
    free_cash_flow: Optional[float] = None

    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Per-share data
    eps: Optional[float] = None
    eps_forward: Optional[float] = None
    book_value_per_share: Optional[float] = None
    revenue_per_share: Optional[float] = None

    # Additional
    beta: Optional[float] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    short_ratio: Optional[float] = None
    short_percent_of_float: Optional[float] = None

    # Raw info for any extra fields
    raw_info: Optional[dict[str, Any]] = None


class CalendarEvents(BaseModel):
    """Upcoming calendar events for a company."""

    symbol: str
    next_earnings_date: Optional[date] = None
    ex_dividend_date: Optional[date] = None
    dividend_date: Optional[date] = None


class MarketData(BaseModel):
    """Combined market data for a ticker."""

    symbol: str
    prices: PriceSeries
    fundamentals: Fundamentals
    calendar: Optional[CalendarEvents] = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Agent Output Schemas
# =============================================================================


class AgentSignal(BaseModel):
    """Output from a single agent for a single ticker."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    agent_type: AgentType
    symbol: str
    signal: Signal
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str = Field(..., description="Explanation of the signal")
    key_factors: list[str] = Field(
        default_factory=list, description="Key factors driving the signal"
    )
    metrics: Optional[dict[str, Any]] = Field(
        None, description="Relevant metrics/calculations"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: float | int) -> float:
        """Normalize confidence to 0-1 range."""
        if isinstance(v, int) and v > 1:
            # Convert 1-10 scale to 0-1
            return v / 10.0
        return float(v)


class PerTickerReport(BaseModel):
    """Aggregated report for a single ticker from all agents."""

    symbol: str
    signals: list[AgentSignal]
    consensus_signal: Signal
    consensus_confidence: float = Field(..., ge=0.0, le=1.0)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    summary: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def agent_agreement(self) -> float:
        """Calculate agreement level among agents (0-1)."""
        if not self.signals:
            return 0.0
        max_count = max(self.bullish_count, self.bearish_count, self.neutral_count)
        return max_count / len(self.signals)


class PortfolioDecision(BaseModel):
    """Final portfolio decision with allocation."""

    symbol: str
    action: Signal
    allocation_pct: float = Field(
        ..., ge=0.0, le=1.0, description="Suggested allocation as % of portfolio"
    )
    position_size: Optional[float] = Field(
        None, description="Dollar amount if portfolio size known"
    )
    stop_loss_pct: Optional[float] = Field(
        None, description="Suggested stop loss percentage"
    )
    take_profit_pct: Optional[float] = Field(
        None, description="Suggested take profit percentage"
    )
    reasoning: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk level 0-1")


class AnalysisBundle(BaseModel):
    """Complete analysis output for all tickers."""

    run_id: str
    mode: LLMMode
    tickers: list[str]
    reports: list[PerTickerReport]
    portfolio_decisions: list[PortfolioDecision]
    total_agents_run: int
    successful_agents: int
    failed_agents: int
    execution_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: list[str] = Field(default_factory=list)


# =============================================================================
# LLM Schemas
# =============================================================================


class LLMTask(BaseModel):
    """Task to be sent to LLM gateway."""

    custom_id: str = Field(..., description="Deterministic ID for tracking")
    agent_id: str
    symbol: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 1000
    temperature: float = 0.7
    require_json: bool = False
    json_schema: Optional[dict[str, Any]] = None


class LLMResult(BaseModel):
    """Result from LLM gateway."""

    custom_id: str
    agent_id: str
    symbol: str
    content: str
    parsed_json: Optional[dict[str, Any]] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    failed: bool = False


class BatchStatus(BaseModel):
    """Status of a batch job."""

    batch_id: str
    status: str
    total_count: int
    completed_count: int
    failed_count: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_file_id: Optional[str] = None


# =============================================================================
# Investor Persona Schemas
# =============================================================================


class InvestorPersona(BaseModel):
    """Configuration for an investor persona."""

    id: str = Field(..., description="Unique persona ID (e.g., 'warren_buffett')")
    name: str = Field(..., description="Display name (e.g., 'Warren Buffett')")
    philosophy: str = Field(..., description="Core investment philosophy")
    focus_areas: list[str] = Field(
        default_factory=list, description="Key areas this investor focuses on"
    )
    system_prompt: str = Field(..., description="System prompt for LLM")
    key_metrics: list[str] = Field(
        default_factory=list, description="Metrics this investor prioritizes"
    )
    risk_tolerance: str = Field("moderate", description="Risk tolerance level")


# =============================================================================
# Technical Analysis Schemas
# =============================================================================


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    symbol: str

    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # RSI
    rsi_14: Optional[float] = None

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None

    # Momentum
    momentum_10: Optional[float] = None
    roc_10: Optional[float] = None

    # Volatility
    atr_14: Optional[float] = None
    volatility_20: Optional[float] = None

    # Volume
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    obv: Optional[float] = None

    # Trend
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None

    # Support/Resistance
    pivot_point: Optional[float] = None
    support_1: Optional[float] = None
    support_2: Optional[float] = None
    resistance_1: Optional[float] = None
    resistance_2: Optional[float] = None

    # Current price context
    current_price: Optional[float] = None
    price_vs_sma_20: Optional[float] = None  # % above/below
    price_vs_sma_50: Optional[float] = None
    price_vs_sma_200: Optional[float] = None


class ValuationMetrics(BaseModel):
    """Valuation analysis output."""

    symbol: str

    # DCF components
    dcf_value: Optional[float] = None
    current_price: Optional[float] = None
    intrinsic_value: Optional[float] = None
    margin_of_safety: Optional[float] = None

    # Relative valuation
    pe_vs_sector: Optional[float] = None
    pb_vs_sector: Optional[float] = None
    ev_ebitda_vs_sector: Optional[float] = None

    # Growth-adjusted
    peg_assessment: Optional[str] = None
    growth_rate_used: Optional[float] = None

    # Owner earnings (Buffett method)
    owner_earnings: Optional[float] = None
    owner_earnings_yield: Optional[float] = None

    # Summary
    valuation_grade: Optional[str] = None  # A, B, C, D, F
    is_undervalued: Optional[bool] = None
    upside_potential: Optional[float] = None


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""

    symbol: str

    # Volatility measures
    volatility_30d: Optional[float] = None
    volatility_90d: Optional[float] = None
    beta: Optional[float] = None

    # Drawdown
    max_drawdown_1y: Optional[float] = None
    current_drawdown: Optional[float] = None

    # Financial risk
    debt_risk_score: Optional[float] = None
    liquidity_risk_score: Optional[float] = None

    # Market risk
    correlation_to_spy: Optional[float] = None
    sector_concentration_risk: Optional[float] = None

    # Aggregate
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_grade: str  # Low, Medium, High, Very High
    risk_factors: list[str] = Field(default_factory=list)


class SentimentMetrics(BaseModel):
    """Sentiment analysis metrics."""

    symbol: str

    # Analyst sentiment
    analyst_rating: Optional[str] = None  # Buy, Hold, Sell
    analyst_count: Optional[int] = None
    target_price_mean: Optional[float] = None
    target_price_high: Optional[float] = None
    target_price_low: Optional[float] = None
    target_upside: Optional[float] = None

    # Institutional
    institutional_ownership: Optional[float] = None
    institutional_change_qoq: Optional[float] = None

    # Short interest
    short_interest_ratio: Optional[float] = None
    short_percent_float: Optional[float] = None
    days_to_cover: Optional[float] = None

    # Insider activity
    insider_net_shares_90d: Optional[float] = None
    insider_sentiment: Optional[str] = None  # Bullish, Neutral, Bearish

    # Aggregate
    overall_sentiment: str  # Very Bullish, Bullish, Neutral, Bearish, Very Bearish
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
