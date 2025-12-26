"""
Pydantic schemas for the hedge fund analysis module.

All data structures used across agents, orchestrator, and LLM gateway.
"""

from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

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
    start_date: date | None = Field(
        None, description="Start date for historical data"
    )
    end_date: date | None = Field(None, description="End date for historical data")

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()


class AnalysisRequest(BaseModel):
    """Full analysis request with multiple tickers."""

    tickers: list[TickerInput] = Field(..., min_length=1)
    run_id: str | None = Field(
        None, description="Optional run ID for batch tracking"
    )
    mode: LLMMode = Field(
        LLMMode.REALTIME, description="LLM execution mode"
    )
    agents: list[str] | None = Field(
        None, description="Specific agents to run (None = all)"
    )
    personas: list[str] | None = Field(
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
    adj_close: float | None = None


class PriceSeries(BaseModel):
    """Historical price series for a ticker."""

    symbol: str
    prices: list[PricePoint]
    currency: str = "USD"

    @property
    def latest(self) -> PricePoint | None:
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
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None

    # Valuation ratios
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_revenue: float | None = None

    # Profitability
    profit_margin: float | None = None
    operating_margin: float | None = None
    gross_margin: float | None = None
    roe: float | None = None
    roa: float | None = None
    roic: float | None = None

    # Growth
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None

    # Financial health
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_assets: float | None = None
    interest_coverage: float | None = None
    free_cash_flow: float | None = None

    # Dividends
    dividend_yield: float | None = None
    payout_ratio: float | None = None

    # Per-share data
    eps: float | None = None
    eps_forward: float | None = None
    book_value_per_share: float | None = None
    revenue_per_share: float | None = None

    # Additional
    beta: float | None = None
    shares_outstanding: float | None = None
    float_shares: float | None = None
    short_ratio: float | None = None
    short_percent_of_float: float | None = None

    # Domain classification (for domain-specific analysis)
    domain: str | None = Field(None, description="Domain type: bank, reit, insurer, etf, etc.")

    # Financial statement data (domain-specific metrics)
    financials: dict[str, Any] | None = Field(
        None,
        description="Financial statement data for domain-specific analysis"
    )

    # Domain-specific computed metrics
    net_interest_income: float | None = Field(None, description="For banks: Net Interest Income")
    net_interest_margin: float | None = Field(None, description="For banks: NIM ratio")
    ffo: float | None = Field(None, description="For REITs: Funds From Operations")
    ffo_per_share: float | None = Field(None, description="For REITs: FFO per share")
    p_ffo: float | None = Field(None, description="For REITs: Price to FFO ratio")
    loss_ratio: float | None = Field(None, description="For insurers: Loss ratio")

    # Raw info for any extra fields
    raw_info: dict[str, Any] | None = None


class CalendarEvents(BaseModel):
    """Upcoming calendar events for a company."""

    symbol: str
    next_earnings_date: date | None = None
    ex_dividend_date: date | None = None
    dividend_date: date | None = None


class MarketData(BaseModel):
    """Combined market data for a ticker."""

    symbol: str
    prices: PriceSeries
    fundamentals: Fundamentals
    calendar: CalendarEvents | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    metrics: dict[str, Any] | None = Field(
        None, description="Relevant metrics/calculations"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

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
    summary: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

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
    position_size: float | None = Field(
        None, description="Dollar amount if portfolio size known"
    )
    stop_loss_pct: float | None = Field(
        None, description="Suggested stop loss percentage"
    )
    take_profit_pct: float | None = Field(
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    json_schema: dict[str, Any] | None = None


class LLMResult(BaseModel):
    """Result from LLM gateway."""

    custom_id: str
    agent_id: str
    symbol: str
    content: str
    parsed_json: dict[str, Any] | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    failed: bool = False


class BatchStatus(BaseModel):
    """Status of a batch job."""

    batch_id: str
    status: str
    total_count: int
    completed_count: int
    failed_count: int
    created_at: datetime
    completed_at: datetime | None = None
    output_file_id: str | None = None


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
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None

    # MACD
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None

    # RSI
    rsi_14: float | None = None

    # Bollinger Bands
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None

    # Momentum
    momentum_10: float | None = None
    roc_10: float | None = None

    # Volatility
    atr_14: float | None = None
    volatility_20: float | None = None

    # Volume
    volume_sma_20: float | None = None
    volume_ratio: float | None = None
    obv: float | None = None

    # Trend
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None

    # Support/Resistance
    pivot_point: float | None = None
    support_1: float | None = None
    support_2: float | None = None
    resistance_1: float | None = None
    resistance_2: float | None = None

    # Current price context
    current_price: float | None = None
    price_vs_sma_20: float | None = None  # % above/below
    price_vs_sma_50: float | None = None
    price_vs_sma_200: float | None = None


class ValuationMetrics(BaseModel):
    """Valuation analysis output."""

    symbol: str

    # DCF components
    dcf_value: float | None = None
    current_price: float | None = None
    intrinsic_value: float | None = None
    margin_of_safety: float | None = None

    # Relative valuation
    pe_vs_sector: float | None = None
    pb_vs_sector: float | None = None
    ev_ebitda_vs_sector: float | None = None

    # Growth-adjusted
    peg_assessment: str | None = None
    growth_rate_used: float | None = None

    # Owner earnings (Buffett method)
    owner_earnings: float | None = None
    owner_earnings_yield: float | None = None

    # Summary
    valuation_grade: str | None = None  # A, B, C, D, F
    is_undervalued: bool | None = None
    upside_potential: float | None = None


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""

    symbol: str

    # Volatility measures
    volatility_30d: float | None = None
    volatility_90d: float | None = None
    beta: float | None = None

    # Drawdown
    max_drawdown_1y: float | None = None
    current_drawdown: float | None = None

    # Financial risk
    debt_risk_score: float | None = None
    liquidity_risk_score: float | None = None

    # Market risk
    correlation_to_spy: float | None = None
    sector_concentration_risk: float | None = None

    # Aggregate
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_grade: str  # Low, Medium, High, Very High
    risk_factors: list[str] = Field(default_factory=list)


class SentimentMetrics(BaseModel):
    """Sentiment analysis metrics."""

    symbol: str

    # Analyst sentiment
    analyst_rating: str | None = None  # Buy, Hold, Sell
    analyst_count: int | None = None
    target_price_mean: float | None = None
    target_price_high: float | None = None
    target_price_low: float | None = None
    target_upside: float | None = None

    # Institutional
    institutional_ownership: float | None = None
    institutional_change_qoq: float | None = None

    # Short interest
    short_interest_ratio: float | None = None
    short_percent_float: float | None = None
    days_to_cover: float | None = None

    # Insider activity
    insider_net_shares_90d: float | None = None
    insider_sentiment: str | None = None  # Bullish, Neutral, Bearish

    # Aggregate
    overall_sentiment: str  # Very Bullish, Bullish, Neutral, Bearish, Very Bearish
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
