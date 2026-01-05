"""Notification system schemas.

Defines TriggerType enum, channel configuration, rule definitions,
and API request/response models for the notification system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS
# =============================================================================


class TriggerType(str, Enum):
    """Notification trigger types derived from available data sources.
    
    Categories:
    - Price & Dip: Stock price movement and dip detection
    - Signals: DipFinder and strategy signals
    - Fundamentals: Valuation and company metrics
    - AI Analysis: AI persona ratings and confidence
    - Portfolio: Portfolio value and risk metrics
    - Watchlist: Watched stock events
    """
    
    # Price & Dip Triggers
    PRICE_DROPS_BELOW = "PRICE_DROPS_BELOW"
    PRICE_RISES_ABOVE = "PRICE_RISES_ABOVE"
    DIP_EXCEEDS_PERCENT = "DIP_EXCEEDS_PERCENT"
    DIP_DURATION_EXCEEDS = "DIP_DURATION_EXCEEDS"
    TAIL_EVENT_DETECTED = "TAIL_EVENT_DETECTED"
    NEW_ATH_REACHED = "NEW_ATH_REACHED"
    
    # Signal Triggers
    DIPFINDER_ALERT = "DIPFINDER_ALERT"
    DIPFINDER_SCORE_ABOVE = "DIPFINDER_SCORE_ABOVE"
    STRATEGY_SIGNAL_BUY = "STRATEGY_SIGNAL_BUY"
    STRATEGY_SIGNAL_SELL = "STRATEGY_SIGNAL_SELL"
    ENTRY_SIGNAL_TRIGGERED = "ENTRY_SIGNAL_TRIGGERED"
    WIN_RATE_ABOVE = "WIN_RATE_ABOVE"
    
    # Fundamental Triggers
    PE_RATIO_BELOW = "PE_RATIO_BELOW"
    PE_RATIO_ABOVE = "PE_RATIO_ABOVE"
    ANALYST_UPGRADE = "ANALYST_UPGRADE"
    ANALYST_DOWNGRADE = "ANALYST_DOWNGRADE"
    PRICE_BELOW_TARGET = "PRICE_BELOW_TARGET"
    EARNINGS_APPROACHING = "EARNINGS_APPROACHING"
    QUALITY_SCORE_ABOVE = "QUALITY_SCORE_ABOVE"
    MOMENTUM_SCORE_ABOVE = "MOMENTUM_SCORE_ABOVE"
    
    # AI Analysis Triggers
    AI_RATING_STRONG_BUY = "AI_RATING_STRONG_BUY"
    AI_RATING_CHANGE = "AI_RATING_CHANGE"
    AI_CONFIDENCE_HIGH = "AI_CONFIDENCE_HIGH"
    AI_CONSENSUS_BUY = "AI_CONSENSUS_BUY"
    
    # Portfolio Triggers
    PORTFOLIO_VALUE_ABOVE = "PORTFOLIO_VALUE_ABOVE"
    PORTFOLIO_VALUE_BELOW = "PORTFOLIO_VALUE_BELOW"
    PORTFOLIO_DRAWDOWN_EXCEEDS = "PORTFOLIO_DRAWDOWN_EXCEEDS"
    POSITION_WEIGHT_EXCEEDS = "POSITION_WEIGHT_EXCEEDS"
    PORTFOLIO_GAIN_EXCEEDS = "PORTFOLIO_GAIN_EXCEEDS"
    PORTFOLIO_LOSS_EXCEEDS = "PORTFOLIO_LOSS_EXCEEDS"
    
    # Watchlist Triggers
    WATCHLIST_STOCK_DIPS = "WATCHLIST_STOCK_DIPS"
    WATCHLIST_OPPORTUNITY = "WATCHLIST_OPPORTUNITY"


class ComparisonOperator(str, Enum):
    """Comparison operators for rule conditions."""
    GT = "GT"      # Greater than
    LT = "LT"      # Less than
    GTE = "GTE"    # Greater than or equal
    LTE = "LTE"    # Less than or equal
    EQ = "EQ"      # Equal
    NEQ = "NEQ"    # Not equal
    CHANGE = "CHANGE"  # Value changed (for change detection triggers)


class ChannelType(str, Enum):
    """Supported notification channel types."""
    DISCORD = "discord"
    TELEGRAM = "telegram"
    EMAIL = "email"
    SLACK = "slack"
    PUSHOVER = "pushover"
    NTFY = "ntfy"
    WEBHOOK = "webhook"


class RulePriority(str, Enum):
    """Rule priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# TRIGGER TYPE METADATA
# =============================================================================


class TriggerTypeInfo(BaseModel):
    """Metadata about a trigger type for UI display."""
    
    type: TriggerType
    name: str
    description: str
    category: str
    requires_symbol: bool = True
    requires_portfolio: bool = False
    default_value: float | None = None
    default_operator: ComparisonOperator = ComparisonOperator.GT
    value_unit: str | None = None  # "percent", "days", "score", "price"
    is_boolean: bool = False  # No threshold needed


# Trigger type metadata registry
TRIGGER_TYPE_INFO: dict[TriggerType, TriggerTypeInfo] = {
    # Price & Dip
    TriggerType.PRICE_DROPS_BELOW: TriggerTypeInfo(
        type=TriggerType.PRICE_DROPS_BELOW,
        name="Price Drops Below",
        description="Notify when stock price drops below a threshold",
        category="Price & Dip",
        default_operator=ComparisonOperator.LT,
        value_unit="price",
    ),
    TriggerType.PRICE_RISES_ABOVE: TriggerTypeInfo(
        type=TriggerType.PRICE_RISES_ABOVE,
        name="Price Rises Above",
        description="Notify when stock price rises above a threshold",
        category="Price & Dip",
        default_value=100.0,
        value_unit="price",
    ),
    TriggerType.DIP_EXCEEDS_PERCENT: TriggerTypeInfo(
        type=TriggerType.DIP_EXCEEDS_PERCENT,
        name="Dip Exceeds Percent",
        description="Notify when dip from ATH exceeds threshold",
        category="Price & Dip",
        default_value=15.0,
        value_unit="percent",
    ),
    TriggerType.DIP_DURATION_EXCEEDS: TriggerTypeInfo(
        type=TriggerType.DIP_DURATION_EXCEEDS,
        name="Dip Duration Exceeds",
        description="Notify when stock has been in dip for X days",
        category="Price & Dip",
        default_value=30.0,
        value_unit="days",
    ),
    TriggerType.TAIL_EVENT_DETECTED: TriggerTypeInfo(
        type=TriggerType.TAIL_EVENT_DETECTED,
        name="Tail Event Detected",
        description="Notify when a rare multi-year dip event is detected",
        category="Price & Dip",
        is_boolean=True,
    ),
    TriggerType.NEW_ATH_REACHED: TriggerTypeInfo(
        type=TriggerType.NEW_ATH_REACHED,
        name="New All-Time High",
        description="Notify when stock reaches a new all-time high",
        category="Price & Dip",
        is_boolean=True,
    ),
    
    # Signals
    TriggerType.DIPFINDER_ALERT: TriggerTypeInfo(
        type=TriggerType.DIPFINDER_ALERT,
        name="DipFinder Alert",
        description="Notify when DipFinder flags a stock as opportunity",
        category="Signals",
        is_boolean=True,
    ),
    TriggerType.DIPFINDER_SCORE_ABOVE: TriggerTypeInfo(
        type=TriggerType.DIPFINDER_SCORE_ABOVE,
        name="DipFinder Score Above",
        description="Notify when DipFinder score exceeds threshold",
        category="Signals",
        default_value=70.0,
        value_unit="score",
    ),
    TriggerType.STRATEGY_SIGNAL_BUY: TriggerTypeInfo(
        type=TriggerType.STRATEGY_SIGNAL_BUY,
        name="Strategy Buy Signal",
        description="Notify when strategy optimizer signals BUY",
        category="Signals",
        is_boolean=True,
    ),
    TriggerType.STRATEGY_SIGNAL_SELL: TriggerTypeInfo(
        type=TriggerType.STRATEGY_SIGNAL_SELL,
        name="Strategy Sell Signal",
        description="Notify when strategy optimizer signals SELL",
        category="Signals",
        is_boolean=True,
    ),
    TriggerType.ENTRY_SIGNAL_TRIGGERED: TriggerTypeInfo(
        type=TriggerType.ENTRY_SIGNAL_TRIGGERED,
        name="Entry Signal Triggered",
        description="Notify when dip entry optimizer signals buy now",
        category="Signals",
        is_boolean=True,
    ),
    TriggerType.WIN_RATE_ABOVE: TriggerTypeInfo(
        type=TriggerType.WIN_RATE_ABOVE,
        name="Win Rate Above",
        description="Notify when signal win rate exceeds threshold",
        category="Signals",
        default_value=60.0,
        value_unit="percent",
    ),
    
    # Fundamentals
    TriggerType.PE_RATIO_BELOW: TriggerTypeInfo(
        type=TriggerType.PE_RATIO_BELOW,
        name="P/E Ratio Below",
        description="Notify when P/E ratio drops below threshold",
        category="Fundamentals",
        default_value=15.0,
        default_operator=ComparisonOperator.LT,
    ),
    TriggerType.PE_RATIO_ABOVE: TriggerTypeInfo(
        type=TriggerType.PE_RATIO_ABOVE,
        name="P/E Ratio Above",
        description="Notify when P/E ratio rises above threshold",
        category="Fundamentals",
        default_value=50.0,
    ),
    TriggerType.ANALYST_UPGRADE: TriggerTypeInfo(
        type=TriggerType.ANALYST_UPGRADE,
        name="Analyst Upgrade",
        description="Notify when analyst rating improves",
        category="Fundamentals",
        is_boolean=True,
        default_operator=ComparisonOperator.CHANGE,
    ),
    TriggerType.ANALYST_DOWNGRADE: TriggerTypeInfo(
        type=TriggerType.ANALYST_DOWNGRADE,
        name="Analyst Downgrade",
        description="Notify when analyst rating worsens",
        category="Fundamentals",
        is_boolean=True,
        default_operator=ComparisonOperator.CHANGE,
    ),
    TriggerType.PRICE_BELOW_TARGET: TriggerTypeInfo(
        type=TriggerType.PRICE_BELOW_TARGET,
        name="Price Below Target",
        description="Notify when price is X% below analyst target",
        category="Fundamentals",
        default_value=10.0,
        value_unit="percent",
    ),
    TriggerType.EARNINGS_APPROACHING: TriggerTypeInfo(
        type=TriggerType.EARNINGS_APPROACHING,
        name="Earnings Approaching",
        description="Notify when earnings date is within X days",
        category="Fundamentals",
        default_value=7.0,
        default_operator=ComparisonOperator.LTE,
        value_unit="days",
    ),
    TriggerType.QUALITY_SCORE_ABOVE: TriggerTypeInfo(
        type=TriggerType.QUALITY_SCORE_ABOVE,
        name="Quality Score Above",
        description="Notify when quality score exceeds threshold",
        category="Fundamentals",
        default_value=70.0,
        value_unit="score",
    ),
    TriggerType.MOMENTUM_SCORE_ABOVE: TriggerTypeInfo(
        type=TriggerType.MOMENTUM_SCORE_ABOVE,
        name="Momentum Score Above",
        description="Notify when momentum score exceeds threshold",
        category="Fundamentals",
        default_value=70.0,
        value_unit="score",
    ),
    
    # AI Analysis
    TriggerType.AI_RATING_STRONG_BUY: TriggerTypeInfo(
        type=TriggerType.AI_RATING_STRONG_BUY,
        name="AI Rating Strong Buy",
        description="Notify when AI rates stock as strong buy",
        category="AI Analysis",
        is_boolean=True,
    ),
    TriggerType.AI_RATING_CHANGE: TriggerTypeInfo(
        type=TriggerType.AI_RATING_CHANGE,
        name="AI Rating Change",
        description="Notify when AI rating changes",
        category="AI Analysis",
        is_boolean=True,
        default_operator=ComparisonOperator.CHANGE,
    ),
    TriggerType.AI_CONFIDENCE_HIGH: TriggerTypeInfo(
        type=TriggerType.AI_CONFIDENCE_HIGH,
        name="AI Confidence High",
        description="Notify when AI confidence exceeds threshold",
        category="AI Analysis",
        default_value=7.0,
        value_unit="score",
    ),
    TriggerType.AI_CONSENSUS_BUY: TriggerTypeInfo(
        type=TriggerType.AI_CONSENSUS_BUY,
        name="AI Consensus Buy",
        description="Notify when multiple AI personas agree BUY",
        category="AI Analysis",
        default_value=3.0,
        value_unit="personas",
    ),
    
    # Portfolio
    TriggerType.PORTFOLIO_VALUE_ABOVE: TriggerTypeInfo(
        type=TriggerType.PORTFOLIO_VALUE_ABOVE,
        name="Portfolio Value Above",
        description="Notify when portfolio value exceeds threshold",
        category="Portfolio",
        requires_symbol=False,
        requires_portfolio=True,
        value_unit="currency",
    ),
    TriggerType.PORTFOLIO_VALUE_BELOW: TriggerTypeInfo(
        type=TriggerType.PORTFOLIO_VALUE_BELOW,
        name="Portfolio Value Below",
        description="Notify when portfolio value drops below threshold",
        category="Portfolio",
        requires_symbol=False,
        requires_portfolio=True,
        default_operator=ComparisonOperator.LT,
        value_unit="currency",
    ),
    TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS: TriggerTypeInfo(
        type=TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS,
        name="Portfolio Drawdown Exceeds",
        description="Notify when portfolio drawdown exceeds threshold",
        category="Portfolio",
        requires_symbol=False,
        requires_portfolio=True,
        default_value=10.0,
        value_unit="percent",
    ),
    TriggerType.POSITION_WEIGHT_EXCEEDS: TriggerTypeInfo(
        type=TriggerType.POSITION_WEIGHT_EXCEEDS,
        name="Position Weight Exceeds",
        description="Notify when single position exceeds % of portfolio",
        category="Portfolio",
        requires_symbol=False,
        requires_portfolio=True,
        default_value=25.0,
        value_unit="percent",
    ),
    TriggerType.PORTFOLIO_GAIN_EXCEEDS: TriggerTypeInfo(
        type=TriggerType.PORTFOLIO_GAIN_EXCEEDS,
        name="Portfolio Gain Exceeds",
        description="Notify when daily/weekly gain exceeds threshold",
        category="Portfolio",
        requires_symbol=False,
        requires_portfolio=True,
        default_value=5.0,
        value_unit="percent",
    ),
    TriggerType.PORTFOLIO_LOSS_EXCEEDS: TriggerTypeInfo(
        type=TriggerType.PORTFOLIO_LOSS_EXCEEDS,
        name="Portfolio Loss Exceeds",
        description="Notify when daily/weekly loss exceeds threshold",
        category="Portfolio",
        requires_symbol=False,
        requires_portfolio=True,
        default_value=5.0,
        value_unit="percent",
    ),
    
    # Watchlist
    TriggerType.WATCHLIST_STOCK_DIPS: TriggerTypeInfo(
        type=TriggerType.WATCHLIST_STOCK_DIPS,
        name="Watchlist Stock Dips",
        description="Notify when any watched stock enters a dip",
        category="Watchlist",
        requires_symbol=False,
        default_value=10.0,
        value_unit="percent",
    ),
    TriggerType.WATCHLIST_OPPORTUNITY: TriggerTypeInfo(
        type=TriggerType.WATCHLIST_OPPORTUNITY,
        name="Watchlist Opportunity",
        description="Notify when any watched stock signals opportunity",
        category="Watchlist",
        requires_symbol=False,
        is_boolean=True,
    ),
}


# =============================================================================
# CHANNEL SCHEMAS
# =============================================================================


class ChannelCreate(BaseModel):
    """Request to create a notification channel."""
    
    name: str = Field(
        ..., min_length=1, max_length=100,
        description="Display name for the channel",
    )
    channel_type: ChannelType = Field(
        ..., description="Type of notification channel",
    )
    apprise_url: str = Field(
        ..., min_length=10,
        description="Apprise URL for the channel (e.g., discord://webhook_id/token)",
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Trading Discord",
                "channel_type": "discord",
                "apprise_url": "discord://webhook_id/token",
            }
        }
    }


class ChannelUpdate(BaseModel):
    """Request to update a notification channel."""
    
    name: str | None = Field(None, min_length=1, max_length=100)
    apprise_url: str | None = Field(None, min_length=10)
    is_active: bool | None = None


class ChannelResponse(BaseModel):
    """Notification channel response."""
    
    id: int
    name: str
    channel_type: ChannelType
    is_verified: bool
    is_active: bool
    error_count: int
    last_error: str | None
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ChannelTestResponse(BaseModel):
    """Response from testing a channel."""
    
    success: bool
    message: str


# =============================================================================
# RULE SCHEMAS
# =============================================================================


class RuleCreate(BaseModel):
    """Request to create a notification rule."""
    
    name: str = Field(
        ..., min_length=1, max_length=100,
        description="Display name for the rule",
    )
    channel_id: int = Field(
        ..., description="ID of the notification channel to use",
    )
    trigger_type: TriggerType = Field(
        ..., description="Type of trigger condition",
    )
    target_symbol: str | None = Field(
        None, max_length=20,
        description="Stock symbol to watch (required for symbol-based triggers)",
    )
    target_portfolio_id: int | None = Field(
        None, description="Portfolio ID (required for portfolio triggers)",
    )
    comparison_operator: ComparisonOperator = Field(
        ComparisonOperator.GT, description="Comparison operator",
    )
    target_value: float | None = Field(
        None, description="Threshold value (not required for boolean triggers)",
    )
    smart_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional trigger configuration",
    )
    cooldown_minutes: int = Field(
        60, ge=5, le=10080,  # 5 min to 1 week
        description="Minimum minutes between alerts for this rule",
    )
    priority: RulePriority = Field(
        RulePriority.NORMAL, description="Alert priority level",
    )
    
    @field_validator("target_symbol")
    @classmethod
    def normalize_symbol(cls, v: str | None) -> str | None:
        if v:
            return v.strip().upper()
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "AAPL Dip Alert",
                "channel_id": 1,
                "trigger_type": "DIP_EXCEEDS_PERCENT",
                "target_symbol": "AAPL",
                "comparison_operator": "GT",
                "target_value": 15.0,
                "cooldown_minutes": 120,
                "priority": "high",
            }
        }
    }


class RuleUpdate(BaseModel):
    """Request to update a notification rule."""
    
    name: str | None = Field(None, min_length=1, max_length=100)
    channel_id: int | None = None
    comparison_operator: ComparisonOperator | None = None
    target_value: float | None = None
    smart_payload: dict[str, Any] | None = None
    cooldown_minutes: int | None = Field(None, ge=5, le=10080)
    priority: RulePriority | None = None
    is_active: bool | None = None


class RuleResponse(BaseModel):
    """Notification rule response."""
    
    id: int
    name: str
    trigger_type: TriggerType
    target_symbol: str | None
    target_portfolio_id: int | None
    comparison_operator: ComparisonOperator
    target_value: float | None
    smart_payload: dict[str, Any]
    cooldown_minutes: int
    priority: RulePriority
    is_active: bool
    last_triggered_at: datetime | None
    trigger_count: int
    channel: ChannelResponse
    created_at: datetime
    updated_at: datetime


class RuleTestResponse(BaseModel):
    """Response from testing a rule."""
    
    would_trigger: bool
    current_value: float | None
    threshold_value: float | None
    message: str


# =============================================================================
# NOTIFICATION LOG SCHEMAS
# =============================================================================


class NotificationLogEntry(BaseModel):
    """A notification log entry."""
    
    id: int
    rule_id: int | None
    rule_name: str | None
    channel_name: str | None
    trigger_type: TriggerType
    trigger_symbol: str | None
    trigger_value: float | None
    threshold_value: float | None
    title: str
    body: str
    status: str  # pending, sent, failed, skipped
    error_message: str | None
    triggered_at: datetime
    sent_at: datetime | None


class NotificationHistoryResponse(BaseModel):
    """Paginated notification history."""
    
    notifications: list[NotificationLogEntry]
    total: int
    page: int
    page_size: int


# =============================================================================
# SUMMARY SCHEMAS
# =============================================================================


class NotificationSummary(BaseModel):
    """Summary of user's notification setup."""
    
    total_channels: int
    active_channels: int
    total_rules: int
    active_rules: int
    notifications_today: int
    notifications_this_week: int
