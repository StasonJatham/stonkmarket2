"""Domain models for strongly-typed data throughout the application.

This module provides Pydantic models that serve as the source of truth for
data structures passed between services, replacing raw dictionaries.

Usage:
    from app.domain import TickerInfo, PriceBar, QualityMetrics

    # Type-safe data with validation
    info: TickerInfo = await yfinance_service.get_ticker_info("AAPL")
    
    # Easy serialization
    data = info.model_dump()
"""

from app.domain.ticker import (
    TickerInfo,
    TickerSearchResult,
)
from app.domain.price import (
    PriceBar,
    PriceHistory,
)
from app.domain.fundamentals import (
    FundamentalsData,
    QualityMetrics,
)
from app.domain.calendar import (
    EarningsEvent,
    IpoEvent,
    SplitEvent,
    EconomicEvent,
    CalendarEventType,
)

__all__ = [
    # Ticker
    "TickerInfo",
    "TickerSearchResult",
    # Price
    "PriceBar",
    "PriceHistory",
    # Fundamentals
    "FundamentalsData",
    "QualityMetrics",
    # Calendar
    "EarningsEvent",
    "IpoEvent",
    "SplitEvent",
    "EconomicEvent",
    "CalendarEventType",
]
