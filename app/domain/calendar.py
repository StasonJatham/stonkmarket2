"""Calendar event domain models.

Type-safe representations of market calendar events (earnings, IPOs, splits, economic).
"""

from __future__ import annotations

from datetime import date as DateType
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CalendarEventType(str, Enum):
    """Types of calendar events."""
    
    EARNINGS = "earnings"
    IPO = "ipo"
    SPLIT = "split"
    ECONOMIC = "economic"


class EarningsEvent(BaseModel):
    """Earnings announcement event.
    
    Represents a company's quarterly earnings report.
    """
    
    symbol: str = Field(..., description="Ticker symbol")
    company_name: str | None = Field(None, description="Company name")
    report_date: DateType = Field(..., description="Earnings report date")
    
    # Estimates
    eps_estimate: float | None = Field(None, description="Expected EPS")
    eps_actual: float | None = Field(None, description="Actual EPS (after report)")
    revenue_estimate: float | None = Field(None, description="Expected revenue")
    revenue_actual: float | None = Field(None, description="Actual revenue (after report)")
    
    # Timing
    time_of_day: Literal["before_market", "after_market", "during_market", "unknown"] = Field(
        default="unknown",
        description="When the report is released",
    )
    fiscal_quarter: str | None = Field(None, description="Fiscal quarter (e.g., Q1 2025)")
    
    model_config = {
        "from_attributes": True,
    }
    
    @property
    def has_reported(self) -> bool:
        """Check if earnings have been reported."""
        return self.eps_actual is not None or self.revenue_actual is not None
    
    @property
    def eps_surprise(self) -> float | None:
        """EPS surprise (actual - estimate)."""
        if self.eps_actual is not None and self.eps_estimate is not None:
            return self.eps_actual - self.eps_estimate
        return None
    
    @property
    def eps_surprise_pct(self) -> float | None:
        """EPS surprise percentage."""
        if self.eps_surprise is not None and self.eps_estimate and self.eps_estimate != 0:
            return self.eps_surprise / abs(self.eps_estimate)
        return None


class IpoEvent(BaseModel):
    """IPO (Initial Public Offering) event.
    
    Represents an upcoming or recent IPO.
    """
    
    symbol: str | None = Field(None, description="Ticker symbol (may not be assigned yet)")
    company_name: str = Field(..., description="Company name")
    ipo_date: DateType = Field(..., description="IPO date")
    
    # Pricing
    price_low: float | None = Field(None, ge=0, description="Low end of price range")
    price_high: float | None = Field(None, ge=0, description="High end of price range")
    price_actual: float | None = Field(None, ge=0, description="Actual IPO price")
    
    # Deal details
    shares_offered: int | None = Field(None, ge=0, description="Number of shares offered")
    exchange: str | None = Field(None, description="Exchange listing")
    
    model_config = {
        "from_attributes": True,
    }
    
    @property
    def price_range(self) -> tuple[float, float] | None:
        """Get price range as tuple."""
        if self.price_low is not None and self.price_high is not None:
            return (self.price_low, self.price_high)
        return None
    
    @property
    def price_midpoint(self) -> float | None:
        """Get midpoint of price range."""
        if self.price_low is not None and self.price_high is not None:
            return (self.price_low + self.price_high) / 2
        return None


class SplitEvent(BaseModel):
    """Stock split event.
    
    Represents a stock split or reverse split.
    """
    
    symbol: str = Field(..., description="Ticker symbol")
    company_name: str | None = Field(None, description="Company name")
    split_date: DateType = Field(..., description="Split effective date")
    
    # Split details
    split_ratio: str = Field(..., description="Split ratio (e.g., '4:1', '1:10')")
    numerator: int = Field(..., ge=1, description="Split numerator (shares received)")
    denominator: int = Field(..., ge=1, description="Split denominator (shares held)")
    
    model_config = {
        "from_attributes": True,
    }
    
    @property
    def is_reverse_split(self) -> bool:
        """Check if this is a reverse split (consolidation)."""
        return self.numerator < self.denominator
    
    @property
    def split_factor(self) -> float:
        """Get split factor (shares multiplier)."""
        return self.numerator / self.denominator


class EconomicEvent(BaseModel):
    """Economic calendar event.
    
    Represents macroeconomic data releases, central bank meetings, etc.
    """
    
    event_name: str = Field(..., description="Event name")
    event_date: DateType = Field(..., description="Event date")
    event_time: str | None = Field(None, description="Event time (HH:MM)")
    
    # Region/Country
    country: str | None = Field(None, description="Country code (US, EU, etc.)")
    currency: str | None = Field(None, description="Affected currency")
    
    # Values
    previous_value: float | None = Field(None, description="Previous period value")
    forecast_value: float | None = Field(None, description="Consensus forecast")
    actual_value: float | None = Field(None, description="Actual value (after release)")
    
    # Impact
    importance: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Event importance level",
    )
    
    model_config = {
        "from_attributes": True,
    }
    
    @property
    def surprise(self) -> float | None:
        """Value surprise (actual - forecast)."""
        if self.actual_value is not None and self.forecast_value is not None:
            return self.actual_value - self.forecast_value
        return None
    
    @property
    def has_released(self) -> bool:
        """Check if the data has been released."""
        return self.actual_value is not None
