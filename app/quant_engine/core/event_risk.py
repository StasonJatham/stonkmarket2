"""
Event Risk Service - Earnings and dividend date awareness.

Prevents buying right before risky events.
Uses earnings/dividend dates from stock_fundamentals table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class EventRiskState:
    """Event risk analysis for a stock."""
    
    # Overall risk level
    risk_level: Literal["CLEAR", "CAUTION", "HIGH_RISK", "BLOCKED"]
    
    # Earnings
    earnings_date: date | None
    days_to_earnings: int | None
    earnings_risk: Literal["none", "upcoming", "imminent"]
    
    # Dividends
    ex_dividend_date: date | None
    days_to_ex_div: int | None
    dividend_opportunity: bool  # True if can capture dividend
    
    # Score multiplier
    score_multiplier: float  # 0.0 to 1.05
    
    # Action guidance
    guidance: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "earnings": {
                "date": str(self.earnings_date) if self.earnings_date else None,
                "days_until": self.days_to_earnings,
                "risk": self.earnings_risk,
            },
            "dividend": {
                "ex_date": str(self.ex_dividend_date) if self.ex_dividend_date else None,
                "days_until": self.days_to_ex_div,
                "can_capture": self.dividend_opportunity,
            },
            "score_multiplier": round(self.score_multiplier, 2),
            "guidance": self.guidance,
        }


class EventRiskService:
    """
    Evaluates event risk from earnings and dividends.
    
    Rules:
    - Earnings in <5 days: BLOCKED (gambling)
    - Earnings in 5-14 days: CAUTION
    - Ex-dividend in <5 days: Opportunity (if buying for income)
    """
    
    def __init__(
        self,
        earnings_block_days: int = 5,
        earnings_caution_days: int = 14,
        dividend_opportunity_days: int = 5,
    ):
        self.earnings_block_days = earnings_block_days
        self.earnings_caution_days = earnings_caution_days
        self.dividend_opportunity_days = dividend_opportunity_days
    
    def analyze_event_risk(
        self,
        earnings_date: date | datetime | str | None = None,
        ex_dividend_date: date | datetime | str | None = None,
        as_of_date: date | None = None,
    ) -> EventRiskState:
        """
        Analyze event risk for a stock.
        
        Args:
            earnings_date: Next earnings announcement date
            ex_dividend_date: Next ex-dividend date
            as_of_date: Date to analyze from (defaults to today)
        """
        today = as_of_date or date.today()
        
        # Normalize dates
        earnings = self._normalize_date(earnings_date)
        ex_div = self._normalize_date(ex_dividend_date)
        
        # Earnings analysis
        days_to_earnings = None
        earnings_risk: Literal["none", "upcoming", "imminent"] = "none"
        if earnings and earnings >= today:
            days_to_earnings = (earnings - today).days
            if days_to_earnings <= self.earnings_block_days:
                earnings_risk = "imminent"
            elif days_to_earnings <= self.earnings_caution_days:
                earnings_risk = "upcoming"
        
        # Dividend analysis
        days_to_ex_div = None
        dividend_opportunity = False
        if ex_div and ex_div >= today:
            days_to_ex_div = (ex_div - today).days
            if days_to_ex_div <= self.dividend_opportunity_days:
                dividend_opportunity = True
        
        # Determine overall risk level and multiplier
        risk_level, multiplier, guidance = self._compute_risk_level(
            earnings_risk, days_to_earnings, dividend_opportunity, days_to_ex_div
        )
        
        return EventRiskState(
            risk_level=risk_level,
            earnings_date=earnings,
            days_to_earnings=days_to_earnings,
            earnings_risk=earnings_risk,
            ex_dividend_date=ex_div,
            days_to_ex_div=days_to_ex_div,
            dividend_opportunity=dividend_opportunity,
            score_multiplier=multiplier,
            guidance=guidance,
        )
    
    def _normalize_date(self, dt: date | datetime | str | None) -> date | None:
        """Convert various date formats to date object."""
        if dt is None:
            return None
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return dt
        if isinstance(dt, datetime):
            return dt.date()
        if isinstance(dt, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(dt.replace('Z', '+00:00')).date()
            except ValueError:
                try:
                    # Try common date format
                    return datetime.strptime(dt[:10], "%Y-%m-%d").date()
                except ValueError:
                    return None
        return None
    
    def _compute_risk_level(
        self,
        earnings_risk: str,
        days_to_earnings: int | None,
        div_opportunity: bool,
        days_to_ex_div: int | None,
    ) -> tuple[Literal["CLEAR", "CAUTION", "HIGH_RISK", "BLOCKED"], float, str]:
        """Compute overall risk level."""
        
        # Earnings imminent = BLOCKED
        if earnings_risk == "imminent":
            return (
                "BLOCKED",
                0.0,  # Zero out score
                f"Earnings in {days_to_earnings} days - wait until after announcement"
            )
        
        # Earnings upcoming = CAUTION or HIGH_RISK
        if earnings_risk == "upcoming":
            # Very close = HIGH_RISK
            if days_to_earnings and days_to_earnings <= 7:
                guidance = f"Earnings in {days_to_earnings} days - high event risk"
                return ("HIGH_RISK", 0.6, guidance)
            
            # Further out = CAUTION
            guidance = f"Earnings in {days_to_earnings} days - consider smaller position"
            
            # But dividend opportunity might offset
            if div_opportunity:
                return (
                    "CAUTION",
                    0.85,
                    guidance + f" Ex-dividend in {days_to_ex_div} days."
                )
            
            return ("CAUTION", 0.75, guidance)
        
        # Dividend opportunity = slight boost
        if div_opportunity:
            return (
                "CLEAR",
                1.05,  # Small boost
                f"Ex-dividend in {days_to_ex_div} days - can capture dividend"
            )
        
        # No events = CLEAR
        return (
            "CLEAR",
            1.0,
            "No imminent earnings or dividend events"
        )


# Singleton
_event_risk_service: EventRiskService | None = None


def get_event_risk_service() -> EventRiskService:
    """Get singleton EventRiskService instance."""
    global _event_risk_service
    if _event_risk_service is None:
        _event_risk_service = EventRiskService()
    return _event_risk_service
