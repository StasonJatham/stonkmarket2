"""Earnings calendar integration.

Checks for upcoming earnings dates and flags stocks that are about to
report. Pre-earnings stocks have higher uncertainty and should be
treated with caution for dip entry.

IMPORTANT: Uses locally stored fundamentals data from scheduled jobs,
NOT ad-hoc yfinance fetches at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from app.core.logging import get_logger

logger = get_logger("dipfinder.earnings_calendar")


@dataclass
class EarningsInfo:
    """Earnings date information for a stock."""

    ticker: str

    # Next earnings date
    next_earnings_date: datetime | None = None
    days_to_earnings: int | None = None

    # Historical earnings dates (for pattern analysis)
    last_earnings_date: datetime | None = None
    days_since_earnings: int | None = None

    # Risk flags
    is_pre_earnings: bool = False  # Within warning window before earnings
    pre_earnings_risk_level: str = "none"  # "none", "low", "medium", "high"

    # Earnings estimates (if available from stored data)
    earnings_estimate_avg: float | None = None
    earnings_estimate_high: float | None = None
    earnings_estimate_low: float | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "next_earnings_date": self.next_earnings_date.isoformat() if self.next_earnings_date else None,
            "days_to_earnings": self.days_to_earnings,
            "last_earnings_date": self.last_earnings_date.isoformat() if self.last_earnings_date else None,
            "days_since_earnings": self.days_since_earnings,
            "is_pre_earnings": self.is_pre_earnings,
            "pre_earnings_risk_level": self.pre_earnings_risk_level,
            "earnings_estimate_avg": self.earnings_estimate_avg,
        }


def classify_pre_earnings_risk(days_to_earnings: int | None) -> tuple[bool, str]:
    """
    Classify pre-earnings risk level based on days to earnings.

    Args:
        days_to_earnings: Days until next earnings

    Returns:
        (is_pre_earnings, risk_level)
    """
    if days_to_earnings is None:
        return False, "unknown"

    if days_to_earnings <= 0:
        # Earnings today or past
        return False, "none"
    elif days_to_earnings <= 3:
        # Very close to earnings - high risk
        return True, "high"
    elif days_to_earnings <= 7:
        # Within a week - medium risk
        return True, "medium"
    elif days_to_earnings <= 14:
        # Within two weeks - low risk but noteworthy
        return True, "low"
    else:
        # More than 2 weeks away - not pre-earnings
        return False, "none"


def compute_earnings_penalty(
    earnings_info: EarningsInfo,
    base_score: float,
) -> float:
    """
    Compute a penalty to apply to dip score based on earnings risk.

    Pre-earnings stocks are riskier because the dip might be:
    1. Anticipation of bad earnings (insiders selling)
    2. About to be overshadowed by earnings move

    Args:
        earnings_info: Earnings date information
        base_score: Current dip attractiveness score

    Returns:
        Penalty to subtract from score (0-15)
    """
    if not earnings_info.is_pre_earnings:
        return 0.0

    if earnings_info.pre_earnings_risk_level == "high":
        # Within 3 days - significant penalty
        return 15.0
    elif earnings_info.pre_earnings_risk_level == "medium":
        # Within 7 days - moderate penalty
        return 10.0
    elif earnings_info.pre_earnings_risk_level == "low":
        # Within 14 days - small penalty
        return 5.0

    return 0.0


def get_earnings_info_from_stored(
    stored_fundamentals: dict[str, Any],
) -> EarningsInfo:
    """
    Get earnings calendar information from locally stored fundamentals.
    
    Uses pre-fetched data from the database. Does NOT make any yfinance calls.
    
    Args:
        stored_fundamentals: Dict from get_fundamentals_from_db() containing
                            next_earnings_date, earnings_date, etc.
        
    Returns:
        EarningsInfo with dates and risk assessment
    """
    ticker = stored_fundamentals.get("symbol", "UNKNOWN")
    result = EarningsInfo(ticker=ticker)
    now = datetime.now()
    
    if not stored_fundamentals:
        return result
    
    # Extract next earnings date
    next_earnings = stored_fundamentals.get("next_earnings_date")
    if next_earnings:
        if isinstance(next_earnings, datetime):
            result.next_earnings_date = next_earnings
        elif isinstance(next_earnings, str):
            try:
                result.next_earnings_date = datetime.fromisoformat(next_earnings.replace("Z", "+00:00"))
            except ValueError:
                pass
    
    # Extract last earnings date
    last_earnings = stored_fundamentals.get("earnings_date")
    if last_earnings:
        if isinstance(last_earnings, datetime):
            result.last_earnings_date = last_earnings
        elif isinstance(last_earnings, str):
            try:
                result.last_earnings_date = datetime.fromisoformat(last_earnings.replace("Z", "+00:00"))
            except ValueError:
                pass
    
    # Calculate days
    if result.next_earnings_date:
        # Make both datetimes naive for comparison if needed
        next_dt = result.next_earnings_date
        if next_dt.tzinfo is not None:
            next_dt = next_dt.replace(tzinfo=None)
        result.days_to_earnings = (next_dt - now).days
    
    if result.last_earnings_date:
        last_dt = result.last_earnings_date
        if last_dt.tzinfo is not None:
            last_dt = last_dt.replace(tzinfo=None)
        result.days_since_earnings = (now - last_dt).days
    
    # Classify risk
    is_pre_earnings, risk_level = classify_pre_earnings_risk(result.days_to_earnings)
    result.is_pre_earnings = is_pre_earnings
    result.pre_earnings_risk_level = risk_level
    
    # Extract earnings estimates
    result.earnings_estimate_avg = stored_fundamentals.get("earnings_estimate_avg")
    result.earnings_estimate_high = stored_fundamentals.get("earnings_estimate_high")
    result.earnings_estimate_low = stored_fundamentals.get("earnings_estimate_low")
    
    return result


async def get_earnings_info(
    symbol: str,
    stored_fundamentals: dict[str, Any] | None = None,
) -> EarningsInfo:
    """
    Get earnings calendar information for a stock.
    
    Uses locally stored fundamentals from scheduled jobs.
    Does NOT make ad-hoc yfinance calls.
    
    Args:
        symbol: Stock ticker
        stored_fundamentals: Pre-loaded fundamentals dict, or None to load from DB
        
    Returns:
        EarningsInfo with dates and risk assessment
    """
    if stored_fundamentals is None:
        # Load from database (NOT ad-hoc yfinance fetch)
        from app.services.fundamentals import get_fundamentals_from_db
        stored_fundamentals = await get_fundamentals_from_db(symbol)
    
    if not stored_fundamentals:
        return EarningsInfo(ticker=symbol)
    
    # Add symbol to dict if not present
    if "symbol" not in stored_fundamentals:
        stored_fundamentals["symbol"] = symbol
    
    return get_earnings_info_from_stored(stored_fundamentals)
