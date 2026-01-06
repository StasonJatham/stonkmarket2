"""Structural decline detection and fundamental momentum analysis.

Distinguishes between temporary dips (buying opportunities) and structural
declines (value traps). Uses quarterly data trends and margin analysis.

IMPORTANT: Uses locally stored fundamentals data from scheduled jobs,
NOT ad-hoc yfinance fetches at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.data_helpers import safe_float as _safe_float, pct_change
from app.core.logging import get_logger

logger = get_logger("dipfinder.structural_analysis")


@dataclass
class FundamentalMomentum:
    """Fundamental momentum metrics tracking quarterly trends."""
    
    # Revenue trend (QoQ and YoY)
    revenue_qoq_change: float | None = None  # Quarter-over-quarter % change
    revenue_yoy_change: float | None = None  # Year-over-year % change
    revenue_declining_quarters: int = 0  # Consecutive quarters of decline
    
    # Earnings/Net Income trend
    earnings_qoq_change: float | None = None
    earnings_yoy_change: float | None = None
    earnings_declining_quarters: int = 0
    
    # Margin trends (critical for structural decline detection)
    gross_margin_trend: float | None = None  # Change in gross margin (pp)
    operating_margin_trend: float | None = None  # Change in operating margin (pp)
    margin_compression_quarters: int = 0  # Consecutive quarters of margin decline
    
    # Overall momentum score (0-100, higher = improving fundamentals)
    momentum_score: float = 50.0
    
    # Structural decline flags
    is_structural_decline: bool = False
    decline_severity: str = "none"  # none, mild, moderate, severe
    decline_reasons: list[str] | None = None
    
    # Confidence in assessment
    data_quality: str = "unknown"  # full, partial, minimal, unknown
    quarters_analyzed: int = 0
    
    def to_dict(self) -> dict:
        return {
            "revenue_qoq_change": self.revenue_qoq_change,
            "revenue_yoy_change": self.revenue_yoy_change,
            "revenue_declining_quarters": self.revenue_declining_quarters,
            "earnings_qoq_change": self.earnings_qoq_change,
            "earnings_yoy_change": self.earnings_yoy_change,
            "earnings_declining_quarters": self.earnings_declining_quarters,
            "gross_margin_trend": self.gross_margin_trend,
            "operating_margin_trend": self.operating_margin_trend,
            "margin_compression_quarters": self.margin_compression_quarters,
            "momentum_score": round(self.momentum_score, 2),
            "is_structural_decline": self.is_structural_decline,
            "decline_severity": self.decline_severity,
            "decline_reasons": self.decline_reasons,
            "data_quality": self.data_quality,
            "quarters_analyzed": self.quarters_analyzed,
        }


def _compute_pct_change(current: float | None, previous: float | None) -> float | None:
    """Compute percentage change between two values (returns as percent, e.g. 10.0 for 10%)."""
    return pct_change(current, previous, as_percent=True)


def _count_declining_periods(values: list[float | None]) -> int:
    """Count consecutive declining periods from most recent."""
    if not values or len(values) < 2:
        return 0
    
    count = 0
    for i in range(len(values) - 1):
        current = values[i]
        prev = values[i + 1]
        if current is None or prev is None:
            break
        if current < prev:
            count += 1
        else:
            break
    return count


def _extract_quarterly_series(
    quarterly_data: dict[str, Any] | None,
    field_name: str,
) -> list[float | None]:
    """
    Extract a time series from stored quarterly financial data.
    
    The quarterly data is stored as a dict with date keys mapping to values,
    or as a list of dicts with period info.
    """
    if not quarterly_data:
        return []
    
    # Handle dict format: {"2024-09-30": value, "2024-06-30": value, ...}
    if isinstance(quarterly_data, dict):
        # Check if it's a nested structure with the field inside
        if field_name in quarterly_data:
            # Direct field in dict
            value = quarterly_data.get(field_name)
            if isinstance(value, dict):
                # It's {date: value, date: value}
                sorted_dates = sorted(value.keys(), reverse=True)
                return [_safe_float(value.get(d)) for d in sorted_dates]
            return [_safe_float(value)]
        
        # Maybe it's quarterly data keyed by date with field inside each
        # e.g., {"2024-09-30": {"TotalRevenue": 123, ...}, ...}
        sorted_dates = sorted(quarterly_data.keys(), reverse=True)
        values = []
        for d in sorted_dates:
            period_data = quarterly_data.get(d)
            if isinstance(period_data, dict):
                values.append(_safe_float(period_data.get(field_name)))
            else:
                values.append(_safe_float(period_data))
        return values if values else []
    
    # Handle list format
    if isinstance(quarterly_data, list):
        return [_safe_float(item.get(field_name)) if isinstance(item, dict) else None 
                for item in quarterly_data]
    
    return []


def analyze_fundamental_momentum_from_stored(
    stored_fundamentals: dict[str, Any],
) -> FundamentalMomentum:
    """
    Analyze fundamental momentum from locally stored fundamentals data.
    
    Uses pre-fetched quarterly income statement data from the database.
    Does NOT make any yfinance calls.
    
    Args:
        stored_fundamentals: Dict from get_fundamentals_from_db() containing
                            income_stmt_quarterly, revenue_growth, etc.
        
    Returns:
        FundamentalMomentum with trend analysis
    """
    result = FundamentalMomentum()
    
    if not stored_fundamentals:
        result.data_quality = "unknown"
        return result
    
    # Use pre-computed growth metrics when available
    revenue_growth = _safe_float(stored_fundamentals.get("revenue_growth"))
    earnings_growth = _safe_float(stored_fundamentals.get("earnings_growth"))
    earnings_qtr_growth = _safe_float(stored_fundamentals.get("earnings_quarterly_growth"))
    
    # Get quarterly income statement for detailed analysis
    income_stmt_quarterly = stored_fundamentals.get("income_stmt_quarterly")
    
    if income_stmt_quarterly:
        # Extract time series
        revenues = _extract_quarterly_series(income_stmt_quarterly, "TotalRevenue")
        if not revenues:
            revenues = _extract_quarterly_series(income_stmt_quarterly, "Total Revenue")
        
        net_incomes = _extract_quarterly_series(income_stmt_quarterly, "NetIncome")
        if not net_incomes:
            net_incomes = _extract_quarterly_series(income_stmt_quarterly, "Net Income")
        
        gross_profits = _extract_quarterly_series(income_stmt_quarterly, "GrossProfit")
        if not gross_profits:
            gross_profits = _extract_quarterly_series(income_stmt_quarterly, "Gross Profit")
        
        operating_incomes = _extract_quarterly_series(income_stmt_quarterly, "OperatingIncome")
        if not operating_incomes:
            operating_incomes = _extract_quarterly_series(income_stmt_quarterly, "Operating Income")
        
        n_quarters = max(len(revenues), len(net_incomes))
        result.quarters_analyzed = n_quarters
        
        # Calculate trends from quarterly data
        if len(revenues) >= 2 and revenues[0] is not None and revenues[1] is not None:
            result.revenue_qoq_change = _compute_pct_change(revenues[0], revenues[1])
        
        if len(revenues) >= 5 and revenues[0] is not None and revenues[4] is not None:
            result.revenue_yoy_change = _compute_pct_change(revenues[0], revenues[4])
        
        if len(net_incomes) >= 2 and net_incomes[0] is not None and net_incomes[1] is not None:
            result.earnings_qoq_change = _compute_pct_change(net_incomes[0], net_incomes[1])
        
        if len(net_incomes) >= 5 and net_incomes[0] is not None and net_incomes[4] is not None:
            result.earnings_yoy_change = _compute_pct_change(net_incomes[0], net_incomes[4])
        
        # Count declining quarters
        result.revenue_declining_quarters = _count_declining_periods(revenues)
        result.earnings_declining_quarters = _count_declining_periods(net_incomes)
        
        # Calculate margin trends
        gross_margins = []
        operating_margins = []
        for i in range(min(len(revenues), len(gross_profits))):
            if revenues[i] and revenues[i] > 0 and gross_profits[i] is not None:
                gross_margins.append(gross_profits[i] / revenues[i])
            else:
                gross_margins.append(None)
                
        for i in range(min(len(revenues), len(operating_incomes))):
            if revenues[i] and revenues[i] > 0 and operating_incomes[i] is not None:
                operating_margins.append(operating_incomes[i] / revenues[i])
            else:
                operating_margins.append(None)
        
        # Margin trend = recent margin - older margin (in percentage points)
        if len(gross_margins) >= 4 and gross_margins[0] is not None and gross_margins[3] is not None:
            result.gross_margin_trend = (gross_margins[0] - gross_margins[3]) * 100
        
        if len(operating_margins) >= 4 and operating_margins[0] is not None and operating_margins[3] is not None:
            result.operating_margin_trend = (operating_margins[0] - operating_margins[3]) * 100
        
        result.margin_compression_quarters = _count_declining_periods(operating_margins)
        result.data_quality = "full" if n_quarters >= 4 else "partial"
    else:
        # Fall back to using just the growth metrics
        result.data_quality = "minimal"
        if revenue_growth is not None:
            result.revenue_yoy_change = revenue_growth * 100  # Convert to percentage
        if earnings_growth is not None:
            result.earnings_yoy_change = earnings_growth * 100
        if earnings_qtr_growth is not None:
            result.earnings_qoq_change = earnings_qtr_growth * 100
    
    # Determine structural decline
    _assess_structural_decline(result)
    
    # Compute momentum score
    _compute_momentum_score(result, revenue_growth, earnings_growth)
    
    return result


def _assess_structural_decline(result: FundamentalMomentum) -> None:
    """Assess if the fundamental trends indicate structural decline."""
    decline_reasons = []
    severity_score = 0
    
    # Revenue declining 3+ quarters = warning sign
    if result.revenue_declining_quarters >= 3:
        decline_reasons.append(f"Revenue declining {result.revenue_declining_quarters} consecutive quarters")
        severity_score += 2
    elif result.revenue_declining_quarters >= 2:
        decline_reasons.append(f"Revenue declining {result.revenue_declining_quarters} consecutive quarters")
        severity_score += 1
    
    # Earnings declining more than revenue = margin pressure
    if result.earnings_declining_quarters >= 3:
        decline_reasons.append(f"Earnings declining {result.earnings_declining_quarters} consecutive quarters")
        severity_score += 2
    
    # Margin compression is the most serious sign
    if result.operating_margin_trend is not None and result.operating_margin_trend < -3:
        decline_reasons.append(f"Operating margin compressed {abs(result.operating_margin_trend):.1f}pp")
        severity_score += 3
    elif result.operating_margin_trend is not None and result.operating_margin_trend < -1:
        severity_score += 1
    
    if result.gross_margin_trend is not None and result.gross_margin_trend < -3:
        decline_reasons.append(f"Gross margin compressed {abs(result.gross_margin_trend):.1f}pp")
        severity_score += 2
    
    # Large YoY revenue decline
    if result.revenue_yoy_change is not None and result.revenue_yoy_change < -10:
        decline_reasons.append(f"Revenue down {abs(result.revenue_yoy_change):.1f}% YoY")
        severity_score += 2
    
    # Determine severity
    if severity_score >= 6:
        result.decline_severity = "severe"
        result.is_structural_decline = True
    elif severity_score >= 4:
        result.decline_severity = "moderate"
        result.is_structural_decline = True
    elif severity_score >= 2:
        result.decline_severity = "mild"
        result.is_structural_decline = False  # Mild might still be a dip opportunity
    else:
        result.decline_severity = "none"
    
    result.decline_reasons = decline_reasons if decline_reasons else None


def _compute_momentum_score(
    result: FundamentalMomentum,
    revenue_growth: float | None,
    earnings_growth: float | None,
) -> None:
    """Compute overall momentum score (0-100)."""
    momentum = 50.0
    
    # Use YoY changes if available from quarterly data, otherwise use stored growth rates
    rev_yoy = result.revenue_yoy_change
    if rev_yoy is None and revenue_growth is not None:
        rev_yoy = revenue_growth * 100
    
    earn_yoy = result.earnings_yoy_change
    if earn_yoy is None and earnings_growth is not None:
        earn_yoy = earnings_growth * 100
    
    # Revenue trend
    if rev_yoy is not None:
        if rev_yoy > 20:
            momentum += 15
        elif rev_yoy > 10:
            momentum += 10
        elif rev_yoy > 0:
            momentum += 5
        elif rev_yoy > -10:
            momentum -= 5
        elif rev_yoy > -20:
            momentum -= 10
        else:
            momentum -= 20
    
    # Earnings trend
    if earn_yoy is not None:
        if earn_yoy > 20:
            momentum += 15
        elif earn_yoy > 0:
            momentum += 8
        elif earn_yoy > -20:
            momentum -= 8
        else:
            momentum -= 15
    
    # Margin trend
    if result.operating_margin_trend is not None:
        if result.operating_margin_trend > 2:
            momentum += 10
        elif result.operating_margin_trend > 0:
            momentum += 5
        elif result.operating_margin_trend > -2:
            momentum -= 5
        else:
            momentum -= 15
    
    result.momentum_score = max(0, min(100, momentum))


async def analyze_fundamental_momentum(
    symbol: str,
    stored_fundamentals: dict[str, Any] | None = None,
) -> FundamentalMomentum:
    """
    Analyze fundamental momentum for a symbol.
    
    Uses locally stored fundamentals from scheduled jobs.
    Does NOT make ad-hoc yfinance calls.
    
    Args:
        symbol: Stock ticker
        stored_fundamentals: Pre-loaded fundamentals dict, or None to load from DB
        
    Returns:
        FundamentalMomentum with trend analysis
    """
    if stored_fundamentals is None:
        # Load from database (NOT ad-hoc yfinance fetch)
        from app.services.fundamentals import get_fundamentals_from_db
        stored_fundamentals = await get_fundamentals_from_db(symbol)
    
    if not stored_fundamentals:
        # No stored data - return unknown
        result = FundamentalMomentum()
        result.data_quality = "unknown"
        logger.debug(f"No stored fundamentals for {symbol}, cannot analyze momentum")
        return result
    
    return analyze_fundamental_momentum_from_stored(stored_fundamentals)


def is_dip_vs_structural_decline(
    momentum: FundamentalMomentum,
    dip_depth_pct: float,
) -> tuple[bool, str]:
    """
    Determine if a price drop is a buying dip or structural decline.
    
    Args:
        momentum: Fundamental momentum analysis
        dip_depth_pct: Current dip depth as percentage (e.g., 20 for -20%)
        
    Returns:
        (is_buy_opportunity, explanation)
    """
    if momentum.data_quality == "unknown":
        return True, "Insufficient data for structural analysis - treating as potential dip"
    
    if momentum.is_structural_decline:
        if momentum.decline_severity == "severe":
            reasons = ", ".join(momentum.decline_reasons[:2]) if momentum.decline_reasons else "multiple concerning trends"
            return False, f"AVOID: Structural decline detected ({reasons})"
        elif momentum.decline_severity == "moderate":
            if dip_depth_pct >= 30:
                return True, "Caution: Moderate fundamental issues but deep dip may be overdone"
            return False, "CAUTION: Moderate structural issues - wait for stabilization"
    
    # Not a structural decline
    if momentum.momentum_score >= 60:
        return True, f"BUY: Strong fundamentals (momentum {momentum.momentum_score:.0f}) with price dip"
    elif momentum.momentum_score >= 40:
        return True, f"CONSIDER: Neutral fundamentals (momentum {momentum.momentum_score:.0f}) - price may be fair"
    else:
        return True, f"CAUTION: Weak fundamentals (momentum {momentum.momentum_score:.0f}) but not structural decline"
