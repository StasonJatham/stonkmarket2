"""Structural decline detection and fundamental momentum analysis.

Distinguishes between temporary dips (buying opportunities) and structural
declines (value traps). Uses quarterly data trends and margin analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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


def _safe_float(value: Any) -> float | None:
    """Safely convert to float."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _compute_pct_change(current: float | None, previous: float | None) -> float | None:
    """Compute percentage change between two values."""
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


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


async def analyze_fundamental_momentum(
    symbol: str,
    quarterly_financials: dict[str, Any] | None = None,
) -> FundamentalMomentum:
    """
    Analyze fundamental momentum from quarterly financial data.
    
    Uses up to 8 quarters of data to detect:
    - Revenue/earnings trajectory (improving vs declining)
    - Margin compression trends
    - Structural decline patterns
    
    Args:
        symbol: Stock ticker
        quarterly_financials: Pre-fetched quarterly data or None to fetch
        
    Returns:
        FundamentalMomentum with trend analysis
    """
    result = FundamentalMomentum()
    
    if quarterly_financials is None:
        # Fetch from yfinance service
        from app.services.data_providers import get_yfinance_service
        svc = get_yfinance_service()
        financials = await svc.get_financials(symbol)
        if not financials:
            result.data_quality = "unknown"
            return result
        quarterly_financials = financials.get("quarterly", {})
    
    income_stmt = quarterly_financials.get("income_statement", {})
    
    if not income_stmt:
        result.data_quality = "unknown"
        return result
    
    # Extract quarterly values (yfinance returns most recent first)
    # We need to handle the dict format from our yfinance service
    revenues = []
    net_incomes = []
    gross_profits = []
    operating_incomes = []
    
    # The quarterly data is a single dict with the most recent quarter
    # For full trend analysis, we'd need the raw DataFrame
    # For now, use what's available
    revenue = _safe_float(income_stmt.get("Total Revenue"))
    net_income = _safe_float(income_stmt.get("Net Income"))
    gross_profit = _safe_float(income_stmt.get("Gross Profit"))
    operating_income = _safe_float(income_stmt.get("Operating Income"))
    
    # Calculate margins if we have the data
    gross_margin = None
    operating_margin = None
    if revenue and revenue > 0:
        if gross_profit is not None:
            gross_margin = gross_profit / revenue
        if operating_income is not None:
            operating_margin = operating_income / revenue
    
    # With single quarter data, we can only do limited analysis
    # Flag as partial data quality
    result.data_quality = "minimal"
    result.quarters_analyzed = 1
    
    # Use YoY growth metrics from ticker info as proxy
    # These are typically available and more reliable
    return result


async def analyze_fundamental_momentum_full(
    symbol: str,
) -> FundamentalMomentum:
    """
    Full fundamental momentum analysis using raw quarterly DataFrames.
    
    This fetches complete quarterly history and computes multi-quarter trends.
    More expensive but more accurate than single-quarter analysis.
    """
    import yfinance as yf
    from app.core.rate_limiter import get_yfinance_limiter
    
    result = FundamentalMomentum()
    limiter = get_yfinance_limiter()
    
    if not limiter.acquire_sync():
        logger.warning(f"Rate limited for {symbol} fundamental momentum")
        return result
    
    try:
        ticker = yf.Ticker(symbol)
        income_stmt = ticker.quarterly_income_stmt
        
        if income_stmt is None or income_stmt.empty:
            result.data_quality = "unknown"
            return result
        
        # Get up to 8 quarters of data
        n_quarters = min(8, income_stmt.shape[1])
        result.quarters_analyzed = n_quarters
        
        # Extract revenue series
        revenues = []
        for i in range(n_quarters):
            try:
                rev = income_stmt.loc["Total Revenue"].iloc[i] if "Total Revenue" in income_stmt.index else None
                revenues.append(_safe_float(rev))
            except (KeyError, IndexError):
                revenues.append(None)
        
        # Extract net income series
        net_incomes = []
        for i in range(n_quarters):
            try:
                ni = income_stmt.loc["Net Income"].iloc[i] if "Net Income" in income_stmt.index else None
                net_incomes.append(_safe_float(ni))
            except (KeyError, IndexError):
                net_incomes.append(None)
        
        # Extract gross profit for margin calculation
        gross_profits = []
        for i in range(n_quarters):
            try:
                gp = income_stmt.loc["Gross Profit"].iloc[i] if "Gross Profit" in income_stmt.index else None
                gross_profits.append(_safe_float(gp))
            except (KeyError, IndexError):
                gross_profits.append(None)
        
        # Extract operating income
        operating_incomes = []
        for i in range(n_quarters):
            try:
                oi = income_stmt.loc["Operating Income"].iloc[i] if "Operating Income" in income_stmt.index else None
                operating_incomes.append(_safe_float(oi))
            except (KeyError, IndexError):
                operating_incomes.append(None)
        
        # Calculate trends
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
            result.gross_margin_trend = (gross_margins[0] - gross_margins[3]) * 100  # pp change
        
        if len(operating_margins) >= 4 and operating_margins[0] is not None and operating_margins[3] is not None:
            result.operating_margin_trend = (operating_margins[0] - operating_margins[3]) * 100
        
        result.margin_compression_quarters = _count_declining_periods(operating_margins)
        
        # Determine structural decline
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
        
        # Compute momentum score (0-100, higher = better)
        # Start at 50 (neutral), adjust based on trends
        momentum = 50.0
        
        # Revenue trend
        if result.revenue_yoy_change is not None:
            if result.revenue_yoy_change > 20:
                momentum += 15
            elif result.revenue_yoy_change > 10:
                momentum += 10
            elif result.revenue_yoy_change > 0:
                momentum += 5
            elif result.revenue_yoy_change > -10:
                momentum -= 5
            elif result.revenue_yoy_change > -20:
                momentum -= 10
            else:
                momentum -= 20
        
        # Earnings trend
        if result.earnings_yoy_change is not None:
            if result.earnings_yoy_change > 20:
                momentum += 15
            elif result.earnings_yoy_change > 0:
                momentum += 8
            elif result.earnings_yoy_change > -20:
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
        result.data_quality = "full" if n_quarters >= 4 else "partial"
        
        return result
        
    except Exception as e:
        logger.warning(f"Error analyzing fundamental momentum for {symbol}: {e}")
        result.data_quality = "unknown"
        return result


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
            return False, f"CAUTION: Moderate structural issues - wait for stabilization"
    
    # Not a structural decline
    if momentum.momentum_score >= 60:
        return True, f"BUY: Strong fundamentals (momentum {momentum.momentum_score:.0f}) with price dip"
    elif momentum.momentum_score >= 40:
        return True, f"CONSIDER: Neutral fundamentals (momentum {momentum.momentum_score:.0f}) - price may be fair"
    else:
        return True, f"CAUTION: Weak fundamentals (momentum {momentum.momentum_score:.0f}) but not structural decline"
