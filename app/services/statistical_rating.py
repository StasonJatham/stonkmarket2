"""Statistical rating service - replaces AI-based dip rating with calculations.

This module provides a deterministic, statistics-based rating system that uses:
- Dip depth and duration
- Signal analysis from the quant engine
- Fundamental metrics when available

No OpenAI API calls - pure computation based on quantitative data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger


logger = get_logger("services.statistical_rating")


@dataclass
class RatingResult:
    """Rating calculation result."""
    rating: str  # strong_buy, buy, hold, sell, strong_sell
    reasoning: str  # Human-readable explanation
    confidence: int  # 1-10 scale
    opportunity_score: float  # 0-100 composite score
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility with old rate_dip API."""
        return {
            "rating": self.rating,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "opportunity_score": self.opportunity_score,
        }


def calculate_rating(
    *,
    symbol: str,
    dip_pct: float | None = None,
    days_in_dip: int | None = None,
    current_price: float | None = None,
    ref_high: float | None = None,
    # Signal data from quant engine
    buy_score: float | None = None,
    opportunity_type: str | None = None,
    best_signal_name: str | None = None,
    best_expected_return: float | None = None,
    zscore_20d: float | None = None,
    rsi_14: float | None = None,
    # Fundamentals
    pe_ratio: float | None = None,
    forward_pe: float | None = None,
    peg_ratio: float | None = None,
    ev_to_ebitda: float | None = None,
    profit_margin: float | None = None,
    revenue_growth: float | None = None,
    debt_to_equity: float | None = None,
    recommendation: str | None = None,
    # Dip classification
    dip_classification: str | None = None,
    quality_score: float | None = None,
    stability_score: float | None = None,
    # Extra context (ignored but accepted for compatibility)
    **extra,
) -> RatingResult:
    """
    Calculate a dip-buy rating using statistical methods.
    
    This replaces the AI-based rate_dip function with deterministic calculations.
    
    Decision rubric (in order of priority):
    1. Dip depth: >= 20% = strong_buy candidate, 10-20% = buy, < 10% = hold
    2. Quality score: >= 70 = +1 confidence, < 40 = downgrade
    3. Stability score: < 30 = -1 confidence (volatile)
    4. Signal strength: buy_score > 70 = upgrade, < 30 = downgrade
    5. Valuation check: P/E > 50 and EV/EBITDA > 25 = downgrade unless dip > 25%
    6. Dip persistence: >= 30 days = +1 confidence, < 14 days = -1 confidence
    
    Returns:
        RatingResult with rating, reasoning, confidence, and opportunity_score
    """
    # Initialize
    opportunity_score = 50.0  # Neutral baseline
    confidence = 6  # Default confidence
    factors: list[str] = []  # Reasoning factors
    
    # Ensure dip_pct is calculated if we have ref_high and current_price
    if dip_pct is None and ref_high and current_price and ref_high > 0:
        dip_pct = ((ref_high - current_price) / ref_high) * 100
    
    # =========================================================================
    # Step 1: Dip depth contribution (up to 30 points)
    # =========================================================================
    if dip_pct is not None and dip_pct > 5:
        # Deep dips get more points
        dip_contribution = min(dip_pct * 1.0, 30)
        opportunity_score += dip_contribution
        
        if dip_pct >= 25:
            factors.append(f"Deep dip {dip_pct:.1f}% (strong opportunity)")
            confidence += 1
        elif dip_pct >= 15:
            factors.append(f"Solid dip {dip_pct:.1f}%")
        elif dip_pct >= 10:
            factors.append(f"Moderate dip {dip_pct:.1f}%")
        else:
            factors.append(f"Minor dip {dip_pct:.1f}%")
    else:
        factors.append("No significant dip")
    
    # =========================================================================
    # Step 2: Signal analysis contribution (up to 25 points)
    # =========================================================================
    if buy_score is not None:
        if buy_score >= 70:
            opportunity_score += 25
            factors.append(f"Strong buy signals (score {buy_score:.0f})")
            confidence += 1
        elif buy_score >= 50:
            opportunity_score += 15
            factors.append(f"Positive signals (score {buy_score:.0f})")
        elif buy_score >= 30:
            opportunity_score += 5
            factors.append(f"Weak signals (score {buy_score:.0f})")
        else:
            opportunity_score -= 10
            factors.append(f"Negative signals (score {buy_score:.0f})")
            confidence -= 1
    
    # Best signal info
    if best_signal_name and best_expected_return:
        if best_expected_return > 0.1:  # > 10% expected return
            opportunity_score += 5
            factors.append(f"{best_signal_name}: {best_expected_return*100:.1f}% expected return")
    
    # =========================================================================
    # Step 3: Quality and stability scores (up to 20 points)
    # =========================================================================
    if quality_score is not None:
        if quality_score >= 70:
            opportunity_score += 15
            factors.append(f"High quality ({quality_score:.0f}/100)")
            confidence += 1
        elif quality_score >= 50:
            opportunity_score += 8
            factors.append(f"Moderate quality ({quality_score:.0f}/100)")
        elif quality_score < 40:
            opportunity_score -= 10
            factors.append(f"Low quality ({quality_score:.0f}/100)")
            confidence -= 1
    
    if stability_score is not None:
        if stability_score >= 60:
            opportunity_score += 5
        elif stability_score < 30:
            opportunity_score -= 5
            factors.append(f"High volatility (stability {stability_score:.0f})")
            confidence -= 1
    
    # =========================================================================
    # Step 4: Dip classification (market vs stock-specific)
    # =========================================================================
    if dip_classification:
        if dip_classification == "STOCK_SPECIFIC":
            # Stock is underperforming market - more cautious
            opportunity_score -= 5
            factors.append("Stock-specific decline (investigate)")
        elif dip_classification == "MARKET_DIP":
            # Stock is down with market - less concerning
            factors.append("Market-wide pullback")
    
    # =========================================================================
    # Step 5: Valuation check
    # =========================================================================
    valuation_warning = False
    if pe_ratio is not None and pe_ratio > 50:
        valuation_warning = True
    if ev_to_ebitda is not None and ev_to_ebitda > 25:
        valuation_warning = True
    
    if valuation_warning:
        if dip_pct is None or dip_pct < 25:
            opportunity_score -= 10
            factors.append("High valuation risk")
            confidence -= 1
    
    # Positive valuation signals
    if forward_pe is not None and forward_pe < 15:
        opportunity_score += 5
        factors.append(f"Attractive forward P/E ({forward_pe:.1f})")
    
    if peg_ratio is not None and peg_ratio < 1.0:
        opportunity_score += 5
        factors.append(f"Good PEG ratio ({peg_ratio:.2f})")
    
    # =========================================================================
    # Step 6: Dip persistence
    # =========================================================================
    if days_in_dip is not None:
        if days_in_dip >= 30:
            opportunity_score += 5
            factors.append(f"Sustained dip ({days_in_dip} days)")
            confidence += 1
        elif days_in_dip < 7:
            opportunity_score -= 5
            factors.append(f"Recent dip ({days_in_dip} days)")
            confidence -= 1
    
    # =========================================================================
    # Step 7: Profitability and growth signals
    # =========================================================================
    if profit_margin is not None and profit_margin > 0.15:
        opportunity_score += 3
    
    if revenue_growth is not None and revenue_growth > 0.10:
        opportunity_score += 3
        factors.append(f"Growing revenue ({revenue_growth*100:.0f}%)")
    
    # =========================================================================
    # Step 8: Technical indicators
    # =========================================================================
    if zscore_20d is not None:
        if zscore_20d < -2.0:
            opportunity_score += 5
            factors.append(f"Oversold (z-score {zscore_20d:.2f})")
        elif zscore_20d > 2.0:
            opportunity_score -= 5
            factors.append(f"Overbought (z-score {zscore_20d:.2f})")
    
    if rsi_14 is not None:
        if rsi_14 < 30:
            opportunity_score += 5
            factors.append(f"RSI oversold ({rsi_14:.0f})")
        elif rsi_14 > 70:
            opportunity_score -= 5
            factors.append(f"RSI overbought ({rsi_14:.0f})")
    
    # =========================================================================
    # Step 9: Analyst consensus
    # =========================================================================
    if recommendation:
        rec_lower = recommendation.lower()
        if rec_lower in ("strong_buy", "strongbuy"):
            opportunity_score += 5
            factors.append("Analysts: Strong Buy")
        elif rec_lower == "buy":
            opportunity_score += 3
            factors.append("Analysts: Buy")
        elif rec_lower in ("sell", "strong_sell", "strongsell"):
            opportunity_score -= 5
            factors.append("Analysts: Sell")
    
    # =========================================================================
    # Final calculations
    # =========================================================================
    
    # Clamp opportunity score to 0-100
    opportunity_score = max(0.0, min(100.0, opportunity_score))
    
    # Clamp confidence to 1-10
    confidence = max(1, min(10, confidence))
    
    # Determine rating from opportunity score
    if opportunity_score >= 75:
        rating = "strong_buy"
    elif opportunity_score >= 60:
        rating = "buy"
    elif opportunity_score >= 40:
        rating = "hold"
    elif opportunity_score >= 25:
        rating = "sell"
    else:
        rating = "strong_sell"
    
    # Override: If dip is massive and quality is decent, never rate as sell
    if dip_pct is not None and dip_pct >= 30:
        if quality_score is None or quality_score >= 40:
            if rating in ("sell", "strong_sell"):
                rating = "hold"
                factors.append("Deep value opportunity despite risks")
    
    # Build reasoning string (max 400 chars like the old AI)
    if factors:
        reasoning = f"{symbol}: " + ". ".join(factors[:5])  # Top 5 factors
        if len(reasoning) > 380:
            reasoning = reasoning[:377] + "..."
    else:
        reasoning = f"{symbol}: Insufficient data for detailed analysis."
    
    logger.debug(f"Statistical rating for {symbol}: {rating} (score={opportunity_score:.1f}, conf={confidence})")
    
    return RatingResult(
        rating=rating,
        reasoning=reasoning,
        confidence=confidence,
        opportunity_score=opportunity_score,
    )


async def get_rating_for_symbol(
    symbol: str,
    *,
    include_signals: bool = True,
    include_fundamentals: bool = True,
) -> RatingResult | None:
    """
    Get a statistical rating for a symbol by fetching all available data.
    
    This is the async wrapper that fetches data and calls calculate_rating().
    
    Args:
        symbol: Stock ticker symbol
        include_signals: Whether to fetch quant engine signals
        include_fundamentals: Whether to fetch fundamentals
    
    Returns:
        RatingResult or None if symbol not found
    """
    from app.repositories import dip_state_orm as dip_state_repo
    from app.services.stock_info import get_stock_info_async
    from app.services.fundamentals import get_fundamentals_for_analysis
    
    # Get dip state
    dip_state = await dip_state_repo.get_dip_state(symbol)
    if not dip_state:
        logger.warning(f"No dip state for {symbol}, cannot calculate rating")
        return None
    
    # Calculate days in dip
    days_in_dip = None
    if dip_state.dip_start_date:
        from datetime import date
        days_in_dip = (date.today() - dip_state.dip_start_date).days
    
    # Get fundamentals if requested
    fundamentals: dict[str, Any] = {}
    if include_fundamentals:
        fundamentals = await get_fundamentals_for_analysis(symbol)
    
    # Get signal data if requested (optional - may not be available)
    signal_data: dict[str, Any] = {}
    if include_signals:
        try:
            from datetime import timedelta
            from app.quant_engine import scan_single_stock
            from app.services.prices import get_price_service
            
            # This is expensive - only do it if we really need signals
            end_date = date.today()
            start_date = end_date - timedelta(days=400)
            price_service = get_price_service()
            df = await price_service.get_prices(symbol, start_date, end_date)
            
            prices = df["Close"] if df is not None and "Close" in df.columns else None
            if prices is not None and len(prices) >= 60:
                opp = scan_single_stock(
                    symbol=symbol,
                    prices=prices,
                    name=fundamentals.get("name", symbol),
                )
                if opp:
                    signal_data = {
                        "buy_score": opp.buy_score,
                        "opportunity_type": opp.opportunity_type,
                        "best_signal_name": opp.best_signal_name,
                        "best_expected_return": opp.best_expected_return,
                        "zscore_20d": opp.zscore_20d,
                        "rsi_14": opp.rsi_14,
                    }
        except Exception as e:
            logger.debug(f"Could not get signals for {symbol}: {e}")
    
    return calculate_rating(
        symbol=symbol,
        dip_pct=float(dip_state.dip_percentage) if dip_state.dip_percentage else None,
        days_in_dip=days_in_dip,
        current_price=float(dip_state.current_price) if dip_state.current_price else None,
        ref_high=float(dip_state.ath_price) if dip_state.ath_price else None,
        **signal_data,
        **fundamentals,
    )
