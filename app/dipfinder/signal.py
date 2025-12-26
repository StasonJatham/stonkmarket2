"""Signal module combining all scores into final dip signal.

Integrates dip metrics, market context, quality, and stability
into actionable signals with alert decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import DipFinderConfig, get_dipfinder_config
from .dip import DipMetrics, compute_dip_metrics
from .fundamentals import QualityMetrics
from .stability import StabilityMetrics


class DipClass(str, Enum):
    """Dip classification based on market context."""

    MARKET_DIP = "MARKET_DIP"  # Dip is mainly due to market decline
    STOCK_SPECIFIC = "STOCK_SPECIFIC"  # Stock is underperforming market significantly
    MIXED = "MIXED"  # Combination of market and stock-specific factors


class AlertLevel(str, Enum):
    """Alert level for the signal."""

    NONE = "NONE"  # Does not meet alert criteria
    GOOD = "GOOD"  # Meets basic alert criteria
    STRONG = "STRONG"  # Meets strong alert criteria


@dataclass
class MarketContext:
    """Market context for dip classification."""

    benchmark_ticker: str
    dip_mkt: float  # Benchmark dip fraction
    dip_stock: float  # Stock dip fraction
    excess_dip: float  # Stock dip - benchmark dip
    dip_class: DipClass
    benchmark_data_available: bool = True  # False if benchmark data was insufficient

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "benchmark_ticker": self.benchmark_ticker,
            "dip_mkt": round(self.dip_mkt, 6),
            "dip_stock": round(self.dip_stock, 6),
            "excess_dip": round(self.excess_dip, 6),
            "dip_class": self.dip_class.value,
            "benchmark_data_available": self.benchmark_data_available,
        }


@dataclass
class DipSignal:
    """Complete dip signal with all components."""

    ticker: str
    window: int
    benchmark: str
    as_of_date: str  # ISO date string

    # Dip metrics
    dip_metrics: DipMetrics

    # Market context
    market_context: MarketContext

    # Scores
    quality_metrics: QualityMetrics
    stability_metrics: StabilityMetrics
    dip_score: float  # 0-100
    final_score: float  # 0-100

    # Alert
    alert_level: AlertLevel
    should_alert: bool
    reason: str  # Human-readable explanation

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "ticker": self.ticker,
            "window": self.window,
            "benchmark": self.benchmark,
            "as_of_date": self.as_of_date,
            # Dip metrics
            "dip_stock": round(self.dip_metrics.dip_pct, 6),
            "peak_stock": round(self.dip_metrics.peak_price, 4),
            "current_price": round(self.dip_metrics.current_price, 4),
            "dip_pctl": round(self.dip_metrics.dip_percentile, 2),
            "dip_vs_typical": round(self.dip_metrics.dip_vs_typical, 4),
            "typical_dip": round(self.dip_metrics.typical_dip, 6),
            "persist_days": self.dip_metrics.persist_days,
            "days_since_peak": self.dip_metrics.days_since_peak,
            "is_meaningful": self.dip_metrics.is_meaningful,
            # Market context
            "dip_mkt": round(self.market_context.dip_mkt, 6),
            "excess_dip": round(self.market_context.excess_dip, 6),
            "dip_class": self.market_context.dip_class.value,
            # Scores
            "quality_score": round(self.quality_metrics.score, 2),
            "stability_score": round(self.stability_metrics.score, 2),
            "dip_score": round(self.dip_score, 2),
            "final_score": round(self.final_score, 2),
            # Alert
            "alert_level": self.alert_level.value,
            "should_alert": self.should_alert,
            "reason": self.reason,
            # Detailed factors
            "quality_factors": self.quality_metrics.to_dict(),
            "stability_factors": self.stability_metrics.to_dict(),
        }

    def to_db_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "ticker": self.ticker,
            "benchmark": self.benchmark,
            "window_days": self.window,
            "as_of_date": self.as_of_date,
            "dip_stock": self.dip_metrics.dip_pct,
            "peak_stock": self.dip_metrics.peak_price,
            "dip_pctl": self.dip_metrics.dip_percentile,
            "dip_vs_typical": self.dip_metrics.dip_vs_typical,
            "persist_days": self.dip_metrics.persist_days,
            "dip_mkt": self.market_context.dip_mkt,
            "excess_dip": self.market_context.excess_dip,
            "dip_class": self.market_context.dip_class.value,
            "quality_score": self.quality_metrics.score,
            "stability_score": self.stability_metrics.score,
            "dip_score": self.dip_score,
            "final_score": self.final_score,
            "alert_level": self.alert_level.value,
            "should_alert": self.should_alert,
            "reason": self.reason,
        }


def classify_dip(
    dip_stock: float,
    dip_mkt: float,
    config: DipFinderConfig | None = None,
) -> DipClass:
    """
    Classify dip based on market context.

    Args:
        dip_stock: Stock dip fraction
        dip_mkt: Market/benchmark dip fraction
        config: Configuration with thresholds

    Returns:
        DipClass enum value
    """
    if config is None:
        config = get_dipfinder_config()

    excess_dip = dip_stock - dip_mkt

    # Market dip AND stock not significantly underperforming
    if dip_mkt >= config.market_dip_threshold and excess_dip < config.excess_dip_market:
        return DipClass.MARKET_DIP

    # Stock significantly underperforming market
    if excess_dip >= config.excess_dip_stock_specific:
        return DipClass.STOCK_SPECIFIC

    # Mixed case
    return DipClass.MIXED


def compute_market_context(
    ticker: str,
    stock_prices: np.ndarray,
    benchmark_prices: np.ndarray,
    benchmark_ticker: str,
    window: int,
    config: DipFinderConfig | None = None,
) -> MarketContext:
    """
    Compute market context for dip classification.

    Args:
        ticker: Stock ticker
        stock_prices: Stock closing prices
        benchmark_prices: Benchmark closing prices
        benchmark_ticker: Benchmark ticker symbol
        window: Window for dip calculation
        config: Configuration

    Returns:
        MarketContext with classification
    """
    from .dip import compute_dip_series_windowed

    if config is None:
        config = get_dipfinder_config()

    # Compute stock dip
    stock_dips = compute_dip_series_windowed(stock_prices, window)
    dip_stock = (
        float(stock_dips[-1])
        if len(stock_dips) > 0 and not np.isnan(stock_dips[-1])
        else 0.0
    )

    # Compute benchmark dip
    benchmark_data_available = len(benchmark_prices) >= window
    if benchmark_data_available:
        benchmark_dips = compute_dip_series_windowed(benchmark_prices, window)
        dip_mkt = float(benchmark_dips[-1]) if not np.isnan(benchmark_dips[-1]) else 0.0
    else:
        dip_mkt = 0.0

    excess_dip = dip_stock - dip_mkt

    dip_class = classify_dip(dip_stock, dip_mkt, config)

    return MarketContext(
        benchmark_ticker=benchmark_ticker,
        dip_mkt=dip_mkt,
        dip_stock=dip_stock,
        excess_dip=excess_dip,
        dip_class=dip_class,
        benchmark_data_available=benchmark_data_available,
    )


def compute_dip_score(
    dip_metrics: DipMetrics,
    market_context: MarketContext,
    config: DipFinderConfig | None = None,
) -> float:
    """
    Compute dip score from dip metrics and market context.

    Args:
        dip_metrics: Computed dip metrics
        market_context: Market context with classification
        config: Configuration

    Returns:
        Dip score 0-100
    """
    if config is None:
        config = get_dipfinder_config()

    # GATE: If dip is below minimum threshold, score is 0
    # This prevents stocks with minimal dips from getting any score
    if dip_metrics.dip_pct < config.min_dip_abs:
        return 0.0

    score = 0.0

    # Base score from dip magnitude (max 40 points)
    # More dip = higher score (for buying opportunity)
    # Scale from min_dip_abs (baseline = 0 points) to 40% (extreme = 40 points)
    # Start at 0, not 20 - minimal dips just above threshold should score low
    magnitude_factor = min((dip_metrics.dip_pct - config.min_dip_abs) / 0.30, 1.0)
    score += magnitude_factor * 40

    # Percentile/rarity score (max 25 points)
    # Higher percentile = rarer dip = higher score
    percentile_factor = dip_metrics.dip_percentile / 100
    score += percentile_factor * 25

    # Dip vs typical score (max 20 points)
    if dip_metrics.dip_vs_typical >= config.dip_vs_typical_threshold:
        typical_factor = min((dip_metrics.dip_vs_typical - 1.0) / 2.0, 1.0)
        score += 10 + typical_factor * 10

    # Persistence score (max 10 points)
    if dip_metrics.persist_days >= config.min_persist_days:
        persist_factor = min(dip_metrics.persist_days / 10, 1.0)
        score += persist_factor * 10

    # Classification adjustment (max 5 points)
    if market_context.dip_class == DipClass.STOCK_SPECIFIC:
        # Stock-specific dips may be more actionable (or more risky)
        score += 5
    elif market_context.dip_class == DipClass.MIXED:
        score += 3
    elif market_context.dip_class == DipClass.MARKET_DIP:
        # Market dips can still be good buying opportunities
        score += 2

    return min(100.0, max(0.0, score))


def generate_reason(
    dip_metrics: DipMetrics,
    market_context: MarketContext,
    quality_metrics: QualityMetrics,
    stability_metrics: StabilityMetrics,
    dip_score: float,
    final_score: float,
    config: DipFinderConfig | None = None,
) -> str:
    """
    Generate human-readable reason for the signal.

    Returns one-line explanation of the opportunity.
    """
    if config is None:
        config = get_dipfinder_config()

    parts = []

    # Dip description
    dip_pct = dip_metrics.dip_pct * 100
    parts.append(f"{dip_pct:.1f}% from peak")

    # Significance
    if dip_metrics.dip_percentile >= 90:
        parts.append("rare dip (top 10%)")
    elif dip_metrics.dip_percentile >= 80:
        parts.append("significant dip (top 20%)")

    # Context
    if market_context.dip_class == DipClass.STOCK_SPECIFIC:
        excess_pct = market_context.excess_dip * 100
        parts.append(f"stock-specific ({excess_pct:.1f}% below market)")
    elif market_context.dip_class == DipClass.MARKET_DIP:
        parts.append("market-wide decline")
    else:
        parts.append("mixed market/stock factors")

    # Quality/stability flags
    if quality_metrics.score >= 80:
        parts.append("excellent fundamentals")
    elif quality_metrics.score >= 70:
        parts.append("strong fundamentals")
    elif quality_metrics.score < config.quality_gate:
        parts.append("weak fundamentals")

    if stability_metrics.score >= 80:
        parts.append("very stable")
    elif stability_metrics.score < config.stability_gate:
        parts.append("higher volatility")

    # Combine
    return "; ".join(parts)


def compute_signal(
    ticker: str,
    stock_prices: np.ndarray,
    benchmark_prices: np.ndarray,
    benchmark_ticker: str,
    window: int,
    quality_metrics: QualityMetrics,
    stability_metrics: StabilityMetrics,
    as_of_date: str,
    config: DipFinderConfig | None = None,
) -> DipSignal:
    """
    Compute complete dip signal for a ticker.

    Args:
        ticker: Stock ticker symbol
        stock_prices: Stock closing prices array
        benchmark_prices: Benchmark closing prices array
        benchmark_ticker: Benchmark ticker symbol
        window: Window for dip calculation
        quality_metrics: Pre-computed quality metrics
        stability_metrics: Pre-computed stability metrics
        as_of_date: Date string (ISO format)
        config: Configuration

    Returns:
        Complete DipSignal
    """
    if config is None:
        config = get_dipfinder_config()

    # Validate/normalize weights to ensure they sum to 1.0
    weight_sum = config.weight_dip + config.weight_quality + config.weight_stability
    if abs(weight_sum - 1.0) > 0.001:
        # Normalize at runtime if weights don't sum to 1
        w_dip = config.weight_dip / weight_sum
        w_quality = config.weight_quality / weight_sum
        w_stability = config.weight_stability / weight_sum
    else:
        w_dip = config.weight_dip
        w_quality = config.weight_quality
        w_stability = config.weight_stability

    # Compute dip metrics
    dip_metrics = compute_dip_metrics(ticker, stock_prices, window, config)

    # Compute market context
    market_context = compute_market_context(
        ticker, stock_prices, benchmark_prices, benchmark_ticker, window, config
    )

    # Compute dip score
    dip_score = compute_dip_score(dip_metrics, market_context, config)

    # Compute final score (weighted combination)
    # GATE: If dip is below minimum threshold, cap final score low
    # This prevents high-quality stocks with minimal dips from appearing as opportunities
    if dip_metrics.dip_pct < config.min_dip_abs:
        # No meaningful dip - score is just quality/stability with low cap
        final_score = min(
            w_quality * quality_metrics.score
            + w_stability * stability_metrics.score,
            25.0  # Cap at 25 for non-dip stocks
        )
    else:
        final_score = (
            w_dip * dip_score
            + w_quality * quality_metrics.score
            + w_stability * stability_metrics.score
        )

    # Determine alert level
    # Convert to native Python bool to avoid numpy bool issues with database
    should_alert = bool(
        final_score >= config.alert_good
        and dip_metrics.is_meaningful
        and quality_metrics.score >= config.quality_gate
        and stability_metrics.score >= config.stability_gate
    )

    if should_alert:
        if final_score >= config.alert_strong:
            alert_level = AlertLevel.STRONG
        else:
            alert_level = AlertLevel.GOOD
    else:
        alert_level = AlertLevel.NONE

    # Generate reason
    reason = generate_reason(
        dip_metrics,
        market_context,
        quality_metrics,
        stability_metrics,
        dip_score,
        final_score,
        config,
    )

    return DipSignal(
        ticker=ticker,
        window=window,
        benchmark=benchmark_ticker,
        as_of_date=as_of_date,
        dip_metrics=dip_metrics,
        market_context=market_context,
        quality_metrics=quality_metrics,
        stability_metrics=stability_metrics,
        dip_score=dip_score,
        final_score=final_score,
        alert_level=alert_level,
        should_alert=should_alert,
        reason=reason,
    )
