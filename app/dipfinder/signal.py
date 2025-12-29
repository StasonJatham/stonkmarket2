"""Signal module combining all scores into final dip signal.

Integrates dip metrics, market context, quality, and stability
into actionable signals with alert decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import DipFinderConfig, get_dipfinder_config
from .dip import DipMetrics, compute_dip_metrics
from .fundamentals import QualityMetrics
from .stability import StabilityMetrics

if TYPE_CHECKING:
    from app.dipfinder.earnings_calendar import EarningsInfo
    from app.dipfinder.sector_valuation import SectorRelativeValuation
    from app.dipfinder.structural_analysis import FundamentalMomentum
    from app.quant_engine.support_resistance import SupportResistanceAnalysis

QUANT_BLEND_WEIGHT = 0.30
FUND_MOM_WARN_THRESHOLD = 0.35
FUND_MOM_SEVERE_THRESHOLD = 0.25
FUND_MOM_PENALTY = 6.0
FUND_MOM_SEVERE_PENALTY = 12.0
EVENT_PENALTY_MULTIPLIER = 0.4

# Enhanced analysis score adjustment weights
VOLUME_CONFIRMATION_BONUS = 5.0  # Bonus for volume-confirmed dips
VOLUME_SPIKE_BONUS_MAX = 8.0  # Max bonus for high volume spike
SELLOFF_INTENSITY_BONUS_MAX = 5.0  # Max bonus for high selloff intensity (capitulation)
SUPPORT_PROXIMITY_BONUS = 10.0  # Bonus for being near strong support
SUPPORT_BREAK_PENALTY = 15.0  # Penalty for trading below all support
VALUATION_BONUS_MAX = 8.0  # Max bonus for undervaluation
VALUATION_PENALTY_MAX = 6.0  # Max penalty for overvaluation
STRUCTURAL_DECLINE_PENALTY_MODERATE = 12.0
STRUCTURAL_DECLINE_PENALTY_SEVERE = 25.0
EARNINGS_DETERIORATION_PENALTY = 10.0  # Penalty for post-earnings decline


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
class QuantContext:
    """Quant engine context for adjusting dipfinder scores."""

    best_score: float
    mode: str | None = None
    gate_pass: bool = False
    fund_mom: float | None = None
    event_risk: bool = False


@dataclass
class EnhancedAnalysisInputs:
    """
    Container for enhanced analysis data used to adjust final score.
    
    All fields are optional - missing data is treated as neutral (no adjustment),
    not as a penalty. The data_quality field indicates overall reliability.
    """
    
    # Structural decline analysis
    structural_momentum: "FundamentalMomentum | None" = None
    
    # Support/resistance analysis
    support_resistance: "SupportResistanceAnalysis | None" = None
    
    # Sector-relative valuation
    sector_valuation: "SectorRelativeValuation | None" = None
    
    # Earnings calendar
    earnings_info: "EarningsInfo | None" = None
    
    # Data quality indicator (full, partial, minimal, unknown)
    data_quality: str = "unknown"


@dataclass
class EnhancedScoreAdjustments:
    """Breakdown of score adjustments from enhanced analysis."""
    
    volume_adjustment: float = 0.0
    support_resistance_adjustment: float = 0.0
    valuation_adjustment: float = 0.0
    structural_adjustment: float = 0.0
    earnings_adjustment: float = 0.0
    total_adjustment: float = 0.0
    
    # Flags for UI/explanations
    volume_confirmed: bool = False
    near_support: bool = False
    below_support: bool = False
    undervalued: bool = False
    overvalued: bool = False
    structural_decline: bool = False
    post_earnings_decline: bool = False
    
    def to_dict(self) -> dict:
        return {
            "volume_adjustment": round(self.volume_adjustment, 2),
            "support_resistance_adjustment": round(self.support_resistance_adjustment, 2),
            "valuation_adjustment": round(self.valuation_adjustment, 2),
            "structural_adjustment": round(self.structural_adjustment, 2),
            "earnings_adjustment": round(self.earnings_adjustment, 2),
            "total_adjustment": round(self.total_adjustment, 2),
            "flags": {
                "volume_confirmed": self.volume_confirmed,
                "near_support": self.near_support,
                "below_support": self.below_support,
                "undervalued": self.undervalued,
                "overvalued": self.overvalued,
                "structural_decline": self.structural_decline,
                "post_earnings_decline": self.post_earnings_decline,
            }
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

    # Enhanced analysis (optional, populated when available)
    # These provide deeper insights into dip quality
    enhanced_analysis: dict[str, Any] | None = None  # Contains structural, support/resistance, sector, earnings data

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        result = {
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
            # Volume confirmation
            "volume_spike_ratio": round(self.dip_metrics.volume_spike_ratio, 2),
            "volume_confirmed": self.dip_metrics.volume_confirmed,
            "selloff_intensity": round(self.dip_metrics.selloff_intensity, 4),
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
        
        # Add enhanced analysis if available
        if self.enhanced_analysis:
            result["enhanced_analysis"] = self.enhanced_analysis
        
        return result

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
        excess_dip = dip_stock - dip_mkt
        dip_class = classify_dip(dip_stock, dip_mkt, config)
    else:
        dip_mkt = 0.0
        excess_dip = 0.0
        dip_class = DipClass.MIXED

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
    min_dip_threshold: float | None = None,
) -> float:
    """
    Compute dip score from dip metrics and market context.

    Args:
        dip_metrics: Computed dip metrics
        market_context: Market context with classification
        config: Configuration
        min_dip_threshold: Per-symbol minimum dip threshold (overrides config.min_dip_abs)

    Returns:
        Dip score 0-100
    """
    if config is None:
        config = get_dipfinder_config()
    
    # Use per-symbol threshold if provided
    effective_min_dip = min_dip_threshold if min_dip_threshold is not None else config.min_dip_abs

    # GATE: If dip is below minimum threshold, score is 0
    # This prevents stocks with minimal dips from getting any score
    if dip_metrics.dip_pct < effective_min_dip:
        return 0.0

    score = 0.0

    # Base score from dip magnitude (max 40 points)
    # More dip = higher score (for buying opportunity)
    # Scale from min_dip threshold (baseline = 0 points) to 40% (extreme = 40 points)
    # Start at 0, not 20 - minimal dips just above threshold should score low
    magnitude_factor = min((dip_metrics.dip_pct - effective_min_dip) / 0.30, 1.0)
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
    if market_context.benchmark_data_available:
        if market_context.dip_class == DipClass.STOCK_SPECIFIC:
            # Stock-specific dips may be more actionable (or more risky)
            score += 5
        elif market_context.dip_class == DipClass.MIXED:
            score += 3
        elif market_context.dip_class == DipClass.MARKET_DIP:
            # Market dips can still be good buying opportunities
            score += 2

    return min(100.0, max(0.0, score))


def _apply_quant_adjustments(
    final_score: float,
    quant_context: QuantContext,
    structural_decline_already_penalized: bool = False,
    in_downtrend: bool = False,
) -> tuple[float, bool]:
    """
    Apply quant engine adjustments to final score.
    
    Args:
        final_score: Current score
        quant_context: Quant engine context
        structural_decline_already_penalized: If True, skip fund_mom penalty to avoid
            double-penalizing fundamental deterioration (structural analysis already applied)
        in_downtrend: If True, stock is in a downtrend regime (apply stricter gating)
    
    Returns:
        (adjusted_score, fundamental_warning)
    """
    blended = final_score * (1 - QUANT_BLEND_WEIGHT) + quant_context.best_score * QUANT_BLEND_WEIGHT
    fundamental_warning = False
    penalty = 0.0

    # Only apply fund_mom penalty if structural decline wasn't already penalized
    # This prevents double-penalization for fundamental deterioration
    if quant_context.fund_mom is not None and not structural_decline_already_penalized:
        if quant_context.fund_mom < FUND_MOM_SEVERE_THRESHOLD:
            penalty = FUND_MOM_SEVERE_PENALTY
            fundamental_warning = True
        elif quant_context.fund_mom < FUND_MOM_WARN_THRESHOLD:
            penalty = FUND_MOM_PENALTY
            fundamental_warning = True
    elif structural_decline_already_penalized:
        # Still flag as fundamental warning, but don't double-penalize
        fundamental_warning = True

    if penalty > 0 and quant_context.event_risk:
        penalty *= EVENT_PENALTY_MULTIPLIER

    adjusted = blended - penalty
    
    # Downtrend regime gating:
    # If stock is in downtrend and gate_pass=False (not certified buy),
    # apply additional penalty to reduce false positives from catching falling knives
    if in_downtrend and not quant_context.gate_pass:
        # Mode B (DIP_ENTRY) in downtrend = more speculative, reduce score
        downtrend_penalty = 10.0
        adjusted -= downtrend_penalty
        fundamental_warning = True  # Flag as risky
    
    return min(100.0, max(0.0, adjusted)), fundamental_warning


def _compute_volume_adjustment(dip_metrics: DipMetrics) -> tuple[float, bool]:
    """
    Compute score adjustment based on volume confirmation.
    
    Volume-confirmed dips (high volume during selloff) indicate capitulation
    and are more likely to mark a bottom.
    
    Returns:
        (adjustment, volume_confirmed_flag)
    """
    adjustment = 0.0
    
    # Base bonus for volume confirmation
    if dip_metrics.volume_confirmed:
        adjustment += VOLUME_CONFIRMATION_BONUS
    
    # Additional bonus for very high volume spike (panic selling = potential bottom)
    if dip_metrics.volume_spike_ratio > 1.5:
        # Scale from 1.5x to 3x volume -> 0 to max bonus
        spike_factor = min((dip_metrics.volume_spike_ratio - 1.5) / 1.5, 1.0)
        adjustment += spike_factor * VOLUME_SPIKE_BONUS_MAX
    
    # Bonus for high selloff intensity (capitulation signal)
    if dip_metrics.selloff_intensity > 0.5:
        intensity_factor = min((dip_metrics.selloff_intensity - 0.5) / 0.5, 1.0)
        adjustment += intensity_factor * SELLOFF_INTENSITY_BONUS_MAX
    
    return adjustment, dip_metrics.volume_confirmed


def _compute_support_resistance_adjustment(
    enhanced: EnhancedAnalysisInputs,
    current_price: float,
) -> tuple[float, bool, bool]:
    """
    Compute score adjustment based on support/resistance levels.
    
    Dips near strong support are more attractive.
    Dips below all support levels are risky (no floor).
    
    Returns:
        (adjustment, near_support, below_support)
    """
    sr = enhanced.support_resistance
    if sr is None:
        return 0.0, False, False
    
    adjustment = 0.0
    near_support = False
    below_support = False
    
    # Check if near support
    if sr.price_position == "near_support":
        near_support = True
        # Bonus based on support strength
        if sr.nearest_support and sr.nearest_support.strength > 50:
            strength_factor = sr.nearest_support.strength / 100
            adjustment += SUPPORT_PROXIMITY_BONUS * strength_factor
        else:
            adjustment += SUPPORT_PROXIMITY_BONUS * 0.5
    
    # Check if below all support (risky)
    elif sr.price_position == "below_support":
        below_support = True
        adjustment -= SUPPORT_BREAK_PENALTY
    
    # Good risk/reward ratio is a bonus
    if sr.risk_reward_ratio is not None and sr.risk_reward_ratio > 2.0:
        # Reward/risk > 2:1 is good
        rr_bonus = min((sr.risk_reward_ratio - 2.0) / 3.0, 1.0) * 5.0
        adjustment += rr_bonus
    
    # Entry quality adjustment
    if sr.entry_quality == "excellent":
        adjustment += 3.0
    elif sr.entry_quality == "good":
        adjustment += 1.5
    elif sr.entry_quality == "poor":
        adjustment -= 3.0
    
    return adjustment, near_support, below_support


def _compute_valuation_adjustment(enhanced: EnhancedAnalysisInputs) -> tuple[float, bool, bool]:
    """
    Compute score adjustment based on sector-relative valuation.
    
    Undervalued stocks get a bonus, overvalued stocks get a penalty.
    
    Returns:
        (adjustment, undervalued_flag, overvalued_flag)
    """
    val = enhanced.sector_valuation
    if val is None:
        return 0.0, False, False
    
    adjustment = 0.0
    
    # Use valuation_score (0-1, higher = more attractive)
    # 0.5 is neutral, >0.5 is undervalued, <0.5 is overvalued
    if val.valuation_score > 0.5:
        # Undervalued: bonus
        factor = (val.valuation_score - 0.5) / 0.5  # 0 to 1
        adjustment += factor * VALUATION_BONUS_MAX
    elif val.valuation_score < 0.5:
        # Overvalued: penalty
        factor = (0.5 - val.valuation_score) / 0.5  # 0 to 1
        adjustment -= factor * VALUATION_PENALTY_MAX
    
    # Extra flags for strong signals
    undervalued = val.is_deeply_undervalued
    overvalued = val.is_overvalued
    
    if undervalued:
        adjustment += 2.0  # Extra bonus for deeply undervalued
    if overvalued:
        adjustment -= 2.0  # Extra penalty for clearly overvalued
    
    return adjustment, undervalued, overvalued


def _compute_structural_adjustment(enhanced: EnhancedAnalysisInputs) -> tuple[float, bool]:
    """
    Compute score adjustment based on structural decline analysis.
    
    Structural declines (deteriorating fundamentals) should be penalized.
    Strong fundamental momentum should be rewarded.
    Penalties are scaled by data quality - minimal/partial data gets reduced weight.
    
    Returns:
        (adjustment, structural_decline_flag)
    """
    momentum = enhanced.structural_momentum
    if momentum is None or momentum.data_quality == "unknown":
        # Missing data = neutral, not penalizing
        return 0.0, False
    
    # Scale penalties/rewards by data quality
    # Full data (4+ quarters): 100% weight
    # Partial data (2-3 quarters): 50% weight
    # Minimal data (YoY only): 0% weight (not enough data to penalize)
    data_quality_weight = {
        "full": 1.0,
        "partial": 0.5,
        "minimal": 0.0,
    }.get(momentum.data_quality, 0.0)
    
    adjustment = 0.0
    is_structural = False
    
    if momentum.is_structural_decline:
        is_structural = True
        if momentum.decline_severity == "severe":
            adjustment -= STRUCTURAL_DECLINE_PENALTY_SEVERE * data_quality_weight
        elif momentum.decline_severity == "moderate":
            adjustment -= STRUCTURAL_DECLINE_PENALTY_MODERATE * data_quality_weight
        # Mild decline: no penalty (might still be opportunity)
    else:
        # Not a structural decline - reward strong momentum
        if momentum.momentum_score >= 70:
            adjustment += 5.0 * data_quality_weight  # Strong fundamentals bonus
        elif momentum.momentum_score >= 60:
            adjustment += 3.0 * data_quality_weight  # Good fundamentals bonus
    
    return adjustment, is_structural


def _compute_earnings_adjustment(enhanced: EnhancedAnalysisInputs) -> tuple[float, bool]:
    """
    Compute score adjustment based on earnings impact.
    
    IMPORTANT: Only apply penalty if:
    1. Earnings were recently reported (within last 30 days), AND
    2. Fundamentals show deterioration (structural decline or negative momentum)
    
    Pre-earnings risk is handled separately via pre_earnings_risk_level.
    
    Returns:
        (adjustment, post_earnings_decline_flag)
    """
    earnings = enhanced.earnings_info
    momentum = enhanced.structural_momentum
    
    if earnings is None:
        return 0.0, False
    
    adjustment = 0.0
    post_earnings_decline = False
    
    # Check for post-earnings deterioration scenario
    # Earnings reported recently AND fundamentals declining
    if earnings.days_since_earnings is not None and earnings.days_since_earnings <= 30:
        # Recent earnings - check if fundamentals deteriorated
        if momentum is not None and momentum.is_structural_decline:
            post_earnings_decline = True
            severity_factor = 1.0
            if momentum.decline_severity == "severe":
                severity_factor = 1.5
            elif momentum.decline_severity == "mild":
                severity_factor = 0.5
            adjustment -= EARNINGS_DETERIORATION_PENALTY * severity_factor
    
    # Note: No pre-earnings penalty. The requirement is to only adjust post-earnings
    # when fundamentals deteriorate. Pre-earnings uncertainty is not penalized because:
    # 1) Dips before earnings could be overreactions to fear, not fundamental issues
    # 2) We don't have EPS/revenue surprise or guidance data to assess pre-earnings risk
    # 3) The structural decline check above handles post-earnings deterioration
    
    return adjustment, post_earnings_decline


def compute_enhanced_adjustments(
    dip_metrics: DipMetrics,
    enhanced: EnhancedAnalysisInputs | None,
) -> EnhancedScoreAdjustments:
    """
    Compute all enhanced analysis score adjustments.
    
    Missing data is treated as neutral (no adjustment), not as a penalty.
    
    Args:
        dip_metrics: Computed dip metrics (for volume data)
        enhanced: Enhanced analysis inputs (can be None)
        
    Returns:
        EnhancedScoreAdjustments with breakdown
    """
    result = EnhancedScoreAdjustments()
    
    # Volume confirmation (always available from dip_metrics)
    result.volume_adjustment, result.volume_confirmed = _compute_volume_adjustment(dip_metrics)
    
    if enhanced is None:
        result.total_adjustment = result.volume_adjustment
        return result
    
    # Support/resistance
    result.support_resistance_adjustment, result.near_support, result.below_support = (
        _compute_support_resistance_adjustment(enhanced, dip_metrics.current_price)
    )
    
    # Valuation
    result.valuation_adjustment, result.undervalued, result.overvalued = (
        _compute_valuation_adjustment(enhanced)
    )
    
    # Structural decline
    result.structural_adjustment, result.structural_decline = (
        _compute_structural_adjustment(enhanced)
    )
    
    # Earnings
    result.earnings_adjustment, result.post_earnings_decline = (
        _compute_earnings_adjustment(enhanced)
    )
    
    # Sum all adjustments
    result.total_adjustment = (
        result.volume_adjustment +
        result.support_resistance_adjustment +
        result.valuation_adjustment +
        result.structural_adjustment +
        result.earnings_adjustment
    )
    
    return result


def generate_reason(
    dip_metrics: DipMetrics,
    market_context: MarketContext,
    quality_metrics: QualityMetrics,
    stability_metrics: StabilityMetrics,
    dip_score: float,
    final_score: float,
    config: DipFinderConfig | None = None,
    quant_context: QuantContext | None = None,
    fundamental_warning: bool = False,
    enhanced_adjustments: EnhancedScoreAdjustments | None = None,
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
    if not market_context.benchmark_data_available:
        parts.append("benchmark data limited")
    elif market_context.dip_class == DipClass.STOCK_SPECIFIC:
        excess_pct = market_context.excess_dip * 100
        parts.append(f"stock-specific ({excess_pct:.1f}% below market)")
    elif market_context.dip_class == DipClass.MARKET_DIP:
        parts.append("market-wide decline")
    else:
        parts.append("mixed market/stock factors")

    # Enhanced analysis insights
    if enhanced_adjustments is not None:
        if enhanced_adjustments.volume_confirmed:
            parts.append("volume-confirmed selloff")
        if enhanced_adjustments.near_support:
            parts.append("near key support")
        if enhanced_adjustments.below_support:
            parts.append("⚠️ below support levels")
        if enhanced_adjustments.undervalued:
            parts.append("deeply undervalued")
        if enhanced_adjustments.overvalued:
            parts.append("expensive valuation")
        if enhanced_adjustments.structural_decline:
            parts.append("⚠️ structural decline")
        if enhanced_adjustments.post_earnings_decline:
            parts.append("⚠️ post-earnings deterioration")

    # Quality/stability flags (only if not already covered by enhanced analysis)
    if fundamental_warning and not (enhanced_adjustments and enhanced_adjustments.structural_decline):
        if quant_context and quant_context.event_risk:
            parts.append("fundamentals weak (earnings noise)")
        else:
            parts.append("fundamentals deteriorating")
    elif not fundamental_warning:
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
    quant_context: QuantContext | None = None,
    volumes: np.ndarray | None = None,
    enhanced_inputs: EnhancedAnalysisInputs | None = None,
    min_dip_threshold: float | None = None,
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
        quant_context: Quant engine context for score adjustments
        volumes: Optional volume data for volume confirmation
        enhanced_inputs: Optional enhanced analysis inputs for score adjustments
        min_dip_threshold: Per-symbol minimum dip threshold (overrides config.min_dip_abs)

    Returns:
        Complete DipSignal
    """
    if config is None:
        config = get_dipfinder_config()

    # Use per-symbol threshold if provided, otherwise fall back to global config
    effective_min_dip = min_dip_threshold if min_dip_threshold is not None else config.min_dip_abs

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

    # Compute dip metrics (with volume data for confirmation)
    dip_metrics = compute_dip_metrics(ticker, stock_prices, window, config, volumes=volumes)

    # Compute market context
    market_context = compute_market_context(
        ticker, stock_prices, benchmark_prices, benchmark_ticker, window, config
    )

    # Compute dip score
    dip_score = compute_dip_score(dip_metrics, market_context, config, min_dip_threshold=effective_min_dip)

    # Compute final score (weighted combination)
    # GATE: If dip is below minimum threshold, cap final score low
    # This prevents high-quality stocks with minimal dips from appearing as opportunities
    if dip_metrics.dip_pct < effective_min_dip:
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

    # Apply enhanced analysis adjustments FIRST (only for meaningful dips)
    # This allows us to check for structural decline before applying quant fund_mom penalty
    enhanced_adjustments: EnhancedScoreAdjustments | None = None
    structural_decline_penalized = False
    if dip_metrics.dip_pct >= effective_min_dip:
        enhanced_adjustments = compute_enhanced_adjustments(dip_metrics, enhanced_inputs)
        final_score += enhanced_adjustments.total_adjustment
        final_score = min(100.0, max(0.0, final_score))
        structural_decline_penalized = enhanced_adjustments.structural_decline

    # Detect downtrend regime:
    # Stock is in downtrend if current price is below 200-day SMA
    # and recent momentum is negative (price trending lower)
    in_downtrend = False
    if len(stock_prices) >= 200:
        sma_200 = np.mean(stock_prices[-200:])
        current_price = float(stock_prices[-1])
        in_downtrend = current_price < sma_200
        
        # Also check short-term momentum (last 20 days)
        if len(stock_prices) >= 20:
            sma_20 = np.mean(stock_prices[-20:])
            # Confirm downtrend: below SMA200 AND short-term trend is down
            in_downtrend = in_downtrend and current_price < sma_20

    # Apply quant adjustments AFTER enhanced (to avoid double-penalizing fund deterioration)
    fundamental_warning = False
    if quant_context and dip_metrics.dip_pct >= effective_min_dip:
        final_score, fundamental_warning = _apply_quant_adjustments(
            final_score, quant_context,
            structural_decline_already_penalized=structural_decline_penalized,
            in_downtrend=in_downtrend,
        )
    
    # Also set fundamental warning from structural decline
    if structural_decline_penalized:
        fundamental_warning = True

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
        quant_context=quant_context,
        fundamental_warning=fundamental_warning,
        enhanced_adjustments=enhanced_adjustments,
    )

    signal = DipSignal(
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
    
    # Attach enhanced analysis details for API response
    if enhanced_adjustments is not None:
        signal.enhanced_analysis = {
            "score_adjustments": enhanced_adjustments.to_dict(),
        }
        # Add component details if available
        if enhanced_inputs is not None:
            if enhanced_inputs.structural_momentum is not None:
                signal.enhanced_analysis["structural"] = enhanced_inputs.structural_momentum.to_dict()
            if enhanced_inputs.support_resistance is not None:
                signal.enhanced_analysis["support_resistance"] = enhanced_inputs.support_resistance.to_dict()
            if enhanced_inputs.sector_valuation is not None:
                signal.enhanced_analysis["valuation"] = enhanced_inputs.sector_valuation.to_dict()
            if enhanced_inputs.earnings_info is not None:
                signal.enhanced_analysis["earnings"] = enhanced_inputs.earnings_info.to_dict()
    
    return signal
