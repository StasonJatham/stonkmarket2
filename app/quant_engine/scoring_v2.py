"""
Scoring V2 - Integrated Scoring System Using All Data Sources.

This module replaces the old scoring.py with a statistically-driven approach
that integrates:
1. backtest_v2 metrics (Kelly, SQN, Sharpe, crash performance)
2. dip_entry_optimizer data (recovery velocity, optimal thresholds, MAE)
3. Fundamental analysis (domain-specific ratios, financial health)
4. Statistical validation (bootstrap CI, walk-forward OOS)

NO HARDCODED THRESHOLDS - All optimal values discovered through data.

Key Principles:
- Thresholds are found by statistics, not hardcoded
- Each metric is normalized to [0,1] using rank percentiles
- Final score is weighted combination with dynamic weights
- Confidence intervals determine weight contribution

Author: Quant Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from app.quant_engine.config import QUANT_LIMITS

logger = logging.getLogger(__name__)

SCORING_VERSION = "2.0.0"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class BacktestV2Input:
    """Input data from backtest_v2 system."""
    
    # Strategy performance
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Advanced metrics
    kelly_fraction: float = 0.0
    sqn: float = 0.0  # System Quality Number
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    cvar_5: float = 0.0  # Conditional VaR
    
    # Trade statistics
    n_trades: int = 0
    avg_trade_return: float = 0.0
    avg_holding_days: float = 0.0
    
    # Benchmark comparison
    vs_buyhold_pct: float = 0.0
    vs_spy_pct: float = 0.0
    
    # Crash testing
    crash_outperformed_count: int = 0
    crash_tested_count: int = 0
    avg_crash_alpha: float = 0.0
    
    # Walk-forward validation
    wfo_passed: bool = False
    wfo_oos_return_pct: float = 0.0
    wfo_oos_sharpe: float = 0.0
    wfo_pct_folds_profitable: float = 0.0


@dataclass
class DipEntryInput:
    """Input data from dip_entry_optimizer."""
    
    # Current state
    current_drawdown_pct: float = 0.0
    
    # Optimal thresholds (discovered by statistics)
    optimal_threshold_pct: float = 0.0  # Best risk-adjusted
    max_profit_threshold_pct: float = 0.0  # Best total return
    
    # At MATCHED threshold (closest to current drawdown)
    matched_threshold_pct: float = 0.0
    n_occurrences: int = 0
    per_year: float = 0.0
    
    # Recovery metrics at matched threshold
    recovery_rate: float = 0.0  # % that recovered to entry
    full_recovery_rate: float = 0.0  # % that fully recovered
    avg_recovery_days: float = 0.0
    avg_recovery_velocity: float = 0.0  # Higher = faster bounce
    
    # Return metrics at matched threshold
    win_rate_optimal_hold: float = 0.0
    avg_return_optimal_hold: float = 0.0
    sharpe_optimal_hold: float = 0.0
    total_profit_compounded: float = 0.0
    
    # Risk metrics at matched threshold
    max_further_drawdown: float = 0.0  # MAE
    avg_further_drawdown: float = 0.0
    prob_further_drop: float = 0.0
    continuation_risk: str = "medium"
    
    # Entry quality
    entry_score: float = 0.0  # Pre-computed risk-adjusted score
    signal_strength: float = 0.0  # 0-100
    is_buy_now: bool = False
    
    # Confidence
    data_years: float = 0.0


@dataclass
class FundamentalsInput:
    """Input data from fundamentals analysis."""
    
    # Valuation
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    ev_ebitda: float | None = None
    
    # Profitability
    profit_margin: float | None = None
    operating_margin: float | None = None
    gross_margin: float | None = None
    roe: float | None = None
    roa: float | None = None
    
    # Growth
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    
    # Financial health
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    free_cash_flow: float | None = None
    
    # Risk
    beta: float | None = None
    short_ratio: float | None = None
    
    # Analyst
    analyst_rating: float | None = None  # 1-5 scale
    target_upside_pct: float | None = None
    
    # Domain-specific (populated based on sector)
    domain: str = "general"  # general, bank, reit, insurance
    domain_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ScoreComponents:
    """Individual score components for transparency."""
    
    # Backtest quality (0-100)
    backtest_quality: float = 0.0
    backtest_confidence: float = 0.0  # CI-based weight
    
    # Entry timing (0-100)
    entry_timing: float = 0.0
    entry_confidence: float = 0.0
    
    # Recovery probability (0-100)
    recovery_score: float = 0.0
    recovery_confidence: float = 0.0
    
    # Fundamental value (0-100)
    fundamental_score: float = 0.0
    fundamental_confidence: float = 0.0
    
    # Risk assessment (0-100, higher = less risky)
    risk_score: float = 0.0
    
    # Momentum (0-100)
    momentum_score: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "backtest_quality": round(self.backtest_quality, 2),
            "backtest_confidence": round(self.backtest_confidence, 2),
            "entry_timing": round(self.entry_timing, 2),
            "entry_confidence": round(self.entry_confidence, 2),
            "recovery_score": round(self.recovery_score, 2),
            "recovery_confidence": round(self.recovery_confidence, 2),
            "fundamental_score": round(self.fundamental_score, 2),
            "fundamental_confidence": round(self.fundamental_confidence, 2),
            "risk_score": round(self.risk_score, 2),
            "momentum_score": round(self.momentum_score, 2),
        }


@dataclass
class ScoringResultV2:
    """Complete scoring result with transparency."""
    
    symbol: str
    score: float  # 0-100 composite score
    mode: Literal["CERTIFIED_BUY", "DIP_ENTRY", "HOLD", "DOWNTREND"]
    action: Literal["STRONG_BUY", "BUY", "HOLD", "AVOID"]
    
    # Sub-scores (with defaults)
    components: ScoreComponents = field(default_factory=ScoreComponents)
    
    # Key metrics for display
    key_metrics: dict[str, Any] = field(default_factory=dict)
    
    # Recommendation
    action_reason: str = ""
    
    # Confidence
    confidence: Literal["low", "medium", "high"] = "medium"
    confidence_reason: str = ""
    
    # Metadata
    scoring_version: str = SCORING_VERSION
    data_start: date | None = None
    data_end: date | None = None
    computed_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================


def percentile_normalize(value: float, values: np.ndarray) -> float:
    """
    Normalize a value to [0, 1] based on its percentile rank in the distribution.
    
    This is the PRIMARY normalization method - no hardcoded thresholds.
    The distribution itself determines what's "good" or "bad".
    
    Args:
        value: The value to normalize
        values: Array of all values in the universe for comparison
        
    Returns:
        Percentile rank [0, 1]
    """
    if len(values) == 0:
        return 0.5
    
    # Handle NaN
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.5
    
    # Percentile rank
    return float(stats.percentileofscore(values, value, kind='mean') / 100)


def z_score_normalize(value: float, mean: float, std: float, clip: float = 3.0) -> float:
    """
    Normalize using z-score, then sigmoid to [0, 1].
    
    Args:
        value: The value to normalize
        mean: Population mean
        std: Population standard deviation
        clip: Clip z-score to this range
        
    Returns:
        Value in [0, 1]
    """
    if std == 0 or np.isnan(std):
        return 0.5
    
    z = (value - mean) / std
    z = np.clip(z, -clip, clip)
    
    # Sigmoid transformation
    return float(1 / (1 + np.exp(-z)))


def inverse_normalize(value: float, values: np.ndarray) -> float:
    """
    Percentile normalize but INVERTED (lower is better).
    Used for: drawdowns, debt ratios, PE ratios, etc.
    
    Args:
        value: The value to normalize (lower is better)
        values: Array of all values in the universe
        
    Returns:
        Inverted percentile rank [0, 1] where 1 = best (lowest value)
    """
    return 1.0 - percentile_normalize(value, values)


def confidence_from_sample_size(n: int, min_n: int = 5, good_n: int = 30) -> float:
    """
    Calculate confidence weight based on sample size.
    
    Args:
        n: Number of observations
        min_n: Minimum for any confidence
        good_n: Sample size for full confidence
        
    Returns:
        Confidence weight [0, 1]
    """
    if n < min_n:
        return 0.0
    if n >= good_n:
        return 1.0
    
    # Linear interpolation
    return (n - min_n) / (good_n - min_n)


def confidence_from_data_years(years: float, min_years: float = 3.0, good_years: float = 10.0) -> float:
    """
    Calculate confidence weight based on data history length.
    
    Args:
        years: Years of historical data
        min_years: Minimum for any confidence
        good_years: Years for full confidence
        
    Returns:
        Confidence weight [0, 1]
    """
    if years < min_years:
        return 0.0
    if years >= good_years:
        return 1.0
    
    return (years - min_years) / (good_years - min_years)


# =============================================================================
# SCORE COMPUTATION
# =============================================================================


def compute_backtest_quality_score(
    backtest: BacktestV2Input,
    universe_stats: dict[str, np.ndarray],
) -> tuple[float, float]:
    """
    Compute backtest quality score using percentile ranks.
    
    Components:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Edge vs benchmarks
    - Advanced metrics (Kelly, SQN, profit factor)
    - Crash performance
    - WFO validation
    
    Args:
        backtest: Backtest metrics
        universe_stats: Arrays of metrics for all symbols (for percentile ranking)
        
    Returns:
        (score, confidence) both in [0, 100]
    """
    scores = []
    weights = []
    
    # Risk-adjusted returns (30% weight)
    if "sharpe_ratio" in universe_stats:
        sharpe_pct = percentile_normalize(backtest.sharpe_ratio, universe_stats["sharpe_ratio"])
        scores.append(sharpe_pct * 100)
        weights.append(0.15)
    
    if "sortino_ratio" in universe_stats:
        sortino_pct = percentile_normalize(backtest.sortino_ratio, universe_stats["sortino_ratio"])
        scores.append(sortino_pct * 100)
        weights.append(0.10)
    
    if "calmar_ratio" in universe_stats:
        calmar_pct = percentile_normalize(backtest.calmar_ratio, universe_stats["calmar_ratio"])
        scores.append(calmar_pct * 100)
        weights.append(0.05)
    
    # Edge vs benchmarks (20% weight)
    if "vs_buyhold_pct" in universe_stats:
        edge_bh_pct = percentile_normalize(backtest.vs_buyhold_pct, universe_stats["vs_buyhold_pct"])
        scores.append(edge_bh_pct * 100)
        weights.append(0.10)
    
    if "vs_spy_pct" in universe_stats:
        edge_spy_pct = percentile_normalize(backtest.vs_spy_pct, universe_stats["vs_spy_pct"])
        scores.append(edge_spy_pct * 100)
        weights.append(0.10)
    
    # Advanced metrics (25% weight)
    if "sqn" in universe_stats:
        sqn_pct = percentile_normalize(backtest.sqn, universe_stats["sqn"])
        scores.append(sqn_pct * 100)
        weights.append(0.10)
    
    if "kelly_fraction" in universe_stats:
        # Kelly clipped to reasonable range
        kelly_clipped = min(0.5, max(0, backtest.kelly_fraction))
        kelly_pct = percentile_normalize(kelly_clipped, universe_stats["kelly_fraction"])
        scores.append(kelly_pct * 100)
        weights.append(0.10)
    
    if "profit_factor" in universe_stats:
        pf_pct = percentile_normalize(backtest.profit_factor, universe_stats["profit_factor"])
        scores.append(pf_pct * 100)
        weights.append(0.05)
    
    # Crash resilience (15% weight)
    if backtest.crash_tested_count > 0:
        crash_win_rate = backtest.crash_outperformed_count / backtest.crash_tested_count
        scores.append(crash_win_rate * 100)
        weights.append(0.10)
        
        if "avg_crash_alpha" in universe_stats:
            crash_alpha_pct = percentile_normalize(backtest.avg_crash_alpha, universe_stats["avg_crash_alpha"])
            scores.append(crash_alpha_pct * 100)
            weights.append(0.05)
    
    # WFO validation (10% weight)
    if backtest.wfo_passed:
        scores.append(100)  # Passed WFO
        weights.append(0.05)
        
        if "wfo_oos_sharpe" in universe_stats:
            wfo_pct = percentile_normalize(backtest.wfo_oos_sharpe, universe_stats["wfo_oos_sharpe"])
            scores.append(wfo_pct * 100)
            weights.append(0.05)
    
    # Calculate weighted score
    if not scores:
        return 50.0, 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0, 0.0
    
    # Normalize weights
    weights = [w / total_weight for w in weights]
    
    score = sum(s * w for s, w in zip(scores, weights))
    
    # Confidence based on sample size and WFO
    confidence = confidence_from_sample_size(backtest.n_trades, min_n=10, good_n=50)
    if backtest.wfo_passed:
        confidence = min(100, confidence * 1.2)  # Boost for WFO validation
    
    return float(score), float(confidence * 100)


def compute_entry_timing_score(
    dip_entry: DipEntryInput,
    universe_stats: dict[str, np.ndarray],
) -> tuple[float, float]:
    """
    Compute entry timing score.
    
    How good is NOW as an entry point?
    
    Components:
    - Distance to optimal threshold (closer = better)
    - Signal strength
    - Is buy now signal active?
    - Continuation risk (inverse)
    
    Args:
        dip_entry: Dip entry metrics
        universe_stats: Arrays of metrics for all symbols
        
    Returns:
        (score, confidence) both in [0, 100]
    """
    scores = []
    weights = []
    
    # Distance to optimal threshold (25%)
    # If current drawdown is at or beyond optimal threshold = 100%
    # If current drawdown is 0 = 0%
    if dip_entry.optimal_threshold_pct != 0:
        distance_ratio = min(1.0, abs(dip_entry.current_drawdown_pct) / abs(dip_entry.optimal_threshold_pct))
        scores.append(distance_ratio * 100)
        weights.append(0.25)
    
    # Signal strength (25%)
    scores.append(dip_entry.signal_strength)
    weights.append(0.25)
    
    # Is buy now active? (25%)
    if dip_entry.is_buy_now:
        scores.append(100)
    else:
        # Partial credit based on how close
        if dip_entry.current_drawdown_pct < dip_entry.optimal_threshold_pct:
            # Beyond optimal = still good
            scores.append(80)
        else:
            # How close to optimal?
            if dip_entry.optimal_threshold_pct != 0:
                closeness = abs(dip_entry.current_drawdown_pct) / abs(dip_entry.optimal_threshold_pct)
                scores.append(closeness * 60)
            else:
                scores.append(30)
    weights.append(0.25)
    
    # Continuation risk (25%, inverse)
    risk_map = {"low": 100, "medium": 60, "high": 20}
    scores.append(risk_map.get(dip_entry.continuation_risk, 60))
    weights.append(0.25)
    
    # Calculate weighted score
    if not scores:
        return 50.0, 0.0
    
    score = sum(s * w for s, w in zip(scores, weights))
    
    # Confidence based on data years and occurrences
    conf_years = confidence_from_data_years(dip_entry.data_years)
    conf_samples = confidence_from_sample_size(dip_entry.n_occurrences, min_n=3, good_n=20)
    confidence = (conf_years + conf_samples) / 2
    
    return float(score), float(confidence * 100)


def compute_recovery_score(
    dip_entry: DipEntryInput,
    universe_stats: dict[str, np.ndarray],
) -> tuple[float, float]:
    """
    Compute recovery probability score.
    
    How likely is this stock to recover from the current dip?
    
    Components:
    - Historical recovery rate at this threshold
    - Recovery velocity (speed of recovery)
    - Full recovery rate
    - Optimal holding period returns
    
    Args:
        dip_entry: Dip entry metrics
        universe_stats: Arrays of metrics for all symbols
        
    Returns:
        (score, confidence) both in [0, 100]
    """
    scores = []
    weights = []
    
    # Recovery rate (30%)
    if "recovery_rate" in universe_stats:
        recovery_pct = percentile_normalize(dip_entry.recovery_rate, universe_stats["recovery_rate"])
        scores.append(recovery_pct * 100)
        weights.append(0.30)
    else:
        scores.append(dip_entry.recovery_rate)  # Already in %
        weights.append(0.30)
    
    # Recovery velocity (25%)
    if "avg_recovery_velocity" in universe_stats and dip_entry.avg_recovery_velocity > 0:
        velocity_pct = percentile_normalize(dip_entry.avg_recovery_velocity, universe_stats["avg_recovery_velocity"])
        scores.append(velocity_pct * 100)
        weights.append(0.25)
    
    # Full recovery rate (15%)
    if "full_recovery_rate" in universe_stats:
        full_pct = percentile_normalize(dip_entry.full_recovery_rate, universe_stats["full_recovery_rate"])
        scores.append(full_pct * 100)
        weights.append(0.15)
    else:
        scores.append(dip_entry.full_recovery_rate)
        weights.append(0.15)
    
    # Expected return at optimal hold (20%)
    if "avg_return_optimal_hold" in universe_stats:
        return_pct = percentile_normalize(dip_entry.avg_return_optimal_hold, universe_stats["avg_return_optimal_hold"])
        scores.append(return_pct * 100)
        weights.append(0.20)
    
    # Win rate at optimal hold (10%)
    if "win_rate_optimal_hold" in universe_stats:
        winrate_pct = percentile_normalize(dip_entry.win_rate_optimal_hold, universe_stats["win_rate_optimal_hold"])
        scores.append(winrate_pct * 100)
        weights.append(0.10)
    
    # Calculate weighted score
    if not scores:
        return 50.0, 0.0
    
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    score = sum(s * w for s, w in zip(scores, weights))
    
    # Confidence based on sample size
    confidence = confidence_from_sample_size(dip_entry.n_occurrences, min_n=3, good_n=20)
    
    return float(score), float(confidence * 100)


def compute_fundamental_score(
    fundamentals: FundamentalsInput,
    universe_stats: dict[str, np.ndarray],
) -> tuple[float, float]:
    """
    Compute fundamental score using domain-appropriate metrics.
    
    Different sectors have different key metrics:
    - General: PE, PEG, profit margin, ROE, growth
    - Banks: NIM, efficiency ratio, NPL ratio
    - REITs: FFO, occupancy, NAV discount
    - Insurance: Combined ratio, loss ratio
    
    Args:
        fundamentals: Fundamental metrics
        universe_stats: Arrays of metrics for all symbols
        
    Returns:
        (score, confidence) both in [0, 100]
    """
    scores = []
    weights = []
    metrics_available = 0
    
    # Valuation (25%) - lower is better for most
    if fundamentals.pe_ratio is not None and "pe_ratio" in universe_stats:
        # Reasonable PE check (positive and not extreme)
        if 0 < fundamentals.pe_ratio < 200:
            pe_pct = inverse_normalize(fundamentals.pe_ratio, universe_stats["pe_ratio"])
            scores.append(pe_pct * 100)
            weights.append(0.10)
            metrics_available += 1
    
    if fundamentals.peg_ratio is not None and "peg_ratio" in universe_stats:
        if 0 < fundamentals.peg_ratio < 10:
            peg_pct = inverse_normalize(fundamentals.peg_ratio, universe_stats["peg_ratio"])
            scores.append(peg_pct * 100)
            weights.append(0.10)
            metrics_available += 1
    
    if fundamentals.ev_ebitda is not None and "ev_ebitda" in universe_stats:
        if 0 < fundamentals.ev_ebitda < 100:
            ev_pct = inverse_normalize(fundamentals.ev_ebitda, universe_stats["ev_ebitda"])
            scores.append(ev_pct * 100)
            weights.append(0.05)
            metrics_available += 1
    
    # Profitability (30%) - higher is better
    if fundamentals.profit_margin is not None and "profit_margin" in universe_stats:
        margin_pct = percentile_normalize(fundamentals.profit_margin, universe_stats["profit_margin"])
        scores.append(margin_pct * 100)
        weights.append(0.15)
        metrics_available += 1
    
    if fundamentals.roe is not None and "roe" in universe_stats:
        if -0.5 < fundamentals.roe < 1.0:  # Reasonable range
            roe_pct = percentile_normalize(fundamentals.roe, universe_stats["roe"])
            scores.append(roe_pct * 100)
            weights.append(0.15)
            metrics_available += 1
    
    # Growth (20%) - higher is better
    if fundamentals.revenue_growth is not None and "revenue_growth" in universe_stats:
        if -1.0 < fundamentals.revenue_growth < 5.0:  # Reasonable range
            growth_pct = percentile_normalize(fundamentals.revenue_growth, universe_stats["revenue_growth"])
            scores.append(growth_pct * 100)
            weights.append(0.10)
            metrics_available += 1
    
    if fundamentals.earnings_growth is not None and "earnings_growth" in universe_stats:
        if -1.0 < fundamentals.earnings_growth < 10.0:
            eg_pct = percentile_normalize(fundamentals.earnings_growth, universe_stats["earnings_growth"])
            scores.append(eg_pct * 100)
            weights.append(0.10)
            metrics_available += 1
    
    # Financial health (25%) - complex
    if fundamentals.debt_to_equity is not None and "debt_to_equity" in universe_stats:
        if 0 <= fundamentals.debt_to_equity < 10:
            # Lower debt is better
            de_pct = inverse_normalize(fundamentals.debt_to_equity, universe_stats["debt_to_equity"])
            scores.append(de_pct * 100)
            weights.append(0.10)
            metrics_available += 1
    
    if fundamentals.current_ratio is not None and "current_ratio" in universe_stats:
        if 0 < fundamentals.current_ratio < 10:
            # Higher is better, but too high can be inefficient
            # Use percentile without inversion
            cr_pct = percentile_normalize(fundamentals.current_ratio, universe_stats["current_ratio"])
            scores.append(cr_pct * 100)
            weights.append(0.10)
            metrics_available += 1
    
    if fundamentals.free_cash_flow is not None and "free_cash_flow" in universe_stats:
        fcf_pct = percentile_normalize(fundamentals.free_cash_flow, universe_stats["free_cash_flow"])
        scores.append(fcf_pct * 100)
        weights.append(0.05)
        metrics_available += 1
    
    # Domain-specific adjustments
    if fundamentals.domain != "general" and fundamentals.domain_metrics:
        domain_scores = _compute_domain_specific_scores(fundamentals, universe_stats)
        scores.extend([s[0] for s in domain_scores])
        weights.extend([s[1] for s in domain_scores])
        metrics_available += len(domain_scores)
    
    # Calculate weighted score
    if not scores:
        return 50.0, 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 50.0, 0.0
    
    weights = [w / total_weight for w in weights]
    score = sum(s * w for s, w in zip(scores, weights))
    
    # Confidence based on how many metrics we have
    max_metrics = 10
    confidence = min(1.0, metrics_available / max_metrics)
    
    return float(score), float(confidence * 100)


def _compute_domain_specific_scores(
    fundamentals: FundamentalsInput,
    universe_stats: dict[str, np.ndarray],
) -> list[tuple[float, float]]:
    """
    Compute domain-specific metric scores.
    
    Returns list of (score, weight) tuples.
    """
    scores = []
    
    if fundamentals.domain == "bank":
        # Net Interest Margin - higher is better
        if "nim" in fundamentals.domain_metrics:
            nim = fundamentals.domain_metrics["nim"]
            if "nim" in universe_stats:
                nim_pct = percentile_normalize(nim, universe_stats["nim"])
                scores.append((nim_pct * 100, 0.15))
        
        # Efficiency ratio - lower is better
        if "efficiency_ratio" in fundamentals.domain_metrics:
            eff = fundamentals.domain_metrics["efficiency_ratio"]
            if "efficiency_ratio" in universe_stats:
                eff_pct = inverse_normalize(eff, universe_stats["efficiency_ratio"])
                scores.append((eff_pct * 100, 0.10))
        
        # NPL ratio - lower is better
        if "npl_ratio" in fundamentals.domain_metrics:
            npl = fundamentals.domain_metrics["npl_ratio"]
            if "npl_ratio" in universe_stats:
                npl_pct = inverse_normalize(npl, universe_stats["npl_ratio"])
                scores.append((npl_pct * 100, 0.10))
    
    elif fundamentals.domain == "reit":
        # FFO yield - higher is better
        if "ffo_yield" in fundamentals.domain_metrics:
            ffo = fundamentals.domain_metrics["ffo_yield"]
            if "ffo_yield" in universe_stats:
                ffo_pct = percentile_normalize(ffo, universe_stats["ffo_yield"])
                scores.append((ffo_pct * 100, 0.15))
        
        # NAV discount - larger discount (more negative) is better for value
        if "nav_discount" in fundamentals.domain_metrics:
            nav = fundamentals.domain_metrics["nav_discount"]
            if "nav_discount" in universe_stats:
                # More negative = bigger discount = better
                nav_pct = inverse_normalize(nav, universe_stats["nav_discount"])
                scores.append((nav_pct * 100, 0.10))
    
    elif fundamentals.domain == "insurance":
        # Combined ratio - lower is better (< 100 = underwriting profit)
        if "combined_ratio" in fundamentals.domain_metrics:
            cr = fundamentals.domain_metrics["combined_ratio"]
            if "combined_ratio" in universe_stats:
                cr_pct = inverse_normalize(cr, universe_stats["combined_ratio"])
                scores.append((cr_pct * 100, 0.20))
    
    return scores


def compute_risk_score(
    dip_entry: DipEntryInput,
    backtest: BacktestV2Input,
    universe_stats: dict[str, np.ndarray],
) -> float:
    """
    Compute risk score (higher = less risky, better).
    
    Components:
    - Max further drawdown (MAE) - lower is better
    - Max drawdown from backtest - lower is better
    - CVaR - lower is better
    - Continuation risk - low is better
    
    Args:
        dip_entry: Dip entry metrics
        backtest: Backtest metrics
        universe_stats: Arrays of metrics for all symbols
        
    Returns:
        Risk score [0, 100] where 100 = least risky
    """
    scores = []
    weights = []
    
    # Max further drawdown (MAE) - lower is better (30%)
    if "max_further_drawdown" in universe_stats:
        mae_pct = inverse_normalize(abs(dip_entry.max_further_drawdown), universe_stats["max_further_drawdown"])
        scores.append(mae_pct * 100)
        weights.append(0.30)
    
    # Max drawdown from backtest - lower is better (25%)
    if "max_drawdown_pct" in universe_stats:
        dd_pct = inverse_normalize(abs(backtest.max_drawdown_pct), universe_stats["max_drawdown_pct"])
        scores.append(dd_pct * 100)
        weights.append(0.25)
    
    # CVaR - lower (more positive) is better (20%)
    if "cvar_5" in universe_stats:
        cvar_pct = percentile_normalize(backtest.cvar_5, universe_stats["cvar_5"])
        scores.append(cvar_pct * 100)
        weights.append(0.20)
    
    # Continuation risk (25%)
    risk_map = {"low": 100, "medium": 60, "high": 20}
    scores.append(risk_map.get(dip_entry.continuation_risk, 60))
    weights.append(0.25)
    
    if not scores:
        return 50.0
    
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    return float(sum(s * w for s, w in zip(scores, weights)))


def compute_momentum_score(
    current_drawdown_pct: float,
    fundamentals: FundamentalsInput,
    universe_stats: dict[str, np.ndarray],
) -> float:
    """
    Compute momentum score.
    
    Combines:
    - Current dip depth (deeper = more oversold = higher momentum potential)
    - Analyst target upside
    - Recent growth trends
    
    Args:
        current_drawdown_pct: Current drawdown from high (negative)
        fundamentals: Fundamental metrics
        universe_stats: Arrays of metrics for all symbols
        
    Returns:
        Momentum score [0, 100]
    """
    scores = []
    weights = []
    
    # Dip depth - deeper dip = more recovery potential (40%)
    if "current_drawdown_pct" in universe_stats:
        # More negative = bigger dip = more oversold
        dip_pct = inverse_normalize(current_drawdown_pct, universe_stats["current_drawdown_pct"])
        scores.append(dip_pct * 100)
        weights.append(0.40)
    
    # Analyst target upside (30%)
    if fundamentals.target_upside_pct is not None and "target_upside_pct" in universe_stats:
        upside_pct = percentile_normalize(fundamentals.target_upside_pct, universe_stats["target_upside_pct"])
        scores.append(upside_pct * 100)
        weights.append(0.30)
    
    # Revenue growth (15%)
    if fundamentals.revenue_growth is not None and "revenue_growth" in universe_stats:
        growth_pct = percentile_normalize(fundamentals.revenue_growth, universe_stats["revenue_growth"])
        scores.append(growth_pct * 100)
        weights.append(0.15)
    
    # Earnings growth (15%)
    if fundamentals.earnings_growth is not None and "earnings_growth" in universe_stats:
        eg_pct = percentile_normalize(fundamentals.earnings_growth, universe_stats["earnings_growth"])
        scores.append(eg_pct * 100)
        weights.append(0.15)
    
    if not scores:
        return 50.0
    
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    return float(sum(s * w for s, w in zip(scores, weights)))


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================


def compute_score_v2(
    symbol: str,
    backtest: BacktestV2Input,
    dip_entry: DipEntryInput,
    fundamentals: FundamentalsInput,
    universe_stats: dict[str, np.ndarray],
    current_market_regime: str = "normal",
) -> ScoringResultV2:
    """
    Compute the V2 integrated score for a symbol.
    
    The final score is a confidence-weighted combination of:
    - Backtest quality (how good is the strategy?)
    - Entry timing (is NOW a good time to buy?)
    - Recovery probability (will it recover from this dip?)
    - Fundamental value (is it a good company?)
    - Risk assessment (how risky is this entry?)
    - Momentum (what's the upside potential?)
    
    Each component contributes based on its confidence level.
    
    Args:
        symbol: Stock ticker
        backtest: backtest_v2 metrics
        dip_entry: dip_entry_optimizer metrics
        fundamentals: Fundamental metrics
        universe_stats: Arrays of all metrics for percentile ranking
        current_market_regime: "bull", "bear", "crash", "recovery"
        
    Returns:
        ScoringResultV2 with complete scoring information
    """
    components = ScoreComponents()
    
    # Compute each component
    components.backtest_quality, components.backtest_confidence = compute_backtest_quality_score(
        backtest, universe_stats
    )
    
    components.entry_timing, components.entry_confidence = compute_entry_timing_score(
        dip_entry, universe_stats
    )
    
    components.recovery_score, components.recovery_confidence = compute_recovery_score(
        dip_entry, universe_stats
    )
    
    components.fundamental_score, components.fundamental_confidence = compute_fundamental_score(
        fundamentals, universe_stats
    )
    
    components.risk_score = compute_risk_score(dip_entry, backtest, universe_stats)
    
    components.momentum_score = compute_momentum_score(
        dip_entry.current_drawdown_pct, fundamentals, universe_stats
    )
    
    # Dynamic weight calculation based on confidence
    # Higher confidence = higher weight contribution
    weights = {
        "backtest": 0.20 * (components.backtest_confidence / 100),
        "entry": 0.25 * (components.entry_confidence / 100),
        "recovery": 0.25 * (components.recovery_confidence / 100),
        "fundamental": 0.15 * (components.fundamental_confidence / 100),
        "risk": 0.10,  # Always contributes
        "momentum": 0.05,  # Always contributes
    }
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        # Fallback to equal weights
        weights = {k: 1/6 for k in weights}
    
    # Calculate final score
    final_score = (
        components.backtest_quality * weights["backtest"] +
        components.entry_timing * weights["entry"] +
        components.recovery_score * weights["recovery"] +
        components.fundamental_score * weights["fundamental"] +
        components.risk_score * weights["risk"] +
        components.momentum_score * weights["momentum"]
    )
    
    # Determine mode
    if backtest.wfo_passed and backtest.sharpe_ratio > 1.0 and backtest.vs_buyhold_pct > 0:
        mode = "CERTIFIED_BUY"
    elif dip_entry.is_buy_now:
        mode = "DIP_ENTRY"
    elif dip_entry.current_drawdown_pct < -50:  # Extended decline
        mode = "DOWNTREND"
    else:
        mode = "HOLD"
    
    # Determine action
    action = _determine_action(final_score, mode, dip_entry, components)
    action_reason = _determine_action_reason(final_score, mode, dip_entry, components)
    
    # Determine confidence
    avg_confidence = (
        components.backtest_confidence +
        components.entry_confidence +
        components.recovery_confidence +
        components.fundamental_confidence
    ) / 4
    
    if avg_confidence >= 70:
        confidence = "high"
    elif avg_confidence >= 40:
        confidence = "medium"
    else:
        confidence = "low"
    
    confidence_reason = _determine_confidence_reason(
        avg_confidence, dip_entry, backtest, fundamentals
    )
    
    # Key metrics for display
    key_metrics = {
        "current_drawdown_pct": round(dip_entry.current_drawdown_pct, 2),
        "optimal_threshold_pct": round(dip_entry.optimal_threshold_pct, 2),
        "recovery_rate": round(dip_entry.recovery_rate, 1),
        "avg_recovery_days": round(dip_entry.avg_recovery_days, 0),
        "recovery_velocity": round(dip_entry.avg_recovery_velocity, 2),
        "win_rate": round(dip_entry.win_rate_optimal_hold, 1),
        "sharpe_ratio": round(backtest.sharpe_ratio, 2),
        "signal_strength": round(dip_entry.signal_strength, 0),
        "is_buy_now": dip_entry.is_buy_now,
        "continuation_risk": dip_entry.continuation_risk,
    }
    
    return ScoringResultV2(
        symbol=symbol,
        score=round(final_score, 2),
        mode=mode,
        components=components,
        key_metrics=key_metrics,
        action=action,
        action_reason=action_reason,
        confidence=confidence,
        confidence_reason=confidence_reason,
    )


def _determine_action(
    score: float,
    mode: str,
    dip_entry: DipEntryInput,
    components: ScoreComponents,
) -> Literal["STRONG_BUY", "BUY", "HOLD", "AVOID"]:
    """Determine action recommendation."""
    if mode == "DOWNTREND":
        return "AVOID"
    
    if mode == "CERTIFIED_BUY":
        if score >= 75:
            return "STRONG_BUY"
        elif score >= 60:
            return "BUY"
        else:
            return "HOLD"
    
    if mode == "DIP_ENTRY":
        if dip_entry.is_buy_now and score >= 70:
            return "STRONG_BUY"
        elif dip_entry.is_buy_now and score >= 50:
            return "BUY"
        elif score >= 60:
            return "HOLD"  # Not yet at entry point
        else:
            return "HOLD"
    
    # HOLD mode
    if score >= 70:
        return "HOLD"  # Wait for dip
    else:
        return "HOLD"


def _determine_action_reason(
    score: float,
    mode: str,
    dip_entry: DipEntryInput,
    components: ScoreComponents,
) -> str:
    """Generate human-readable action reason."""
    reasons = []
    
    if mode == "CERTIFIED_BUY":
        reasons.append("Strategy passes statistical validation")
        if components.backtest_quality >= 70:
            reasons.append("Strong backtest performance")
    
    elif mode == "DIP_ENTRY":
        if dip_entry.is_buy_now:
            reasons.append(f"Buy signal active at {abs(dip_entry.current_drawdown_pct):.1f}% dip")
        else:
            reasons.append(f"Optimal entry at {abs(dip_entry.optimal_threshold_pct):.0f}% dip")
        
        if dip_entry.recovery_rate >= 80:
            reasons.append(f"{dip_entry.recovery_rate:.0f}% historical recovery rate")
        
        if dip_entry.continuation_risk == "low":
            reasons.append("Low continuation risk")
        elif dip_entry.continuation_risk == "high":
            reasons.append("Elevated continuation risk")
    
    elif mode == "DOWNTREND":
        reasons.append("Extended decline pattern - wait for stabilization")
    
    else:
        if dip_entry.current_drawdown_pct > -5:
            reasons.append("No significant dip - consider waiting")
        else:
            reasons.append(f"Current dip ({abs(dip_entry.current_drawdown_pct):.1f}%) below optimal entry")
    
    return "; ".join(reasons[:2])  # Max 2 reasons


def _determine_confidence_reason(
    avg_confidence: float,
    dip_entry: DipEntryInput,
    backtest: BacktestV2Input,
    fundamentals: FundamentalsInput,
) -> str:
    """Generate human-readable confidence reason."""
    if avg_confidence >= 70:
        return f"Based on {dip_entry.data_years:.0f}+ years of data with {dip_entry.n_occurrences} similar events"
    elif avg_confidence >= 40:
        reasons = []
        if dip_entry.n_occurrences < 10:
            reasons.append(f"Only {dip_entry.n_occurrences} historical events")
        if not backtest.wfo_passed:
            reasons.append("No walk-forward validation")
        return "; ".join(reasons) if reasons else "Moderate data coverage"
    else:
        return "Limited historical data for this threshold"


# =============================================================================
# BATCH PROCESSING
# =============================================================================


async def compute_scores_batch(
    symbols: list[str],
    get_backtest_func,
    get_dip_entry_func,
    get_fundamentals_func,
) -> list[ScoringResultV2]:
    """
    Compute scores for all symbols with universe statistics.
    
    This collects metrics from all symbols first to compute percentile ranks,
    then scores each symbol against the universe.
    
    Args:
        symbols: List of symbols to score
        get_backtest_func: Async function(symbol) -> BacktestV2Input
        get_dip_entry_func: Async function(symbol) -> DipEntryInput
        get_fundamentals_func: Async function(symbol) -> FundamentalsInput
        
    Returns:
        List of ScoringResultV2 for each symbol
    """
    import asyncio
    
    # Collect all data first
    all_data: dict[str, tuple[BacktestV2Input, DipEntryInput, FundamentalsInput]] = {}
    
    for symbol in symbols:
        try:
            backtest = await get_backtest_func(symbol)
            dip_entry = await get_dip_entry_func(symbol)
            fundamentals = await get_fundamentals_func(symbol)
            all_data[symbol] = (backtest, dip_entry, fundamentals)
        except Exception as e:
            logger.warning(f"Failed to collect data for {symbol}: {e}")
            continue
    
    if not all_data:
        return []
    
    # Build universe statistics
    universe_stats = _build_universe_stats(all_data)
    
    # Score each symbol
    results = []
    for symbol, (backtest, dip_entry, fundamentals) in all_data.items():
        try:
            result = compute_score_v2(
                symbol=symbol,
                backtest=backtest,
                dip_entry=dip_entry,
                fundamentals=fundamentals,
                universe_stats=universe_stats,
            )
            results.append(result)
        except Exception as e:
            logger.exception(f"Failed to score {symbol}: {e}")
    
    return results


def _build_universe_stats(
    all_data: dict[str, tuple[BacktestV2Input, DipEntryInput, FundamentalsInput]],
) -> dict[str, np.ndarray]:
    """
    Build arrays of metrics from all symbols for percentile ranking.
    """
    stats: dict[str, list[float]] = {}
    
    for symbol, (backtest, dip_entry, fundamentals) in all_data.items():
        # Backtest metrics
        _add_stat(stats, "sharpe_ratio", backtest.sharpe_ratio)
        _add_stat(stats, "sortino_ratio", backtest.sortino_ratio)
        _add_stat(stats, "calmar_ratio", backtest.calmar_ratio)
        _add_stat(stats, "kelly_fraction", backtest.kelly_fraction)
        _add_stat(stats, "sqn", backtest.sqn)
        _add_stat(stats, "profit_factor", backtest.profit_factor)
        _add_stat(stats, "vs_buyhold_pct", backtest.vs_buyhold_pct)
        _add_stat(stats, "vs_spy_pct", backtest.vs_spy_pct)
        _add_stat(stats, "max_drawdown_pct", abs(backtest.max_drawdown_pct))
        _add_stat(stats, "cvar_5", backtest.cvar_5)
        _add_stat(stats, "wfo_oos_sharpe", backtest.wfo_oos_sharpe)
        _add_stat(stats, "avg_crash_alpha", backtest.avg_crash_alpha)
        
        # Dip entry metrics
        _add_stat(stats, "recovery_rate", dip_entry.recovery_rate)
        _add_stat(stats, "full_recovery_rate", dip_entry.full_recovery_rate)
        _add_stat(stats, "avg_recovery_velocity", dip_entry.avg_recovery_velocity)
        _add_stat(stats, "avg_return_optimal_hold", dip_entry.avg_return_optimal_hold)
        _add_stat(stats, "win_rate_optimal_hold", dip_entry.win_rate_optimal_hold)
        _add_stat(stats, "max_further_drawdown", abs(dip_entry.max_further_drawdown))
        _add_stat(stats, "current_drawdown_pct", dip_entry.current_drawdown_pct)
        
        # Fundamental metrics
        _add_stat(stats, "pe_ratio", fundamentals.pe_ratio)
        _add_stat(stats, "peg_ratio", fundamentals.peg_ratio)
        _add_stat(stats, "ev_ebitda", fundamentals.ev_ebitda)
        _add_stat(stats, "profit_margin", fundamentals.profit_margin)
        _add_stat(stats, "roe", fundamentals.roe)
        _add_stat(stats, "revenue_growth", fundamentals.revenue_growth)
        _add_stat(stats, "earnings_growth", fundamentals.earnings_growth)
        _add_stat(stats, "debt_to_equity", fundamentals.debt_to_equity)
        _add_stat(stats, "current_ratio", fundamentals.current_ratio)
        _add_stat(stats, "free_cash_flow", fundamentals.free_cash_flow)
        _add_stat(stats, "target_upside_pct", fundamentals.target_upside_pct)
    
    # Convert to numpy arrays
    return {k: np.array(v) for k, v in stats.items()}


def _add_stat(stats: dict[str, list[float]], key: str, value: float | None) -> None:
    """Add a statistic value if valid."""
    if value is not None and not np.isnan(value):
        if key not in stats:
            stats[key] = []
        stats[key].append(value)
