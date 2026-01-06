"""Extreme Value Analysis (EVA) module for tail event detection.

Implements Peaks Over Threshold (POT) method with Generalized Pareto Distribution
(GPD) fitting for proper statistical handling of extreme dip events.

The key insight: extreme events like a 90% crash shouldn't define "normal" behavior.
Instead, we model the tail separately and calculate return periods to quantify
how rare an event truly is.

Example:
    - HOOD's 90% IPO crash → return_period = 50+ years (extraordinary)
    - HOOD's 22% current dip → regime_percentile = 75% (significant in normal regime)
    - Both are opportunities, but classified differently
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TailAnalysis:
    """Results from Extreme Value Analysis."""

    # Threshold detection
    tail_threshold: float  # Threshold separating normal from extreme (e.g., 0.50 = 50%)
    threshold_percentile: float  # What percentile the threshold represents (e.g., 95)

    # Current dip classification
    is_tail_event: bool  # Is current dip beyond the tail threshold?
    return_period_years: float | None  # How rare? (years between events of this magnitude)

    # Regime-aware percentile (excludes tail events)
    regime_dip_percentile: float  # Percentile within "normal" regime only
    n_tail_events: int  # Number of historical tail events detected
    n_normal_events: int  # Number of normal regime observations

    # GPD fit quality (if fitted)
    gpd_shape: float | None = None  # Shape parameter (xi) - negative = bounded tail
    gpd_scale: float | None = None  # Scale parameter (sigma)
    gpd_fit_success: bool = False  # Whether GPD fitting succeeded


def compute_tail_threshold(
    dip_series: np.ndarray,
    threshold_percentile: float = 95.0,
    min_threshold: float = 0.40,
    max_threshold: float = 0.70,
) -> tuple[float, float]:
    """
    Compute the threshold separating normal dips from tail events.

    Uses percentile-based threshold with bounds to ensure sensible values.
    The threshold defines where "extreme" begins.

    Args:
        dip_series: Historical dip values (positive = dip from peak)
        threshold_percentile: Percentile to use for threshold (default 95th)
        min_threshold: Minimum threshold (default 40% dip)
        max_threshold: Maximum threshold (default 70% dip)

    Returns:
        (threshold, actual_percentile) - the threshold value and its percentile
    """
    valid = dip_series[~np.isnan(dip_series)]

    # Only consider actual dips (> 5%)
    dips_only = valid[valid >= 0.05]

    if len(dips_only) < 20:
        # Not enough data - use minimum threshold
        return min_threshold, 95.0

    # Calculate percentile-based threshold
    raw_threshold = float(np.percentile(dips_only, threshold_percentile))

    # Clamp to sensible bounds
    threshold = np.clip(raw_threshold, min_threshold, max_threshold)

    # Calculate what percentile this threshold actually represents
    actual_percentile = float(np.sum(dips_only < threshold) / len(dips_only) * 100)

    return threshold, actual_percentile


def fit_gpd_to_exceedances(
    dip_series: np.ndarray,
    threshold: float,
    min_exceedances: int = 5,
) -> tuple[float | None, float | None, bool]:
    """
    Fit Generalized Pareto Distribution to exceedances over threshold.

    The GPD models the distribution of extreme values above a high threshold.
    This allows us to extrapolate and estimate return periods for rare events.

    Args:
        dip_series: Historical dip values
        threshold: Tail threshold
        min_exceedances: Minimum exceedances needed for fitting

    Returns:
        (shape, scale, success) - GPD parameters and fit success flag
    """
    valid = dip_series[~np.isnan(dip_series)]

    # Get exceedances (values above threshold)
    exceedances = valid[valid > threshold] - threshold

    if len(exceedances) < min_exceedances:
        return None, None, False

    try:
        # Fit GPD using MLE
        # genpareto.fit returns (shape, loc, scale) but we fix loc=0
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

        # Sanity checks
        if not np.isfinite(shape) or not np.isfinite(scale):
            return None, None, False

        if scale <= 0:
            return None, None, False

        # Shape should be reasonable for financial data
        # Typically -0.5 < shape < 0.5 for most assets
        if abs(shape) > 1.0:
            return None, None, False

        return float(shape), float(scale), True

    except Exception:
        return None, None, False


def compute_return_period(
    current_dip: float,
    threshold: float,
    gpd_shape: float | None,
    gpd_scale: float | None,
    lambda_rate: float,
    trading_days_per_year: int = 252,
) -> float | None:
    """
    Compute return period for a given dip magnitude.

    Return period = expected time between events of this magnitude or greater.
    E.g., "50-year event" means we expect one such event every 50 years on average.

    Uses the POT model:
        P(X > x | X > u) = (1 + shape * (x - u) / scale) ^ (-1/shape)  [if shape != 0]
        P(X > x | X > u) = exp(-(x - u) / scale)  [if shape == 0]

        Return period = 1 / (lambda * P(X > x | X > u))

    Args:
        current_dip: Current dip value
        threshold: Tail threshold
        gpd_shape: GPD shape parameter (xi)
        gpd_scale: GPD scale parameter (sigma)
        lambda_rate: Rate of threshold exceedances (events per day)
        trading_days_per_year: Trading days per year

    Returns:
        Return period in years, or None if can't compute
    """
    if current_dip <= threshold:
        # Below threshold - not a tail event
        return None

    if gpd_shape is None or gpd_scale is None:
        # GPD not fitted - use empirical approximation
        return None

    if lambda_rate <= 0:
        return None

    exceedance = current_dip - threshold

    try:
        # Survival probability using GPD
        # P(X > x | X > u) = 1 - F_GPD(x - u)
        survival_prob = 1 - stats.genpareto.cdf(exceedance, gpd_shape, scale=gpd_scale)

        if survival_prob <= 0 or not np.isfinite(survival_prob):
            # Beyond the support of the distribution (for bounded tails)
            # This is an extremely rare event
            return 100.0  # Cap at 100 years

        # Return period in days
        return_period_days = 1 / (lambda_rate * survival_prob)

        # Convert to years
        return_period_years = return_period_days / trading_days_per_year

        # Cap at reasonable values
        return min(float(return_period_years), 100.0)

    except Exception:
        return None


def compute_empirical_return_period(
    current_dip: float,
    dip_series: np.ndarray,
    trading_days_per_year: int = 252,
) -> float | None:
    """
    Compute return period using empirical distribution (fallback method).

    Simpler than GPD fitting but less accurate for extrapolation.

    Args:
        current_dip: Current dip value
        dip_series: Historical dip values
        trading_days_per_year: Trading days per year

    Returns:
        Return period in years based on historical frequency
    """
    valid = dip_series[~np.isnan(dip_series)]

    if len(valid) < 100:
        return None

    # Count events at least as extreme as current
    n_as_extreme = np.sum(valid >= current_dip)

    if n_as_extreme == 0:
        # Never seen before - estimate based on tail
        # Use simple extrapolation
        return float(len(valid)) / trading_days_per_year

    # Average days between such events
    days_per_event = len(valid) / n_as_extreme

    return float(days_per_event) / trading_days_per_year


def compute_regime_percentile(
    dip_series: np.ndarray,
    current_dip: float,
    threshold: float,
    exclude_last: bool = True,
) -> tuple[float, int, int]:
    """
    Compute percentile rank within "normal" regime (excluding tail events).

    This gives a fair comparison of the current dip against typical behavior,
    without extreme outliers skewing the distribution.

    Args:
        dip_series: Historical dip values
        current_dip: Current dip value
        threshold: Tail threshold (events above this are excluded)
        exclude_last: Exclude the last observation from comparison

    Returns:
        (regime_percentile, n_tail_events, n_normal_events)
    """
    valid = dip_series[~np.isnan(dip_series)]

    if exclude_last and len(valid) > 1:
        valid = valid[:-1]

    # Split into tail and normal
    tail_mask = valid > threshold
    n_tail_events = int(np.sum(tail_mask))

    # Normal regime = events at or below threshold
    normal_dips = valid[~tail_mask]
    n_normal_events = len(normal_dips)

    if n_normal_events == 0:
        return 50.0, n_tail_events, 0

    # If current dip is a tail event, it's at 100th percentile of normal
    if current_dip > threshold:
        return 100.0, n_tail_events, n_normal_events

    # Compute percentile within normal regime
    below_count = np.sum(normal_dips < current_dip)
    regime_percentile = (below_count / n_normal_events) * 100

    return float(regime_percentile), n_tail_events, n_normal_events


def analyze_tail_events(
    dip_series: np.ndarray,
    current_dip: float,
    threshold_percentile: float = 95.0,
    min_threshold: float = 0.40,
    max_threshold: float = 0.70,
    trading_days_per_year: int = 252,
) -> TailAnalysis:
    """
    Perform complete Extreme Value Analysis on dip distribution.

    This is the main entry point that computes all EVA metrics.

    Args:
        dip_series: Historical dip values
        current_dip: Current dip value
        threshold_percentile: Percentile for threshold detection
        min_threshold: Minimum tail threshold
        max_threshold: Maximum tail threshold
        trading_days_per_year: Trading days per year

    Returns:
        TailAnalysis with all EVA metrics
    """
    # Step 1: Compute tail threshold
    tail_threshold, actual_percentile = compute_tail_threshold(
        dip_series,
        threshold_percentile=threshold_percentile,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )

    # Step 2: Determine if current dip is a tail event
    is_tail_event = current_dip > tail_threshold

    # Step 3: Compute regime percentile (excluding tail events)
    regime_percentile, n_tail, n_normal = compute_regime_percentile(
        dip_series,
        current_dip,
        tail_threshold,
        exclude_last=True,
    )

    # Step 4: Fit GPD to exceedances (if enough data)
    gpd_shape, gpd_scale, fit_success = fit_gpd_to_exceedances(
        dip_series,
        tail_threshold,
        min_exceedances=5,
    )

    # Step 5: Compute return period
    return_period: float | None = None

    if is_tail_event:
        valid = dip_series[~np.isnan(dip_series)]
        n_exceedances = np.sum(valid > tail_threshold)
        lambda_rate = n_exceedances / len(valid) if len(valid) > 0 else 0

        if fit_success:
            # Use GPD-based return period
            return_period = compute_return_period(
                current_dip,
                tail_threshold,
                gpd_shape,
                gpd_scale,
                lambda_rate,
                trading_days_per_year,
            )

        if return_period is None:
            # Fallback to empirical
            return_period = compute_empirical_return_period(
                current_dip,
                dip_series,
                trading_days_per_year,
            )

    return TailAnalysis(
        tail_threshold=tail_threshold,
        threshold_percentile=actual_percentile,
        is_tail_event=is_tail_event,
        return_period_years=return_period,
        regime_dip_percentile=regime_percentile,
        n_tail_events=n_tail,
        n_normal_events=n_normal,
        gpd_shape=gpd_shape,
        gpd_scale=gpd_scale,
        gpd_fit_success=fit_success,
    )


def format_return_period(years: float | None) -> str:
    """
    Format return period for display.

    Args:
        years: Return period in years

    Returns:
        Human-readable string like "10-year event" or "Once in 50 years"
    """
    if years is None:
        return ""

    if years < 1:
        months = int(years * 12)
        if months <= 1:
            return "~monthly event"
        return f"~{months}-month event"
    elif years < 2:
        return "~annual event"
    elif years < 10:
        return f"~{int(years)}-year event"
    elif years < 50:
        return f"~{int(round(years / 5) * 5)}-year event"
    else:
        return "Rare event (50+ years)"
