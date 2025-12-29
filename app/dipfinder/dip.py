"""Dip calculation module with O(n) peak-to-current algorithm.

Implements efficient dip series computation using a monotonic max deque
for sliding window maximum. Provides percentile ranking and persistence
detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import DipFinderConfig


@dataclass
class DipMetrics:
    """Computed dip metrics for a single ticker."""

    ticker: str
    window: int

    # Current dip values
    dip_pct: float  # Current dip as fraction (0.15 = 15%)
    peak_price: float  # Peak price in window
    current_price: float  # Current price

    # Significance metrics
    dip_percentile: float  # Percentile rank (0-100) in historical distribution
    dip_vs_typical: float  # Ratio of current dip to typical (median) dip
    typical_dip: float  # Median historical dip

    # Persistence
    persist_days: int  # Days dip condition has held (below threshold)
    days_since_peak: int  # Days since the peak price was hit

    # Volume confirmation (new fields)
    volume_spike_ratio: float = 1.0  # Avg volume during dip vs normal (>1.5 = spike)
    volume_confirmed: bool = False  # True if dip had significant volume confirmation
    selloff_intensity: float = 0.0  # Volume-weighted price decline intensity

    # Flags
    is_meaningful: bool = False  # Meets all dip criteria

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "window": self.window,
            "dip_pct": round(self.dip_pct, 6),
            "peak_price": round(self.peak_price, 4),
            "current_price": round(self.current_price, 4),
            "dip_percentile": round(self.dip_percentile, 2),
            "dip_vs_typical": round(self.dip_vs_typical, 4),
            "typical_dip": round(self.typical_dip, 6),
            "persist_days": self.persist_days,
            "days_since_peak": self.days_since_peak,
            "volume_spike_ratio": round(self.volume_spike_ratio, 2),
            "volume_confirmed": self.volume_confirmed,
            "selloff_intensity": round(self.selloff_intensity, 4),
            "is_meaningful": self.is_meaningful,
        }


def compute_dip_series_windowed(
    close_prices: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute peak-to-current dip series for a given window.

    Uses O(n) monotonic deque algorithm for sliding window maximum.

    Args:
        close_prices: Array of closing prices (oldest first)
        window: Window size in days

    Returns:
        Array of dip fractions (same length as input, NaN for first window-1)
        Positive values indicate dip (0.15 = 15% below peak)
    """
    n = len(close_prices)
    if n == 0:
        return np.array([])

    if window <= 0:
        raise ValueError("Window must be positive")

    # Result array
    dips = np.full(n, np.nan)

    # Monotonic decreasing deque storing (index, value)
    # Front always has the maximum in current window
    max_deque: deque = deque()

    for i in range(n):
        price = close_prices[i]

        # Remove elements outside window
        while max_deque and max_deque[0][0] <= i - window:
            max_deque.popleft()

        # Remove elements smaller than current (they can't be max)
        while max_deque and max_deque[-1][1] <= price:
            max_deque.pop()

        # Add current element
        max_deque.append((i, price))

        # Compute dip once we have enough data
        if i >= window - 1:
            peak = max_deque[0][1]
            if peak > 0:
                dips[i] = (peak - price) / peak
            else:
                dips[i] = 0.0

    return dips


def compute_dip_series_multi_window(
    close_prices: np.ndarray,
    windows: list[int],
) -> dict[int, np.ndarray]:
    """
    Compute dip series for multiple windows efficiently.

    Args:
        close_prices: Array of closing prices
        windows: List of window sizes

    Returns:
        Dict mapping window -> dip series
    """
    return {w: compute_dip_series_windowed(close_prices, w) for w in windows}


def compute_dip_percentile(
    dip_series: np.ndarray,
    current_dip: float,
    exclude_last: bool = True,
) -> float:
    """
    Compute percentile rank of current dip in historical distribution.

    Args:
        dip_series: Historical dip series
        current_dip: Current dip value
        exclude_last: If True, exclude last value from comparison

    Returns:
        Percentile rank (0-100), higher = more severe/rare
    """
    # Get valid (non-NaN) values
    if exclude_last:
        valid = dip_series[:-1] if len(dip_series) > 1 else dip_series
    else:
        valid = dip_series

    valid = valid[~np.isnan(valid)]

    if len(valid) == 0:
        return 50.0  # Neutral if no history

    # Count how many values are less than current
    # (higher percentile = more severe dip)
    below_count = np.sum(valid < current_dip)
    percentile = (below_count / len(valid)) * 100

    return float(percentile)


def compute_typical_dip(
    dip_series: np.ndarray,
    use_median: bool = True,
    min_dip_threshold: float = 0.01,
) -> float:
    """
    Compute typical (median or mean) dip from historical series.

    Only considers positive dips above min_dip_threshold to avoid
    including zero/no-dip days which would skew the result.

    Args:
        dip_series: Historical dip series (dip values >= 0)
        use_median: If True, use median; else use mean
        min_dip_threshold: Minimum dip to include (default 1%)

    Returns:
        Typical dip value, or 0.0 if no valid dips
    """
    valid = dip_series[~np.isnan(dip_series)]

    # Filter to positive dips only (above threshold)
    positive_dips = valid[valid >= min_dip_threshold]

    if len(positive_dips) == 0:
        return 0.0

    if use_median:
        return float(np.median(positive_dips))
    else:
        return float(np.mean(positive_dips))


def compute_volume_confirmation(
    close_prices: np.ndarray,
    volumes: np.ndarray | None,
    dip_series: np.ndarray,
    dip_threshold: float = 0.05,
    lookback: int = 20,
    volume_spike_threshold: float = 1.5,
) -> tuple[float, bool, float]:
    """
    Compute volume confirmation metrics for a dip.

    A confirmed dip should have higher-than-normal volume on down days,
    indicating real selling pressure (capitulation) rather than just
    low-volume drift.

    Args:
        close_prices: Array of closing prices
        volumes: Array of trading volumes (optional)
        dip_series: Pre-computed dip series
        dip_threshold: Minimum dip to consider for volume analysis
        lookback: Days to look back for average volume
        volume_spike_threshold: Multiplier for volume spike detection

    Returns:
        (volume_spike_ratio, volume_confirmed, selloff_intensity)
    """
    if volumes is None or len(volumes) == 0:
        return 1.0, False, 0.0

    if len(close_prices) != len(volumes):
        return 1.0, False, 0.0

    n = len(close_prices)
    if n < lookback + 5:
        return 1.0, False, 0.0

    # Calculate daily returns
    returns = np.zeros(n)
    returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]

    # Get average volume over lookback period (excluding recent dip period)
    avg_volume_period = max(0, n - lookback - 10)  # Go back before the dip
    if avg_volume_period < 20:
        avg_volume_period = 0
    avg_volume = float(np.mean(volumes[avg_volume_period:n - 5]))

    if avg_volume <= 0:
        return 1.0, False, 0.0

    # Find down days during dip period
    dip_days = []
    for i in range(max(0, n - 30), n):  # Last 30 days
        if not np.isnan(dip_series[i]) and dip_series[i] >= dip_threshold:
            if returns[i] < -0.005:  # Down at least 0.5%
                dip_days.append(i)

    if not dip_days:
        return 1.0, False, 0.0

    # Calculate average volume on dip down days vs normal
    dip_day_volumes = [volumes[i] for i in dip_days]
    avg_dip_volume = float(np.mean(dip_day_volumes))
    volume_spike_ratio = avg_dip_volume / avg_volume

    # Volume confirmed if spike ratio exceeds threshold
    volume_confirmed = volume_spike_ratio >= volume_spike_threshold

    # Selloff intensity: volume-weighted price decline
    # Higher = more intense selling pressure
    selloff_intensity = 0.0
    for i in dip_days:
        if returns[i] < 0:
            # Contribution = |return| * (volume / avg_volume)
            vol_weight = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            selloff_intensity += abs(returns[i]) * vol_weight

    return float(volume_spike_ratio), volume_confirmed, float(selloff_intensity)


def compute_persistence(
    close_prices: np.ndarray,
    dip_threshold: float,
    window: int,
) -> int:
    """
    Count consecutive days the dip condition has held.

    Args:
        close_prices: Array of closing prices
        dip_threshold: Minimum dip fraction to count as "in dip"
        window: Window for dip calculation

    Returns:
        Number of consecutive days at end of series where dip >= threshold
    """
    dip_series = compute_dip_series_windowed(close_prices, window)

    # Count from end backwards
    persist_days = 0
    for i in range(len(dip_series) - 1, -1, -1):
        if np.isnan(dip_series[i]):
            break
        if dip_series[i] >= dip_threshold:
            persist_days += 1
        else:
            break

    return persist_days


def compute_dip_metrics(
    ticker: str,
    close_prices: np.ndarray,
    window: int,
    config: DipFinderConfig | None = None,
    volumes: np.ndarray | None = None,
) -> DipMetrics:
    """
    Compute complete dip metrics for a ticker.

    Args:
        ticker: Ticker symbol
        close_prices: Array of closing prices (oldest first)
        window: Window size in days
        config: DipFinder configuration (uses defaults if None)
        volumes: Optional array of trading volumes for volume confirmation

    Returns:
        DipMetrics with all computed values
    """
    from .config import get_dipfinder_config

    if config is None:
        config = get_dipfinder_config()

    if len(close_prices) == 0:
        return DipMetrics(
            ticker=ticker,
            window=window,
            dip_pct=0.0,
            peak_price=0.0,
            current_price=0.0,
            dip_percentile=50.0,
            dip_vs_typical=0.0,
            typical_dip=0.0,
            persist_days=0,
            days_since_peak=0,
            is_meaningful=False,
        )

    # Compute dip series
    dip_series = compute_dip_series_windowed(close_prices, window)

    # Current values
    current_price = float(close_prices[-1])
    current_dip = float(dip_series[-1]) if not np.isnan(dip_series[-1]) else 0.0

    # Peak price (from window ending at current day)
    start_idx = max(0, len(close_prices) - window)
    window_prices = close_prices[start_idx:]
    peak_price = float(np.max(window_prices))
    peak_idx_in_window = int(np.argmax(window_prices))
    days_since_peak = len(window_prices) - 1 - peak_idx_in_window

    # Use recent history for percentile/typical to keep window-consistent comparisons.
    baseline_series = dip_series[-365:] if len(dip_series) >= 365 else dip_series

    dip_percentile = compute_dip_percentile(baseline_series, current_dip, exclude_last=True)
    typical_dip = compute_typical_dip(baseline_series)

    # Dip vs typical ratio
    dip_vs_typical = current_dip / typical_dip if typical_dip > 0 else 0.0

    # Persistence
    persist_days = compute_persistence(close_prices, config.min_dip_abs, window)

    # Volume confirmation
    volume_spike_ratio, volume_confirmed, selloff_intensity = compute_volume_confirmation(
        close_prices=close_prices,
        volumes=volumes,
        dip_series=dip_series,
        dip_threshold=config.min_dip_abs,
    )

    # Is meaningful check
    is_meaningful = (
        current_dip >= config.min_dip_abs
        and persist_days >= config.min_persist_days
        and (
            dip_percentile >= config.dip_percentile_threshold * 100
            or dip_vs_typical >= config.dip_vs_typical_threshold
        )
    )

    return DipMetrics(
        ticker=ticker,
        window=window,
        dip_pct=current_dip,
        peak_price=peak_price,
        current_price=current_price,
        dip_percentile=dip_percentile,
        dip_vs_typical=dip_vs_typical,
        typical_dip=typical_dip,
        persist_days=persist_days,
        days_since_peak=days_since_peak,
        volume_spike_ratio=volume_spike_ratio,
        volume_confirmed=volume_confirmed,
        selloff_intensity=selloff_intensity,
        is_meaningful=is_meaningful,
    )


class IncrementalDipTracker:
    """
    Incremental dip tracker for streaming updates.

    Maintains state for efficient single-day updates without
    recomputing the entire history.
    """

    def __init__(self, window: int):
        """Initialize tracker with window size."""
        self.window = window
        self._prices: deque = deque(maxlen=window)
        self._max_deque: deque = deque()  # (index, value)
        self._day_index = 0

    def update(self, price: float) -> float | None:
        """
        Add a new price and return current dip.

        Args:
            price: New closing price

        Returns:
            Current dip fraction, or None if not enough data
        """
        self._prices.append(price)

        # Remove elements outside window
        while (
            self._max_deque and self._max_deque[0][0] <= self._day_index - self.window
        ):
            self._max_deque.popleft()

        # Remove elements smaller than current
        while self._max_deque and self._max_deque[-1][1] <= price:
            self._max_deque.pop()

        # Add current
        self._max_deque.append((self._day_index, price))

        self._day_index += 1

        if len(self._prices) >= self.window:
            peak = self._max_deque[0][1]
            if peak > 0:
                return (peak - price) / peak
            return 0.0

        return None

    @property
    def current_dip(self) -> float | None:
        """Get current dip without adding new data."""
        if len(self._prices) < self.window:
            return None

        if not self._max_deque:
            return None

        peak = self._max_deque[0][1]
        price = self._prices[-1]

        if peak > 0:
            return (peak - price) / peak
        return 0.0

    @property
    def current_peak(self) -> float | None:
        """Get current peak price."""
        if not self._max_deque:
            return None
        return self._max_deque[0][1]

    def reset(self) -> None:
        """Reset tracker state."""
        self._prices.clear()
        self._max_deque.clear()
        self._day_index = 0
