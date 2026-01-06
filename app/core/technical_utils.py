"""
Simple Technical Indicator Utilities.

Provides lightweight technical indicator calculations for simple Python lists.
Use this for agents and simple calculations where pandas overhead is not needed.

For pandas-based calculations with full indicator suites, use:
    from app.quant_engine.core import TechnicalService

Usage:
    from app.core.technical_utils import sma, ema, rsi, atr, volatility, std
    
    closes = [100.0, 101.5, 99.0, 102.0, ...]
    rsi_value = rsi(closes, period=14)
"""

from __future__ import annotations

import math
from typing import Sequence


def sma(values: Sequence[float], period: int) -> float | None:
    """
    Calculate Simple Moving Average.
    
    Args:
        values: Price or value series
        period: Lookback period
        
    Returns:
        SMA value or None if insufficient data
    """
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: Sequence[float], period: int) -> float | None:
    """
    Calculate Exponential Moving Average.
    
    Uses the standard smoothing multiplier: 2 / (period + 1)
    
    Args:
        values: Price or value series
        period: Lookback period
        
    Returns:
        EMA value or None if insufficient data
    """
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    ema_value = sum(values[:period]) / period  # Start with SMA

    for price in values[period:]:
        ema_value = (price - ema_value) * multiplier + ema_value

    return ema_value


def std(values: Sequence[float], period: int) -> float | None:
    """
    Calculate standard deviation.
    
    Args:
        values: Price or value series
        period: Lookback period
        
    Returns:
        Standard deviation or None if insufficient data
    """
    if len(values) < period:
        return None

    subset = list(values[-period:])
    mean = sum(subset) / period
    variance = sum((x - mean) ** 2 for x in subset) / period
    return math.sqrt(variance)


def rsi(closes: Sequence[float], period: int = 14) -> float | None:
    """
    Calculate Relative Strength Index.
    
    Args:
        closes: Closing prices
        period: RSI period (default 14)
        
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    # Separate gains and losses
    gains = [max(0, c) for c in changes]
    losses = [abs(min(0, c)) for c in changes]

    # Initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smooth averages (Wilder's smoothing)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """
    Calculate Average True Range.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Closing prices
        period: ATR period (default 14)
        
    Returns:
        ATR value or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)

    return sum(trs[-period:]) / period


def volatility(closes: Sequence[float], period: int = 20) -> float | None:
    """
    Calculate historical volatility (annualized).
    
    Args:
        closes: Closing prices
        period: Lookback period (default 20)
        
    Returns:
        Annualized volatility or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    # Daily returns
    returns = [
        (closes[i] - closes[i-1]) / closes[i-1]
        for i in range(1, len(closes))
        if closes[i-1] != 0
    ]

    if len(returns) < period:
        return None

    subset = returns[-period:]
    mean = sum(subset) / len(subset)
    variance = sum((r - mean) ** 2 for r in subset) / len(subset)
    daily_vol = math.sqrt(variance)

    # Annualize (252 trading days)
    return daily_vol * math.sqrt(252)


def macd(
    closes: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        closes: Closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram) or (None, None, None)
    """
    if len(closes) < slow_period:
        return None, None, None

    ema_fast = ema(closes, fast_period)
    ema_slow = ema(closes, slow_period)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_line = ema_fast - ema_slow
    
    # Calculate MACD series for signal line
    macd_values = _macd_series(closes, fast_period, slow_period)
    if len(macd_values) < signal_period:
        return macd_line, None, None
    
    signal_line = ema(macd_values, signal_period)
    histogram = macd_line - signal_line if signal_line else None
    
    return macd_line, signal_line, histogram


def _macd_series(
    closes: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
) -> list[float]:
    """Calculate MACD line series (internal helper)."""
    if len(closes) < slow_period:
        return []

    result = []
    for i in range(slow_period, len(closes) + 1):
        subset = closes[:i]
        ema_fast = ema(subset, fast_period)
        ema_slow = ema(subset, slow_period)
        if ema_fast and ema_slow:
            result.append(ema_fast - ema_slow)

    return result


def bollinger_bands(
    closes: Sequence[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """
    Calculate Bollinger Bands.
    
    Args:
        closes: Closing prices
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2.0)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band) or (None, None, None)
    """
    middle = sma(closes, period)
    std_dev = std(closes, period)
    
    if middle is None or std_dev is None:
        return None, None, None
    
    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev
    
    return upper, middle, lower


__all__ = [
    "sma",
    "ema",
    "std",
    "rsi",
    "atr",
    "volatility",
    "macd",
    "bollinger_bands",
]
