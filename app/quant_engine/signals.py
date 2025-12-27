"""
Technical Signal Scanner - Per-Stock Optimized Trading Signals.

This module implements practical, backtested technical signals that answer:
- "Which stock is the best buy opportunity right now?"
- "Is this dip a statistical overreaction?"
- "What signals have historically worked for this specific stock?"
- "What's the optimal holding period after buying?"

Key features:
- Hyperparameter optimization per stock (holding period, thresholds)
- Multiple signal types (momentum, mean reversion, volume, MACD, etc.)
- Backtested win rates and expected returns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OptimizedSignal:
    """A signal optimized for a specific stock."""
    
    name: str
    description: str
    
    # Current state
    current_value: float
    is_buy_signal: bool
    signal_strength: float  # 0-1
    
    # Optimized parameters
    optimal_threshold: float
    optimal_holding_days: int
    
    # Backtest results at optimal params
    win_rate: float
    avg_return_pct: float
    max_return_pct: float
    min_return_pct: float
    n_signals: int
    
    # Comparison to default params
    default_win_rate: float = 0.0
    improvement_pct: float = 0.0  # How much better is optimized vs default


@dataclass 
class StockOpportunity:
    """Complete analysis for a single stock."""
    
    symbol: str
    name: str
    
    # Overall score
    buy_score: float  # 0-100
    opportunity_type: str  # STRONG_BUY, BUY, WEAK_BUY, NEUTRAL, AVOID
    opportunity_reason: str
    
    # Current price metrics
    current_price: float
    price_vs_52w_high_pct: float
    price_vs_52w_low_pct: float
    
    # Statistical metrics
    zscore_20d: float
    zscore_60d: float
    rsi_14: float
    
    # Optimized signals (sorted by expected value)
    signals: list[OptimizedSignal] = field(default_factory=list)
    
    # Active buy signals
    active_signals: list[OptimizedSignal] = field(default_factory=list)
    
    # Best action recommendation
    best_signal_name: str = ""
    best_holding_days: int = 0
    best_expected_return: float = 0.0


@dataclass
class ScanResult:
    """Complete scan results."""
    
    scanned_at: str
    n_stocks: int
    
    # Stocks ranked by opportunity
    stocks: list[StockOpportunity] = field(default_factory=list)
    
    # Summary
    top_opportunities: list[str] = field(default_factory=list)
    n_active_signals: int = 0


# =============================================================================
# Signal Computation Functions
# =============================================================================


def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window).mean()


def compute_ema(prices: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=window, adjust=False).mean()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal line, and Histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper, Middle, Lower Bollinger Bands."""
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_zscore(prices: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score (how many std devs from mean)."""
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return (prices - mean) / std.replace(0, np.nan)


def compute_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Drawdown from rolling peak."""
    peak = prices.rolling(window, min_periods=1).max()
    return (prices - peak) / peak


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume vs average volume."""
    avg_volume = volume.rolling(window).mean()
    return volume / avg_volume.replace(0, np.nan)


def compute_price_vs_sma(prices: pd.Series, window: int) -> pd.Series:
    """Price relative to SMA (% above/below)."""
    sma = prices.rolling(window).mean()
    return (prices - sma) / sma


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Stochastic Oscillator %K."""
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    return k


def _compute_volume_dip_signal(p: dict) -> pd.Series:
    """Combine volume spike with price dip."""
    if "volume" not in p:
        return pd.Series(0.0, index=p["close"].index)
    
    volume_ratio = compute_volume_ratio(p["volume"], 20)
    zscore = compute_zscore(p["close"], 20)
    
    # Signal = volume ratio when z-score is negative
    signal = volume_ratio.where(zscore < 0, 0)
    return signal.fillna(0)


def _compute_consecutive_down(prices: pd.Series) -> pd.Series:
    """Count consecutive down days."""
    returns = prices.pct_change()
    is_down = (returns < 0).astype(int)
    
    # Count consecutive downs
    consecutive = is_down.copy()
    for i in range(1, len(consecutive)):
        if is_down.iloc[i] == 1:
            consecutive.iloc[i] = consecutive.iloc[i-1] + 1
        else:
            consecutive.iloc[i] = 0
    
    return consecutive


# =============================================================================
# Signal Definitions
# =============================================================================


SIGNAL_CONFIGS = [
    # Z-Score Mean Reversion
    {
        "name": "Z-Score 20D Oversold",
        "description": "Price is significantly below 20-day mean",
        "compute": lambda p, **kw: compute_zscore(p["close"], 20),
        "default_threshold": -2.0,
        "direction": "below",
        "threshold_range": [-3.0, -1.5],
    },
    {
        "name": "Z-Score 60D Oversold",
        "description": "Price is significantly below 60-day mean",
        "compute": lambda p, **kw: compute_zscore(p["close"], 60),
        "default_threshold": -1.5,
        "direction": "below",
        "threshold_range": [-2.5, -1.0],
    },
    # RSI
    {
        "name": "RSI Oversold",
        "description": "RSI indicates oversold conditions",
        "compute": lambda p, **kw: compute_rsi(p["close"], 14),
        "default_threshold": 30.0,
        "direction": "below",
        "threshold_range": [20.0, 35.0],
    },
    {
        "name": "RSI Extremely Oversold",
        "description": "RSI indicates extremely oversold",
        "compute": lambda p, **kw: compute_rsi(p["close"], 14),
        "default_threshold": 25.0,
        "direction": "below",
        "threshold_range": [15.0, 30.0],
    },
    # Bollinger Bands
    {
        "name": "Below Bollinger Lower",
        "description": "Price below lower Bollinger Band",
        "compute": lambda p, **kw: _compute_bollinger_pct(p["close"]),
        "default_threshold": 0.0,
        "direction": "below",
        "threshold_range": [-0.05, 0.02],
    },
    # Drawdown from Peak
    {
        "name": "Drawdown 15%+",
        "description": "Price 15%+ below 52-week high",
        "compute": lambda p, **kw: compute_drawdown(p["close"], 252),
        "default_threshold": -0.15,
        "direction": "below",
        "threshold_range": [-0.25, -0.10],
    },
    {
        "name": "Drawdown 25%+",
        "description": "Price 25%+ below 52-week high",
        "compute": lambda p, **kw: compute_drawdown(p["close"], 252),
        "default_threshold": -0.25,
        "direction": "below",
        "threshold_range": [-0.40, -0.15],
    },
    # Price vs SMA
    {
        "name": "Below SMA 50",
        "description": "Price below 50-day moving average",
        "compute": lambda p, **kw: compute_price_vs_sma(p["close"], 50),
        "default_threshold": -0.05,
        "direction": "below",
        "threshold_range": [-0.15, -0.02],
    },
    {
        "name": "Below SMA 200",
        "description": "Price below 200-day moving average",
        "compute": lambda p, **kw: compute_price_vs_sma(p["close"], 200),
        "default_threshold": -0.05,
        "direction": "below",
        "threshold_range": [-0.20, -0.02],
    },
    # MACD
    {
        "name": "MACD Bullish Cross",
        "description": "MACD histogram turning positive",
        "compute": lambda p, **kw: compute_macd(p["close"])[2],
        "default_threshold": 0.0,
        "direction": "cross_above",
        "threshold_range": [-0.5, 0.5],
    },
    # Stochastic (use RSI as fallback if no high/low data)
    {
        "name": "Stochastic Oversold",
        "description": "Stochastic %K indicates oversold",
        "compute": lambda p, **kw: _compute_stochastic_safe(p),
        "default_threshold": 20.0,
        "direction": "below",
        "threshold_range": [10.0, 30.0],
    },
    # Volume Spike on Dip
    {
        "name": "Volume Spike on Dip",
        "description": "High volume during price drop (potential capitulation)",
        "compute": _compute_volume_dip_signal,
        "default_threshold": 1.5,
        "direction": "above",
        "threshold_range": [1.2, 2.5],
    },
    # Consecutive Down Days
    {
        "name": "Consecutive Down Days",
        "description": "Multiple consecutive red days",
        "compute": lambda p, **kw: _compute_consecutive_down(p["close"]),
        "default_threshold": 4,
        "direction": "above",
        "threshold_range": [3, 6],
    },
    # Golden Cross (longer-term trend)
    {
        "name": "Below SMA 20/50 Cross",
        "description": "Price below both short and medium-term averages",
        "compute": lambda p, **kw: _compute_dual_sma_below(p["close"]),
        "default_threshold": -0.03,
        "direction": "below",
        "threshold_range": [-0.10, -0.01],
    },
]


def _compute_bollinger_pct(prices: pd.Series) -> pd.Series:
    """Compute percentage below lower Bollinger band."""
    upper, middle, lower = compute_bollinger_bands(prices, 20, 2.0)
    return (prices - lower) / lower


def _compute_stochastic_safe(p: dict) -> pd.Series:
    """Compute stochastic with fallback to RSI if no high/low data."""
    close = p["close"]
    if "high" in p and "low" in p:
        return compute_stochastic(p["high"], p["low"], close, 14)
    # Fallback: use close as proxy for high/low
    return compute_stochastic(close, close, close, 14)


def _compute_dual_sma_below(prices: pd.Series) -> pd.Series:
    """Compute min of (price vs SMA 20) and (price vs SMA 50)."""
    vs_sma20 = compute_price_vs_sma(prices, 20)
    vs_sma50 = compute_price_vs_sma(prices, 50)
    return pd.concat([vs_sma20, vs_sma50], axis=1).max(axis=1)  # Less negative = closer to SMA


# =============================================================================
# Backtesting and Optimization
# =============================================================================


def backtest_signal(
    prices: pd.Series,
    signal: pd.Series,
    threshold: float,
    direction: str,
    holding_days: int,
) -> dict[str, float]:
    """
    Backtest a signal with specific parameters.
    
    Returns dict with win_rate, avg_return, max_return, min_return, n_signals.
    """
    # Identify signal triggers
    if direction == "below":
        triggers = signal < threshold
    elif direction == "above":
        triggers = signal > threshold
    elif direction == "cross_above":
        # Signal crosses above threshold from below
        prev_signal = signal.shift(1)
        triggers = (signal > threshold) & (prev_signal <= threshold)
    else:
        triggers = signal < threshold  # Default to below
    
    # Compute forward returns
    fwd_ret = prices.pct_change(holding_days).shift(-holding_days)
    
    # Get returns when signal triggered
    signal_returns = fwd_ret[triggers].dropna()
    
    if len(signal_returns) < 3:
        return {
            "win_rate": 0.0,
            "avg_return": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
            "n_signals": 0,
        }
    
    return {
        "win_rate": float((signal_returns > 0).mean()),
        "avg_return": float(signal_returns.mean()),
        "max_return": float(signal_returns.max()),
        "min_return": float(signal_returns.min()),
        "n_signals": len(signal_returns),
    }


def optimize_signal_params(
    prices: pd.Series,
    signal: pd.Series,
    direction: str,
    threshold_range: list[float],
    holding_days_options: list[int] = [5, 10, 20, 40, 60],
    min_signals: int = 5,
) -> tuple[float, int, dict]:
    """
    Find optimal threshold and holding period for a signal.
    
    Optimizes for: win_rate * avg_return (expected value)
    
    Returns: (optimal_threshold, optimal_holding_days, backtest_results)
    """
    best_score = -np.inf
    best_threshold = threshold_range[0]
    best_holding = 20
    best_results = {}
    
    # Grid search over thresholds
    n_threshold_steps = 5
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_threshold_steps)
    
    for threshold in thresholds:
        for holding_days in holding_days_options:
            results = backtest_signal(prices, signal, threshold, direction, holding_days)
            
            if results["n_signals"] < min_signals:
                continue
            
            # Score = expected value (win_rate * avg_return)
            # But penalize very few signals slightly
            signal_penalty = min(1.0, results["n_signals"] / 20)
            score = results["win_rate"] * results["avg_return"] * 100 * signal_penalty
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_holding = holding_days
                best_results = results
    
    return best_threshold, best_holding, best_results


def evaluate_signal_for_stock(
    price_data: dict[str, pd.Series],
    signal_config: dict,
    holding_days_options: list[int] = [5, 10, 20, 40, 60],
) -> OptimizedSignal | None:
    """Evaluate and optimize a single signal for a stock."""
    try:
        # Compute signal values
        signal_values = signal_config["compute"](price_data)
        
        if signal_values.isna().all():
            return None
        
        current_value = signal_values.iloc[-1]
        if pd.isna(current_value):
            return None
        
        prices = price_data["close"]
        direction = signal_config["direction"]
        
        # Get default backtest
        default_results = backtest_signal(
            prices,
            signal_values,
            signal_config["default_threshold"],
            direction,
            20,  # Default holding period
        )
        
        # Optimize parameters
        opt_threshold, opt_holding, opt_results = optimize_signal_params(
            prices,
            signal_values,
            direction,
            signal_config["threshold_range"],
            holding_days_options,
        )
        
        # Skip if no valid results
        if opt_results.get("n_signals", 0) < 3:
            return None
        
        # Check if signal is currently active
        if direction == "below":
            is_buy = current_value < opt_threshold
            if opt_threshold != 0:
                strength = max(0, min(1, (opt_threshold - current_value) / abs(opt_threshold)))
            else:
                strength = 1.0 if is_buy else 0.0
        elif direction == "cross_above":
            # For crossover, check if just crossed
            prev_value = signal_values.iloc[-2] if len(signal_values) > 1 else current_value
            is_buy = (current_value > opt_threshold) and (prev_value <= opt_threshold)
            strength = 1.0 if is_buy else 0.0
        else:  # above
            is_buy = current_value > opt_threshold
            if opt_threshold != 0:
                strength = max(0, min(1, (current_value - opt_threshold) / abs(opt_threshold)))
            else:
                strength = 1.0 if is_buy else 0.0
        
        # Calculate improvement
        default_ev = default_results.get("win_rate", 0) * default_results.get("avg_return", 0) * 100
        opt_ev = opt_results["win_rate"] * opt_results["avg_return"] * 100
        improvement = ((opt_ev - default_ev) / abs(default_ev) * 100) if default_ev != 0 else 0
        
        return OptimizedSignal(
            name=signal_config["name"],
            description=signal_config["description"],
            current_value=float(current_value),
            is_buy_signal=is_buy,
            signal_strength=strength if is_buy else 0.0,
            optimal_threshold=float(opt_threshold),
            optimal_holding_days=opt_holding,
            win_rate=opt_results["win_rate"],
            avg_return_pct=opt_results["avg_return"] * 100,
            max_return_pct=opt_results["max_return"] * 100,
            min_return_pct=opt_results["min_return"] * 100,
            n_signals=opt_results["n_signals"],
            default_win_rate=default_results.get("win_rate", 0),
            improvement_pct=improvement,
        )
        
    except Exception as e:
        logger.warning(f"Failed to evaluate signal {signal_config['name']}: {e}")
        return None


def analyze_stock(
    symbol: str,
    name: str,
    price_data: dict[str, pd.Series],
    holding_days_options: list[int] = [5, 10, 20, 40, 60],
) -> StockOpportunity:
    """
    Complete analysis for a single stock.
    
    Evaluates all signals, optimizes parameters, and generates recommendations.
    """
    prices = price_data["close"]
    
    if len(prices) < 252:
        return StockOpportunity(
            symbol=symbol,
            name=name,
            buy_score=0,
            opportunity_type="INSUFFICIENT_DATA",
            opportunity_reason="Less than 1 year of data",
            current_price=float(prices.iloc[-1]) if len(prices) > 0 else 0,
            price_vs_52w_high_pct=0,
            price_vs_52w_low_pct=0,
            zscore_20d=0,
            zscore_60d=0,
            rsi_14=50,
        )
    
    # Compute current metrics
    current_price = float(prices.iloc[-1])
    high_52w = float(prices.iloc[-252:].max())
    low_52w = float(prices.iloc[-252:].min())
    
    zscore_20d = compute_zscore(prices, 20).iloc[-1]
    zscore_60d = compute_zscore(prices, 60).iloc[-1]
    rsi_14 = compute_rsi(prices, 14).iloc[-1]
    
    # Evaluate all signals
    signals = []
    for config in SIGNAL_CONFIGS:
        result = evaluate_signal_for_stock(price_data, config, holding_days_options)
        if result is not None:
            signals.append(result)
    
    # Sort by expected value (win_rate * avg_return)
    signals.sort(
        key=lambda s: s.win_rate * s.avg_return_pct if s.n_signals >= 5 else 0,
        reverse=True,
    )
    
    # Get active signals
    active_signals = [s for s in signals if s.is_buy_signal and s.n_signals >= 5]
    
    # Calculate buy score
    buy_score = 0.0
    if active_signals:
        for sig in active_signals:
            # Score contribution = strength * win_rate * avg_return (capped)
            contribution = sig.signal_strength * sig.win_rate * min(sig.avg_return_pct, 20) / 20
            buy_score += contribution * 25  # Scale to 0-100
        buy_score = min(100, buy_score)
    
    # Determine opportunity type
    if buy_score >= 70:
        opp_type = "STRONG_BUY"
        opp_reason = f"{len(active_signals)} strong signals, best: {active_signals[0].name} ({active_signals[0].win_rate*100:.0f}% win rate)"
    elif buy_score >= 40:
        opp_type = "BUY"
        opp_reason = f"{len(active_signals)} buy signals active"
    elif buy_score > 10:
        opp_type = "WEAK_BUY"
        opp_reason = "Some signals active, but weak"
    elif rsi_14 > 70:
        opp_type = "OVERBOUGHT"
        opp_reason = f"RSI at {rsi_14:.0f} indicates overbought"
    else:
        opp_type = "NEUTRAL"
        opp_reason = "No strong buy signals currently"
    
    # Best recommendation
    best_signal_name = ""
    best_holding_days = 0
    best_expected_return = 0.0
    if signals:
        best = signals[0]
        best_signal_name = best.name
        best_holding_days = best.optimal_holding_days
        best_expected_return = best.win_rate * best.avg_return_pct
    
    return StockOpportunity(
        symbol=symbol,
        name=name,
        buy_score=buy_score,
        opportunity_type=opp_type,
        opportunity_reason=opp_reason,
        current_price=current_price,
        price_vs_52w_high_pct=(current_price / high_52w - 1) * 100,
        price_vs_52w_low_pct=(current_price / low_52w - 1) * 100,
        zscore_20d=float(zscore_20d) if pd.notna(zscore_20d) else 0,
        zscore_60d=float(zscore_60d) if pd.notna(zscore_60d) else 0,
        rsi_14=float(rsi_14) if pd.notna(rsi_14) else 50,
        signals=signals[:10],  # Top 10 signals
        active_signals=active_signals,
        best_signal_name=best_signal_name,
        best_holding_days=best_holding_days,
        best_expected_return=best_expected_return,
    )


def scan_all_stocks(
    price_data: dict[str, pd.Series],
    stock_names: dict[str, str],
    holding_days_options: list[int] = [5, 10, 20, 40, 60],
) -> list[StockOpportunity]:
    """
    Scan all stocks and rank by opportunity.
    
    Parameters
    ----------
    price_data : dict[str, pd.Series]
        Price series per symbol.
    stock_names : dict[str, str]
        Symbol -> company name mapping.
    holding_days_options : list[int]
        Holding periods to test during optimization.
    
    Returns
    -------
    list[StockOpportunity]
        Stocks ranked by buy_score (highest first).
    """
    results = []
    
    for symbol, prices in price_data.items():
        name = stock_names.get(symbol, symbol)
        
        # Build price dict
        if isinstance(prices, pd.Series):
            data = {"close": prices}
        else:
            data = prices
        
        opportunity = analyze_stock(symbol, name, data, holding_days_options)
        results.append(opportunity)
    
    # Sort by buy score
    results.sort(key=lambda x: x.buy_score, reverse=True)
    
    return results


# =============================================================================
# Legacy compatibility (for existing API endpoint)
# =============================================================================


@dataclass
class SignalResult:
    """Legacy signal result for API compatibility."""
    name: str
    value: float
    is_buy_signal: bool
    strength: float
    win_rate: float
    avg_return_pct: float
    n_signals: int
    description: str = ""


@dataclass 
class StockSignalSummary:
    """Legacy summary for API compatibility."""
    symbol: str
    name: str
    signals: list[SignalResult] = field(default_factory=list)
    buy_score: float = 0.0
    opportunity_type: str = "NEUTRAL"
    opportunity_reason: str = ""
    current_price: float = 0.0
    price_vs_52w_high_pct: float = 0.0
    price_vs_52w_low_pct: float = 0.0
    zscore_20d: float = 0.0
    zscore_60d: float = 0.0
