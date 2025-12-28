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


def apply_cooldown_to_triggers(triggers: pd.Series, cooldown_days: int) -> pd.Series:
    """
    Apply cooldown period to trigger signals.
    
    After a trigger fires, suppress all triggers for the next `cooldown_days` days.
    This prevents "threshold whiplash" where oscillating signals create multiple
    entries in quick succession (e.g., drawdown bouncing around -25%).
    
    Args:
        triggers: Boolean series with True on trigger days
        cooldown_days: Number of days to wait before allowing another trigger
        
    Returns:
        Filtered trigger series with cooldown applied
    """
    if cooldown_days <= 0:
        return triggers
    
    result = triggers.copy()
    trigger_dates = triggers[triggers].index.tolist()
    
    last_trigger_date = None
    for date in trigger_dates:
        if last_trigger_date is not None:
            days_since = (date - last_trigger_date).days
            if days_since < cooldown_days:
                # Still in cooldown - suppress this trigger
                result.loc[date] = False
                continue
        # Allow this trigger and reset cooldown
        last_trigger_date = date
    
    return result
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
    # Price vs SMA - Event-based crossover signals
    {
        "name": "Cross Below SMA 50",
        "description": "Price crosses below 50-day moving average",
        "compute": lambda p, **kw: compute_price_vs_sma(p["close"], 50),
        "default_threshold": -0.02,
        "direction": "cross_below",
        "threshold_range": [-0.10, 0.0],
    },
    {
        "name": "Cross Below SMA 200",
        "description": "Price crosses below 200-day moving average",
        "compute": lambda p, **kw: compute_price_vs_sma(p["close"], 200),
        "default_threshold": -0.02,
        "direction": "cross_below",
        "threshold_range": [-0.15, 0.0],
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
# Transaction Cost Constants (imported from trade_engine concept)
# =============================================================================

DEFAULT_SLIPPAGE_BPS = 5  # 5 basis points (0.05%) per side
TOTAL_ROUND_TRIP_COST = 2 * DEFAULT_SLIPPAGE_BPS / 10000  # 0.1% total


# =============================================================================
# Signal Decay Tracking (I6 Fix)
# =============================================================================

def compute_signal_half_life(
    prices: pd.Series,
    signal: pd.Series,
    direction: str,
    threshold: float,
    max_lag: int = 60,
) -> int:
    """
    Compute signal half-life (how quickly signal predictive power decays).
    
    Signals that decay quickly (low half-life) should be acted on fast.
    Signals with long half-life are more robust for longer holds.
    
    Returns:
        int: Number of days until signal autocorrelation drops below 0.5
    """
    if len(prices) < max_lag + 60:
        return max_lag // 2  # Default to half of max
    
    # Compute returns at different lags
    autocorrs = []
    for lag in range(1, max_lag + 1):
        fwd_ret = prices.pct_change(lag).shift(-lag)
        
        # Identify when signal is active
        if direction == "below":
            active = signal < threshold
        elif direction == "above":
            active = signal > threshold
        else:
            active = signal < threshold
        
        # Average return when signal is active vs inactive
        active_rets = fwd_ret[active].dropna()
        inactive_rets = fwd_ret[~active].dropna()
        
        if len(active_rets) < 10 or len(inactive_rets) < 10:
            autocorrs.append(0)
            continue
        
        # Compute correlation of signal with future returns
        # Higher = signal still predictive
        signal_effect = active_rets.mean() - inactive_rets.mean()
        # Normalize by day-1 effect
        autocorrs.append(signal_effect)
    
    if len(autocorrs) < 2 or autocorrs[0] == 0:
        return max_lag // 2
    
    # Find half-life (when effect drops to 50% of initial)
    initial_effect = abs(autocorrs[0])
    half_effect = initial_effect * 0.5
    
    for i, effect in enumerate(autocorrs):
        if abs(effect) < half_effect:
            return i + 1
    
    return max_lag


def compute_signal_turnover_rate(
    signal: pd.Series,
    direction: str,
    threshold: float,
    lookback: int = 252,
) -> float:
    """
    Compute how often a signal flips (high turnover = more trading costs).
    
    Returns:
        float: Average number of signal flips per year (252 trading days)
    """
    if len(signal) < lookback:
        lookback = len(signal)
    
    recent = signal.iloc[-lookback:]
    
    if direction == "below":
        active = recent < threshold
    elif direction == "above":
        active = recent > threshold
    else:
        active = recent < threshold
    
    # Count state changes
    changes = (active != active.shift(1)).sum()
    
    # Annualize
    turnover_rate = changes / lookback * 252
    
    return float(turnover_rate)


# =============================================================================
# Backtesting and Optimization (Fixed for Look-Ahead Bias)
# =============================================================================


def _get_individual_trade_returns(
    prices: pd.Series,
    signal: pd.Series,
    threshold: float,
    direction: str,
    holding_days: int,
) -> list[float]:
    """
    Get individual trade returns (not averaged) for proper OOS aggregation.
    
    Returns a list of individual trade returns, each adjusted for transaction costs.
    Uses EDGE detection - only counts trades on the first day signal triggers.
    """
    # Identify signal triggers using EDGE detection
    # CRITICAL: Only trigger on the FIRST day a condition becomes true.
    if direction == "below":
        is_condition_true = signal < threshold
        was_condition_false = signal.shift(1) >= threshold
        triggers = is_condition_true & was_condition_false
    elif direction == "above":
        is_condition_true = signal > threshold
        was_condition_false = signal.shift(1) <= threshold
        triggers = is_condition_true & was_condition_false
    elif direction == "cross_above":
        prev_signal = signal.shift(1)
        triggers = (signal > threshold) & (prev_signal <= threshold)
    elif direction == "cross_below":
        prev_signal = signal.shift(1)
        triggers = (signal < threshold) & (prev_signal >= threshold)
    else:
        is_condition_true = signal < threshold
        was_condition_false = signal.shift(1) >= threshold
        triggers = is_condition_true & was_condition_false
    
    # Apply cooldown to prevent overlapping trades
    triggers = apply_cooldown_to_triggers(triggers, holding_days)
    
    # Compute forward returns with transaction costs
    future_prices = prices.shift(-holding_days)
    # Apply slippage: buy at higher, sell at lower
    entry_prices = prices * (1 + DEFAULT_SLIPPAGE_BPS / 10000)
    exit_prices = future_prices * (1 - DEFAULT_SLIPPAGE_BPS / 10000)
    fwd_ret = (exit_prices / entry_prices) - 1
    
    # Exclude last holding_days points
    fwd_ret.iloc[-holding_days:] = np.nan
    
    # Get individual returns where signal triggered
    signal_returns = fwd_ret[triggers].dropna()
    
    return signal_returns.tolist()


def backtest_signal(
    prices: pd.Series,
    signal: pd.Series,
    threshold: float,
    direction: str,
    holding_days: int,
) -> dict[str, float]:
    """
    Backtest a signal with specific parameters.
    
    FIXED: No look-ahead bias. Forward returns are computed correctly:
    - For signal on day t, return is (price[t+holding_days] - price[t]) / price[t]
    - We exclude the last `holding_days` signals since we can't know future returns
    
    Returns dict with win_rate, avg_return, max_return, min_return, n_signals.
    """
    # Identify signal triggers using EDGE detection
    # CRITICAL: Only trigger on the FIRST day a condition becomes true.
    # This models real trading behavior - you enter once on the signal, 
    # not re-enter every day the condition remains true.
    if direction == "below":
        # Edge: yesterday was >= threshold, today is < threshold
        is_condition_true = signal < threshold
        was_condition_false = signal.shift(1) >= threshold
        triggers = is_condition_true & was_condition_false
    elif direction == "above":
        # Edge: yesterday was <= threshold, today is > threshold
        is_condition_true = signal > threshold
        was_condition_false = signal.shift(1) <= threshold
        triggers = is_condition_true & was_condition_false
    elif direction == "cross_above":
        # Signal crosses above threshold from below
        prev_signal = signal.shift(1)
        triggers = (signal > threshold) & (prev_signal <= threshold)
    elif direction == "cross_below":
        # Signal crosses below threshold from above (event-based buy signal)
        prev_signal = signal.shift(1)
        triggers = (signal < threshold) & (prev_signal >= threshold)
    else:
        # Default to below edge detection
        is_condition_true = signal < threshold
        was_condition_false = signal.shift(1) >= threshold
        triggers = is_condition_true & was_condition_false
    
    # Apply cooldown: After entering a trade, wait for the holding period
    # before allowing another entry. This prevents "threshold whiplash"
    # where oscillating signals create multiple overlapping positions.
    triggers = apply_cooldown_to_triggers(triggers, holding_days)
    
    # FIXED: Compute forward returns correctly (no look-ahead bias)
    # Now includes transaction costs (slippage on entry and exit)
    future_prices = prices.shift(-holding_days)
    # Apply slippage: entry at slightly higher price, exit at slightly lower
    entry_prices = prices * (1 + DEFAULT_SLIPPAGE_BPS / 10000)
    exit_prices = future_prices * (1 - DEFAULT_SLIPPAGE_BPS / 10000)
    fwd_ret = (exit_prices / entry_prices) - 1
    
    # CRITICAL: Exclude the last holding_days points - we can't know those returns yet
    fwd_ret.iloc[-holding_days:] = np.nan
    
    # Get returns when signal triggered (only where we have complete forward data)
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


def _grid_search_params(
    prices: pd.Series,
    signal: pd.Series,
    direction: str,
    threshold_range: list[float],
    holding_days_options: list[int],
    min_signals: int,
) -> tuple[float, int, dict]:
    """
    Internal grid search for optimal parameters on a single data window.
    """
    best_score = -np.inf
    best_threshold = threshold_range[0]
    best_holding = 20
    best_results = {}
    
    n_threshold_steps = 5
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_threshold_steps)
    
    for threshold in thresholds:
        for holding_days in holding_days_options:
            results = backtest_signal(prices, signal, threshold, direction, holding_days)
            
            if results["n_signals"] < min_signals:
                continue
            
            # Score by Expected Value (EV) = win_rate × avg_return
            # REMOVED: signal_penalty that biased toward more frequent (lower quality) signals
            # The min_signals requirement already ensures statistical significance.
            # We want quality trades, not quantity of trades.
            score = results["win_rate"] * results["avg_return"] * 100
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_holding = holding_days
                best_results = results
    
    return best_threshold, best_holding, best_results


def optimize_signal_params_walkforward(
    prices: pd.Series,
    signal: pd.Series,
    direction: str,
    threshold_range: list[float],
    holding_days_options: list[int] = [10, 20, 40, 60, 90, 120],
    min_signals: int = 3,
    n_folds: int = 4,
    train_ratio: float = 0.7,
) -> tuple[float, int, dict, dict]:
    """
    Walk-forward optimization to avoid overfitting.
    
    Splits data into rolling train/test windows:
    - Train on 70% of each fold to find optimal params
    - Test on remaining 30% (out-of-sample) to get TRUE performance
    
    Returns: (optimal_threshold, optimal_holding_days, oos_results, stability_info)
    """
    n = len(prices)
    if n < 252:  # Need at least 1 year
        return threshold_range[0], 20, {}, {"is_stable": False}
    
    fold_size = n // n_folds
    oos_returns = []
    fold_params = []
    
    for fold in range(n_folds - 1):
        # Training window: from start to end of this fold
        train_end = fold_size * (fold + 2)
        train_size = int(train_end * train_ratio)
        
        # Test window: remaining 30% of training data
        test_start = train_size
        test_end = train_end
        
        if test_end - test_start < 60:
            continue
        
        # Optimize on training data only
        train_prices = prices.iloc[:train_size]
        train_signal = signal.iloc[:train_size]
        
        if len(train_prices) < 120:
            continue
            
        opt_thresh, opt_hold, _ = _grid_search_params(
            train_prices, train_signal, direction, 
            threshold_range, holding_days_options, min_signals
        )
        fold_params.append((opt_thresh, opt_hold))
        
        # Test on out-of-sample window with the optimized params
        test_prices = prices.iloc[test_start:test_end]
        test_signal = signal.iloc[test_start:test_end]
        
        oos_result = backtest_signal(
            test_prices, test_signal, opt_thresh, direction, opt_hold
        )
        
        if oos_result["n_signals"] > 0:
            # FIXED: Get actual individual trade returns, not averaged proxies
            # Re-compute to get individual returns
            individual_returns = _get_individual_trade_returns(
                test_prices, test_signal, opt_thresh, direction, opt_hold
            )
            oos_returns.extend(individual_returns)
    
    # Aggregate OOS performance (this is the TRUE expected performance)
    if len(oos_returns) < 3:
        return threshold_range[0], 20, {
            "win_rate": 0.0, "avg_return": 0.0, "max_return": 0.0,
            "min_return": 0.0, "n_signals": 0, "is_oos": True
        }, {"is_stable": False}
    
    oos_returns_arr = np.array(oos_returns)
    oos_results = {
        "win_rate": float((oos_returns_arr > 0).mean()),
        "avg_return": float(oos_returns_arr.mean()),
        "max_return": float(oos_returns_arr.max()),
        "min_return": float(oos_returns_arr.min()),
        "n_signals": len(oos_returns),
        "is_oos": True,  # Flag that this is out-of-sample
    }
    
    # Parameter stability analysis
    if len(fold_params) >= 2:
        thresh_values = [p[0] for p in fold_params]
        hold_values = [p[1] for p in fold_params]
        thresh_mean = np.mean(thresh_values)
        thresh_std = np.std(thresh_values)
        hold_std = np.std(hold_values)
        
        # Stable if threshold variance is <30% of mean
        is_stable = (thresh_std / abs(thresh_mean) < 0.3) if thresh_mean != 0 else True
        
        stability_info = {
            "is_stable": is_stable,
            "threshold_cv": thresh_std / abs(thresh_mean) if thresh_mean != 0 else 0,
            "holding_std": hold_std,
            "n_folds": len(fold_params),
        }
    else:
        stability_info = {"is_stable": False, "n_folds": len(fold_params)}
    
    # Use the most recent fold's parameters as the "optimal" ones
    if fold_params:
        final_thresh, final_hold = fold_params[-1]
    else:
        final_thresh, final_hold = threshold_range[0], 20
    
    return final_thresh, final_hold, oos_results, stability_info


def optimize_signal_params(
    prices: pd.Series,
    signal: pd.Series,
    direction: str,
    threshold_range: list[float],
    holding_days_options: list[int] = [10, 20, 40, 60, 90, 120],
    min_signals: int = 5,
    use_walkforward: bool = True,
) -> tuple[float, int, dict]:
    """
    Find optimal threshold and holding period for a signal.
    
    Uses walk-forward validation by default to avoid overfitting.
    Set use_walkforward=False for legacy in-sample optimization (NOT recommended).
    
    Returns: (optimal_threshold, optimal_holding_days, backtest_results)
    """
    if use_walkforward and len(prices) >= 252:
        # Use walk-forward validation (recommended)
        opt_thresh, opt_hold, oos_results, stability = optimize_signal_params_walkforward(
            prices, signal, direction, threshold_range, holding_days_options, min_signals
        )
        
        # Add stability info to results
        oos_results["is_stable"] = stability.get("is_stable", False)
        return opt_thresh, opt_hold, oos_results
    
    # Fallback to in-sample grid search for short data
    return _grid_search_params(
        prices, signal, direction, threshold_range, holding_days_options, min_signals
    )


def evaluate_signal_for_stock(
    price_data: dict[str, pd.Series],
    signal_config: dict,
    holding_days_options: list[int] = [10, 20, 40, 60, 90, 120],
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
            # For crossover, check if just crossed above
            prev_value = signal_values.iloc[-2] if len(signal_values) > 1 else current_value
            is_buy = (current_value > opt_threshold) and (prev_value <= opt_threshold)
            strength = 1.0 if is_buy else 0.0
        elif direction == "cross_below":
            # For crossover, check if just crossed below (event-based buy signal)
            prev_value = signal_values.iloc[-2] if len(signal_values) > 1 else current_value
            is_buy = (current_value < opt_threshold) and (prev_value >= opt_threshold)
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


# =============================================================================
# Correlation-Aware Signal Weighting (S3 Fix)
# =============================================================================

def _compute_correlation_adjusted_weights(
    price_data: dict[str, pd.Series],
    active_signals: list,
) -> dict[str, float]:
    """
    Compute weights that downweight correlated signals.
    
    If RSI and Stochastic are both triggered and highly correlated,
    we don't want to double-count their contribution.
    
    Returns dict mapping signal name to weight (0 to 1).
    """
    if len(active_signals) <= 1:
        return {s.name: 1.0 for s in active_signals}
    
    prices = price_data.get("close")
    if prices is None or len(prices) < 60:
        return {s.name: 1.0 for s in active_signals}
    
    # Compute signal trigger series for each active signal
    signal_triggers = {}
    
    for sig in active_signals:
        config = None
        for cfg in SIGNAL_CONFIGS:
            if cfg["name"] == sig.name:
                config = cfg
                break
        
        if config is None:
            signal_triggers[sig.name] = pd.Series(0, index=prices.index)
            continue
        
        try:
            values = config["compute"](price_data)
            threshold = sig.optimal_threshold
            direction = config["direction"]
            
            if direction == "below":
                trigger = (values < threshold).astype(float)
            elif direction == "above":
                trigger = (values > threshold).astype(float)
            elif direction == "cross_above":
                prev = values.shift(1)
                trigger = ((values > threshold) & (prev <= threshold)).astype(float)
            elif direction == "cross_below":
                prev = values.shift(1)
                trigger = ((values < threshold) & (prev >= threshold)).astype(float)
            else:
                trigger = (values < threshold).astype(float)
            
            signal_triggers[sig.name] = trigger
        except Exception:
            signal_triggers[sig.name] = pd.Series(0, index=prices.index)
    
    # Compute pairwise correlations
    if len(signal_triggers) < 2:
        return {s.name: 1.0 for s in active_signals}
    
    trigger_df = pd.DataFrame(signal_triggers)
    corr_matrix = trigger_df.corr().fillna(0)
    
    # Compute weight as 1 / (1 + sum of correlations with other signals)
    # Higher correlation with others = lower weight
    weights = {}
    for sig_name in corr_matrix.columns:
        # Sum of absolute correlations with OTHER signals
        other_corr = corr_matrix[sig_name].drop(sig_name).abs().sum()
        n_others = len(corr_matrix) - 1
        avg_corr = other_corr / n_others if n_others > 0 else 0
        
        # Weight inversely proportional to avg correlation
        # High correlation (0.8) -> weight ~0.55
        # Low correlation (0.2) -> weight ~0.83
        weight = 1.0 / (1.0 + avg_corr)
        weights[sig_name] = weight
    
    # Normalize so max weight is 1.0
    max_weight = max(weights.values()) if weights else 1.0
    weights = {k: v / max_weight for k, v in weights.items()}
    
    return weights


def analyze_stock(
    symbol: str,
    name: str,
    price_data: dict[str, pd.Series],
    holding_days_options: list[int] = [10, 20, 40, 60, 90, 120],
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
    
    # FIXED (S3): Correlation-aware buy score calculation
    # Downweight signals that are correlated to avoid double-counting
    buy_score = 0.0
    if active_signals:
        # Compute signal correlations based on their trigger patterns
        signal_weights = _compute_correlation_adjusted_weights(price_data, active_signals)
        
        for i, sig in enumerate(active_signals):
            # Score contribution = strength * win_rate * avg_return (capped)
            # Now weighted by independence from other signals
            raw_contribution = sig.signal_strength * sig.win_rate * min(sig.avg_return_pct, 20) / 20
            weight = signal_weights.get(sig.name, 1.0)
            contribution = raw_contribution * weight
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
    holding_days_options: list[int] = [10, 20, 40, 60, 90, 120],
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
# Historical Signal Triggers (for Chart Markers)
# =============================================================================


@dataclass
class SignalTrigger:
    """A historical signal trigger point."""
    date: str  # ISO date string
    signal_name: str
    price: float
    win_rate: float
    avg_return_pct: float
    holding_days: int
    drawdown_pct: float = 0.0  # The threshold that triggered (for drawdown signals)
    signal_type: str = "entry"  # "entry" for buy signals, "exit" for sell signals


def get_historical_triggers(
    price_data: dict[str, pd.Series],
    lookback_days: int = 365,
    min_signals: int = 3,  # Reduced minimum for chart markers
) -> list[SignalTrigger]:
    """
    Get historical signal trigger points for chart markers.
    
    IMPORTANT: Only returns triggers for the TOP/BEST signal for this stock.
    The best signal is determined by expected value (win_rate * avg_return).
    
    This uses ONLY past data at each point - no look-ahead bias.
    The backtest statistics are computed on the FULL available history,
    which is the same approach used for current signal evaluation.
    
    Returns list of SignalTrigger sorted by date.
    """
    prices = price_data.get("close")
    if prices is None or len(prices) < 60:  # Need at least 60 days
        return []
    
    # First, find the BEST signal for this stock
    best_signal_config = None
    best_signal_ev = -float("inf")
    best_signal_bt = None
    best_signal_threshold = None
    
    for config in SIGNAL_CONFIGS:
        try:
            signal_values = config["compute"](price_data)
            if signal_values.isna().all():
                continue
            
            direction = config["direction"]
            
            # Optimize parameters for this signal
            opt_threshold, opt_holding, opt_results = optimize_signal_params(
                prices,
                signal_values,
                direction,
                config["threshold_range"],
                holding_days_options=[10, 20, 40, 60, 90, 120],
            )
            
            if opt_results.get("n_signals", 0) < min_signals:
                continue
            
            # Expected value = win_rate * avg_return
            ev = opt_results["win_rate"] * opt_results["avg_return"] * 100
            
            if ev > best_signal_ev:
                best_signal_ev = ev
                best_signal_config = config
                best_signal_bt = opt_results
                best_signal_threshold = opt_threshold
                
        except Exception as e:
            logger.warning(f"Error evaluating signal {config['name']}: {e}")
            continue
    
    if best_signal_config is None or best_signal_bt is None:
        return []
    
    # Now get trigger points for the BEST signal only
    triggers: list[SignalTrigger] = []
    
    try:
        signal_values = best_signal_config["compute"](price_data)
        direction = best_signal_config["direction"]
        threshold = best_signal_threshold
        
        # Find trigger points using the OPTIMIZED threshold
        # CRITICAL: Use EDGE detection for all directions - only trigger on the 
        # FIRST day of condition becoming true, not every day it's true.
        # This converts "state conditions" into "event signals".
        
        if direction == "below":
            # Edge detection: yesterday was >= threshold, today is < threshold
            is_condition_true = signal_values < threshold
            was_condition_false = signal_values.shift(1) >= threshold
            is_triggered = is_condition_true & was_condition_false
        elif direction == "above":
            # Edge detection: yesterday was <= threshold, today is > threshold
            is_condition_true = signal_values > threshold
            was_condition_false = signal_values.shift(1) <= threshold
            is_triggered = is_condition_true & was_condition_false
        elif direction == "cross_above":
            prev_signal = signal_values.shift(1)
            is_triggered = (signal_values > threshold) & (prev_signal <= threshold)
        elif direction == "cross_below":
            prev_signal = signal_values.shift(1)
            is_triggered = (signal_values < threshold) & (prev_signal >= threshold)
        else:
            # Default to below edge detection
            is_condition_true = signal_values < threshold
            was_condition_false = signal_values.shift(1) >= threshold
            is_triggered = is_condition_true & was_condition_false
        
        # Optimal holding period from backtest
        optimal_holding = best_signal_bt.get("holding_days", 20)
        
        # Apply cooldown: Don't trigger again until we've exited the previous position
        # This prevents "threshold whiplash" (e.g., drawdown oscillating around -25%)
        is_triggered = apply_cooldown_to_triggers(is_triggered, optimal_holding)
        
        # Get trigger dates in lookback period
        trigger_mask = is_triggered.iloc[-lookback_days:]
        trigger_dates = trigger_mask[trigger_mask].index
        
        # Add entry triggers with historical stats from the best signal's backtest
        for date_idx in trigger_dates:
            price_at_trigger = float(prices.loc[date_idx])
            triggers.append(SignalTrigger(
                date=str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx),
                signal_name=best_signal_config["name"],
                price=price_at_trigger,
                win_rate=best_signal_bt["win_rate"],
                avg_return_pct=best_signal_bt["avg_return"] * 100,
                holding_days=optimal_holding,
                drawdown_pct=abs(threshold) if "Drawdown" in best_signal_config["name"] else 0.0,
                signal_type="entry",
            ))
            
            # Add corresponding exit trigger based on holding period
            try:
                exit_idx = date_idx + pd.Timedelta(days=optimal_holding)
                # Find the nearest trading day in our price data
                exit_prices = prices[prices.index >= exit_idx]
                if len(exit_prices) > 0:
                    exit_date = exit_prices.index[0]
                    exit_price = float(exit_prices.iloc[0])
                    entry_return = ((exit_price / price_at_trigger) - 1) * 100
                    triggers.append(SignalTrigger(
                        date=str(exit_date.date()) if hasattr(exit_date, "date") else str(exit_date),
                        signal_name=f"Exit: {best_signal_config['name']}",
                        price=exit_price,
                        win_rate=best_signal_bt["win_rate"],
                        avg_return_pct=entry_return,  # Actual return for this specific trade
                        holding_days=optimal_holding,
                        drawdown_pct=0.0,
                        signal_type="exit",
                    ))
            except Exception:
                pass  # Skip exits that fall outside available data
    except Exception as e:
        logger.warning(f"Error computing triggers for best signal: {e}")
    
    # Sort by date
    triggers.sort(key=lambda t: t.date)
    
    return triggers


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
