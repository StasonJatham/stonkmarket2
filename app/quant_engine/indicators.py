"""
Shared Technical Indicators Module

This module provides standardized technical indicator computation used by both:
- backtest_engine.py (batch optimization)
- trade_engine.py (real-time analysis)

Extracted to avoid duplication and ensure consistent indicator calculations.
"""

import numpy as np
import pandas as pd


# =============================================================================
# Price Data Preparation
# =============================================================================

def prepare_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare price DataFrame for analysis.
    
    - Converts column names to lowercase
    - Uses Adj Close as the primary "close" column (for dividends/splits)
    - Ensures all required columns exist
    
    Args:
        df: Raw price DataFrame (may have mixed case columns)
        
    Returns:
        Cleaned DataFrame with lowercase columns, using adj_close for close
    """
    df = df.copy()
    
    # Handle multi-level columns from yfinance
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    
    # Convert to lowercase
    col_map = {str(c): str(c).lower().replace(' ', '_') for c in df.columns}
    df = df.rename(columns=col_map)
    
    # Use Adj Close as the primary close (accounts for dividends/splits)
    if 'adj_close' in df.columns and df['adj_close'].notna().any():
        df['close'] = df['adj_close'].combine_first(df.get('close', df['adj_close']))
    
    # Ensure required columns exist
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' or 'adj_close' column")
    
    # Fill missing OHLV from close if needed
    if 'open' not in df.columns:
        df['open'] = df['close']
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']
    if 'volume' not in df.columns:
        df['volume'] = 1
    
    return df


# =============================================================================
# Technical Indicators (using ta library)
# =============================================================================

def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators.
    
    Uses the 'ta' library for standardized, tested implementations.
    NOTE: Automatically normalizes price data using prepare_price_dataframe.
    
    Indicators computed:
    - Moving Averages: SMA (5, 10, 20, 50, 200), EMA (12, 26)
    - Momentum: RSI (7, 14, 21), MACD, Stochastic, Momentum
    - Volatility: Bollinger Bands, ATR, Z-scores
    - Trend: ADX, Golden/Death cross signals
    - Price-relative: vs MA levels, drawdowns
    
    Args:
        prices: DataFrame with OHLCV data (or just close)
        
    Returns:
        DataFrame with all indicators added as columns
    """
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    
    # Normalize price data (use adj_close, lowercase columns)
    df = prepare_price_dataframe(prices)
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1, index=close.index))
    
    # Moving Averages
    df["sma_5"] = SMAIndicator(close, window=5).sma_indicator()
    df["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close, window=200).sma_indicator()
    df["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close, window=26).ema_indicator()
    
    # RSI
    df["rsi_7"] = RSIIndicator(close, window=7).rsi()
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    df["rsi_21"] = RSIIndicator(close, window=21).rsi()
    
    # MACD
    macd = MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()  # 0-1, where 0 = lower band, 1 = upper
    
    # Stochastic
    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # ADX (trend strength)
    adx = ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    
    # ATR (volatility)
    atr = AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / close  # ATR as % of price
    
    # Price-based
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    
    # Drawdown from peak
    rolling_max = close.rolling(252, min_periods=1).max()
    df["drawdown"] = (close - rolling_max) / rolling_max
    df["drawdown_20"] = (close - close.rolling(20).max()) / close.rolling(20).max()
    df["drawdown_50"] = (close - close.rolling(50).max()) / close.rolling(50).max()
    
    # Z-scores
    df["zscore_10"] = (close - close.rolling(10).mean()) / close.rolling(10).std()
    df["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    df["zscore_50"] = (close - close.rolling(50).mean()) / close.rolling(50).std()
    
    # Momentum
    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)
    df["momentum_20"] = close.pct_change(20)
    
    # Volume (if available)
    if volume.sum() > len(volume):  # Not all 1s
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]
    
    # Price relative to MAs
    df["price_vs_sma_20"] = (close - df["sma_20"]) / df["sma_20"]
    df["price_vs_sma_50"] = (close - df["sma_50"]) / df["sma_50"]
    df["price_vs_sma_200"] = (close - df["sma_200"]) / df["sma_200"]
    
    # Golden/Death cross signals
    df["sma_20_above_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
    df["sma_50_above_200"] = (df["sma_50"] > df["sma_200"]).astype(int)
    
    return df


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_confidence_interval(
    returns: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    metric: str = "mean",
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters:
        returns: List of trade returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        metric: "mean", "median", "sharpe", or "win_rate"
    
    Returns:
        tuple: (point_estimate, ci_lower, ci_upper)
    """
    if len(returns) < 5:
        return 0.0, 0.0, 0.0
    
    returns_arr = np.array(returns)
    n = len(returns_arr)
    
    # Generate bootstrap samples
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns_arr, size=n, replace=True)
        
        if metric == "mean":
            stat = np.mean(sample)
        elif metric == "median":
            stat = np.median(sample)
        elif metric == "sharpe":
            if np.std(sample) > 0:
                stat = np.mean(sample) / np.std(sample) * np.sqrt(12)  # Annualized
            else:
                stat = 0
        elif metric == "win_rate":
            stat = (sample > 0).mean()
        else:
            stat = np.mean(sample)
        
        boot_stats.append(stat)
    
    boot_stats_arr = np.array(boot_stats)
    
    # Compute percentiles
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats_arr, alpha * 100))
    ci_upper = float(np.percentile(boot_stats_arr, (1 - alpha) * 100))
    
    # Point estimate
    if metric == "mean":
        point = float(np.mean(returns_arr))
    elif metric == "median":
        point = float(np.median(returns_arr))
    elif metric == "sharpe":
        point = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(12)) if np.std(returns_arr) > 0 else 0
    elif metric == "win_rate":
        point = float((returns_arr > 0).mean())
    else:
        point = float(np.mean(returns_arr))
    
    return point, ci_lower, ci_upper


# =============================================================================
# Simple Indicator Functions (for signals.py compatibility)
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
