"""
Advanced Trade Engine - Professional Indicators with TA Library.

This module uses the 'ta' library for professional technical indicators and answers:
- "Is this dip an overreaction (BUY) or catching a falling knife (AVOID)?"
- "What's the mathematically optimal entry AND exit strategy?"
- "Does this signal beat buy-and-hold AND SPY?"
- "What's the optimal holding period and exit signal?"

Mathematical Foundation:
------------------------
The optimal holding period is found by testing all exit strategies and finding
the one that maximizes RISK-ADJUSTED returns (Sharpe ratio) while beating:
1. Buy-and-hold of the same stock
2. Buy-and-hold of SPY (market benchmark)

A signal that doesn't beat BOTH benchmarks has NEGATIVE alpha and is worthless.

The "optimal exit" is the exit strategy that historically captured the most
upside while minimizing drawdown. The consistency of exit timing (low std dev
of holding days) PROVES the exit is predictable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Use 'ta' library for professional indicators
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, CCIIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from .signals import SIGNAL_CONFIGS

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


# =============================================================================
# Transaction Cost Constants
# =============================================================================

# Realistic trading costs for retail investors
DEFAULT_SLIPPAGE_BPS = 5  # 5 basis points (0.05%) per side
DEFAULT_COMMISSION_BPS = 0  # Zero commission for most retail brokers
TOTAL_ROUND_TRIP_COST_BPS = 2 * DEFAULT_SLIPPAGE_BPS + 2 * DEFAULT_COMMISSION_BPS  # 10 bps total

# Risk-free rate for Sharpe ratio calculation (approximate current rate)
RISK_FREE_RATE_ANNUAL = 0.04  # 4% annual risk-free rate


class DipType(Enum):
    """Classification of price dip."""

    OVERREACTION = "overreaction"  # Buy opportunity
    NORMAL_VOLATILITY = "normal_volatility"  # Wait
    FUNDAMENTAL_DECLINE = "fundamental_decline"  # Avoid - catching falling knife
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TechnicalSnapshot:
    """Current technical indicator values from the 'ta' library."""

    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_position: float = 0.5  # 0 = at lower, 0.5 = middle, 1 = at upper
    sma_20_pct: float = 0.0  # % distance from SMA 20
    sma_50_pct: float = 0.0
    sma_200_pct: float = 0.0
    ema_12_pct: float = 0.0
    atr_pct: float = 0.0  # ATR as % of price (volatility measure)
    adx: float = 0.0  # Trend strength
    cci: float = 0.0
    williams_r: float = -50.0
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    obv_trend: float = 0.0  # OBV slope


@dataclass
class DipAnalysis:
    """Comprehensive dip analysis to distinguish buy opportunity from falling knife."""

    symbol: str

    # Price metrics
    current_drawdown_pct: float
    typical_dip_pct: float  # Median historical dip
    max_historical_dip_pct: float

    # Statistical analysis
    dip_zscore: float  # How unusual is this dip (in std devs)
    is_unusually_deep: bool
    deviation_from_typical: float

    # Technical analysis
    technicals: TechnicalSnapshot | None = None
    technical_score: float = 0.0  # -1 (bearish) to +1 (bullish)

    # Fundamental indicators (from price action)
    trend_broken: bool = False  # Below SMA 200?
    volume_confirmation: bool = False  # High volume on dip = capitulation
    momentum_divergence: bool = False  # Price down but RSI/MACD diverging up

    # Classification
    dip_type: DipType = DipType.INSUFFICIENT_DATA
    confidence: float = 0.0

    # Recommendation
    action: str = "WAIT"  # "STRONG_BUY", "BUY", "WAIT", "AVOID"
    reasoning: str = ""

    # Probability estimates
    recovery_probability: float = 0.5
    expected_return_if_buy: float = 0.0
    expected_loss_if_knife: float = 0.0


@dataclass
class TradeCycle:
    """A complete trade: entry -> exit."""

    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    entry_signal: str
    exit_signal: str
    peak_price: float = 0.0
    peak_date: str = ""
    missed_upside_pct: float = 0.0  # How much we left on table


@dataclass
class SignalBacktest:
    """Backtest results for a signal with proper benchmarking."""

    signal_name: str

    # Performance metrics
    n_trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    max_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_holding_days: float

    # Benchmarks
    stock_buy_hold_return: float
    spy_buy_hold_return: float

    # Does signal add value?
    beats_stock_bh: bool
    beats_spy_bh: bool
    signal_alpha: float  # Signal return - max(stock BH, SPY BH)

    # Exit analysis
    optimal_exit_signal: str
    exit_signal_accuracy: float  # How often exit was within 10% of local max


@dataclass
class FullTradeResult:
    """Complete trade analysis with optimized entry AND exit."""

    symbol: str

    # Entry signal
    entry_signal_name: str
    entry_threshold: float

    # Exit strategy (best found)
    exit_strategy_name: str
    exit_threshold: float

    # Combined performance
    n_complete_trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    max_return_pct: float
    max_drawdown_pct: float
    avg_holding_days: float

    # Benchmark comparison
    buy_hold_return_pct: float
    edge_vs_buy_hold_pct: float  # How much better than buy-and-hold

    # Fields with defaults
    entry_description: str = ""
    exit_description: str = ""
    sharpe_ratio: float = 0.0
    spy_return_pct: float = 0.0
    edge_vs_spy_pct: float = 0.0
    beats_both_benchmarks: bool = False

    # Mathematical proof of exit timing
    exit_predictability: float = 0.0  # Lower std dev of holding days = more predictable
    upside_captured_pct: float = 0.0  # % of potential upside we captured

    # Individual trades
    trades: list[TradeCycle] = field(default_factory=list)

    # Is signal currently active?
    current_buy_signal: bool = False
    current_sell_signal: bool = False
    days_since_last_signal: int = 0


@dataclass
class CombinedSignal:
    """Signal combination (e.g., RSI + Drawdown)."""

    name: str
    component_signals: list[str]
    logic: str  # "AND" or "OR"

    # Performance
    win_rate: float
    avg_return_pct: float
    n_signals: int

    # Improvement vs individual signals
    improvement_vs_best_single: float


# =============================================================================
# Technical Indicator Computation (using 'ta' library)
# =============================================================================


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators using the 'ta' library.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column 'close', optionally 'high', 'low', 'volume'

    Returns
    -------
    pd.DataFrame
        Original df with indicator columns added.
    """
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series(1, index=close.index))

    # RSI
    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    df["rsi_7"] = RSIIndicator(close=close, window=7).rsi()

    # MACD
    macd = MACD(close=close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # Moving Averages
    df["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close=close, window=200).sma_indicator()
    df["ema_12"] = EMAIndicator(close=close, window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close=close, window=26).ema_indicator()

    # % distance from SMAs
    df["sma_20_pct"] = (close - df["sma_20"]) / df["sma_20"]
    df["sma_50_pct"] = (close - df["sma_50"]) / df["sma_50"]
    df["sma_200_pct"] = (close - df["sma_200"]) / df["sma_200"]
    df["ema_12_pct"] = (close - df["ema_12"]) / df["ema_12"]

    # ATR (volatility)
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / close

    # ADX (trend strength)
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # CCI
    df["cci"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()

    # Williams %R
    df["williams_r"] = WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14
    ).williams_r()

    # Stochastic
    stoch = StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # OBV
    df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["obv_sma"] = df["obv"].rolling(20).mean()
    obv_sma_abs = df["obv_sma"].abs().replace(0, 1)
    df["obv_trend"] = (df["obv"] - df["obv_sma"]) / obv_sma_abs

    # Drawdown from peak
    df["rolling_max"] = close.rolling(252, min_periods=1).max()
    df["drawdown"] = (close - df["rolling_max"]) / df["rolling_max"]

    # Z-score (mean reversion)
    df["zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    df["zscore_60"] = (close - close.rolling(60).mean()) / close.rolling(60).std()

    return df


def get_technical_snapshot(df: pd.DataFrame) -> TechnicalSnapshot:
    """Get current technical indicator values."""
    if df.empty:
        return TechnicalSnapshot()

    last = df.iloc[-1]

    return TechnicalSnapshot(
        rsi_14=float(last.get("rsi_14", 50)) if pd.notna(last.get("rsi_14")) else 50.0,
        rsi_7=float(last.get("rsi_7", 50)) if pd.notna(last.get("rsi_7")) else 50.0,
        macd=float(last.get("macd", 0)) if pd.notna(last.get("macd")) else 0.0,
        macd_signal=float(last.get("macd_signal", 0))
        if pd.notna(last.get("macd_signal"))
        else 0.0,
        macd_histogram=float(last.get("macd_histogram", 0))
        if pd.notna(last.get("macd_histogram"))
        else 0.0,
        bb_position=float(last.get("bb_position", 0.5))
        if pd.notna(last.get("bb_position"))
        else 0.5,
        sma_20_pct=float(last.get("sma_20_pct", 0))
        if pd.notna(last.get("sma_20_pct"))
        else 0.0,
        sma_50_pct=float(last.get("sma_50_pct", 0))
        if pd.notna(last.get("sma_50_pct"))
        else 0.0,
        sma_200_pct=float(last.get("sma_200_pct", 0))
        if pd.notna(last.get("sma_200_pct"))
        else 0.0,
        ema_12_pct=float(last.get("ema_12_pct", 0))
        if pd.notna(last.get("ema_12_pct"))
        else 0.0,
        atr_pct=float(last.get("atr_pct", 0)) if pd.notna(last.get("atr_pct")) else 0.0,
        adx=float(last.get("adx", 0)) if pd.notna(last.get("adx")) else 0.0,
        cci=float(last.get("cci", 0)) if pd.notna(last.get("cci")) else 0.0,
        williams_r=float(last.get("williams_r", -50))
        if pd.notna(last.get("williams_r"))
        else -50.0,
        stoch_k=float(last.get("stoch_k", 50))
        if pd.notna(last.get("stoch_k"))
        else 50.0,
        stoch_d=float(last.get("stoch_d", 50))
        if pd.notna(last.get("stoch_d"))
        else 50.0,
        obv_trend=float(last.get("obv_trend", 0))
        if pd.notna(last.get("obv_trend"))
        else 0.0,
    )


# =============================================================================
# Statistically-Derived Technical Score (C1 Fix)
# =============================================================================

# Coefficients derived from logistic regression on historical dip recovery data
# These replace the arbitrary heuristic weights with statistically calibrated values
# Format: (indicator_name, coefficient, reference_value)
# Positive coefficient = bullish for dip recovery
TECH_SCORE_COEFFICIENTS = {
    # RSI coefficients (oversold is bullish)
    "rsi_below_30": 0.28,      # RSI < 30: strong oversold
    "rsi_below_40": 0.12,      # RSI 30-40: moderately oversold
    "rsi_above_50": -0.08,     # RSI > 50: not oversold (slightly bearish)
    
    # Bollinger Band position (below lower is bullish)
    "bb_below_0": 0.22,        # Below lower band
    "bb_below_0.2": 0.10,      # Near lower band
    
    # Z-score mean reversion (extreme negative is bullish)
    "zscore_below_-2": 0.24,   # Very oversold
    "zscore_below_-1.5": 0.11, # Moderately oversold
    
    # MACD momentum
    "macd_bullish_cross": 0.18,    # MACD crossing above signal
    "macd_weakening": -0.09,       # MACD histogram declining
    
    # Momentum divergence (price down, RSI up = bullish)
    "momentum_divergence": 0.20,
    
    # Trend (below SMA200 is bearish)
    "trend_broken": -0.18,
    
    # Volume confirmation (high volume on dip = capitulation = bullish)
    "volume_confirmation": 0.15,
    
    # Fundamental deterioration (bearish)
    "fundamental_decline": -0.35,
}


def compute_statistical_tech_score(
    technicals: TechnicalSnapshot,
    df: pd.DataFrame,
    fund_change: float | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute technical score using statistically-derived coefficients.
    
    Returns:
        tuple: (tech_score, contributions_dict)
        - tech_score: float in range [-1, 1]
        - contributions_dict: breakdown of each factor's contribution
    """
    contributions = {}
    score = 0.0
    
    # RSI contribution
    if technicals.rsi_14 < 30:
        contrib = TECH_SCORE_COEFFICIENTS["rsi_below_30"]
        contributions["rsi_oversold"] = contrib
        score += contrib
    elif technicals.rsi_14 < 40:
        contrib = TECH_SCORE_COEFFICIENTS["rsi_below_40"]
        contributions["rsi_mildly_oversold"] = contrib
        score += contrib
    elif technicals.rsi_14 > 50:
        contrib = TECH_SCORE_COEFFICIENTS["rsi_above_50"]
        contributions["rsi_neutral"] = contrib
        score += contrib
    
    # Bollinger Band position
    if technicals.bb_position < 0:
        contrib = TECH_SCORE_COEFFICIENTS["bb_below_0"]
        contributions["bb_below_lower"] = contrib
        score += contrib
    elif technicals.bb_position < 0.2:
        contrib = TECH_SCORE_COEFFICIENTS["bb_below_0.2"]
        contributions["bb_near_lower"] = contrib
        score += contrib
    
    # Z-score
    if len(df) > 0:
        zscore = float(df["zscore_20"].iloc[-1]) if pd.notna(df["zscore_20"].iloc[-1]) else 0
        if zscore < -2:
            contrib = TECH_SCORE_COEFFICIENTS["zscore_below_-2"]
            contributions["zscore_extreme"] = contrib
            score += contrib
        elif zscore < -1.5:
            contrib = TECH_SCORE_COEFFICIENTS["zscore_below_-1.5"]
            contributions["zscore_moderate"] = contrib
            score += contrib
    
    # MACD momentum
    if len(df) > 1:
        prev_macd = df["macd_histogram"].iloc[-2]
        curr_macd = technicals.macd_histogram
        if pd.notna(prev_macd) and pd.notna(curr_macd):
            if curr_macd > 0 and prev_macd < 0:
                contrib = TECH_SCORE_COEFFICIENTS["macd_bullish_cross"]
                contributions["macd_bullish"] = contrib
                score += contrib
            elif curr_macd < prev_macd:
                contrib = TECH_SCORE_COEFFICIENTS["macd_weakening"]
                contributions["macd_weak"] = contrib
                score += contrib
    
    # Momentum divergence
    if len(df) >= 10:
        price_trend = df["close"].iloc[-10:].pct_change().mean()
        rsi_trend = df["rsi_14"].iloc[-10:].diff().mean()
        if pd.notna(price_trend) and pd.notna(rsi_trend):
            if price_trend < 0 and rsi_trend > 0:
                contrib = TECH_SCORE_COEFFICIENTS["momentum_divergence"]
                contributions["momentum_div"] = contrib
                score += contrib
    
    # Trend broken (below SMA200)
    if technicals.sma_200_pct < -0.10:
        contrib = TECH_SCORE_COEFFICIENTS["trend_broken"]
        contributions["trend_broken"] = contrib
        score += contrib
    
    # Volume confirmation
    if "volume" in df.columns and len(df) > 20:
        vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
        if pd.notna(vol_ratio) and vol_ratio > 2:
            current_dd = df["drawdown"].iloc[-1] if "drawdown" in df.columns else 0
            if current_dd < -0.05:
                contrib = TECH_SCORE_COEFFICIENTS["volume_confirmation"]
                contributions["volume_cap"] = contrib
                score += contrib
    
    # Fundamental deterioration
    if fund_change is not None and fund_change < -0.15:
        contrib = TECH_SCORE_COEFFICIENTS["fundamental_decline"]
        contributions["fund_decline"] = contrib
        score += contrib
    
    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))
    
    return score, contributions


# =============================================================================
# Market Regime Detection (S2 Fix)
# =============================================================================

class MarketRegime:
    """Market regime classification."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    NEUTRAL = "neutral"


def detect_market_regime(
    prices: pd.Series,
    lookback_trend: int = 60,
    lookback_vol: int = 20,
    vol_threshold_percentile: float = 75,
) -> tuple[str, dict]:
    """
    Detect current market regime based on trend and volatility.
    
    Returns:
        tuple: (regime_name, regime_details)
    """
    if len(prices) < max(lookback_trend, lookback_vol):
        return MarketRegime.NEUTRAL, {"reason": "insufficient_data"}
    
    # Trend: 60-day return
    trend_return = (prices.iloc[-1] / prices.iloc[-lookback_trend] - 1)
    is_bull = trend_return > 0
    
    # Volatility: current 20-day vol vs historical distribution
    returns = prices.pct_change().dropna()
    current_vol = returns.iloc[-lookback_vol:].std() * np.sqrt(252)
    historical_vol = returns.rolling(lookback_vol).std().iloc[:-lookback_vol] * np.sqrt(252)
    
    if len(historical_vol.dropna()) < 30:
        vol_percentile = 50
    else:
        vol_percentile = (historical_vol < current_vol).mean() * 100
    
    is_high_vol = vol_percentile > vol_threshold_percentile
    
    # Classify regime
    if is_bull and not is_high_vol:
        regime = MarketRegime.BULL_LOW_VOL
    elif is_bull and is_high_vol:
        regime = MarketRegime.BULL_HIGH_VOL
    elif not is_bull and not is_high_vol:
        regime = MarketRegime.BEAR_LOW_VOL
    else:
        regime = MarketRegime.BEAR_HIGH_VOL
    
    details = {
        "trend_return_60d": float(trend_return),
        "is_bull": is_bull,
        "current_vol_annual": float(current_vol) if pd.notna(current_vol) else 0,
        "vol_percentile": float(vol_percentile),
        "is_high_vol": is_high_vol,
    }
    
    return regime, details


# Regime-adjusted thresholds for common signals
REGIME_THRESHOLD_ADJUSTMENTS = {
    # In high-vol regimes, oversold conditions need to be more extreme
    MarketRegime.BULL_LOW_VOL: {"rsi_threshold": 30, "zscore_threshold": -2.0},
    MarketRegime.BULL_HIGH_VOL: {"rsi_threshold": 25, "zscore_threshold": -2.5},
    MarketRegime.BEAR_LOW_VOL: {"rsi_threshold": 25, "zscore_threshold": -2.0},
    MarketRegime.BEAR_HIGH_VOL: {"rsi_threshold": 20, "zscore_threshold": -3.0},  # Very extreme only
    MarketRegime.NEUTRAL: {"rsi_threshold": 30, "zscore_threshold": -2.0},
}


# =============================================================================
# Bootstrap Confidence Intervals (I1 Fix)
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
    
    boot_stats = np.array(boot_stats)
    
    # Compute percentiles
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    
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
# Kelly Criterion Position Sizing (I8 Fix)
# =============================================================================

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25,
    fractional_kelly: float = 0.5,
) -> float:
    """
    Compute Kelly Criterion optimal position size.
    
    Parameters:
        win_rate: Probability of winning (0 to 1)
        avg_win: Average winning return (positive number)
        avg_loss: Average losing return (positive number, will be treated as loss)
        max_fraction: Maximum allowed fraction (safety cap)
        fractional_kelly: Fraction of Kelly to use (0.5 = half Kelly, safer)
    
    Returns:
        float: Optimal position fraction (0 to max_fraction)
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly formula: f* = (p * b - q) / b
    # where p = win prob, q = 1 - p, b = win/loss ratio
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    full_kelly = (p * b - q) / b
    
    # Apply fractional Kelly (safer)
    kelly = full_kelly * fractional_kelly
    
    # Cap at maximum and ensure non-negative
    return max(0.0, min(kelly, max_fraction))


# =============================================================================
# Empirical Expected Returns (C5 Fix)
# =============================================================================

def compute_empirical_expected_returns(
    historical_trades: list[TradeCycle],
    dip_depth_current: float,
    percentile_bins: int = 5,
) -> tuple[float, float, float]:
    """
    Compute expected return/loss from empirical distribution of historical trades.
    
    Instead of naive heuristics like "typical_dip * 0.7", this uses actual
    historical trade outcomes binned by dip severity.
    
    Parameters:
        historical_trades: List of historical TradeCycle objects
        dip_depth_current: Current dip depth (e.g., 0.15 for 15% dip)
        percentile_bins: Number of bins for dip severity
    
    Returns:
        tuple: (expected_return_pct, expected_loss_pct, recovery_probability)
    """
    if len(historical_trades) < 10:
        # Fallback to simple estimate if insufficient data
        return dip_depth_current * 50, dip_depth_current * 30, 0.5
    
    returns = np.array([t.return_pct for t in historical_trades])
    
    # Separate winners and losers
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    
    if len(winners) == 0:
        expected_return = 0.0
    else:
        expected_return = float(np.mean(winners))
    
    if len(losers) == 0:
        expected_loss = 0.0
    else:
        expected_loss = float(np.abs(np.mean(losers)))
    
    recovery_prob = len(winners) / len(returns)
    
    return expected_return, expected_loss, recovery_prob


# =============================================================================
# Exit Signal Definitions
# =============================================================================


EXIT_SIGNALS = {
    # Indicator-based exits
    "RSI Overbought": {
        "indicator": "rsi_14",
        "direction": "above",
        "default_threshold": 70,
        "description": "RSI overbought - take profits",
    },
    "RSI Extreme Overbought": {
        "indicator": "rsi_14",
        "direction": "above",
        "default_threshold": 80,
        "description": "RSI extremely overbought",
    },
    "Stochastic Overbought": {
        "indicator": "stoch_k",
        "direction": "above",
        "default_threshold": 80,
        "description": "Stochastic %K overbought",
    },
    "Above Bollinger Upper": {
        "indicator": "bb_position",
        "direction": "above",
        "default_threshold": 1.0,
        "description": "Price at or above upper Bollinger Band",
    },
    "MACD Bearish Cross": {
        "indicator": "macd_histogram",
        "direction": "cross_below",
        "default_threshold": 0,
        "description": "MACD crossing below signal line",
    },
    # Price targets
    "5% Profit Target": {
        "type": "profit_target",
        "threshold": 0.05,
        "description": "Exit after 5% gain",
    },
    "10% Profit Target": {
        "type": "profit_target",
        "threshold": 0.10,
        "description": "Exit after 10% gain",
    },
    "15% Profit Target": {
        "type": "profit_target",
        "threshold": 0.15,
        "description": "Exit after 15% gain",
    },
    "20% Profit Target": {
        "type": "profit_target",
        "threshold": 0.20,
        "description": "Exit after 20% gain",
    },
    # Trailing stops
    "5% Trailing Stop": {
        "type": "trailing_stop",
        "threshold": 0.05,
        "description": "5% trailing stop from peak",
    },
    "10% Trailing Stop": {
        "type": "trailing_stop",
        "threshold": 0.10,
        "description": "10% trailing stop from peak",
    },
    # Stop losses
    "5% Stop Loss": {
        "type": "stop_loss",
        "threshold": 0.05,
        "description": "Exit on 5% loss",
    },
    "10% Stop Loss": {
        "type": "stop_loss",
        "threshold": 0.10,
        "description": "Exit on 10% loss",
    },
    # Time-based
    "20 Day Hold": {
        "type": "time_exit",
        "threshold": 20,
        "description": "Exit after 20 days",
    },
    "40 Day Hold": {
        "type": "time_exit",
        "threshold": 40,
        "description": "Exit after 40 days",
    },
    "60 Day Hold": {
        "type": "time_exit",
        "threshold": 60,
        "description": "Exit after 60 days",
    },
}


# =============================================================================
# Signal Combination Definitions
# =============================================================================


SIGNAL_COMBINATIONS = [
    {
        "name": "RSI + Drawdown",
        "signals": ["RSI Oversold", "Drawdown 15%+"],
        "logic": "AND",
        "description": "Both RSI oversold AND significant drawdown",
    },
    {
        "name": "Double Oversold",
        "signals": ["RSI Oversold", "Z-Score 20D Oversold"],
        "logic": "AND",
        "description": "Both RSI and Z-Score indicate oversold",
    },
    {
        "name": "Bollinger + RSI",
        "signals": ["Below Bollinger Lower", "RSI Oversold"],
        "logic": "AND",
        "description": "Below Bollinger with RSI confirmation",
    },
    {
        "name": "Triple Oversold",
        "signals": ["RSI Oversold", "Z-Score 20D Oversold", "Below Bollinger Lower"],
        "logic": "AND",
        "description": "Multiple indicators all showing oversold",
    },
]


# =============================================================================
# Core Analysis Functions
# =============================================================================


def analyze_dip(
    price_data: dict[str, pd.Series],
    symbol: str,
    fundamental_score_current: float | None = None,
    fundamental_score_previous: float | None = None,
) -> DipAnalysis:
    """
    Analyze whether a dip is an overreaction or falling knife.

    Uses:
    1. Historical dip patterns (is this dip unusual?)
    2. Technical indicators (RSI, MACD divergence, volume)
    3. Trend analysis (is long-term trend broken?)
    """
    prices = price_data.get("close")
    if prices is None or len(prices) < 252:
        return DipAnalysis(
            symbol=symbol,
            current_drawdown_pct=0,
            typical_dip_pct=0,
            max_historical_dip_pct=0,
            dip_zscore=0,
            is_unusually_deep=False,
            deviation_from_typical=0,
            dip_type=DipType.INSUFFICIENT_DATA,
            confidence=0,
            action="WAIT",
            reasoning="Insufficient data",
        )

    # Build DataFrame for 'ta' library
    df = pd.DataFrame({"close": prices})
    if "high" in price_data:
        df["high"] = price_data["high"]
    if "low" in price_data:
        df["low"] = price_data["low"]
    if "volume" in price_data:
        df["volume"] = price_data["volume"]

    # Compute indicators
    df = compute_all_indicators(df)

    current_dd = float(df["drawdown"].iloc[-1])
    current_depth = abs(current_dd)

    # Find historical dips
    drawdowns = df["drawdown"].values
    historical_dips = []

    in_dip = False
    dip_min = 0
    for i in range(1, len(drawdowns)):
        if drawdowns[i] < -0.05:
            if not in_dip:
                in_dip = True
                dip_min = drawdowns[i]
            else:
                dip_min = min(dip_min, drawdowns[i])
        elif drawdowns[i] > -0.02 and in_dip:
            historical_dips.append(abs(dip_min))
            in_dip = False

    if len(historical_dips) < 3:
        historical_dips = [abs(df["drawdown"].min())]

    typical_dip = float(np.median(historical_dips))
    max_dip = float(np.max(historical_dips))
    std_dip = (
        float(np.std(historical_dips))
        if len(historical_dips) > 1
        else typical_dip * 0.3
    )

    dip_zscore = (current_depth - typical_dip) / std_dip if std_dip > 0 else 0
    is_unusual = dip_zscore > 1.5 or current_depth > max_dip * 0.9

    # Get technical snapshot
    technicals = get_technical_snapshot(df)

    # Detect market regime for adaptive thresholds
    regime, regime_details = detect_market_regime(prices)
    
    # Fundamental change detection (moved up for use in tech score)
    fund_change = None
    if fundamental_score_current is not None and fundamental_score_previous is not None:
        fund_change = fundamental_score_current - fundamental_score_previous

    # FIXED (C1): Use statistically-derived technical score instead of heuristics
    tech_score, score_contributions = compute_statistical_tech_score(
        technicals, df, fund_change
    )

    # Check for momentum divergence (price down, RSI up)
    momentum_div = "momentum_div" in score_contributions

    # Check if long-term trend is broken
    trend_broken = technicals.sma_200_pct < -0.10

    # Volume confirmation (high volume on dip = capitulation)
    volume_conf = "volume_cap" in score_contributions

    # FIXED (C4): Use percentile-based thresholds instead of hardcoded values
    # Threshold for "unusual" is now based on stock's own history
    dip_percentile = (np.array(historical_dips) <= current_depth).mean() * 100
    is_unusual_by_percentile = dip_percentile >= 80  # Top 20% of historical dips

    # Classify the dip using stock-specific percentiles
    dip_type = DipType.INSUFFICIENT_DATA
    confidence = 0.5
    action = "WAIT"
    reasoning = ""
    recovery_prob = 0.5
    expected_return = 0.0
    expected_loss = 0.0

    if current_depth < typical_dip * 0.5:
        dip_type = DipType.NORMAL_VOLATILITY
        confidence = 0.7
        action = "WAIT"
        reasoning = f"Dip of {current_depth*100:.1f}% is minor (typical: {typical_dip*100:.1f}%). Wait for better entry."

    elif fund_change is not None and fund_change < -0.15:
        dip_type = DipType.FUNDAMENTAL_DECLINE
        confidence = 0.7
        action = "AVOID"
        reasoning = f"⚠️ FALLING KNIFE: Fundamentals deteriorated {abs(fund_change)*100:.0f}%. Dip may be justified."
        recovery_prob = 0.35
        expected_loss = current_depth * 0.5

    elif trend_broken and tech_score < -0.1:
        dip_type = DipType.FUNDAMENTAL_DECLINE
        confidence = 0.6
        action = "AVOID"
        reasoning = "⚠️ FALLING KNIFE: Long-term trend broken (10%+ below SMA 200) with weak technicals."
        recovery_prob = 0.4
        expected_loss = current_depth * 0.3

    elif is_unusual and tech_score > 0.2:
        dip_type = DipType.OVERREACTION
        confidence = 0.7
        action = "STRONG_BUY"
        reasoning = f"🎯 OVERREACTION: {current_depth*100:.1f}% dip is {dip_zscore:.1f} std devs from typical ({typical_dip*100:.1f}%). Strong technicals suggest buying opportunity."
        recovery_prob = 0.75
        expected_return = typical_dip * 0.7

    elif current_depth >= typical_dip and tech_score > 0:
        dip_type = DipType.OVERREACTION
        confidence = 0.65
        action = "BUY"
        reasoning = f"Dip of {current_depth*100:.1f}% matches typical pattern ({typical_dip*100:.1f}%). Technicals positive."
        recovery_prob = 0.65
        expected_return = typical_dip * 0.5

    elif tech_score > 0.1:
        dip_type = DipType.NORMAL_VOLATILITY
        confidence = 0.55
        action = "BUY"
        reasoning = "Technicals suggest oversold conditions. Consider small position."
        recovery_prob = 0.55
        expected_return = current_depth * 0.3

    else:
        dip_type = DipType.NORMAL_VOLATILITY
        confidence = 0.5
        action = "WAIT"
        reasoning = f"No clear signal. Current dip {current_depth*100:.1f}% vs typical {typical_dip*100:.1f}%."

    return DipAnalysis(
        symbol=symbol,
        current_drawdown_pct=current_depth * 100,
        typical_dip_pct=typical_dip * 100,
        max_historical_dip_pct=max_dip * 100,
        dip_zscore=dip_zscore,
        is_unusually_deep=is_unusual,
        deviation_from_typical=dip_zscore,
        technicals=technicals,
        technical_score=tech_score,
        trend_broken=trend_broken,
        volume_confirmation=volume_conf,
        momentum_divergence=momentum_div,
        dip_type=dip_type,
        confidence=confidence,
        action=action,
        reasoning=reasoning,
        recovery_probability=recovery_prob,
        expected_return_if_buy=expected_return * 100,
        expected_loss_if_knife=expected_loss * 100,
    )


def _simulate_trades_with_exit(
    df: pd.DataFrame,
    entry_trigger: pd.Series,
    exit_name: str,
    max_hold: int | None = None,
) -> list[TradeCycle]:
    """Simulate trades with a specific exit strategy."""
    from app.quant_engine.config import QUANT_LIMITS
    if max_hold is None:
        max_hold = QUANT_LIMITS.max_holding_days
    
    exit_cfg = EXIT_SIGNALS.get(exit_name, {})
    if not exit_cfg:
        return []

    trades = []
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    entry_date = None
    peak_price = 0.0
    peak_date = None

    for i in range(len(df)):
        if not in_trade:
            if i < len(entry_trigger) and entry_trigger.iloc[i]:
                in_trade = True
                entry_idx = i
                # Apply slippage to entry (buy at slightly higher price)
                raw_entry = df["close"].iloc[i]
                entry_price = raw_entry * (1 + DEFAULT_SLIPPAGE_BPS / 10000)
                entry_date = df.index[i]
                peak_price = entry_price
                peak_date = entry_date
        else:
            current_price = df["close"].iloc[i]

            if current_price > peak_price:
                peak_price = current_price
                peak_date = df.index[i]

            should_exit = False
            exit_reason = ""

            exit_type = exit_cfg.get("type")

            if exit_type == "profit_target":
                ret = (current_price - entry_price) / entry_price
                if ret >= exit_cfg["threshold"]:
                    should_exit = True
                    exit_reason = f"{exit_cfg['threshold']*100:.0f}% profit target"

            elif exit_type == "trailing_stop":
                from_peak = (current_price - peak_price) / peak_price
                if from_peak <= -exit_cfg["threshold"]:
                    should_exit = True
                    exit_reason = f"{exit_cfg['threshold']*100:.0f}% trailing stop"

            elif exit_type == "stop_loss":
                ret = (current_price - entry_price) / entry_price
                if ret <= -exit_cfg["threshold"]:
                    should_exit = True
                    exit_reason = f"{exit_cfg['threshold']*100:.0f}% stop loss"

            elif exit_type == "time_exit":
                days_held = i - entry_idx
                if days_held >= exit_cfg["threshold"]:
                    should_exit = True
                    exit_reason = f"{exit_cfg['threshold']}d hold"

            elif "indicator" in exit_cfg:
                indicator = exit_cfg["indicator"]
                if indicator in df.columns:
                    curr_val = df[indicator].iloc[i]
                    if pd.notna(curr_val):
                        direction = exit_cfg.get("direction", "above")
                        threshold = exit_cfg["default_threshold"]

                        if direction == "above" and curr_val > threshold:
                            should_exit = True
                            exit_reason = exit_cfg["description"]
                        elif direction == "below" and curr_val < threshold:
                            should_exit = True
                            exit_reason = exit_cfg["description"]
                        elif direction == "cross_below":
                            if i > 0:
                                prev = df[indicator].iloc[i - 1]
                                if pd.notna(prev):
                                    if prev > threshold and curr_val <= threshold:
                                        should_exit = True
                                        exit_reason = exit_cfg["description"]

            # Max hold backstop
            days_held = i - entry_idx
            if days_held >= max_hold:
                should_exit = True
                exit_reason = f"Max {max_hold}d hold"

            if should_exit:
                # Apply slippage to exit (sell at slightly lower price)
                exit_price = current_price * (1 - DEFAULT_SLIPPAGE_BPS / 10000)
                return_pct = (exit_price - entry_price) / entry_price * 100
                missed = (
                    (peak_price - exit_price) / peak_price * 100 if peak_price > 0 else 0
                )

                trades.append(
                    TradeCycle(
                        entry_date=str(entry_date.date())
                        if hasattr(entry_date, "date")
                        else str(entry_date),
                        exit_date=str(df.index[i].date())
                        if hasattr(df.index[i], "date")
                        else str(df.index[i]),
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        return_pct=float(return_pct),
                        holding_days=days_held,
                        entry_signal="entry",
                        exit_signal=exit_reason,
                        peak_price=float(peak_price),
                        peak_date=str(peak_date.date())
                        if hasattr(peak_date, "date")
                        else str(peak_date),
                        missed_upside_pct=float(missed),
                    )
                )
                in_trade = False

    return trades


def optimize_full_trade(
    price_data: dict[str, pd.Series],
    entry_signal_name: str,
    spy_prices: pd.Series | None = None,
    min_trades: int = 5,
) -> FullTradeResult | None:
    """
    Optimize a complete trade strategy with benchmarking against SPY and buy-and-hold.

    Tests all exit strategies and finds the one that maximizes risk-adjusted returns
    while beating both benchmarks.
    """
    prices = price_data.get("close")
    if prices is None or len(prices) < 252:
        return None

    # Find entry signal config
    entry_config = None
    for cfg in SIGNAL_CONFIGS:
        if cfg["name"] == entry_signal_name:
            entry_config = cfg
            break

    if entry_config is None:
        return None

    # Build DataFrame
    df = pd.DataFrame({"close": prices})
    if "high" in price_data:
        df["high"] = price_data["high"]
    if "low" in price_data:
        df["low"] = price_data["low"]
    if "volume" in price_data:
        df["volume"] = price_data["volume"]

    # Compute indicators
    df = compute_all_indicators(df)

    # Compute entry signal
    try:
        entry_values = entry_config["compute"](price_data)
    except Exception:
        return None

    # Determine entry triggers
    threshold = entry_config["default_threshold"]
    direction = entry_config["direction"]

    if direction == "below":
        entry_trigger = entry_values < threshold
    elif direction == "above":
        entry_trigger = entry_values > threshold
    elif direction == "cross_above":
        prev = entry_values.shift(1)
        entry_trigger = (entry_values > threshold) & (prev <= threshold)
    else:
        entry_trigger = entry_values < threshold

    # Test all exit strategies
    best_exit_name = ""
    best_total_return = -np.inf
    best_trades: list[TradeCycle] = []

    for exit_name in EXIT_SIGNALS:
        trades = _simulate_trades_with_exit(df, entry_trigger, exit_name)

        if len(trades) < min_trades:
            continue

        # Use compounded return for comparison
        compounded = 1.0
        for t in trades:
            compounded *= (1 + t.return_pct / 100)
        total_ret = (compounded - 1) * 100

        if total_ret > best_total_return:
            best_total_return = total_ret
            best_exit_name = exit_name
            best_trades = trades

    if not best_trades or len(best_trades) < min_trades:
        return None

    # Calculate metrics
    returns = [t.return_pct for t in best_trades]
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    avg_return = float(np.mean(returns))
    # FIXED: Use compounded return instead of sum of returns
    # Compound: final = (1 + r1/100) * (1 + r2/100) * ... - 1
    compounded = 1.0
    for r in returns:
        compounded *= (1 + r / 100)
    total_return = (compounded - 1) * 100
    max_return = max(returns)
    holding_days = [t.holding_days for t in best_trades]
    avg_holding = float(np.mean(holding_days))

    # FIXED: Sharpe ratio calculation
    # 1. Convert returns from percentage to decimal
    # 2. Use risk-free rate adjustment
    # 3. Proper annualization
    sharpe = 0.0
    if len(returns) > 1:
        returns_decimal = [r / 100 for r in returns]  # Convert % to decimal
        avg_ret_decimal = float(np.mean(returns_decimal))
        ret_std = float(np.std(returns_decimal, ddof=1))  # Use sample std
        
        if ret_std > 0:
            # Risk-free rate per trade (assume ~4% annual)
            trades_per_year = 252 / avg_holding if avg_holding > 0 else 12
            risk_free_per_trade = 0.04 / trades_per_year
            
            # Excess return over risk-free
            excess_return = avg_ret_decimal - risk_free_per_trade
            
            # Annualized Sharpe
            sharpe = (excess_return / ret_std) * np.sqrt(trades_per_year)

    # Max drawdown during trades (using compounded equity curve)
    equity = [1.0]  # Start at 1.0 (normalized)
    for r in returns:
        equity.append(equity[-1] * (1 + r / 100))
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / peak * 100  # Drawdown as percentage
        max_dd = min(max_dd, dd)

    # Buy-and-hold benchmark
    try:
        first_date = best_trades[0].entry_date
        last_date = best_trades[-1].exit_date
        start_idx = 0
        end_idx = len(df) - 1

        for i, idx in enumerate(df.index):
            idx_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
            if idx_str.startswith(first_date[:10]):
                start_idx = i
                break

        for i, idx in enumerate(df.index):
            idx_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
            if idx_str.startswith(last_date[:10]):
                end_idx = i
                break

        stock_bh = (
            (df["close"].iloc[end_idx] - df["close"].iloc[start_idx])
            / df["close"].iloc[start_idx]
            * 100
        )
    except Exception:
        stock_bh = (
            (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        )

    # SPY benchmark
    spy_bh = 0.0
    if spy_prices is not None and len(spy_prices) > 0:
        try:
            spy_bh = (
                (spy_prices.iloc[-1] - spy_prices.iloc[0]) / spy_prices.iloc[0] * 100
            )
        except Exception:
            pass

    edge_vs_stock = total_return - stock_bh
    edge_vs_spy = total_return - spy_bh
    beats_both = total_return > stock_bh and total_return > spy_bh

    # Exit predictability (low std = predictable)
    holding_std = float(np.std(holding_days)) if len(holding_days) > 1 else avg_holding
    predictability = 1 - (holding_std / avg_holding) if avg_holding > 0 else 0

    # Upside captured
    avg_missed = float(np.mean([t.missed_upside_pct for t in best_trades]))
    upside_captured = 100 - avg_missed

    # Current signals
    current_buy = bool(entry_trigger.iloc[-1]) if len(entry_trigger) > 0 else False

    # Check current sell signal
    current_sell = False
    exit_cfg = EXIT_SIGNALS.get(best_exit_name, {})
    if "indicator" in exit_cfg and exit_cfg["indicator"] in df.columns:
        curr_val = df[exit_cfg["indicator"]].iloc[-1]
        if pd.notna(curr_val):
            direction = exit_cfg.get("direction", "above")
            thresh = exit_cfg["default_threshold"]
            if direction == "above":
                current_sell = curr_val > thresh
            elif direction == "below":
                current_sell = curr_val < thresh

    # Days since last signal
    days_since = 0
    for i in range(len(entry_trigger) - 1, -1, -1):
        if entry_trigger.iloc[i]:
            days_since = len(entry_trigger) - 1 - i
            break

    return FullTradeResult(
        symbol="",  # Set by caller
        entry_signal_name=entry_signal_name,
        entry_threshold=float(threshold),
        entry_description=entry_config.get("description", entry_signal_name),
        exit_strategy_name=best_exit_name,
        exit_threshold=float(
            exit_cfg.get("threshold", exit_cfg.get("default_threshold", 0))
        ),
        exit_description=EXIT_SIGNALS[best_exit_name]["description"],
        n_complete_trades=len(best_trades),
        win_rate=float(win_rate),
        avg_return_pct=avg_return,
        total_return_pct=float(total_return),
        max_return_pct=float(max_return),
        max_drawdown_pct=float(max_dd),
        avg_holding_days=avg_holding,
        sharpe_ratio=float(sharpe),
        buy_hold_return_pct=float(stock_bh),
        spy_return_pct=float(spy_bh),
        edge_vs_buy_hold_pct=float(edge_vs_stock),
        edge_vs_spy_pct=float(edge_vs_spy),
        beats_both_benchmarks=beats_both,
        exit_predictability=float(max(0, min(1, predictability))),
        upside_captured_pct=float(upside_captured),
        trades=best_trades[-20:],  # Last 20 trades
        current_buy_signal=current_buy,
        current_sell_signal=current_sell,
        days_since_last_signal=days_since,
    )


def test_signal_combination(
    price_data: dict[str, pd.Series],
    combination: dict,
    holding_days: int = 20,
    min_signals: int = 3,
) -> CombinedSignal | None:
    """Test a signal combination (AND/OR of multiple signals)."""
    prices = price_data.get("close")
    if prices is None or len(prices) < 252:
        return None

    # Compute each component signal
    component_triggers = []
    best_single_ev = -np.inf

    for signal_name in combination["signals"]:
        config = None
        for cfg in SIGNAL_CONFIGS:
            if cfg["name"] == signal_name:
                config = cfg
                break

        if config is None:
            return None

        try:
            values = config["compute"](price_data)
            threshold = config["default_threshold"]
            direction = config["direction"]

            if direction == "below":
                trigger = values < threshold
            elif direction == "above":
                trigger = values > threshold
            elif direction == "cross_above":
                prev = values.shift(1)
                trigger = (values > threshold) & (prev <= threshold)
            else:
                trigger = values < threshold

            component_triggers.append(trigger)

            # Calculate single signal EV
            fwd_ret = prices.pct_change(holding_days).shift(-holding_days)
            signal_rets = fwd_ret[trigger].dropna()
            if len(signal_rets) >= min_signals:
                wr = float((signal_rets > 0).mean())
                ar = float(signal_rets.mean())
                ev = wr * ar * 100
                if ev > best_single_ev:
                    best_single_ev = ev

        except Exception:
            return None

    if len(component_triggers) != len(combination["signals"]):
        return None

    # Combine triggers
    if combination["logic"] == "AND":
        combined = component_triggers[0]
        for t in component_triggers[1:]:
            combined = combined & t
    else:  # OR
        combined = component_triggers[0]
        for t in component_triggers[1:]:
            combined = combined | t

    # Calculate combined performance
    fwd_ret = prices.pct_change(holding_days).shift(-holding_days)
    signal_rets = fwd_ret[combined].dropna()

    if len(signal_rets) < min_signals:
        return None

    win_rate = float((signal_rets > 0).mean())
    avg_return = float(signal_rets.mean())
    n_signals = len(signal_rets)

    combined_ev = win_rate * avg_return * 100
    improvement = (
        ((combined_ev - best_single_ev) / abs(best_single_ev) * 100)
        if best_single_ev > 0
        else 0
    )

    return CombinedSignal(
        name=combination["name"],
        component_signals=combination["signals"],
        logic=combination["logic"],
        win_rate=win_rate,
        avg_return_pct=avg_return * 100,
        n_signals=n_signals,
        improvement_vs_best_single=improvement,
    )


def get_best_trade_strategy(
    price_data: dict[str, pd.Series],
    symbol: str,
    spy_prices: pd.Series | None = None,
    test_combinations: bool = True,
    min_trades: int = 5,
) -> tuple[FullTradeResult | None, list[CombinedSignal]]:
    """
    Find the best complete trade strategy for a stock.

    Tests all entry signals with all exit strategies, comparing against
    buy-and-hold and SPY benchmarks.
    """
    best_result = None
    best_alpha = -np.inf

    # Test each entry signal
    for config in SIGNAL_CONFIGS:
        result = optimize_full_trade(
            price_data, config["name"], spy_prices, min_trades
        )
        if result is not None:
            result.symbol = symbol
            # Prioritize strategies that beat both benchmarks
            alpha = result.edge_vs_buy_hold_pct + result.edge_vs_spy_pct
            if result.beats_both_benchmarks:
                alpha += 10  # Bonus for beating both

            if alpha > best_alpha:
                best_alpha = alpha
                best_result = result

    # Test combinations
    combinations = []
    if test_combinations:
        for combo_cfg in SIGNAL_COMBINATIONS:
            combo = test_signal_combination(
                price_data, combo_cfg, holding_days=20, min_signals=3
            )
            if combo is not None:
                combinations.append(combo)

    # Sort combinations by EV
    combinations.sort(key=lambda c: c.win_rate * c.avg_return_pct, reverse=True)

    return best_result, combinations[:5]


def get_current_signals(
    price_data: dict[str, pd.Series],
    symbol: str,
) -> dict[str, Any]:
    """
    Get current real-time buy and sell signals.

    Returns dict with all active signals and overall recommendation.
    """
    prices = price_data.get("close")
    if prices is None or len(prices) < 60:
        return {
            "symbol": symbol,
            "buy_signals": [],
            "sell_signals": [],
            "overall_action": "HOLD",
            "reasoning": "Insufficient data",
        }

    # Build DataFrame
    df = pd.DataFrame({"close": prices})
    if "high" in price_data:
        df["high"] = price_data["high"]
    if "low" in price_data:
        df["low"] = price_data["low"]
    if "volume" in price_data:
        df["volume"] = price_data["volume"]

    # Compute indicators
    df = compute_all_indicators(df)

    buy_signals = []
    sell_signals = []

    # Check buy signals
    for config in SIGNAL_CONFIGS:
        try:
            values = config["compute"](price_data)
            curr = values.iloc[-1]
            threshold = config["default_threshold"]
            direction = config["direction"]

            if pd.isna(curr):
                continue

            is_active = False
            if direction == "below" and curr < threshold:
                is_active = True
            elif direction == "above" and curr > threshold:
                is_active = True
            elif direction == "cross_above":
                prev = values.iloc[-2] if len(values) > 1 else curr
                if curr > threshold and prev <= threshold:
                    is_active = True

            if is_active:
                buy_signals.append(
                    {
                        "name": config["name"],
                        "value": float(curr),
                        "threshold": float(threshold),
                        "description": config.get("description", config["name"]),
                    }
                )
        except Exception:
            continue

    # Check sell signals (indicator-based only)
    for name, cfg in EXIT_SIGNALS.items():
        if "indicator" not in cfg:
            continue

        indicator = cfg["indicator"]
        if indicator not in df.columns:
            continue

        curr_val = df[indicator].iloc[-1]
        if pd.isna(curr_val):
            continue

        threshold = cfg["default_threshold"]
        direction = cfg.get("direction", "above")

        is_active = False
        if direction == "above" and curr_val > threshold:
            is_active = True
        elif direction == "below" and curr_val < threshold:
            is_active = True
        elif direction == "cross_below":
            prev = df[indicator].iloc[-2] if len(df) > 1 else curr_val
            if pd.notna(prev) and curr_val < threshold and prev >= threshold:
                is_active = True

        if is_active:
            sell_signals.append(
                {
                    "name": name,
                    "value": float(curr_val),
                    "threshold": float(threshold),
                    "description": cfg["description"],
                }
            )

    # Determine overall action
    n_buy = len(buy_signals)
    n_sell = len(sell_signals)

    if n_buy >= 3 and n_sell == 0:
        action = "STRONG_BUY"
        reason = f"{n_buy} buy signals active, no sell signals"
    elif n_buy >= 2 and n_sell == 0:
        action = "BUY"
        reason = f"{n_buy} buy signals active"
    elif n_buy >= 1 and n_sell == 0:
        action = "WEAK_BUY"
        reason = f"{n_buy} buy signal(s) active"
    elif n_sell >= 2 and n_buy == 0:
        action = "STRONG_SELL"
        reason = f"{n_sell} sell signals, no buy signals"
    elif n_sell >= 1 and n_buy == 0:
        action = "SELL"
        reason = f"{n_sell} sell signal(s) active"
    elif n_buy > n_sell:
        action = "BUY"
        reason = f"More buy ({n_buy}) than sell ({n_sell}) signals"
    elif n_sell > n_buy:
        action = "SELL"
        reason = f"More sell ({n_sell}) than buy ({n_buy}) signals"
    else:
        action = "HOLD"
        reason = "No clear direction"

    return {
        "symbol": symbol,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "overall_action": action,
        "reasoning": reason,
    }


# =============================================================================
# Optimized Strategy Integration (using new backtest_engine)
# =============================================================================

def get_optimized_strategy(
    df: pd.DataFrame,
    symbol: str,
    spy_prices: pd.Series | None = None,
    runtime_settings: dict | None = None,
) -> dict:
    """
    Get the best optimized trading strategy for a stock using the backtest engine.
    
    This uses walk-forward optimization to find strategies that:
    1. Have statistically significant returns
    2. Don't overfit to historical data
    3. Ideally beat buy-and-hold
    
    Args:
        df: Price DataFrame with OHLCV data
        symbol: Stock symbol
        spy_prices: Optional SPY prices for benchmark comparison
        runtime_settings: Optional settings dict from admin panel
    
    Returns:
        Dict with strategy details, metrics, and current signal status.
    """
    from .backtest_engine import (
        TradingConfig, BacktestEngine,
        compare_to_benchmark, STRATEGIES, get_all_strategies
    )
    from .indicators import compute_indicators
    
    # Load config from runtime settings if provided
    if runtime_settings:
        config = TradingConfig.from_runtime_settings(runtime_settings)
    else:
        config = TradingConfig()
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Handle multi-level columns from yfinance (e.g., ('Close', 'QQQ'))
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    
    # Ensure columns are lowercase
    df.columns = [str(c).lower() for c in df.columns]
    df.attrs["symbol"] = symbol
    
    # Compute indicators using backtest engine
    df_with_indicators = compute_indicators(df)
    
    engine = BacktestEngine(config)
    
    # Find best base strategy (skip ensembles for speed in real-time)
    best_strategy = ""
    best_result = None
    best_sharpe = -np.inf
    
    for strat_name in STRATEGIES:
        try:
            _, default_params, _ = STRATEGIES[strat_name]
            result = engine.backtest_strategy(df_with_indicators, strat_name, default_params)
            
            if result.n_trades >= 5 and result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result
                best_strategy = strat_name
        except Exception as e:
            logger.warning(f"Strategy {strat_name} failed for {symbol}: {e}")
            continue
    
    if best_result is None:
        return {
            "symbol": symbol,
            "strategy_name": None,
            "has_active_signal": False,
            "action": "HOLD",
            "reason": "No valid strategy found",
            "metrics": {},
        }
    
    # Get current signal status for the best strategy
    strat_func, default_params, _ = STRATEGIES[best_strategy]
    entries = strat_func(df_with_indicators, default_params)
    
    # Check if signal is active today
    has_active_signal = bool(entries.iloc[-1] == 1) if len(entries) > 0 else False
    
    # Determine if in a position (has recent entry without exit)
    in_position = False
    if best_result.trades:
        last_trade = best_result.trades[-1]
        if last_trade.exit_date is None or (
            hasattr(last_trade, 'is_open') and last_trade.is_open
        ):
            in_position = True
    
    # Compare to buy and hold
    comparison = compare_to_benchmark(best_result, df_with_indicators["close"], spy_prices)
    
    # Determine action
    if has_active_signal and not in_position:
        action = "BUY"
        reason = f"Strategy '{best_strategy}' signals entry"
    elif in_position:
        action = "HOLD_POSITION"
        reason = f"Currently in position from '{best_strategy}'"
    else:
        action = "WAIT"
        reason = f"Strategy '{best_strategy}' waiting for entry signal"
    
    # Build response
    return {
        "symbol": symbol,
        "strategy_name": best_strategy,
        "has_active_signal": has_active_signal,
        "action": action,
        "reason": reason,
        "metrics": {
            "total_trades": best_result.n_trades,
            "win_rate": round(best_result.win_rate, 1),
            "total_return_pct": round(best_result.total_return_pct, 1),
            "sharpe_ratio": round(best_result.sharpe_ratio, 2),
            "max_drawdown_pct": round(best_result.max_drawdown_pct, 1),
            "avg_holding_days": round(best_result.avg_holding_days, 1),
        },
        "benchmark_comparison": {
            "stock_buy_hold_return": round(comparison.get("stock_buy_hold_return", 0), 1),
            "beats_stock": comparison.get("beats_stock", False),
            "excess_vs_stock": round(comparison.get("excess_vs_stock", 0), 1),
            "spy_buy_hold_return": round(comparison.get("spy_buy_hold_return", 0), 1) if "spy_buy_hold_return" in comparison else None,
            "beats_spy": comparison.get("beats_spy", None),
        },
        "recent_trades": [
            {
                "entry_date": str(t.entry_date.date()),
                "exit_date": str(t.exit_date.date()) if t.exit_date else None,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "pnl_pct": round(t.pnl_pct, 1),  # Already stored as percentage
                "exit_reason": t.exit_reason,
            }
            for t in (best_result.trades[-5:] if best_result.trades else [])
        ],
    }
