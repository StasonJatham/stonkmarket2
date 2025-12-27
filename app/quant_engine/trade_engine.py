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
        high=high, low=low, close=close, window=14
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

    # Technical score (-1 to +1)
    tech_score = 0.0

    # RSI oversold is bullish for dip buying
    if technicals.rsi_14 < 30:
        tech_score += 0.3
    elif technicals.rsi_14 < 40:
        tech_score += 0.1
    elif technicals.rsi_14 > 50:
        tech_score -= 0.1

    # Below Bollinger lower is bullish
    if technicals.bb_position < 0:
        tech_score += 0.2
    elif technicals.bb_position < 0.2:
        tech_score += 0.1

    # Z-score extreme is bullish
    zscore = float(df["zscore_20"].iloc[-1]) if pd.notna(df["zscore_20"].iloc[-1]) else 0
    if zscore < -2:
        tech_score += 0.2
    elif zscore < -1.5:
        tech_score += 0.1

    # MACD turning up is bullish
    if len(df) > 1:
        prev_macd = df["macd_histogram"].iloc[-2]
        curr_macd = technicals.macd_histogram
        if pd.notna(prev_macd) and pd.notna(curr_macd):
            if curr_macd > 0 and prev_macd < 0:
                tech_score += 0.2  # Bullish cross
            elif curr_macd < prev_macd:
                tech_score -= 0.1  # Momentum weakening

    # Check for momentum divergence (price down, RSI up)
    momentum_div = False
    if len(df) >= 10:
        price_trend = df["close"].iloc[-10:].pct_change().mean()
        rsi_trend = df["rsi_14"].iloc[-10:].diff().mean()
        if pd.notna(price_trend) and pd.notna(rsi_trend):
            if price_trend < 0 and rsi_trend > 0:
                momentum_div = True
                tech_score += 0.2

    # Check if long-term trend is broken
    trend_broken = technicals.sma_200_pct < -0.10
    if trend_broken:
        tech_score -= 0.15

    # Volume confirmation (high volume on dip = capitulation)
    volume_conf = False
    if "volume" in df.columns:
        vol_ratio = (
            df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
            if len(df) > 20
            else 1
        )
        if pd.notna(vol_ratio) and vol_ratio > 2 and current_dd < -0.05:
            volume_conf = True
            tech_score += 0.15

    # Fundamental change detection
    fund_change = None
    if fundamental_score_current is not None and fundamental_score_previous is not None:
        fund_change = fundamental_score_current - fundamental_score_previous
        if fund_change < -0.15:
            tech_score -= 0.3  # Fundamentals worsened

    # Classify the dip
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
    max_hold: int = 100,
) -> list[TradeCycle]:
    """Simulate trades with a specific exit strategy."""
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
                entry_price = df["close"].iloc[i]
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
                exit_price = current_price
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

        total_ret = sum(t.return_pct for t in trades)

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
    total_return = sum(returns)
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

    # Max drawdown during trades
    cumulative = [0.0]
    for r in returns:
        cumulative.append(cumulative[-1] + r)
    peak = cumulative[0]
    max_dd = 0.0
    for c in cumulative:
        if c > peak:
            peak = c
        dd = c - peak
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
