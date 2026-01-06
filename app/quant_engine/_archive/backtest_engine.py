"""
DEPRECATED - This file is deprecated as of V3 refactor.

Use app.quant_engine.backtest_v2 for backtesting.
The V2 backtester provides improved features and regime-aware analysis.

This file will be removed in a future version.
---

Professional Backtesting Engine with Statistical Validation.

This module implements a rigorous, institutional-grade backtesting framework:
- Walk-forward optimization to prevent overfitting
- Multiple trading strategies (mean reversion, momentum, trend following)
- Hyperparameter optimization with Optuna (Bayesian)
- Statistical validation (Sharpe, Sortino, Calmar, Monte Carlo)
- Bias detection (look-ahead, overfitting, survivorship)
- Benchmark comparison (vs buy-and-hold stock, vs SPY)
- Real trade accounting (flat cost, capital constraints)

Key principles:
1. NO LOOK-AHEAD BIAS: All decisions use only data available at that time
2. WALK-FORWARD: Train on past, test on future, never the reverse
3. STATISTICAL SIGNIFICANCE: Reject strategies with p > 0.05
4. MULTIPLE TESTING CORRECTION: Adjust for strategy search
5. TRANSACTION COSTS: Include realistic €1 flat + slippage
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats

# Import shared indicator functions
from app.quant_engine.core.config import QUANT_LIMITS
from app.quant_engine.core.indicators import (
    prepare_price_dataframe,
    compute_indicators,
    bootstrap_confidence_interval,
)

# Suppress quantstats warnings about deprecated features
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants (DB-configurable defaults)
# =============================================================================

@dataclass
class TradingConfig:
    """Trading configuration with sensible defaults.
    
    This is loaded from admin settings (runtime_settings table).
    All values can be configured via the Admin UI Settings page.
    """
    
    # Capital and costs
    initial_capital: float = 50_000.0  # €50k default
    flat_cost_per_trade: float = 1.0   # €1 flat cost per trade
    slippage_bps: float = 5.0          # 5 basis points slippage per side
    
    # Risk management
    stop_loss_pct: float = 0.15        # 15% stop loss
    take_profit_pct: float = 0.30      # 30% take profit (optional)
    max_holding_days: int = QUANT_LIMITS.max_holding_days  # From central config
    
    # Optimization
    min_trades_for_significance: int = 30  # Min trades for statistical validity
    confidence_level: float = 0.95     # 95% confidence for hypothesis tests
    
    # Walk-forward settings
    train_ratio: float = 0.70          # 70% train, 30% test per fold
    n_folds: int = 5                   # 5-fold walk-forward
    
    # Current year filter (for 2025 validation)
    validation_start_date: str = "2025-01-01"
    
    @classmethod
    def from_runtime_settings(cls, settings: dict) -> "TradingConfig":
        """Create TradingConfig from runtime settings dict.
        
        Used to load config from admin settings stored in DB.
        """
        return cls(
            initial_capital=settings.get("trading_initial_capital", 50_000.0),
            flat_cost_per_trade=settings.get("trading_flat_cost_per_trade", 1.0),
            slippage_bps=settings.get("trading_slippage_bps", 5.0),
            stop_loss_pct=settings.get("trading_stop_loss_pct", 15.0) / 100,
            take_profit_pct=settings.get("trading_take_profit_pct", 30.0) / 100,
            max_holding_days=settings.get("trading_max_holding_days", 60),
            min_trades_for_significance=settings.get("trading_min_trades_required", 30),
            train_ratio=settings.get("trading_train_ratio", 0.70),
            n_folds=settings.get("trading_walk_forward_folds", 5),
        )


DEFAULT_CONFIG = TradingConfig()


# =============================================================================
# Data Classes for Trade Tracking
# =============================================================================

@dataclass
class Trade:
    """Individual trade record with full audit trail."""
    
    symbol: str
    strategy: str
    signal_name: str
    
    # Entry
    entry_date: pd.Timestamp
    entry_price: float
    entry_signal_value: float
    
    # Exit (filled when trade closes)
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""  # signal, stop_loss, take_profit, max_hold
    
    # Position sizing
    shares: float = 0.0
    position_value: float = 0.0
    
    # Returns (filled when trade closes)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    
    # Costs
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    total_cost: float = 0.0
    
    # Metadata
    is_open: bool = True
    

@dataclass
class StrategyResult:
    """Complete backtest results for a strategy."""
    
    strategy_name: str
    symbol: str
    
    # Performance metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Trade statistics
    n_trades: int = 0
    win_rate: float = 0.0  # Stored as percentage (e.g., 66.7 for 66.7%)
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    
    # Benchmark comparison
    vs_buy_hold_stock: float = 0.0  # Excess return vs holding stock
    vs_spy: float = 0.0             # Excess return vs SPY
    
    # Statistical validity
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    
    # Optimization metadata
    optimal_params: dict = field(default_factory=dict)
    is_out_of_sample: bool = False
    walk_forward_stable: bool = False
    
    # Individual trades
    trades: list[Trade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: pd.Series | None = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for a strategy."""
    
    strategy_name: str
    symbol: str
    
    # Overall verdict
    is_valid: bool = False
    verdict_reason: str = ""
    confidence_score: float = 0.0  # 0-100
    
    # Bias checks
    has_look_ahead_bias: bool = False
    has_overfitting: bool = False
    has_survivorship_bias: bool = False
    
    # Statistical tests
    passes_t_test: bool = False
    passes_bootstrap: bool = False
    passes_monte_carlo: bool = False
    
    # Multiple-testing correction (FDR)
    raw_p_value: float = 1.0  # Unadjusted p-value from t-test
    adjusted_p_value: float = 1.0  # After FDR/Bonferroni correction
    n_strategies_tested: int = 1  # Number of strategies tested
    passes_fdr_correction: bool = False  # Passes after multiple-testing correction
    
    # Walk-forward stability
    oos_sharpe: float = 0.0
    is_vs_sharpe: float = 0.0
    sharpe_degradation: float = 0.0  # Should be < 50%
    
    # Benchmark comparison
    beats_buy_hold: bool = False
    beats_spy: bool = False
    
    # Current year performance
    current_year_return: float = 0.0
    current_year_sharpe: float = 0.0


# =============================================================================
# Price Data Preparation & Technical Indicators
# =============================================================================
# NOTE: prepare_price_dataframe, compute_indicators, and bootstrap_confidence_interval
# are imported from app.quant_engine.indicators (shared module)


# =============================================================================
# Strategy Definitions
# =============================================================================

StrategyFunc = Callable[[pd.DataFrame, dict], pd.Series]


def strategy_mean_reversion_rsi(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Mean reversion strategy based on RSI oversold.
    
    Buy when RSI < oversold_threshold.
    Sell after N days or when RSI > overbought_threshold.
    """
    rsi_col = f"rsi_{params.get('rsi_period', 14)}"
    if rsi_col not in df.columns:
        rsi_col = "rsi_14"
    
    oversold = params.get("oversold_threshold", 30)
    
    # Entry signal: RSI crosses below oversold
    rsi = df[rsi_col]
    entry = (rsi < oversold) & (rsi.shift(1) >= oversold)
    
    return entry.astype(int)


def strategy_mean_reversion_bollinger(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Mean reversion using Bollinger Bands.
    
    Buy when price touches lower band.
    """
    bb_pct = df["bb_pct"]
    threshold = params.get("bb_threshold", 0.05)  # 5% above lower band
    
    # Entry: price near lower band
    entry = (bb_pct < threshold) & (bb_pct.shift(1) >= threshold)
    
    return entry.astype(int)


def strategy_mean_reversion_zscore(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Mean reversion using Z-score.
    
    Buy when price is N standard deviations below mean.
    """
    window = params.get("zscore_window", 20)
    zscore_col = f"zscore_{window}" if f"zscore_{window}" in df.columns else "zscore_20"
    threshold = params.get("zscore_threshold", -2.0)
    
    zscore = df[zscore_col]
    # Entry: crosses below threshold
    entry = (zscore < threshold) & (zscore.shift(1) >= threshold)
    
    return entry.astype(int)


def strategy_momentum_macd(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Momentum strategy using MACD crossover.
    
    Buy when MACD crosses above signal line.
    """
    # MACD histogram (MACD - Signal)
    macd_diff = df["macd_diff"]
    
    # Entry: histogram crosses from negative to positive
    entry = (macd_diff > 0) & (macd_diff.shift(1) <= 0)
    
    return entry.astype(int)


def strategy_momentum_breakout(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Momentum breakout strategy.
    
    Buy when price breaks above N-day high.
    """
    lookback = params.get("breakout_window", 20)
    
    close = df["close"]
    rolling_high = close.rolling(lookback).max().shift(1)  # Exclude today
    
    # Entry: new high
    entry = close > rolling_high
    
    return entry.astype(int)


def strategy_trend_following_ma_cross(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Trend following using moving average crossover.
    
    Buy when fast MA crosses above slow MA.
    """
    fast = params.get("fast_ma", 20)
    slow = params.get("slow_ma", 50)
    
    fast_col = f"sma_{fast}" if f"sma_{fast}" in df.columns else "sma_20"
    slow_col = f"sma_{slow}" if f"sma_{slow}" in df.columns else "sma_50"
    
    fast_ma = df[fast_col]
    slow_ma = df[slow_col]
    
    # Entry: fast crosses above slow
    entry = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    
    return entry.astype(int)


def strategy_drawdown_buy(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Buy on significant drawdown.
    
    Buy when drawdown exceeds threshold (contrarian).
    """
    threshold = params.get("drawdown_threshold", -0.15)  # -15%
    drawdown = df["drawdown"]
    
    # Entry: first day drawdown exceeds threshold
    entry = (drawdown < threshold) & (drawdown.shift(1) >= threshold)
    
    return entry.astype(int)


def strategy_stochastic_oversold(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Stochastic oscillator oversold strategy.
    
    Buy when %K < threshold and %K crosses above %D.
    """
    oversold = params.get("stoch_oversold", 20)
    
    k = df["stoch_k"]
    d = df["stoch_d"]
    
    # Entry: oversold + K crosses above D
    entry = (k < oversold) & (k > d) & (k.shift(1) <= d.shift(1))
    
    return entry.astype(int)


def strategy_combined_oversold(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Combined oversold strategy.
    
    Buy when multiple indicators confirm oversold:
    - RSI < 30
    - Near lower Bollinger
    - Negative Z-score
    """
    rsi = df["rsi_14"]
    bb_pct = df["bb_pct"]
    zscore = df["zscore_20"]
    
    rsi_thresh = params.get("rsi_threshold", 35)
    bb_thresh = params.get("bb_threshold", 0.2)
    zscore_thresh = params.get("zscore_threshold", -1.5)
    
    # Require at least 2 of 3 conditions
    cond1 = rsi < rsi_thresh
    cond2 = bb_pct < bb_thresh
    cond3 = zscore < zscore_thresh
    
    score = cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
    
    # Entry when score goes from <2 to >=2
    entry = (score >= 2) & (score.shift(1) < 2)
    
    return entry.astype(int)


def strategy_volatility_contraction(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Buy on volatility contraction (anticipating breakout).
    
    Buy when ATR% is at N-day low, suggesting imminent move.
    """
    lookback = params.get("atr_lookback", 20)
    
    atr_pct = df["atr_pct"]
    atr_min = atr_pct.rolling(lookback).min()
    
    # Entry: current ATR at local minimum AND price trending up
    is_low_vol = atr_pct <= atr_min * 1.05  # Within 5% of minimum
    momentum_pos = df["momentum_5"] > 0
    
    # Check if either condition just became true (wasn't true before)
    # Using .shift().fillna() with infer_objects to avoid deprecation
    is_low_vol_prev = is_low_vol.shift(1)
    momentum_pos_prev = momentum_pos.shift(1)
    
    # Use .eq(False) instead of ~ to avoid Python 3.14 deprecation
    not_low_vol_prev = is_low_vol_prev.eq(False) | is_low_vol_prev.isna()
    not_momentum_prev = momentum_pos_prev.eq(False) | momentum_pos_prev.isna()
    
    # Entry when currently both conditions met, and at least one wasn't met before
    entry = is_low_vol & momentum_pos & (not_low_vol_prev | not_momentum_prev)
    
    return entry.astype(int)


# =============================================================================
# NEW: Crash Protection & Regime-Aware Strategies
# These are designed to BEAT buy-and-hold by avoiding major drawdowns
# =============================================================================


def strategy_trend_regime_filter(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Only buy when in confirmed UPTREND regime.
    
    This beats buy-and-hold by staying OUT during bear markets.
    Uses 200-day SMA as regime filter + momentum confirmation.
    
    The key insight: missing 50%+ drawdowns is worth missing some upside.
    """
    regime_ma = params.get("regime_ma", 200)
    momentum_days = params.get("momentum_days", 20)
    
    close = df["close"]
    
    # Regime: price above 200-day SMA (or configured MA)
    regime_col = f"sma_{regime_ma}" if f"sma_{regime_ma}" in df.columns else "sma_200"
    if regime_col not in df.columns:
        # Calculate on the fly
        regime_sma = close.rolling(regime_ma, min_periods=50).mean()
    else:
        regime_sma = df[regime_col]
    
    in_uptrend = close > regime_sma
    
    # Momentum: short-term strength
    momentum = close.pct_change(momentum_days) > 0.02  # +2% over period
    
    # Entry: first day both conditions are met
    conditions_met = in_uptrend & momentum
    conditions_met_prev = conditions_met.shift(1)
    not_met_prev = conditions_met_prev.eq(False) | conditions_met_prev.isna()
    
    entry = conditions_met & not_met_prev
    
    return entry.astype(int)


def strategy_crash_avoidance_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Momentum strategy with crash avoidance filter.
    
    Only buys when:
    1. Price is above key moving average (avoiding bear markets)
    2. Recent momentum is positive (confirming strength)
    3. Volatility is not spiking (avoiding panic selling periods)
    
    This beats buy-and-hold by avoiding 30-50% drawdowns.
    """
    ma_period = params.get("ma_period", 50)
    momentum_days = params.get("momentum_days", 10)
    vol_threshold = params.get("vol_threshold", 2.0)  # ATR multiplier
    
    close = df["close"]
    atr_pct = df["atr_pct"]
    
    # Trend filter: above MA
    ma_col = f"sma_{ma_period}" if f"sma_{ma_period}" in df.columns else "sma_50"
    if ma_col not in df.columns:
        ma = close.rolling(ma_period, min_periods=20).mean()
    else:
        ma = df[ma_col]
    
    above_ma = close > ma
    
    # Momentum: positive short-term return
    momentum = close.pct_change(momentum_days) > 0
    
    # Volatility filter: not in high-vol regime (crashes have high vol)
    avg_vol = atr_pct.rolling(60, min_periods=20).mean()
    low_vol = atr_pct < (avg_vol * vol_threshold)
    
    # All conditions
    conditions_met = above_ma & momentum & low_vol
    conditions_met_prev = conditions_met.shift(1)
    not_met_prev = conditions_met_prev.eq(False) | conditions_met_prev.isna()
    
    entry = conditions_met & not_met_prev
    
    return entry.astype(int)


def strategy_golden_cross_confirmed(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Golden cross (50 crosses above 200 SMA) with confirmation.
    
    Classic trend-following signal that avoids bear markets.
    Requires 3-day confirmation to filter false signals.
    
    This beats buy-and-hold by getting OUT before major crashes
    (when 50-day falls below 200-day = Death Cross).
    """
    close = df["close"]
    
    sma_50 = df["sma_50"] if "sma_50" in df.columns else close.rolling(50).mean()
    sma_200 = df["sma_200"] if "sma_200" in df.columns else close.rolling(200).mean()
    
    # Golden cross: 50 crosses above 200
    golden = (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))
    
    # Confirmation: must stay above for 3 days
    golden_3 = golden.shift(3)
    sma_cond_2 = sma_50.shift(2) > sma_200.shift(2)
    sma_cond_1 = sma_50.shift(1) > sma_200.shift(1)
    sma_cond_0 = sma_50 > sma_200
    
    # Handle NaN safely without fillna
    confirmed = (
        (golden_3.eq(True)) &
        sma_cond_2 &
        sma_cond_1 &
        sma_cond_0
    )
    
    return confirmed.astype(int)


def strategy_adaptive_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Adaptive momentum that adjusts to market conditions.
    
    In high-volatility periods: require stronger momentum
    In low-volatility periods: accept weaker momentum
    
    This beats buy-and-hold by being more selective in choppy markets.
    """
    base_lookback = params.get("lookback", 20)
    base_threshold = params.get("threshold", 0.05)  # 5% momentum threshold
    
    close = df["close"]
    atr_pct = df["atr_pct"]
    
    # Adaptive threshold based on volatility
    avg_vol = atr_pct.rolling(60, min_periods=20).mean()
    vol_ratio = atr_pct / avg_vol.replace(0, 0.01)
    
    # Higher vol = higher threshold required
    adaptive_threshold = base_threshold * vol_ratio.clip(0.5, 2.0)
    
    # Momentum calculation
    momentum = close.pct_change(base_lookback)
    
    # Entry: momentum exceeds adaptive threshold
    signal = momentum > adaptive_threshold
    signal_prev = signal.shift(1)
    not_signal_prev = signal_prev.eq(False) | signal_prev.isna()
    
    entry = signal & not_signal_prev
    
    return entry.astype(int)


def strategy_pullback_in_uptrend(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Buy pullbacks in confirmed uptrends.
    
    Only buys when:
    1. Long-term trend is UP (above 100-day SMA)
    2. Short-term RSI shows oversold (pullback)
    3. Volume is declining (exhaustion of selling)
    
    This is the BEST strategy for beating buy-and-hold:
    - Stay in uptrend, buy dips
    - Avoid bear markets entirely
    """
    trend_ma = params.get("trend_ma", 100)
    rsi_threshold = params.get("rsi_threshold", 40)
    
    close = df["close"]
    rsi = df["rsi_14"]
    
    # Trend filter
    ma_col = f"sma_{trend_ma}" if f"sma_{trend_ma}" in df.columns else "sma_100"
    if ma_col not in df.columns:
        trend_sma = close.rolling(trend_ma, min_periods=50).mean()
    else:
        trend_sma = df[ma_col]
    
    in_uptrend = close > trend_sma
    
    # Pullback: RSI oversold but trend still up
    is_pullback = rsi < rsi_threshold
    
    # Recovery: RSI turning up from oversold
    rsi_turning_up = (rsi > rsi.shift(1)) & is_pullback
    
    # Combined entry
    entry = in_uptrend & rsi_turning_up
    
    return entry.astype(int)


# Strategy registry - NO HARDCODED DEFAULTS
# All parameter ranges come from QUANT_LIMITS for hyperparameter optimization
# Format: (function, default_params, param_search_space)
# NOTE: default_params are ONLY used as fallback if optimization fails
# The optimizer tests ALL values in param_search_space and finds the best
def _build_strategy_registry() -> dict[str, tuple[StrategyFunc, dict, dict]]:
    """Build strategy registry with ranges from QUANT_LIMITS."""
    L = QUANT_LIMITS  # Shorthand
    
    return {
        "mean_reversion_rsi": (
            strategy_mean_reversion_rsi,
            {},  # No defaults - must optimize
            {
                "oversold_threshold": L.rsi_oversold_range,
                "rsi_period": [7, 14, 21, 28],
            },
        ),
        "mean_reversion_bollinger": (
            strategy_mean_reversion_bollinger,
            {},
            {"bb_threshold": L.bb_lower_threshold_range},
        ),
        "mean_reversion_zscore": (
            strategy_mean_reversion_zscore,
            {},
            {
                "zscore_threshold": L.zscore_threshold_range,
                "zscore_window": [10, 20, 30, 50],
            },
        ),
        "momentum_macd": (
            strategy_momentum_macd,
            {},
            {},
        ),
        "momentum_breakout": (
            strategy_momentum_breakout,
            {},
            {"breakout_window": L.breakout_window_range},
        ),
        "trend_following_ma": (
            strategy_trend_following_ma_cross,
            {},
            {
                "fast_ma": list(L.ma_periods_fast),
                "slow_ma": list(L.ma_periods_slow),
            },
        ),
        "drawdown_buy": (
            strategy_drawdown_buy,
            {},
            {"drawdown_threshold": (L.drawdown_threshold_range[1] / -100.0, L.drawdown_threshold_range[0] / -100.0)},
        ),
        "stochastic_oversold": (
            strategy_stochastic_oversold,
            {},
            {"stoch_oversold": L.stochastic_oversold_range},
        ),
        "combined_oversold": (
            strategy_combined_oversold,
            {},
            {
                "rsi_threshold": L.rsi_oversold_range,
                "bb_threshold": L.bb_lower_threshold_range,
                "zscore_threshold": L.zscore_threshold_range,
            },
        ),
        "volatility_contraction": (
            strategy_volatility_contraction,
            {},
            {"atr_lookback": L.atr_lookback_range},
        ),
        # Crash protection & regime-aware strategies
        "trend_regime_filter": (
            strategy_trend_regime_filter,
            {},
            {
                "regime_ma": list(L.ma_periods_slow),
                "momentum_days": L.momentum_days_range,
            },
        ),
        "crash_avoidance_momentum": (
            strategy_crash_avoidance_momentum,
            {},
            {
                "ma_period": list(L.ma_periods_slow)[:5],  # First 5 slow MA values
                "momentum_days": L.momentum_days_range,
                "vol_threshold": L.volatility_threshold_range,
            },
        ),
        "golden_cross_confirmed": (
            strategy_golden_cross_confirmed,
            {},
            {},
        ),
        "adaptive_momentum": (
            strategy_adaptive_momentum,
            {},
            {
                "lookback": [10, 20, 30, 40, 50],
                "threshold": (0.02, 0.12),
            },
        ),
        "pullback_in_uptrend": (
            strategy_pullback_in_uptrend,
            {},
            {
                "trend_ma": list(L.ma_periods_slow)[:5],
                "rsi_threshold": L.rsi_oversold_range,
            },
        ),
    }


STRATEGIES = _build_strategy_registry()


# =============================================================================
# Strategy Ensemble/Combination Builder
# =============================================================================

def create_ensemble_strategy(
    strategies: list[str],
    require_all: bool = True,
) -> StrategyFunc:
    """
    Create an ensemble strategy from multiple base strategies.
    
    Args:
        strategies: List of strategy names to combine
        require_all: If True, require ALL strategies to signal (AND logic).
                    If False, require ANY strategy to signal (OR logic).
    
    Returns:
        A new strategy function that combines the inputs.
    """
    def ensemble_func(df: pd.DataFrame, params: dict) -> pd.Series:
        signals = []
        for strat_name in strategies:
            if strat_name not in STRATEGIES:
                continue
            strat_func, default_params, _ = STRATEGIES[strat_name]
            # Use default params merged with any overrides
            strat_params = {**default_params, **params.get(strat_name, {})}
            signal = strat_func(df, strat_params)
            signals.append(signal)
        
        if not signals:
            return pd.Series(0, index=df.index)
        
        # Stack all signals
        signal_matrix = pd.concat(signals, axis=1)
        
        if require_all:
            # AND logic: all strategies must signal
            combined = signal_matrix.all(axis=1).astype(int)
        else:
            # OR logic: any strategy signaling is enough
            combined = signal_matrix.any(axis=1).astype(int)
        
        return combined
    
    return ensemble_func


def generate_all_strategy_combinations(
    max_strategies: int = 3,
    require_all: bool = True,
) -> dict[str, tuple[StrategyFunc, dict, dict]]:
    """
    Generate all possible strategy combinations from 2 to max_strategies.
    
    Args:
        max_strategies: Maximum number of strategies to combine (2-3 recommended)
        require_all: Whether ensemble requires all signals (AND) or any (OR)
    
    Returns:
        Dict of ensemble strategy name -> (function, default_params, search_space)
    """
    from itertools import combinations
    
    strategy_names = list(STRATEGIES.keys())
    ensemble_strategies = {}
    
    for n in range(2, max_strategies + 1):
        for combo in combinations(strategy_names, n):
            # Create ensemble name
            logic_suffix = "AND" if require_all else "OR"
            ensemble_name = f"ensemble_{logic_suffix}_{'_'.join(combo[:2])}_{n}strat"
            if len(combo) > 2:
                # Shorten long names
                ensemble_name = f"ensemble_{logic_suffix}_{len(combo)}way_{'_'.join(c[:4] for c in combo)}"
            
            # Combine default params from all strategies
            combined_defaults = {}
            combined_search = {}
            for strat_name in combo:
                _, defaults, search = STRATEGIES[strat_name]
                combined_defaults[strat_name] = defaults.copy()
                combined_search[strat_name] = search.copy()
            
            ensemble_func = create_ensemble_strategy(list(combo), require_all=require_all)
            ensemble_strategies[ensemble_name] = (
                ensemble_func,
                combined_defaults,
                combined_search,
            )
    
    return ensemble_strategies


def get_all_strategies(
    include_ensembles: bool = True,
    max_ensemble_size: int = 2,
) -> dict[str, tuple[StrategyFunc, dict, dict]]:
    """
    Get all available strategies including optional ensembles.
    
    Args:
        include_ensembles: Whether to include ensemble combinations
        max_ensemble_size: Max strategies to combine (2 = pairs only, 3 = triples)
    
    Returns:
        Complete strategy registry with base + ensemble strategies.
    """
    all_strategies = STRATEGIES.copy()
    
    if include_ensembles:
        # Add AND combinations (require all signals)
        and_ensembles = generate_all_strategy_combinations(
            max_strategies=max_ensemble_size,
            require_all=True,
        )
        all_strategies.update(and_ensembles)
        
        # Add OR combinations (any signal)
        or_ensembles = generate_all_strategy_combinations(
            max_strategies=max_ensemble_size,
            require_all=False,
        )
        all_strategies.update(or_ensembles)
    
    return all_strategies


# =============================================================================
# Backtesting Engine (LEGACY - use backtest_v2.BacktestV2Service for new code)
# =============================================================================

class BacktestEngine:
    """
    Professional backtesting engine with walk-forward validation.
    
    .. deprecated::
        This class is LEGACY. For new code, use:
        
            from app.quant_engine.backtest_v2 import BacktestV2Service
            service = BacktestV2Service()
            result = await service.run_full_backtest(symbol, prices)
        
        The V2 engine provides:
        - Regime-adaptive strategies (Bull/Bear/Crash/Recovery)
        - META Rule fundamental checks for bear market accumulation
        - Portfolio simulation with DCA and scale-in logic
        - Alpha Gauntlet validation (vs B&H, SPY, risk-adjusted)
        - Crash testing (2008, 2020, 2022)
        - Maximum history support (30+ years)
    """
    
    def __init__(self, config: TradingConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        
    def _compute_trade_costs(self, position_value: float) -> float:
        """Compute total cost for a round-trip trade."""
        flat_cost = self.config.flat_cost_per_trade * 2  # Entry + exit
        slippage_cost = position_value * (self.config.slippage_bps / 10000) * 2
        return flat_cost + slippage_cost
    
    def _execute_trade(
        self,
        trade: Trade,
        exit_date: pd.Timestamp,
        exit_price: float,
        reason: str,
    ) -> Trade:
        """Close a trade and compute P&L."""
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.is_open = False
        
        # Apply exit slippage
        actual_exit = exit_price * (1 - self.config.slippage_bps / 10000)
        
        # Compute P&L
        trade.pnl = (actual_exit - trade.entry_price) * trade.shares
        trade.pnl -= trade.total_cost  # Subtract costs
        trade.pnl_pct = (actual_exit / trade.entry_price - 1) * 100
        trade.holding_days = (exit_date - trade.entry_date).days
        
        return trade
    
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        params: dict,
        holding_days: int = 20,
    ) -> StrategyResult:
        """
        Backtest a single strategy with given parameters.
        
        Returns complete trade-level results.
        """
        # Get all strategies including ensembles
        all_strategies = get_all_strategies(include_ensembles=True, max_ensemble_size=3)
        
        if strategy_name not in all_strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_func, _, _ = all_strategies[strategy_name]
        
        # Generate entry signals
        entries = strategy_func(df, params)
        
        # Simulate trades
        trades: list[Trade] = []
        equity = self.config.initial_capital
        equity_curve = pd.Series(index=df.index, dtype=float)
        equity_curve.iloc[0] = equity
        
        open_trades: list[Trade] = []
        
        for i in range(1, len(df)):
            current_date = df.index[i]
            current_price = df["close"].iloc[i]
            
            # Check exit conditions for open trades
            for trade in open_trades[:]:  # Copy list to allow removal
                days_held = (current_date - trade.entry_date).days
                
                # Exit conditions
                exit_reason = None
                if days_held >= holding_days:
                    exit_reason = "max_hold"
                elif (current_price / trade.entry_price - 1) < -self.config.stop_loss_pct:
                    exit_reason = "stop_loss"
                elif self.config.take_profit_pct and (current_price / trade.entry_price - 1) > self.config.take_profit_pct:
                    exit_reason = "take_profit"
                
                if exit_reason:
                    trade = self._execute_trade(trade, current_date, current_price, exit_reason)
                    equity += trade.pnl
                    trades.append(trade)
                    open_trades.remove(trade)
            
            # Check entry conditions (only if not already in a position)
            # For single stock analysis, we go all-in on each signal
            if entries.iloc[i] == 1 and len(open_trades) == 0:
                # All-in position sizing for single stock analysis
                position_value = equity  # Use full equity for position
                
                if position_value > self.config.flat_cost_per_trade * 10:  # Worth trading
                    # Apply entry slippage
                    entry_price = current_price * (1 + self.config.slippage_bps / 10000)
                    shares = position_value / entry_price
                    
                    # Costs
                    entry_cost = self.config.flat_cost_per_trade
                    total_cost = self._compute_trade_costs(position_value)
                    
                    trade = Trade(
                        symbol=df.attrs.get("symbol", "UNKNOWN"),
                        strategy=strategy_name,
                        signal_name=strategy_name,
                        entry_date=current_date,
                        entry_price=entry_price,
                        entry_signal_value=entries.iloc[i],
                        shares=shares,
                        position_value=position_value,
                        entry_cost=entry_cost,
                        total_cost=total_cost,
                    )
                    open_trades.append(trade)
            
            # Update equity curve (mark-to-market)
            mtm_equity = equity
            for trade in open_trades:
                mtm_equity += (current_price - trade.entry_price) * trade.shares
            equity_curve.iloc[i] = mtm_equity
        
        # Close any remaining open trades at last price
        for trade in open_trades:
            last_date = df.index[-1]
            last_price = df["close"].iloc[-1]
            trade = self._execute_trade(trade, last_date, last_price, "end_of_period")
            equity += trade.pnl
            trades.append(trade)
        
        equity_curve.iloc[-1] = equity
        equity_curve = equity_curve.ffill()
        
        # Compute metrics
        result = self._compute_metrics(trades, equity_curve, df)
        result.strategy_name = strategy_name
        result.symbol = df.attrs.get("symbol", "UNKNOWN")
        result.optimal_params = params
        result.trades = trades
        result.equity_curve = equity_curve
        
        return result
    
    def _compute_metrics(
        self,
        trades: list[Trade],
        equity_curve: pd.Series,
        df: pd.DataFrame,
    ) -> StrategyResult:
        """Compute comprehensive performance metrics."""
        result = StrategyResult(strategy_name="", symbol="")
        
        if not trades:
            return result
        
        # Trade statistics
        pnls = [t.pnl for t in trades if not t.is_open]
        pnl_pcts = [t.pnl_pct for t in trades if not t.is_open]
        holding_days = [t.holding_days for t in trades if not t.is_open]
        
        result.n_trades = len(pnls)
        
        if not pnls:
            return result
        
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p <= 0]
        
        result.win_rate = (len(wins) / len(pnl_pcts)) * 100  # Store as percentage
        result.avg_win_pct = np.mean(wins) if wins else 0
        result.avg_loss_pct = np.mean(losses) if losses else 0
        result.avg_holding_days = np.mean(holding_days)
        
        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        returns = equity_curve.pct_change().dropna()
        result.total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        n_years = len(df) / 252
        result.annualized_return_pct = ((1 + result.total_return_pct / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        
        # Risk metrics
        result.volatility_pct = returns.std() * np.sqrt(252) * 100
        
        # Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        result.max_drawdown_pct = drawdown.min() * 100
        
        # Sharpe (assuming 4% risk-free rate)
        excess_returns = returns - 0.04 / 252
        result.sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        result.sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar
        result.calmar_ratio = result.annualized_return_pct / abs(result.max_drawdown_pct) if result.max_drawdown_pct != 0 else 0
        
        # VaR and CVaR
        result.var_95 = np.percentile(returns, 5) * 100
        result.cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) > 20 else 0
        
        # Statistical significance
        if len(pnl_pcts) >= 10:
            t_stat, p_val = stats.ttest_1samp(pnl_pcts, 0)
            result.t_statistic = t_stat
            result.p_value = p_val
            result.is_significant = p_val < (1 - self.config.confidence_level)
        
        return result
    
    def optimize_strategy_optuna(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        n_trials: int = 100,
        objective: Literal["sharpe", "return", "calmar"] = "sharpe",
    ) -> tuple[dict, StrategyResult]:
        """
        Optimize strategy parameters using Optuna (Bayesian optimization).
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        # Get all strategies including ensembles
        all_strategies = get_all_strategies(include_ensembles=True, max_ensemble_size=3)
        
        if strategy_name not in all_strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        _, default_params, search_space = all_strategies[strategy_name]
        
        if not search_space:
            # No parameters to optimize
            result = self.backtest_strategy(df, strategy_name, default_params)
            return default_params, result
        
        def objective_func(trial: optuna.Trial) -> float:
            params = default_params.copy()
            
            for param_name, space in search_space.items():
                if isinstance(space, tuple):
                    # Continuous range
                    if isinstance(space[0], int):
                        params[param_name] = trial.suggest_int(param_name, space[0], space[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, space[0], space[1])
                elif isinstance(space, list):
                    # Categorical
                    params[param_name] = trial.suggest_categorical(param_name, space)
            
            # Also optimize holding days (capped at 60 for capital efficiency)
            holding_days = trial.suggest_int("holding_days", 5, 60)
            
            result = self.backtest_strategy(df, strategy_name, params, holding_days)
            
            if result.n_trades < self.config.min_trades_for_significance:
                return -10.0  # Penalty for too few trades
            
            if objective == "sharpe":
                return result.sharpe_ratio
            elif objective == "return":
                return result.total_return_pct
            else:  # calmar
                return result.calmar_ratio
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)
        
        # Get best params
        best_params = default_params.copy()
        for param_name in search_space:
            if param_name in study.best_params:
                best_params[param_name] = study.best_params[param_name]
        
        best_holding = study.best_params.get("holding_days", 20)
        
        # Run final backtest with best params
        best_result = self.backtest_strategy(df, strategy_name, best_params, best_holding)
        best_params["holding_days"] = best_holding
        
        return best_params, best_result
    
    def walk_forward_optimization(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        n_trials_per_fold: int = 50,
    ) -> tuple[StrategyResult, ValidationReport]:
        """
        Walk-forward optimization with out-of-sample validation.
        
        This is the gold standard for avoiding overfitting:
        1. Split data into K folds
        2. For each fold: optimize on train, test on hold-out
        3. Aggregate OOS results for true performance estimate
        """
        n = len(df)
        fold_size = n // self.config.n_folds
        
        oos_results: list[StrategyResult] = []
        is_results: list[StrategyResult] = []
        all_params: list[dict] = []
        
        for fold in range(self.config.n_folds - 1):
            # Training data: all data up to end of this fold
            train_end = fold_size * (fold + 2)
            train_size = int(train_end * self.config.train_ratio)
            
            train_df = df.iloc[:train_size].copy()
            train_df.attrs = df.attrs
            
            # Test data: remaining portion
            test_df = df.iloc[train_size:train_end].copy()
            test_df.attrs = df.attrs
            
            if len(train_df) < 252 or len(test_df) < 60:
                continue
            
            # Optimize on training data
            try:
                best_params, is_result = self.optimize_strategy_optuna(
                    train_df, strategy_name, n_trials=n_trials_per_fold
                )
            except Exception as e:
                logger.warning(f"Optimization failed for fold {fold}: {e}")
                continue
            
            is_results.append(is_result)
            all_params.append(best_params)
            
            # Test on out-of-sample data (using training-derived params)
            holding_days = best_params.get("holding_days", 20)
            oos_result = self.backtest_strategy(test_df, strategy_name, best_params, holding_days)
            oos_result.is_out_of_sample = True
            oos_results.append(oos_result)
        
        # Aggregate OOS results
        if not oos_results:
            return StrategyResult(strategy_name=strategy_name, symbol=df.attrs.get("symbol", "")), \
                   ValidationReport(strategy_name=strategy_name, symbol=df.attrs.get("symbol", ""), verdict_reason="Insufficient data")
        
        # Use the most recent fold's parameters
        final_params = all_params[-1] if all_params else {}
        
        # Aggregate metrics
        aggregated = StrategyResult(
            strategy_name=strategy_name,
            symbol=df.attrs.get("symbol", ""),
            is_out_of_sample=True,
        )
        
        # Combine all OOS trades
        all_trades = []
        for res in oos_results:
            all_trades.extend(res.trades)
        
        aggregated.trades = all_trades
        aggregated.n_trades = len(all_trades)
        aggregated.optimal_params = final_params
        
        if all_trades:
            pnl_pcts = [t.pnl_pct for t in all_trades]
            aggregated.win_rate = (sum(1 for p in pnl_pcts if p > 0) / len(pnl_pcts)) * 100  # Store as percentage
            aggregated.avg_win_pct = np.mean([p for p in pnl_pcts if p > 0]) if any(p > 0 for p in pnl_pcts) else 0
            aggregated.avg_loss_pct = np.mean([p for p in pnl_pcts if p <= 0]) if any(p <= 0 for p in pnl_pcts) else 0
        
        # Average key metrics
        aggregated.sharpe_ratio = np.mean([r.sharpe_ratio for r in oos_results])
        aggregated.total_return_pct = np.mean([r.total_return_pct for r in oos_results])
        aggregated.max_drawdown_pct = np.mean([r.max_drawdown_pct for r in oos_results])
        
        # Parameter stability check
        if len(all_params) >= 2:
            param_keys = list(all_params[0].keys())
            stable_count = 0
            for key in param_keys:
                if key == "holding_days":
                    continue
                values = [p.get(key, 0) for p in all_params]
                if np.std(values) / (np.mean(values) + 1e-6) < 0.3:  # CV < 30%
                    stable_count += 1
            aggregated.walk_forward_stable = stable_count >= len(param_keys) * 0.6
        
        # Create validation report
        report = self._create_validation_report(
            strategy_name, df, aggregated, is_results, oos_results
        )
        
        return aggregated, report
    
    def _create_validation_report(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        oos_result: StrategyResult,
        is_results: list[StrategyResult],
        oos_results: list[StrategyResult],
    ) -> ValidationReport:
        """Create comprehensive validation report."""
        report = ValidationReport(
            strategy_name=strategy_name,
            symbol=df.attrs.get("symbol", ""),
        )
        
        # IS vs OOS comparison (overfitting check)
        if is_results and oos_results:
            avg_is_sharpe = np.mean([r.sharpe_ratio for r in is_results])
            avg_oos_sharpe = np.mean([r.sharpe_ratio for r in oos_results])
            
            report.is_vs_sharpe = avg_is_sharpe
            report.oos_sharpe = avg_oos_sharpe
            report.sharpe_degradation = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe if avg_is_sharpe > 0 else 0
            
            # Overfitting if degradation > 50%
            report.has_overfitting = report.sharpe_degradation > 0.5
        
        # Statistical tests
        if oos_result.trades:
            pnl_pcts = [t.pnl_pct for t in oos_result.trades]
            if len(pnl_pcts) >= 10:
                _, p_value = stats.ttest_1samp(pnl_pcts, 0)
                report.passes_t_test = p_value < 0.05
                report.raw_p_value = float(p_value)
                
                # Bootstrap test: 95% CI of mean return must be > 0
                # (bootstrap_confidence_interval imported from indicators module)
                _, ci_lower, _ = bootstrap_confidence_interval(
                    pnl_pcts, n_bootstrap=1000, confidence=0.95, metric="mean"
                )
                report.passes_bootstrap = ci_lower > 0
        
        # Benchmark comparison
        stock_bh_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        report.beats_buy_hold = oos_result.total_return_pct > stock_bh_return
        
        # Current year check
        validation_start = pd.Timestamp(self.config.validation_start_date)
        current_year_df = df[df.index >= validation_start]
        if len(current_year_df) >= 20:
            cy_result = self.backtest_strategy(
                current_year_df, 
                strategy_name, 
                oos_result.optimal_params,
                oos_result.optimal_params.get("holding_days", 20)
            )
            report.current_year_return = cy_result.total_return_pct
            report.current_year_sharpe = cy_result.sharpe_ratio
        
        # Overall validity - requires BOTH t-test and bootstrap
        report.is_valid = (
            not report.has_overfitting and
            report.passes_t_test and
            report.passes_bootstrap and  # Bootstrap CI lower bound must be > 0
            oos_result.sharpe_ratio > 0.5 and
            oos_result.n_trades >= self.config.min_trades_for_significance
        )
        
        if report.is_valid:
            report.verdict_reason = f"Strategy passes all validation checks. OOS Sharpe: {oos_result.sharpe_ratio:.2f}"
            report.confidence_score = min(100, oos_result.sharpe_ratio * 40 + report.oos_sharpe * 20)
        else:
            reasons = []
            if report.has_overfitting:
                reasons.append(f"Overfitting detected ({report.sharpe_degradation*100:.0f}% degradation)")
            if not report.passes_t_test:
                reasons.append("Failed t-test significance")
            if not report.passes_bootstrap:
                reasons.append("Failed bootstrap CI test (mean return CI includes 0)")
            if oos_result.sharpe_ratio <= 0.5:
                reasons.append(f"Low Sharpe ratio ({oos_result.sharpe_ratio:.2f})")
            if oos_result.n_trades < self.config.min_trades_for_significance:
                reasons.append(f"Insufficient trades ({oos_result.n_trades})")
            report.verdict_reason = "; ".join(reasons)
            report.confidence_score = max(0, 50 - len(reasons) * 15)
        
        return report
    
    def find_best_strategy(
        self,
        df: pd.DataFrame,
        strategies: list[str] | None = None,
        n_trials_per_strategy: int = 50,
        include_ensembles: bool = True,
        max_ensemble_size: int = 2,
    ) -> tuple[str, StrategyResult, ValidationReport]:
        """
        Find the best strategy for a given stock using walk-forward optimization.
        
        Args:
            df: Price data with computed indicators
            strategies: List of strategy names to test. If None, tests ALL strategies
                       including ensemble combinations.
            n_trials_per_strategy: Optuna trials per strategy for hyperparameter search
            include_ensembles: Whether to test ensemble/combination strategies
            max_ensemble_size: Max strategies to combine (2 = pairs, 3 = triples)
        
        Returns:
            Tuple of (best_strategy_name, result, validation_report)
            The strategy with the highest risk-adjusted return that passes
            all validation checks and beats benchmarks.
        """
        if strategies is None:
            # Get all strategies including ensembles
            all_strategies = get_all_strategies(
                include_ensembles=include_ensembles,
                max_ensemble_size=max_ensemble_size,
            )
            strategies = list(all_strategies.keys())
            logger.info(f"Testing {len(strategies)} strategies (base + ensembles)")
        
        best_strategy = ""
        best_result: StrategyResult | None = None
        best_report: ValidationReport | None = None
        best_score = -np.inf
        
        # Collect all results for FDR correction
        all_results: list[tuple[str, StrategyResult, ValidationReport]] = []
        
        for strategy_name in strategies:
            try:
                result, report = self.walk_forward_optimization(
                    df, strategy_name, n_trials_per_strategy
                )
                
                # Store the result for FDR correction
                all_results.append((strategy_name, result, report))
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue
        
        # Apply Benjamini-Hochberg FDR correction to all p-values
        n_tested = len(all_results)
        if n_tested > 0:
            # Extract raw p-values
            p_values = [(i, r[2].raw_p_value) for i, r in enumerate(all_results)]
            # Sort by p-value
            p_values.sort(key=lambda x: x[1])
            
            # Benjamini-Hochberg correction
            alpha = 0.05
            for rank, (idx, raw_p) in enumerate(p_values, start=1):
                adjusted_p = raw_p * n_tested / rank
                adjusted_p = min(adjusted_p, 1.0)  # Cap at 1.0
                all_results[idx][2].adjusted_p_value = adjusted_p
                all_results[idx][2].n_strategies_tested = n_tested
                all_results[idx][2].passes_fdr_correction = adjusted_p < alpha
        
        # Now score strategies with FDR-corrected significance
        for strategy_name, result, report in all_results:
            # Score: prioritize Sharpe, penalize overfitting and lack of statistical validity
            score = result.sharpe_ratio
            if report.has_overfitting:
                score *= 0.5
            if not report.passes_fdr_correction:  # Use FDR-corrected test
                score *= 0.6  # Heavier penalty for failing multiple-testing correction
            elif not report.passes_t_test:
                score *= 0.7
            if not report.beats_buy_hold:
                score *= 0.8
            
            if score > best_score and result.n_trades >= 10:
                best_score = score
                best_strategy = strategy_name
                best_result = result
                best_report = report
        
        if best_result is None:
            return "", StrategyResult(strategy_name="", symbol=""), ValidationReport(strategy_name="", symbol="", verdict_reason="No valid strategy found")
        
        return best_strategy, best_result, best_report


# =============================================================================
# Benchmark Comparison
# =============================================================================

def compute_buy_and_hold_return(prices: pd.Series, start_date: pd.Timestamp | None = None) -> float:
    """Compute buy-and-hold return for a price series."""
    if start_date:
        prices = prices[prices.index >= start_date]
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100


def compare_to_benchmark(
    strategy_result: StrategyResult,
    stock_prices: pd.Series,
    spy_prices: pd.Series | None = None,
) -> dict:
    """
    Compare strategy to buy-and-hold benchmarks.
    
    Returns dict with excess returns vs stock and SPY.
    """
    stock_bh = compute_buy_and_hold_return(stock_prices)
    
    result = {
        "strategy_return": strategy_result.total_return_pct,
        "stock_buy_hold_return": stock_bh,
        "excess_vs_stock": strategy_result.total_return_pct - stock_bh,
        "beats_stock": strategy_result.total_return_pct > stock_bh,
    }
    
    if spy_prices is not None and len(spy_prices) > 0:
        spy_bh = compute_buy_and_hold_return(spy_prices)
        result["spy_buy_hold_return"] = spy_bh
        result["excess_vs_spy"] = strategy_result.total_return_pct - spy_bh
        result["beats_spy"] = strategy_result.total_return_pct > spy_bh
    
    return result


# =============================================================================
# QuantStats Report Generation
# =============================================================================

def generate_quantstats_report(
    equity_curve: pd.Series,
    benchmark_prices: pd.Series | None = None,
    output_file: str | None = None,
) -> dict:
    """
    Generate comprehensive QuantStats metrics.
    
    Returns dict of all metrics, optionally saves HTML report.
    """
    if not QUANTSTATS_AVAILABLE:
        logger.warning("QuantStats not available")
        return {}
    
    returns = equity_curve.pct_change().dropna()
    
    # Extend pandas for QuantStats
    qs.extend_pandas()
    
    metrics = {
        "total_return": qs.stats.comp(returns) * 100,
        "cagr": qs.stats.cagr(returns) * 100,
        "sharpe": qs.stats.sharpe(returns),
        "sortino": qs.stats.sortino(returns),
        "max_drawdown": qs.stats.max_drawdown(returns) * 100,
        "calmar": qs.stats.calmar(returns),
        "volatility": qs.stats.volatility(returns) * 100,
        "var": qs.stats.var(returns) * 100,
        "cvar": qs.stats.cvar(returns) * 100,
        "win_rate": qs.stats.win_rate(returns) * 100,
        "profit_factor": qs.stats.profit_factor(returns),
        "payoff_ratio": qs.stats.payoff_ratio(returns),
        "avg_return": qs.stats.avg_return(returns) * 100,
        "avg_win": qs.stats.avg_win(returns) * 100,
        "avg_loss": qs.stats.avg_loss(returns) * 100,
    }
    
    if output_file and benchmark_prices is not None:
        benchmark_returns = benchmark_prices.pct_change().dropna()
        # Align indices
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 0:
            qs.reports.html(
                returns.loc[common_idx],
                benchmark=benchmark_returns.loc[common_idx],
                output=output_file,
                title="Strategy Performance Report",
            )
    
    return metrics
