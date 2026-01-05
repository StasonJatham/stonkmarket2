"""
AlphaFactory - Evolutionary Strategy Discovery Engine.

Uses Optuna for Bayesian optimization to discover optimal trading strategies
through a massive search space of TA indicators.

Key Components:
1. IndicatorMatrix - Pre-computes ALL technical indicators once for vectorized access
2. StrategyGenome - Encodes a complete trading strategy as optimizable parameters
3. VectorizedBacktester - Fast numpy-based P&L simulation (1000s/second)
4. AlphaFactory - Optuna-powered strategy evolution with walk-forward validation

Philosophy:
- Pre-compute everything possible for speed
- Use regime-specific configurations (Bull_Config vs Bear_Config)
- Walk-forward prevents overfitting
- Fitness = (Sharpe * 0.4) + (Sortino * 0.4) + (Calmar * 0.2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# Use the already-installed 'ta' library
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import (
    MACD, EMAIndicator, SMAIndicator, ADXIndicator, 
    AroonIndicator, CCIIndicator, DPOIndicator,
)
from ta.volatility import (
    BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel,
)
from ta.volume import (
    OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator,
    AccDistIndexIndicator, MFIIndicator,
)

from app.quant_engine.backtest_v2.regime_filter import MarketRegime

logger = logging.getLogger(__name__)


# =============================================================================
# Logic Gate Enum - How indicators are compared
# =============================================================================

class LogicGate(str, Enum):
    """How to compare indicator value to threshold."""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CROSS_OVER = "cross_over"
    CROSS_UNDER = "cross_under"
    BETWEEN = "between"


class IndicatorType(str, Enum):
    """Available indicator types for strategy construction."""
    # Trend indicators
    SMA = "sma"
    EMA = "ema"
    ADX = "adx"
    AROON = "aroon"
    CCI = "cci"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"
    MACD_HIST = "macd_hist"
    
    # Oscillators
    RSI = "rsi"
    STOCH_K = "stoch_k"
    STOCH_D = "stoch_d"
    WILLIAMS_R = "williams_r"
    MFI = "mfi"
    ROC = "roc"
    
    # Volatility
    ATR = "atr"
    ATR_PCT = "atr_pct"
    BB_UPPER = "bb_upper"
    BB_LOWER = "bb_lower"
    BB_PCT = "bb_pct"  # 0 = lower, 1 = upper
    KELTNER_UPPER = "keltner_upper"
    KELTNER_LOWER = "keltner_lower"
    DONCHIAN_UPPER = "donchian_upper"
    DONCHIAN_LOWER = "donchian_lower"
    
    # Volume
    OBV = "obv"
    ADL = "adl"  # Accumulation/Distribution
    CMF = "cmf"  # Chaikin Money Flow
    
    # Price-relative
    PRICE_VS_SMA = "price_vs_sma"
    PRICE_VS_EMA = "price_vs_ema"
    ZSCORE = "zscore"
    DRAWDOWN = "drawdown"
    
    # Momentum
    MOMENTUM = "momentum"
    RETURNS = "returns"


# =============================================================================
# Indicator Matrix - Pre-compute ALL indicators once
# =============================================================================

class IndicatorMatrix:
    """
    Pre-computes all technical indicators with various parameters.
    
    This enables ultra-fast vectorized backtesting since we only compute once
    and then just lookup values during optimization trials.
    """
    
    # Parameter ranges for each indicator type
    PERIODS = [5, 7, 9, 10, 12, 14, 20, 21, 26, 50, 100, 200]
    
    def __init__(self, prices: pd.DataFrame) -> None:
        """
        Initialize with OHLCV price data.
        
        Args:
            prices: DataFrame with columns [open, high, low, close, volume]
                    (will auto-normalize column names)
        """
        self.df = self._normalize_prices(prices)
        self.close = self.df["close"]
        self.high = self.df["high"]
        self.low = self.df["low"]
        self.open = self.df["open"]
        self.volume = self.df.get("volume", pd.Series(1, index=self.df.index))
        
        # Pre-computed indicator storage
        self._indicators: dict[str, pd.Series] = {}
        
        # Compute all on init
        self._compute_all()
    
    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and handle adj_close."""
        df = df.copy()
        
        # Handle multi-level columns
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        
        # Lowercase
        col_map = {str(c): str(c).lower().replace(' ', '_') for c in df.columns}
        df = df.rename(columns=col_map)
        
        # Use adj_close if available
        if 'adj_close' in df.columns and df['adj_close'].notna().any():
            df['close'] = df['adj_close'].combine_first(df.get('close', df['adj_close']))
        
        # Fill missing OHLV
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' or 'adj_close' column")
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 1
        
        return df
    
    def _compute_all(self) -> None:
        """Pre-compute all indicators with all parameter variations."""
        logger.info(f"Pre-computing indicators for {len(self.df)} bars...")
        
        # Returns (always compute)
        self._indicators["returns"] = self.close.pct_change()
        self._indicators["log_returns"] = np.log(self.close / self.close.shift(1))
        
        for period in self.PERIODS:
            self._compute_for_period(period)
        
        # MACD variants
        self._compute_macd_variants()
        
        logger.info(f"Computed {len(self._indicators)} indicator series")
    
    def _compute_for_period(self, period: int) -> None:
        """Compute all indicators for a given period."""
        close = self.close
        high = self.high
        low = self.low
        volume = self.volume
        n_bars = len(close)
        
        # Skip if not enough data for this period
        # ADX needs 2*period + 1 bars to compute properly
        if n_bars < period * 2 + 1:
            logger.debug(f"Skipping period {period}: only {n_bars} bars available")
            return
        
        # Moving Averages
        self._indicators[f"sma_{period}"] = SMAIndicator(close, window=period).sma_indicator()
        self._indicators[f"ema_{period}"] = EMAIndicator(close, window=period).ema_indicator()
        
        # RSI
        if period >= 5:
            self._indicators[f"rsi_{period}"] = RSIIndicator(close, window=period).rsi()
        
        # Rate of Change
        self._indicators[f"roc_{period}"] = ROCIndicator(close, window=period).roc()
        
        # Williams %R
        if period >= 5:
            self._indicators[f"williams_r_{period}"] = WilliamsRIndicator(
                high, low, close, lbp=period
            ).williams_r()
        
        # ADX
        if period >= 5:
            adx = ADXIndicator(high, low, close, window=period)
            self._indicators[f"adx_{period}"] = adx.adx()
            self._indicators[f"adx_pos_{period}"] = adx.adx_pos()
            self._indicators[f"adx_neg_{period}"] = adx.adx_neg()
        
        # Aroon
        if period >= 5:
            aroon = AroonIndicator(high, low, window=period)
            self._indicators[f"aroon_up_{period}"] = aroon.aroon_up()
            self._indicators[f"aroon_down_{period}"] = aroon.aroon_down()
            self._indicators[f"aroon_ind_{period}"] = aroon.aroon_indicator()
        
        # CCI
        if period >= 5:
            self._indicators[f"cci_{period}"] = CCIIndicator(
                high, low, close, window=period
            ).cci()
        
        # Stochastic
        if period >= 5:
            stoch = StochasticOscillator(high, low, close, window=period, smooth_window=3)
            self._indicators[f"stoch_k_{period}"] = stoch.stoch()
            self._indicators[f"stoch_d_{period}"] = stoch.stoch_signal()
        
        # MFI (Money Flow Index)
        if period >= 5 and self.volume.sum() > len(self.volume):
            self._indicators[f"mfi_{period}"] = MFIIndicator(
                high, low, close, volume, window=period
            ).money_flow_index()
        
        # ATR
        if period >= 5:
            atr = AverageTrueRange(high, low, close, window=period)
            self._indicators[f"atr_{period}"] = atr.average_true_range()
            self._indicators[f"atr_pct_{period}"] = atr.average_true_range() / close
        
        # Bollinger Bands
        if period >= 10:
            bb = BollingerBands(close, window=period, window_dev=2)
            self._indicators[f"bb_upper_{period}"] = bb.bollinger_hband()
            self._indicators[f"bb_lower_{period}"] = bb.bollinger_lband()
            self._indicators[f"bb_pct_{period}"] = bb.bollinger_pband()
        
        # Keltner Channel
        if period >= 10:
            kc = KeltnerChannel(high, low, close, window=period)
            self._indicators[f"keltner_upper_{period}"] = kc.keltner_channel_hband()
            self._indicators[f"keltner_lower_{period}"] = kc.keltner_channel_lband()
        
        # Donchian Channel
        if period >= 5:
            dc = DonchianChannel(high, low, close, window=period)
            self._indicators[f"donchian_upper_{period}"] = dc.donchian_channel_hband()
            self._indicators[f"donchian_lower_{period}"] = dc.donchian_channel_lband()
        
        # Price relative to MAs
        sma = self._indicators[f"sma_{period}"]
        ema = self._indicators[f"ema_{period}"]
        self._indicators[f"price_vs_sma_{period}"] = (close - sma) / sma
        self._indicators[f"price_vs_ema_{period}"] = (close - ema) / ema
        
        # Z-score
        mean = close.rolling(period).mean()
        std = close.rolling(period).std()
        self._indicators[f"zscore_{period}"] = (close - mean) / std.replace(0, np.nan)
        
        # Drawdown from rolling max
        rolling_max = close.rolling(period, min_periods=1).max()
        self._indicators[f"drawdown_{period}"] = (close - rolling_max) / rolling_max
        
        # Momentum
        self._indicators[f"momentum_{period}"] = close.pct_change(period)
    
    def _compute_macd_variants(self) -> None:
        """Compute MACD with different parameters."""
        close = self.close
        
        # Standard MACD (12, 26, 9)
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        self._indicators["macd_12_26"] = macd.macd()
        self._indicators["macd_signal_12_26"] = macd.macd_signal()
        self._indicators["macd_hist_12_26"] = macd.macd_diff()
        
        # Fast MACD (8, 17, 9)
        macd_fast = MACD(close, window_slow=17, window_fast=8, window_sign=9)
        self._indicators["macd_8_17"] = macd_fast.macd()
        self._indicators["macd_signal_8_17"] = macd_fast.macd_signal()
        self._indicators["macd_hist_8_17"] = macd_fast.macd_diff()
        
        # Slow MACD (19, 39, 9)
        macd_slow = MACD(close, window_slow=39, window_fast=19, window_sign=9)
        self._indicators["macd_19_39"] = macd_slow.macd()
        self._indicators["macd_signal_19_39"] = macd_slow.macd_signal()
        self._indicators["macd_hist_19_39"] = macd_slow.macd_diff()
        
        # Volume indicators
        if self.volume.sum() > len(self.volume):
            self._indicators["obv"] = OnBalanceVolumeIndicator(
                self.close, self.volume
            ).on_balance_volume()
            
            self._indicators["adl"] = AccDistIndexIndicator(
                self.high, self.low, self.close, self.volume
            ).acc_dist_index()
            
            self._indicators["cmf_20"] = ChaikinMoneyFlowIndicator(
                self.high, self.low, self.close, self.volume, window=20
            ).chaikin_money_flow()
    
    def get(self, indicator_type: IndicatorType | str, period: int = 14) -> pd.Series:
        """
        Get a pre-computed indicator series.
        
        Args:
            indicator_type: Type of indicator (from IndicatorType enum or string)
            period: Period parameter (ignored for fixed indicators like MACD)
            
        Returns:
            pd.Series with indicator values (or NaN series if not available)
        """
        if isinstance(indicator_type, IndicatorType):
            indicator_type = indicator_type.value
        
        # Handle fixed indicators
        if indicator_type in ["macd", "macd_signal", "macd_hist"]:
            key = f"{indicator_type}_12_26"  # Default MACD
        elif indicator_type in ["obv", "adl", "returns", "log_returns"]:
            key = indicator_type
        elif indicator_type == "cmf":
            key = "cmf_20"
        else:
            key = f"{indicator_type}_{period}"
        
        if key not in self._indicators:
            # Return NaN series instead of raising - allows graceful degradation
            logger.debug(f"Indicator '{key}' not available, returning NaN series")
            return pd.Series(np.nan, index=self.df.index)
        
        return self._indicators[key]
    
    def get_numpy(self, indicator_type: IndicatorType | str, period: int = 14) -> np.ndarray:
        """Get indicator as numpy array for vectorized operations."""
        return self.get(indicator_type, period).values
    
    @property
    def available_indicators(self) -> list[str]:
        """List all available pre-computed indicators."""
        return list(self._indicators.keys())
    
    @property
    def available_periods(self) -> list[int]:
        """List of periods that were computed (based on data length)."""
        # Find which periods have SMA (a simple indicator that should exist)
        periods = []
        for p in self.PERIODS:
            if f"sma_{p}" in self._indicators:
                periods.append(p)
        return periods
    
    def __len__(self) -> int:
        """Number of data points."""
        return len(self.df)


# =============================================================================
# Strategy Genome - Encodes a complete trading strategy
# =============================================================================

@dataclass
class ConditionGenome:
    """A single condition in a strategy (e.g., RSI > 30)."""
    indicator_type: IndicatorType
    period: int
    logic_gate: LogicGate
    threshold: float
    threshold_upper: float | None = None  # For BETWEEN logic
    
    def evaluate(self, matrix: IndicatorMatrix) -> np.ndarray:
        """
        Evaluate this condition across all bars.
        
        Returns:
            Boolean numpy array (True = condition met)
        """
        values = matrix.get_numpy(self.indicator_type, self.period)
        
        if self.logic_gate == LogicGate.GREATER_THAN:
            return values > self.threshold
        elif self.logic_gate == LogicGate.LESS_THAN:
            return values < self.threshold
        elif self.logic_gate == LogicGate.BETWEEN:
            upper = self.threshold_upper or (self.threshold + 20)
            return (values > self.threshold) & (values < upper)
        elif self.logic_gate == LogicGate.CROSS_OVER:
            # Price crosses above threshold
            prev = np.roll(values, 1)
            prev[0] = np.nan
            return (prev < self.threshold) & (values >= self.threshold)
        elif self.logic_gate == LogicGate.CROSS_UNDER:
            # Price crosses below threshold
            prev = np.roll(values, 1)
            prev[0] = np.nan
            return (prev > self.threshold) & (values <= self.threshold)
        else:
            return np.ones(len(values), dtype=bool)


@dataclass
class StrategyGenome:
    """
    Complete strategy encoded as optimizable parameters.
    
    A strategy consists of:
    - Entry conditions (all must be True for entry signal)
    - Exit conditions (any True triggers exit)
    - Position sizing rules
    """
    name: str
    entry_conditions: list[ConditionGenome] = field(default_factory=list)
    exit_conditions: list[ConditionGenome] = field(default_factory=list)
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.20  # 20% take profit
    holding_period_max: int = 252  # Max 1 year hold
    regime: MarketRegime | None = None  # For regime-specific strategies
    
    def generate_entry_signals(self, matrix: IndicatorMatrix) -> np.ndarray:
        """
        Generate entry signals (all conditions must be True).
        
        Returns:
            Boolean array where True = entry signal
        """
        if not self.entry_conditions:
            return np.zeros(len(matrix), dtype=bool)
        
        # All conditions must be True (AND logic)
        result = np.ones(len(matrix), dtype=bool)
        for condition in self.entry_conditions:
            result &= condition.evaluate(matrix)
        
        return result
    
    def generate_exit_signals(self, matrix: IndicatorMatrix) -> np.ndarray:
        """
        Generate exit signals (any condition True triggers exit).
        
        Returns:
            Boolean array where True = exit signal
        """
        if not self.exit_conditions:
            return np.zeros(len(matrix), dtype=bool)
        
        # Any condition True (OR logic)
        result = np.zeros(len(matrix), dtype=bool)
        for condition in self.exit_conditions:
            result |= condition.evaluate(matrix)
        
        return result
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "name": self.name,
            "entry_conditions": [
                {
                    "indicator": c.indicator_type.value,
                    "period": c.period,
                    "logic": c.logic_gate.value,
                    "threshold": c.threshold,
                    "threshold_upper": c.threshold_upper,
                }
                for c in self.entry_conditions
            ],
            "exit_conditions": [
                {
                    "indicator": c.indicator_type.value,
                    "period": c.period,
                    "logic": c.logic_gate.value,
                    "threshold": c.threshold,
                    "threshold_upper": c.threshold_upper,
                }
                for c in self.exit_conditions
            ],
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "holding_period_max": self.holding_period_max,
            "regime": self.regime.value if self.regime else None,
        }


# =============================================================================
# Vectorized Backtester - Fast P&L Simulation
# =============================================================================

@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_holding_days: float = 0.0
    profitable_years_pct: float = 0.0
    fitness_score: float = 0.0
    
    def calculate_fitness(self) -> float:
        """
        Calculate composite fitness score.
        
        Score = (Sharpe * 0.4) + (Sortino * 0.4) + (Calmar * 0.2)
        With constraints for minimum trade count and profitability.
        """
        # Constraints (must pass or score is zeroed)
        if self.num_trades < 20:
            return 0.0
        if self.profitable_years_pct < 0.7:  # 70% of years profitable
            return 0.0
        if self.max_drawdown < -0.5:  # Max 50% drawdown
            return 0.0
        
        # Clip ratios to reasonable range
        sharpe = np.clip(self.sharpe_ratio, -3, 3)
        sortino = np.clip(self.sortino_ratio, -5, 5)
        calmar = np.clip(self.calmar_ratio, -3, 3)
        
        self.fitness_score = (sharpe * 0.4) + (sortino * 0.4) + (calmar * 0.2)
        return self.fitness_score


class VectorizedBacktester:
    """
    Ultra-fast numpy-based backtester.
    
    Designed for 1000s of simulations per second during optimization.
    """
    
    def __init__(
        self,
        matrix: IndicatorMatrix,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,  # 0.1% per trade
    ) -> None:
        """
        Initialize backtester.
        
        Args:
            matrix: Pre-computed indicator matrix
            initial_capital: Starting capital
            commission_pct: Commission per trade (as decimal)
        """
        self.matrix = matrix
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        
        # Cache price data as numpy for speed
        self.prices = matrix.close.values
        self.returns = matrix.get_numpy("returns")
        self.dates = matrix.df.index
    
    def run(self, genome: StrategyGenome) -> BacktestMetrics:
        """
        Run vectorized backtest for a strategy genome.
        
        Args:
            genome: Strategy to test
            
        Returns:
            BacktestMetrics with all performance stats
        """
        n = len(self.prices)
        
        # Generate signals
        entry_signals = genome.generate_entry_signals(self.matrix)
        exit_signals = genome.generate_exit_signals(self.matrix)
        
        # Track positions (vectorized state machine)
        in_position = np.zeros(n, dtype=bool)
        entry_prices = np.zeros(n)
        entry_bars = np.zeros(n, dtype=int)
        
        # Trade tracking
        trade_returns: list[float] = []
        trade_holding_days: list[int] = []
        
        # Simulate (still needs loop for state tracking, but minimal ops)
        position = False
        entry_price = 0.0
        entry_bar = 0
        
        for i in range(1, n):
            price = self.prices[i]
            
            if not position:
                # Check entry
                if entry_signals[i]:
                    position = True
                    entry_price = price * (1 + self.commission_pct)  # Slippage
                    entry_bar = i
            else:
                # Check exits
                holding_days = i - entry_bar
                pnl_pct = (price - entry_price) / entry_price
                
                should_exit = (
                    exit_signals[i] or
                    pnl_pct <= -genome.stop_loss_pct or
                    pnl_pct >= genome.take_profit_pct or
                    holding_days >= genome.holding_period_max
                )
                
                if should_exit:
                    exit_price = price * (1 - self.commission_pct)
                    trade_return = (exit_price - entry_price) / entry_price
                    trade_returns.append(trade_return)
                    trade_holding_days.append(holding_days)
                    position = False
        
        # Calculate metrics
        return self._calculate_metrics(trade_returns, trade_holding_days)
    
    def _calculate_metrics(
        self, 
        trade_returns: list[float], 
        trade_holding_days: list[int]
    ) -> BacktestMetrics:
        """Calculate performance metrics from trades."""
        metrics = BacktestMetrics()
        
        if not trade_returns:
            return metrics
        
        returns_arr = np.array(trade_returns)
        metrics.num_trades = len(returns_arr)
        
        # Basic stats
        metrics.total_return = float(np.prod(1 + returns_arr) - 1)
        metrics.avg_trade_return = float(np.mean(returns_arr))
        metrics.win_rate = float((returns_arr > 0).mean())
        
        if trade_holding_days:
            metrics.avg_holding_days = float(np.mean(trade_holding_days))
        
        # Profit factor
        gains = returns_arr[returns_arr > 0].sum() if (returns_arr > 0).any() else 0
        losses = abs(returns_arr[returns_arr < 0].sum()) if (returns_arr < 0).any() else 0.001
        metrics.profit_factor = float(gains / losses)
        
        # Sharpe ratio (annualized assuming ~20 trades per year)
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            trades_per_year = 252 / (metrics.avg_holding_days or 20)
            metrics.sharpe_ratio = float(
                np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(trades_per_year)
            )
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_arr[returns_arr < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                trades_per_year = 252 / (metrics.avg_holding_days or 20)
                metrics.sortino_ratio = float(
                    np.mean(returns_arr) / downside_std * np.sqrt(trades_per_year)
                )
        
        # Max drawdown (from cumulative returns)
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        metrics.max_drawdown = float(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        if metrics.max_drawdown < 0:
            # Annualized return / max drawdown
            years = metrics.num_trades * (metrics.avg_holding_days or 20) / 252
            if years > 0:
                annualized_return = (1 + metrics.total_return) ** (1 / years) - 1
                metrics.calmar_ratio = float(annualized_return / abs(metrics.max_drawdown))
        
        # Profitable years (estimate by grouping trades)
        # Simplified: check if rolling 20-trade windows are profitable
        if len(returns_arr) >= 20:
            window_size = min(20, len(returns_arr) // 3)
            num_windows = len(returns_arr) // window_size
            profitable_windows = 0
            for i in range(num_windows):
                window = returns_arr[i*window_size:(i+1)*window_size]
                if window.sum() > 0:
                    profitable_windows += 1
            metrics.profitable_years_pct = float(profitable_windows / num_windows)
        else:
            metrics.profitable_years_pct = 1.0 if metrics.total_return > 0 else 0.0
        
        # Calculate fitness
        metrics.calculate_fitness()
        
        return metrics


# =============================================================================
# AlphaFactory - Optuna-Powered Strategy Evolution
# =============================================================================

@dataclass
class AlphaFactoryConfig:
    """Configuration for the optimization engine."""
    # Optuna settings
    n_trials: int = 100
    n_jobs: int = 1  # Parallel trials
    timeout_seconds: int | None = None
    
    # Strategy constraints
    max_entry_conditions: int = 3
    max_exit_conditions: int = 2
    
    # Walk-forward settings
    train_years: int = 3
    validate_years: int = 1
    min_trades_required: int = 20
    min_profitable_years_pct: float = 0.7
    
    # Regime-specific optimization
    regime: MarketRegime | None = None


@dataclass
class OptimizationResult:
    """Result from AlphaFactory optimization."""
    best_genome: StrategyGenome
    best_metrics: BacktestMetrics
    validation_metrics: BacktestMetrics | None
    n_trials_completed: int
    optimization_time_seconds: float
    all_trial_scores: list[float] = field(default_factory=list)


class AlphaFactory:
    """
    Evolutionary strategy discovery engine using Optuna.
    
    Uses Bayesian optimization (TPE sampler) to efficiently explore
    the massive search space of technical indicator combinations.
    """
    
    # Available indicators for optimization
    TREND_INDICATORS = [
        IndicatorType.SMA, IndicatorType.EMA, IndicatorType.ADX,
        IndicatorType.AROON, IndicatorType.CCI, IndicatorType.MACD,
    ]
    OSCILLATORS = [
        IndicatorType.RSI, IndicatorType.STOCH_K, IndicatorType.STOCH_D,
        IndicatorType.WILLIAMS_R, IndicatorType.MFI, IndicatorType.ROC,
    ]
    VOLATILITY_INDICATORS = [
        IndicatorType.ATR_PCT, IndicatorType.BB_PCT,
        IndicatorType.ZSCORE, IndicatorType.DRAWDOWN,
    ]
    MOMENTUM_INDICATORS = [
        IndicatorType.MOMENTUM, IndicatorType.RETURNS,
        IndicatorType.PRICE_VS_SMA, IndicatorType.PRICE_VS_EMA,
    ]
    
    ALL_INDICATORS = (
        TREND_INDICATORS + OSCILLATORS + VOLATILITY_INDICATORS + MOMENTUM_INDICATORS
    )
    
    # Available periods
    PERIODS = [7, 10, 14, 20, 50, 100, 200]
    
    def __init__(
        self,
        train_prices: pd.DataFrame,
        validate_prices: pd.DataFrame | None = None,
        config: AlphaFactoryConfig | None = None,
    ) -> None:
        """
        Initialize AlphaFactory.
        
        Args:
            train_prices: Training data (OHLCV DataFrame)
            validate_prices: Optional validation data for walk-forward
            config: Optimization configuration
        """
        self.config = config or AlphaFactoryConfig()
        
        # Pre-compute indicators
        logger.info("Building indicator matrix for training data...")
        self.train_matrix = IndicatorMatrix(train_prices)
        
        # Get available periods based on data length
        self._available_periods = self.train_matrix.available_periods
        if not self._available_periods:
            # Fallback to smallest periods if data is very short
            self._available_periods = [7, 10, 14]
        logger.info(f"Available periods for optimization: {self._available_periods}")
        
        self.validate_matrix: IndicatorMatrix | None = None
        if validate_prices is not None:
            logger.info("Building indicator matrix for validation data...")
            self.validate_matrix = IndicatorMatrix(validate_prices)
        
        # Create backtesters
        self.train_backtester = VectorizedBacktester(self.train_matrix)
        self.validate_backtester: VectorizedBacktester | None = None
        if self.validate_matrix:
            self.validate_backtester = VectorizedBacktester(self.validate_matrix)
        
        # For tracking
        self._best_genome: StrategyGenome | None = None
        self._best_score: float = float('-inf')
    
    def _create_condition(
        self, 
        trial: optuna.Trial, 
        prefix: str,
        condition_idx: int,
    ) -> ConditionGenome | None:
        """Create a single condition from Optuna trial parameters."""
        # Decide if this condition is active
        active = trial.suggest_categorical(f"{prefix}_{condition_idx}_active", [True, False])
        if not active:
            return None
        
        # Choose indicator type
        indicator_name = trial.suggest_categorical(
            f"{prefix}_{condition_idx}_indicator",
            [i.value for i in self.ALL_INDICATORS]
        )
        indicator_type = IndicatorType(indicator_name)
        
        # Choose period from available periods (data-length dependent)
        period = trial.suggest_categorical(
            f"{prefix}_{condition_idx}_period",
            self._available_periods
        )
        
        # Choose logic gate
        logic_name = trial.suggest_categorical(
            f"{prefix}_{condition_idx}_logic",
            [g.value for g in LogicGate]
        )
        logic_gate = LogicGate(logic_name)
        
        # Choose threshold based on indicator type
        threshold = self._suggest_threshold(trial, f"{prefix}_{condition_idx}", indicator_type)
        
        # Upper threshold for BETWEEN
        threshold_upper = None
        if logic_gate == LogicGate.BETWEEN:
            threshold_upper = trial.suggest_float(
                f"{prefix}_{condition_idx}_threshold_upper",
                threshold + 1,
                threshold + 50,
            )
        
        return ConditionGenome(
            indicator_type=indicator_type,
            period=period,
            logic_gate=logic_gate,
            threshold=threshold,
            threshold_upper=threshold_upper,
        )
    
    def _suggest_threshold(
        self, 
        trial: optuna.Trial, 
        prefix: str,
        indicator_type: IndicatorType,
    ) -> float:
        """Suggest appropriate threshold range based on indicator type."""
        # RSI, Stochastic, Williams %R, MFI: 0-100 range
        if indicator_type in [
            IndicatorType.RSI, IndicatorType.STOCH_K, IndicatorType.STOCH_D,
            IndicatorType.MFI,
        ]:
            return trial.suggest_float(f"{prefix}_threshold", 10, 90)
        
        # Williams %R: -100 to 0
        if indicator_type == IndicatorType.WILLIAMS_R:
            return trial.suggest_float(f"{prefix}_threshold", -90, -10)
        
        # ADX: 0-100 (typically 20-40 significant)
        if indicator_type == IndicatorType.ADX:
            return trial.suggest_float(f"{prefix}_threshold", 15, 50)
        
        # Aroon: 0-100
        if indicator_type == IndicatorType.AROON:
            return trial.suggest_float(f"{prefix}_threshold", 20, 80)
        
        # CCI: typically -200 to 200
        if indicator_type == IndicatorType.CCI:
            return trial.suggest_float(f"{prefix}_threshold", -150, 150)
        
        # BB_PCT: 0-1 range
        if indicator_type == IndicatorType.BB_PCT:
            return trial.suggest_float(f"{prefix}_threshold", 0.0, 1.0)
        
        # Z-score: typically -3 to 3
        if indicator_type == IndicatorType.ZSCORE:
            return trial.suggest_float(f"{prefix}_threshold", -3.0, 3.0)
        
        # Drawdown: -1 to 0
        if indicator_type == IndicatorType.DRAWDOWN:
            return trial.suggest_float(f"{prefix}_threshold", -0.5, 0.0)
        
        # ATR_PCT, Momentum, Returns: small percentages
        if indicator_type in [
            IndicatorType.ATR_PCT, IndicatorType.MOMENTUM, 
            IndicatorType.RETURNS, IndicatorType.PRICE_VS_SMA,
            IndicatorType.PRICE_VS_EMA, IndicatorType.ROC,
        ]:
            return trial.suggest_float(f"{prefix}_threshold", -0.2, 0.2)
        
        # Default fallback
        return trial.suggest_float(f"{prefix}_threshold", 0, 100)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - returns fitness score to maximize."""
        # Build entry conditions
        entry_conditions = []
        for i in range(self.config.max_entry_conditions):
            condition = self._create_condition(trial, "entry", i)
            if condition:
                entry_conditions.append(condition)
        
        # Must have at least one entry condition
        if not entry_conditions:
            return float('-inf')
        
        # Build exit conditions
        exit_conditions = []
        for i in range(self.config.max_exit_conditions):
            condition = self._create_condition(trial, "exit", i)
            if condition:
                exit_conditions.append(condition)
        
        # Risk parameters
        stop_loss = trial.suggest_float("stop_loss", 0.02, 0.15)
        take_profit = trial.suggest_float("take_profit", 0.05, 0.50)
        max_holding = trial.suggest_int("max_holding_days", 5, 252)
        
        # Create genome
        genome = StrategyGenome(
            name=f"trial_{trial.number}",
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            holding_period_max=max_holding,
            regime=self.config.regime,
        )
        
        # Backtest on training data
        metrics = self.train_backtester.run(genome)
        fitness = metrics.fitness_score
        
        # Track best
        if fitness > self._best_score:
            self._best_score = fitness
            self._best_genome = genome
        
        return fitness
    
    def optimize(self) -> OptimizationResult:
        """
        Run optimization to find the best strategy.
        
        Returns:
            OptimizationResult with best genome and metrics
        """
        import time
        start_time = time.time()
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )
        
        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info(f"Starting optimization with {self.config.n_trials} trials...")
        
        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout_seconds,
            show_progress_bar=True,
        )
        
        elapsed = time.time() - start_time
        
        # Get best genome (might be from callback tracking)
        if self._best_genome is None:
            logger.warning("No valid strategy found during optimization")
            return OptimizationResult(
                best_genome=StrategyGenome(name="none"),
                best_metrics=BacktestMetrics(),
                validation_metrics=None,
                n_trials_completed=len(study.trials),
                optimization_time_seconds=elapsed,
            )
        
        # Re-run best genome for full metrics
        train_metrics = self.train_backtester.run(self._best_genome)
        
        # Validate if we have validation data
        validation_metrics = None
        if self.validate_backtester and self._best_genome:
            validation_metrics = self.validate_backtester.run(self._best_genome)
            logger.info(
                f"Validation: Sharpe={validation_metrics.sharpe_ratio:.2f}, "
                f"Return={validation_metrics.total_return:.2%}"
            )
        
        # Collect trial scores
        all_scores = [t.value for t in study.trials if t.value is not None]
        
        logger.info(
            f"Optimization complete in {elapsed:.1f}s. "
            f"Best fitness: {train_metrics.fitness_score:.3f}"
        )
        
        return OptimizationResult(
            best_genome=self._best_genome,
            best_metrics=train_metrics,
            validation_metrics=validation_metrics,
            n_trials_completed=len(study.trials),
            optimization_time_seconds=elapsed,
            all_trial_scores=all_scores,
        )
    
    def walk_forward_optimize(
        self,
        full_prices: pd.DataFrame,
        n_windows: int = 5,
    ) -> list[OptimizationResult]:
        """
        Walk-forward optimization across multiple time windows.
        
        Splits data into rolling train/validate windows and optimizes each,
        returning a list of results to evaluate strategy stability.
        
        Args:
            full_prices: Complete price history
            n_windows: Number of walk-forward windows
            
        Returns:
            List of OptimizationResult, one per window
        """
        results = []
        
        # Calculate window sizes
        total_days = len(full_prices)
        train_days = int(self.config.train_years * 252)
        validate_days = int(self.config.validate_years * 252)
        window_size = train_days + validate_days
        step = validate_days  # Slide by validation period
        
        if total_days < window_size:
            logger.warning(f"Not enough data for walk-forward. Need {window_size} days, have {total_days}")
            return results
        
        logger.info(f"Running walk-forward optimization with {n_windows} windows...")
        
        for i in range(n_windows):
            start_idx = i * step
            end_idx = start_idx + window_size
            
            if end_idx > total_days:
                break
            
            train_end = start_idx + train_days
            
            # Split window
            train_data = full_prices.iloc[start_idx:train_end]
            validate_data = full_prices.iloc[train_end:end_idx]
            
            logger.info(
                f"Window {i+1}: Train {train_data.index[0].date()} to {train_data.index[-1].date()}, "
                f"Validate to {validate_data.index[-1].date()}"
            )
            
            # Create factory for this window
            factory = AlphaFactory(
                train_prices=train_data,
                validate_prices=validate_data,
                config=self.config,
            )
            
            # Optimize
            result = factory.optimize()
            results.append(result)
            
            # Log validation performance
            if result.validation_metrics:
                val = result.validation_metrics
                logger.info(
                    f"  Window {i+1} validation: Sharpe={val.sharpe_ratio:.2f}, "
                    f"Return={val.total_return:.2%}, Trades={val.num_trades}"
                )
        
        return results


# =============================================================================
# Factory Functions
# =============================================================================

def create_alpha_factory(
    prices: pd.DataFrame,
    config: AlphaFactoryConfig | None = None,
) -> AlphaFactory:
    """
    Create an AlphaFactory instance.
    
    Args:
        prices: OHLCV price data
        config: Optional configuration
        
    Returns:
        Configured AlphaFactory
    """
    return AlphaFactory(train_prices=prices, config=config)


def quick_optimize(
    prices: pd.DataFrame,
    n_trials: int = 50,
    regime: MarketRegime | None = None,
) -> OptimizationResult:
    """
    Quick optimization with sensible defaults.
    
    Args:
        prices: OHLCV price data
        n_trials: Number of optimization trials
        regime: Optional regime-specific optimization
        
    Returns:
        OptimizationResult with best strategy found
    """
    config = AlphaFactoryConfig(
        n_trials=n_trials,
        regime=regime,
    )
    factory = AlphaFactory(train_prices=prices, config=config)
    return factory.optimize()
