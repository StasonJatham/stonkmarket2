"""
Regime Detection and Strategy Mode Selection.

This module detects market regimes and selects the appropriate strategy mode.

CRITICAL CHANGE from V1:
- Bear markets are BUYING opportunities, NOT blocks
- Different strategies apply based on regime
- Fundamental checks required in bear mode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification."""

    BULL = "BULL"  # Normal uptrend - use technical strategies
    BEAR = "BEAR"  # Downtrend - switch to accumulation mode
    CRASH = "CRASH"  # Extreme panic - maximum accumulation opportunity
    RECOVERY = "RECOVERY"  # Transitioning from bear to bull


class StrategyMode(str, Enum):
    """Strategy operating mode based on regime."""

    TREND_FOLLOWING = "TREND_FOLLOWING"  # Bull mode - technicals
    CAUTIOUS_TREND = "CAUTIOUS_TREND"  # Recovery mode - careful entries
    VALUE_ACCUMULATION = "VALUE_ACCUMULATION"  # Bear mode - fundamentals + scale-in
    AGGRESSIVE_ACCUMULATION = "AGGRESSIVE_ACCUMULATION"  # Crash mode - max accumulation


@dataclass
class StrategyConfig:
    """Configuration for strategy based on regime."""

    # Strategy type
    use_technicals: bool = True
    use_fundamentals: bool = False

    # Risk management
    stop_loss_pct: float | None = 10.0  # None = no stop loss
    take_profit_pct: float | None = None

    # Position sizing
    position_size_pct: float = 100.0  # % of target position to take initially
    scale_in: bool = False
    scale_in_levels: list[float] = field(default_factory=list)  # Drop % to add

    # Signal handling
    ignore_sell_signals: bool = False  # In bear mode, hold through volatility


@dataclass
class RegimeState:
    """Current market regime with full context."""

    # Regime classification
    regime: MarketRegime
    strategy_mode: StrategyMode

    # SPY metrics
    spy_price: float
    spy_sma200: float
    spy_sma50: float
    drawdown_pct: float  # From 52-week high

    # Trend metrics
    above_sma200: bool
    above_sma50: bool
    sma50_above_sma200: bool  # Golden cross indicator
    momentum_20d: float  # 20-day momentum %

    # Volatility
    volatility_regime: Literal["low", "normal", "high"]
    vix_equivalent: float  # Realized volatility annualized

    # Strategy config for this regime
    strategy_config: StrategyConfig

    # Human-readable
    description: str


class RegimeDetector:
    """
    Detects market regime and provides strategy configuration.

    PHILOSOPHY:
    - Bear markets are the BEST time to buy quality assets
    - We don't "block" buys, we ADAPT the strategy
    - Fundamentals become MORE important in bear markets
    - Technicals become LESS important in bear markets
    """

    # Thresholds for regime detection
    CRASH_DRAWDOWN_PCT = -25.0  # -25% from high = crash
    BEAR_SMA_THRESHOLD = 0.0  # Below SMA200 = bear
    HIGH_VOL_MULTIPLIER = 1.5  # Vol > 1.5x normal = high vol

    def __init__(self, spy_prices: pd.Series | None = None):
        """Initialize with optional SPY prices for regime detection."""
        self.spy_prices = spy_prices
        self._regime_state: RegimeState | None = None

    def set_spy_prices(self, spy_prices: pd.Series) -> None:
        """Update SPY prices for regime detection."""
        self.spy_prices = spy_prices
        self._regime_state = None  # Reset cached state

    def detect(self, as_of_date: pd.Timestamp | None = None) -> RegimeState:
        """
        Detect current market regime.

        Args:
            as_of_date: Optional date for historical regime detection.
                       If None, uses most recent data.

        Returns:
            RegimeState with full regime context and strategy config.
        """
        if self.spy_prices is None or len(self.spy_prices) < 200:
            # Insufficient data - default to cautious mode
            return self._default_regime_state()

        # Get data up to as_of_date
        if as_of_date is not None:
            prices = self.spy_prices.loc[:as_of_date]
        else:
            prices = self.spy_prices

        if len(prices) < 200:
            return self._default_regime_state()

        # Calculate indicators
        close = prices
        sma200 = close.rolling(200).mean()
        sma50 = close.rolling(50).mean()

        current = float(close.iloc[-1])
        sma200_val = float(sma200.iloc[-1])
        sma50_val = float(sma50.iloc[-1])

        # Drawdown from 52-week high
        high_52w = float(close.rolling(252).max().iloc[-1])
        drawdown = (current / high_52w - 1) * 100

        # Momentum
        momentum_20d = float(close.pct_change(20).iloc[-1] * 100)

        # Volatility regime
        returns = close.pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)
        vol_1y = float(returns.tail(252).std() * np.sqrt(252) * 100) if len(returns) >= 252 else vol_20d

        if vol_20d > vol_1y * self.HIGH_VOL_MULTIPLIER:
            vol_regime: Literal["low", "normal", "high"] = "high"
        elif vol_20d < vol_1y * 0.7:
            vol_regime = "low"
        else:
            vol_regime = "normal"

        # Determine regime
        above_sma200 = current > sma200_val
        above_sma50 = current > sma50_val
        sma50_above_sma200 = sma50_val > sma200_val

        if drawdown <= self.CRASH_DRAWDOWN_PCT:
            regime = MarketRegime.CRASH
            strategy_mode = StrategyMode.AGGRESSIVE_ACCUMULATION
            description = f"CRASH: {drawdown:.1f}% drawdown - maximum accumulation opportunity"
        elif not above_sma200:
            if momentum_20d > 5:
                regime = MarketRegime.RECOVERY
                strategy_mode = StrategyMode.CAUTIOUS_TREND
                description = f"RECOVERY: Below SMA200 but momentum turning positive"
            else:
                regime = MarketRegime.BEAR
                strategy_mode = StrategyMode.VALUE_ACCUMULATION
                description = f"BEAR: SPY ${current:.0f} < SMA200 ${sma200_val:.0f} - accumulate quality"
        else:
            regime = MarketRegime.BULL
            strategy_mode = StrategyMode.TREND_FOLLOWING
            description = f"BULL: SPY ${current:.0f} > SMA200 ${sma200_val:.0f} - trend following"

        # Get strategy config for this mode
        strategy_config = self._get_strategy_config(strategy_mode)

        return RegimeState(
            regime=regime,
            strategy_mode=strategy_mode,
            spy_price=current,
            spy_sma200=sma200_val,
            spy_sma50=sma50_val,
            drawdown_pct=drawdown,
            above_sma200=above_sma200,
            above_sma50=above_sma50,
            sma50_above_sma200=sma50_above_sma200,
            momentum_20d=momentum_20d,
            volatility_regime=vol_regime,
            vix_equivalent=vol_20d,
            strategy_config=strategy_config,
            description=description,
        )

    def detect_at_date(self, date: pd.Timestamp) -> RegimeState:
        """Detect regime at a specific historical date (for backtesting)."""
        return self.detect(as_of_date=date)

    def _get_strategy_config(self, mode: StrategyMode) -> StrategyConfig:
        """Get strategy configuration for the given mode."""

        if mode == StrategyMode.TREND_FOLLOWING:
            # Bull mode: Standard technical trading
            return StrategyConfig(
                use_technicals=True,
                use_fundamentals=False,
                stop_loss_pct=10.0,
                take_profit_pct=None,
                position_size_pct=100.0,
                scale_in=False,
                scale_in_levels=[],
                ignore_sell_signals=False,
            )

        elif mode == StrategyMode.CAUTIOUS_TREND:
            # Recovery mode: Careful entries, smaller positions
            return StrategyConfig(
                use_technicals=True,
                use_fundamentals=True,  # Also check fundamentals
                stop_loss_pct=15.0,
                take_profit_pct=20.0,
                position_size_pct=50.0,  # Half position
                scale_in=True,
                scale_in_levels=[10.0],  # Add at -10%
                ignore_sell_signals=False,
            )

        elif mode == StrategyMode.VALUE_ACCUMULATION:
            # Bear mode: Focus on fundamentals, scale in
            return StrategyConfig(
                use_technicals=False,  # IGNORE technicals
                use_fundamentals=True,  # REQUIRE fundamental checks
                stop_loss_pct=35.0,  # Much wider stops
                take_profit_pct=None,
                position_size_pct=25.0,  # Start with 25%
                scale_in=True,
                scale_in_levels=[10.0, 20.0, 30.0],  # Add at -10%, -20%, -30%
                ignore_sell_signals=True,  # Hold through volatility
            )

        elif mode == StrategyMode.AGGRESSIVE_ACCUMULATION:
            # Crash mode: Maximum accumulation
            return StrategyConfig(
                use_technicals=False,
                use_fundamentals=True,  # Still require quality
                stop_loss_pct=None,  # NO stop loss in crash
                take_profit_pct=None,
                position_size_pct=25.0,  # Start small
                scale_in=True,
                scale_in_levels=[10.0, 20.0, 30.0, 40.0],  # More aggressive scaling
                ignore_sell_signals=True,
            )

        else:
            # Fallback to conservative
            return StrategyConfig()

    def _default_regime_state(self) -> RegimeState:
        """Return default regime state when insufficient data."""
        return RegimeState(
            regime=MarketRegime.BEAR,  # Default to cautious
            strategy_mode=StrategyMode.VALUE_ACCUMULATION,
            spy_price=0.0,
            spy_sma200=0.0,
            spy_sma50=0.0,
            drawdown_pct=0.0,
            above_sma200=False,
            above_sma50=False,
            sma50_above_sma200=False,
            momentum_20d=0.0,
            volatility_regime="normal",
            vix_equivalent=20.0,
            strategy_config=self._get_strategy_config(StrategyMode.VALUE_ACCUMULATION),
            description="UNKNOWN: Insufficient SPY data - defaulting to cautious mode",
        )

    def get_regime_history(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Get regime history for a date range.

        Useful for backtesting to know what regime was active at each point.
        """
        if self.spy_prices is None:
            return pd.DataFrame()

        prices = self.spy_prices
        if start_date:
            prices = prices.loc[start_date:]
        if end_date:
            prices = prices.loc[:end_date]

        regimes = []
        for date in prices.index:
            state = self.detect_at_date(date)
            regimes.append(
                {
                    "date": date,
                    "regime": state.regime.value,
                    "strategy_mode": state.strategy_mode.value,
                    "spy_price": state.spy_price,
                    "drawdown_pct": state.drawdown_pct,
                    "above_sma200": state.above_sma200,
                }
            )

        return pd.DataFrame(regimes).set_index("date")


def identify_crash_periods(spy_prices: pd.Series, threshold_pct: float = -20.0) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Identify historical crash periods for stress testing.

    Returns list of (start_date, end_date) tuples for periods where
    SPY dropped more than threshold_pct from its rolling high.
    """
    rolling_high = spy_prices.rolling(252).max()
    drawdown = (spy_prices / rolling_high - 1) * 100

    in_crash = drawdown <= threshold_pct
    crash_periods = []

    start = None
    for date, is_crash in in_crash.items():
        if is_crash and start is None:
            start = date
        elif not is_crash and start is not None:
            crash_periods.append((start, date))
            start = None

    # Handle ongoing crash at end of data
    if start is not None:
        crash_periods.append((start, spy_prices.index[-1]))

    return crash_periods
