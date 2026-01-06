"""
RegimeService - Single source of truth for market regime detection.

This consolidates the regime logic from:
- backtest/regime_filter.py (RegimeDetector)
- scoring.py (detect_regimes)
- analytics.py (detect_regime)

All components (backtester, scorer, API) use THIS service for regime.

PHILOSOPHY (from backtest):
- Bear markets are BUYING opportunities, NOT blocks
- Different strategies apply based on regime
- Fundamental checks required in bear mode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from functools import lru_cache
from typing import Literal, Any

import numpy as np
import pandas as pd

from app.quant_engine.core.technical_service import TechnicalService, get_technical_service

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification."""
    BULL = "BULL"           # SPY > SMA200, positive momentum
    BEAR = "BEAR"           # SPY < SMA200, negative momentum
    CRASH = "CRASH"         # SPY drawdown > 25%, high volatility
    RECOVERY = "RECOVERY"   # SPY < SMA200 but momentum turning positive
    CORRECTION = "CORRECTION"  # SPY drawdown 10-20%


class StrategyMode(str, Enum):
    """Strategy mode based on regime."""
    TECHNICAL = "TECHNICAL"             # Use technical signals (bull market)
    CAUTIOUS_TREND = "CAUTIOUS_TREND"   # Recovery mode - careful entries
    ACCUMULATE = "ACCUMULATE"           # Scale-in buying (bear/crash)
    DEFENSIVE = "DEFENSIVE"             # Reduce exposure (high vol)
    HOLD = "HOLD"                       # No new positions


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
    """
    Complete regime state for decision making.
    
    This is the single source of truth used by:
    - BacktestV2Service
    - ScoringOrchestrator
    - API endpoints
    """
    
    regime: MarketRegime
    strategy_mode: StrategyMode
    strategy_config: StrategyConfig
    
    # SPY metrics
    spy_price: float
    spy_sma200: float
    spy_sma50: float
    spy_drawdown_pct: float  # From 52-week high
    
    # Volatility
    vix_level: float | None  # If available
    volatility_regime: Literal["LOW", "NORMAL", "HIGH", "EXTREME"]
    
    # Trend
    spy_above_sma200: bool
    spy_above_sma50: bool
    sma50_above_sma200: bool  # Golden/death cross
    momentum_20d: float  # 20-day return %
    
    # Signals
    allow_new_longs: bool
    allow_dip_buying: bool
    reduce_position_size: bool
    require_fundamentals: bool
    
    # Metadata
    description: str
    as_of_date: date
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "regime": self.regime.value,
            "strategy_mode": self.strategy_mode.value,
            "spy_price": round(self.spy_price, 2),
            "spy_sma200": round(self.spy_sma200, 2),
            "spy_sma50": round(self.spy_sma50, 2),
            "spy_drawdown_pct": round(self.spy_drawdown_pct, 2),
            "vix_level": round(self.vix_level, 1) if self.vix_level else None,
            "volatility_regime": self.volatility_regime,
            "spy_above_sma200": self.spy_above_sma200,
            "spy_above_sma50": self.spy_above_sma50,
            "sma50_above_sma200": self.sma50_above_sma200,
            "momentum_20d": round(self.momentum_20d, 2),
            "allow_new_longs": self.allow_new_longs,
            "allow_dip_buying": self.allow_dip_buying,
            "reduce_position_size": self.reduce_position_size,
            "require_fundamentals": self.require_fundamentals,
            "description": self.description,
            "as_of_date": str(self.as_of_date),
        }


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    
    # Drawdown thresholds
    correction_threshold: float = -10.0  # 10% drawdown = correction
    bear_threshold: float = -20.0        # 20% drawdown = bear market
    crash_threshold: float = -25.0       # 25% drawdown = crash
    
    # VIX thresholds (if available)
    vix_elevated: float = 20.0
    vix_high: float = 30.0
    vix_extreme: float = 40.0
    
    # Momentum thresholds
    momentum_bullish: float = 5.0   # 5% 20-day return = bullish
    momentum_bearish: float = -5.0  # -5% 20-day return = bearish


# Strategy configurations by regime
REGIME_STRATEGY_CONFIGS = {
    MarketRegime.BULL: StrategyConfig(
        use_technicals=True,
        use_fundamentals=False,
        stop_loss_pct=8.0,
        position_size_pct=100.0,
        scale_in=False,
    ),
    MarketRegime.CORRECTION: StrategyConfig(
        use_technicals=True,
        use_fundamentals=True,
        stop_loss_pct=10.0,
        position_size_pct=75.0,
        scale_in=True,
        scale_in_levels=[-5.0, -10.0],
    ),
    MarketRegime.BEAR: StrategyConfig(
        use_technicals=False,
        use_fundamentals=True,
        stop_loss_pct=None,  # No stop loss in bear - hold through
        position_size_pct=50.0,
        scale_in=True,
        scale_in_levels=[-10.0, -20.0, -30.0],
        ignore_sell_signals=True,
    ),
    MarketRegime.CRASH: StrategyConfig(
        use_technicals=False,
        use_fundamentals=True,
        stop_loss_pct=None,
        position_size_pct=25.0,  # Small initial position
        scale_in=True,
        scale_in_levels=[-10.0, -15.0, -20.0, -30.0],
        ignore_sell_signals=True,
    ),
    MarketRegime.RECOVERY: StrategyConfig(
        use_technicals=True,
        use_fundamentals=True,
        stop_loss_pct=10.0,
        position_size_pct=75.0,
        scale_in=False,
    ),
}


class RegimeService:
    """
    Unified market regime detection service.
    
    Usage:
        service = RegimeService()
        state = service.get_current_regime(spy_prices)
        
        # For backtesting with point-in-time
        historical_state = service.get_regime_at_date(spy_prices, date)
    """
    
    def __init__(
        self,
        config: RegimeConfig | None = None,
        technical_service: TechnicalService | None = None,
    ):
        self.config = config or RegimeConfig()
        self.tech = technical_service or get_technical_service()
        self._cache: dict[str, RegimeState] = {}
    
    def get_current_regime(
        self,
        spy_prices: pd.DataFrame,
        vix_level: float | None = None,
    ) -> RegimeState:
        """
        Get current market regime state.
        
        Args:
            spy_prices: DataFrame with SPY OHLCV data
            vix_level: Optional current VIX level
            
        Returns:
            RegimeState with complete regime information
        """
        return self._compute_regime(spy_prices, vix_level, as_of_date=None)
    
    def get_regime_at_date(
        self,
        spy_prices: pd.DataFrame,
        as_of_date: date | pd.Timestamp,
        vix_series: pd.Series | None = None,
    ) -> RegimeState:
        """
        Get regime state at a specific historical date (for backtesting).
        
        This ensures no look-ahead bias - only uses data available on that date.
        """
        # Truncate data to as_of_date
        if isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
            as_of_date = pd.Timestamp(as_of_date)
        
        historical_prices = spy_prices.loc[:as_of_date]
        
        vix_level = None
        if vix_series is not None and as_of_date in vix_series.index:
            vix_level = float(vix_series.loc[as_of_date])
        
        return self._compute_regime(
            historical_prices,
            vix_level,
            as_of_date.date() if hasattr(as_of_date, 'date') else as_of_date
        )
    
    def get_regime_series(
        self,
        spy_prices: pd.DataFrame,
        vix_series: pd.Series | None = None,
    ) -> pd.Series:
        """
        Compute regime for each date in the series (for backtesting).
        
        Returns:
            Series with MarketRegime value for each date
        """
        regimes = []
        
        for dt in spy_prices.index:
            state = self.get_regime_at_date(spy_prices, dt, vix_series)
            regimes.append(state.regime.value)
        
        return pd.Series(regimes, index=spy_prices.index, name="regime")
    
    def _compute_regime(
        self,
        spy_prices: pd.DataFrame,
        vix_level: float | None,
        as_of_date: date | None,
    ) -> RegimeState:
        """Core regime computation logic."""
        if len(spy_prices) < 50:
            logger.warning("Insufficient SPY data for regime detection")
            return self._default_regime(as_of_date or date.today())
        
        # Normalize column names
        spy_prices = spy_prices.copy()
        spy_prices.columns = [c.lower() for c in spy_prices.columns]
        
        close = spy_prices.get("close", spy_prices.get("adj close", spy_prices.iloc[:, 0]))
        current_price = float(close.iloc[-1])
        
        # Moving averages
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        
        # Drawdown from 52-week high
        high_52w = float(close.rolling(252).max().iloc[-1]) if len(close) >= 252 else float(close.max())
        drawdown_pct = ((current_price / high_52w) - 1) * 100
        
        # Momentum
        momentum_20d = float(close.pct_change(20).iloc[-1] * 100) if len(close) > 20 else 0.0
        
        # Trend checks
        above_sma200 = current_price > sma_200
        above_sma50 = current_price > sma_50
        sma50_above_sma200 = sma_50 > sma_200
        
        # Volatility regime (using realized vol)
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            realized_vol_20d = returns.iloc[-20:].std() * np.sqrt(252) * 100
            if realized_vol_20d > 40:
                volatility_regime = "EXTREME"
            elif realized_vol_20d > 25:
                volatility_regime = "HIGH"
            elif realized_vol_20d < 12:
                volatility_regime = "LOW"
            else:
                volatility_regime = "NORMAL"
        else:
            volatility_regime = "NORMAL"
        
        # VIX override if available
        if vix_level is not None:
            if vix_level >= self.config.vix_extreme:
                volatility_regime = "EXTREME"
            elif vix_level >= self.config.vix_high:
                volatility_regime = "HIGH"
            elif vix_level <= 15:
                volatility_regime = "LOW"
        
        # Determine regime
        regime, strategy_mode, description = self._classify_regime(
            drawdown_pct=drawdown_pct,
            above_sma200=above_sma200,
            momentum_20d=momentum_20d,
            vix_level=vix_level,
            volatility_regime=volatility_regime,
            sma50_above_sma200=sma50_above_sma200,
        )
        
        # Get strategy config for this regime
        strategy_config = REGIME_STRATEGY_CONFIGS.get(regime, REGIME_STRATEGY_CONFIGS[MarketRegime.BULL])
        
        # Determine allowed actions
        allow_new_longs, allow_dip_buying, reduce_size, require_fundamentals = self._determine_actions(
            regime, strategy_mode, volatility_regime
        )
        
        return RegimeState(
            regime=regime,
            strategy_mode=strategy_mode,
            strategy_config=strategy_config,
            spy_price=current_price,
            spy_sma200=sma_200,
            spy_sma50=sma_50,
            spy_drawdown_pct=drawdown_pct,
            vix_level=vix_level,
            volatility_regime=volatility_regime,
            spy_above_sma200=above_sma200,
            spy_above_sma50=above_sma50,
            sma50_above_sma200=sma50_above_sma200,
            momentum_20d=momentum_20d,
            allow_new_longs=allow_new_longs,
            allow_dip_buying=allow_dip_buying,
            reduce_position_size=reduce_size,
            require_fundamentals=require_fundamentals,
            description=description,
            as_of_date=as_of_date or date.today(),
        )
    
    def _classify_regime(
        self,
        drawdown_pct: float,
        above_sma200: bool,
        momentum_20d: float,
        vix_level: float | None,
        volatility_regime: str,
        sma50_above_sma200: bool,
    ) -> tuple[MarketRegime, StrategyMode, str]:
        """Classify market regime based on metrics."""
        cfg = self.config
        
        # CRASH: Severe drawdown with high volatility
        if drawdown_pct <= cfg.crash_threshold and volatility_regime in ("HIGH", "EXTREME"):
            return (
                MarketRegime.CRASH,
                StrategyMode.ACCUMULATE,
                f"CRASH: SPY down {abs(drawdown_pct):.1f}% with extreme volatility. Maximum accumulation opportunity.",
            )
        
        # BEAR: Below SMA200 with significant drawdown
        if not above_sma200 and drawdown_pct <= cfg.bear_threshold:
            # Check for recovery signs
            if momentum_20d > cfg.momentum_bullish:
                return (
                    MarketRegime.RECOVERY,
                    StrategyMode.CAUTIOUS_TREND,
                    f"RECOVERY: Bear market but momentum turning positive (+{momentum_20d:.1f}%). Cautious entries.",
                )
            return (
                MarketRegime.BEAR,
                StrategyMode.ACCUMULATE,
                f"BEAR: SPY below SMA200, down {abs(drawdown_pct):.1f}%. Accumulation mode - fundamentals required.",
            )
        
        # CORRECTION: Moderate drawdown
        if drawdown_pct <= cfg.correction_threshold:
            if volatility_regime in ("HIGH", "EXTREME"):
                return (
                    MarketRegime.CORRECTION,
                    StrategyMode.DEFENSIVE,
                    f"CORRECTION: SPY down {abs(drawdown_pct):.1f}% with high volatility. Defensive positioning.",
                )
            return (
                MarketRegime.CORRECTION,
                StrategyMode.TECHNICAL,
                f"CORRECTION: SPY down {abs(drawdown_pct):.1f}%. Watching for stabilization.",
            )
        
        # RECOVERY: Below SMA200 but positive momentum
        if not above_sma200 and momentum_20d > 0:
            return (
                MarketRegime.RECOVERY,
                StrategyMode.CAUTIOUS_TREND,
                f"RECOVERY: SPY below SMA200 but momentum positive (+{momentum_20d:.1f}%).",
            )
        
        # BULL: Above SMA200 with positive momentum
        if above_sma200 and sma50_above_sma200:
            if volatility_regime == "LOW":
                return (
                    MarketRegime.BULL,
                    StrategyMode.TECHNICAL,
                    f"BULL: Strong uptrend. SPY above SMA200, golden cross, low volatility.",
                )
            return (
                MarketRegime.BULL,
                StrategyMode.TECHNICAL,
                f"BULL: SPY above SMA200, momentum positive. Technical signals active.",
            )
        
        # Default: Neutral/mixed
        if above_sma200:
            return (
                MarketRegime.BULL,
                StrategyMode.TECHNICAL,
                "BULL (cautious): SPY above SMA200 but no golden cross. Standard technical mode.",
            )
        else:
            return (
                MarketRegime.RECOVERY,
                StrategyMode.CAUTIOUS_TREND,
                "Mixed signals: SPY below SMA200. Watching for direction.",
            )
    
    def _determine_actions(
        self,
        regime: MarketRegime,
        strategy_mode: StrategyMode,
        volatility_regime: str,
    ) -> tuple[bool, bool, bool, bool]:
        """
        Determine allowed trading actions based on regime.
        
        Returns:
            (allow_new_longs, allow_dip_buying, reduce_position_size, require_fundamentals)
        """
        
        if regime == MarketRegime.BULL:
            return (True, True, volatility_regime in ("HIGH", "EXTREME"), False)
        
        elif regime == MarketRegime.CORRECTION:
            return (False, True, True, True)
        
        elif regime == MarketRegime.RECOVERY:
            return (True, True, True, True)
        
        elif regime == MarketRegime.BEAR:
            # Bear market: Dip buying only with strong fundamentals
            return (False, True, True, True)
        
        elif regime == MarketRegime.CRASH:
            # Crash: Maximum accumulation opportunity for high-quality names
            return (False, True, True, True)
        
        else:
            # Default to conservative
            return (False, False, True, True)
    
    def _default_regime(self, as_of_date: date) -> RegimeState:
        """Return default (conservative) regime when data is insufficient."""
        return RegimeState(
            regime=MarketRegime.CORRECTION,
            strategy_mode=StrategyMode.DEFENSIVE,
            strategy_config=REGIME_STRATEGY_CONFIGS[MarketRegime.CORRECTION],
            spy_price=0.0,
            spy_sma200=0.0,
            spy_sma50=0.0,
            spy_drawdown_pct=0.0,
            vix_level=None,
            volatility_regime="NORMAL",
            spy_above_sma200=False,
            spy_above_sma50=False,
            sma50_above_sma200=False,
            momentum_20d=0.0,
            allow_new_longs=False,
            allow_dip_buying=False,
            reduce_position_size=True,
            require_fundamentals=True,
            description="Insufficient data - defaulting to defensive positioning",
            as_of_date=as_of_date,
        )
    
    def get_regime_score(self, regime_state: RegimeState) -> float:
        """
        Compute a regime score (0-100) for scoring integration.
        
        Higher score = more favorable regime for buying.
        """
        base_scores = {
            MarketRegime.BULL: 80.0,
            MarketRegime.CORRECTION: 55.0,
            MarketRegime.RECOVERY: 65.0,
            MarketRegime.BEAR: 40.0,
            MarketRegime.CRASH: 30.0,  # Low but not zero - accumulation opportunity
        }
        
        score = base_scores.get(regime_state.regime, 50.0)
        
        # Volatility adjustment
        vol_adjustments = {
            "LOW": 10.0,
            "NORMAL": 0.0,
            "HIGH": -10.0,
            "EXTREME": -15.0,
        }
        score += vol_adjustments.get(regime_state.volatility_regime, 0.0)
        
        # Momentum adjustment
        if regime_state.momentum_20d > 5:
            score += 5
        elif regime_state.momentum_20d < -5:
            score -= 5
        
        # Golden cross bonus
        if regime_state.sma50_above_sma200:
            score += 5
        
        return float(np.clip(score, 0, 100))


# Singleton instance
_regime_service: RegimeService | None = None


def get_regime_service(config: RegimeConfig | None = None) -> RegimeService:
    """Get singleton RegimeService instance."""
    global _regime_service
    if _regime_service is None or config is not None:
        _regime_service = RegimeService(config)
    return _regime_service
