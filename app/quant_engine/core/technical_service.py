"""
TechnicalService - Single source of truth for all technical indicators.

This service consolidates all indicator calculations that were previously
scattered across trade_engine.py, backtest_engine.py, alpha_factory.py,
and dipfinder modules.

All indicators use the same calculation methods, ensuring consistency
between backtesting and live scoring.

Uses the 'ta' library for professional-grade calculations where available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd

# Use 'ta' library for professional indicators
try:
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.trend import MACD, SMAIndicator, EMAIndicator, CCIIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSnapshot:
    """
    Complete technical analysis for a symbol at a point in time.
    
    This replaces the scattered TechnicalSnapshot classes in trade_engine.py
    and the inline calculations in backtest modules.
    """
    
    # Price context
    current_price: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    
    # Momentum
    rsi_14: float = 50.0
    rsi_7: float | None = None
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Volatility
    atr_14: float = 0.0
    atr_pct: float = 0.0  # ATR as % of price
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_pct_b: float = 0.5  # Position within bands (0-1)
    
    # Advanced momentum
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    williams_r: float = -50.0
    cci: float = 0.0
    adx: float = 0.0  # Trend strength
    
    # Volume
    volume_sma_20: float = 0.0
    volume_ratio: float = 1.0  # Current vs SMA
    obv_trend: float = 0.0
    
    # Trend
    trend_direction: Literal["UP", "DOWN", "SIDEWAYS"] = "SIDEWAYS"
    above_sma_20: bool = False
    above_sma_50: bool = False
    above_sma_200: bool = False
    golden_cross: bool = False  # SMA50 > SMA200
    death_cross: bool = False   # SMA50 < SMA200
    
    # Derived scores
    momentum_score: float = 0.0  # -1 to +1
    trend_score: float = 0.0     # -1 to +1
    volatility_regime: Literal["LOW", "NORMAL", "HIGH", "EXTREME"] = "NORMAL"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "current_price": round(self.current_price, 2),
            "rsi_14": round(self.rsi_14, 1),
            "rsi_7": round(self.rsi_7, 1) if self.rsi_7 else None,
            "macd": round(self.macd, 3),
            "macd_signal": round(self.macd_signal, 3),
            "macd_histogram": round(self.macd_histogram, 3),
            "bollinger_pct_b": round(self.bollinger_pct_b, 2),
            "stoch_k": round(self.stoch_k, 1),
            "adx": round(self.adx, 1),
            "atr_pct": round(self.atr_pct, 2),
            "trend_direction": self.trend_direction,
            "above_sma_200": self.above_sma_200,
            "golden_cross": self.golden_cross,
            "death_cross": self.death_cross,
            "momentum_score": round(self.momentum_score, 2),
            "trend_score": round(self.trend_score, 2),
            "volatility_regime": self.volatility_regime,
        }


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""
    
    rsi_period: int = 14
    rsi_fast_period: int = 7
    sma_short: int = 20
    sma_medium: int = 50
    sma_long: int = 200
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volume_period: int = 20
    stoch_period: int = 14
    adx_period: int = 14
    cci_period: int = 20


class TechnicalService:
    """
    Unified technical indicator service.
    
    Usage:
        service = TechnicalService()
        snapshot = service.get_snapshot(prices)
        
        # Or get individual indicators
        rsi = service.compute_rsi(prices['Close'], period=14)
    """
    
    def __init__(self, config: IndicatorConfig | None = None):
        self.config = config or IndicatorConfig()
    
    def get_snapshot(
        self,
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp | None = None,
    ) -> TechnicalSnapshot:
        """
        Compute complete technical snapshot for a price series.
        
        Args:
            prices: DataFrame with 'Close', 'High', 'Low', 'Volume' columns
            as_of_date: Optional date to compute as-of (for backtesting)
            
        Returns:
            TechnicalSnapshot with all indicators
        """
        if as_of_date is not None:
            prices = prices.loc[:as_of_date]
        
        if len(prices) < 20:
            logger.warning(f"Insufficient data ({len(prices)} bars) for snapshot")
            return TechnicalSnapshot()
        
        # Ensure column names are normalized
        prices = self._normalize_columns(prices)
        
        close = prices["close"]
        high = prices.get("high", close)
        low = prices.get("low", close)
        volume = prices.get("volume", pd.Series([0] * len(close), index=close.index))
        
        # Handle None/NaN values in close price
        last_close = close.iloc[-1]
        if last_close is None or (hasattr(last_close, '__iter__') is False and pd.isna(last_close)):
            logger.warning("Last close price is None/NaN")
            return TechnicalSnapshot()
        current_price = float(last_close)
        
        # Use TA library if available, otherwise fallback to custom
        if TA_AVAILABLE:
            return self._compute_with_ta_library(prices, close, high, low, volume, current_price)
        else:
            return self._compute_fallback(close, high, low, volume, current_price)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase and handle duplicates."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # If both 'close' and 'adj close' exist, prefer 'adj close' and drop 'close'
        if "adj close" in df.columns and "close" in df.columns:
            df = df.drop(columns=["close"])
        if "adj_close" in df.columns and "close" in df.columns:
            df = df.drop(columns=["close"])
        
        # Handle common variations
        rename_map = {
            "adj close": "close",
            "adj_close": "close",
        }
        df = df.rename(columns=rename_map)
        
        return df
    
    def _compute_with_ta_library(
        self,
        prices: pd.DataFrame,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        current_price: float,
    ) -> TechnicalSnapshot:
        """Compute indicators using the professional 'ta' library."""
        cfg = self.config
        
        # Moving Averages
        sma_20 = SMAIndicator(close, window=cfg.sma_short).sma_indicator().iloc[-1]
        sma_50 = SMAIndicator(close, window=cfg.sma_medium).sma_indicator().iloc[-1]
        sma_200 = SMAIndicator(close, window=cfg.sma_long).sma_indicator().iloc[-1] if len(close) >= cfg.sma_long else close.mean()
        ema_12 = EMAIndicator(close, window=cfg.ema_fast).ema_indicator().iloc[-1]
        ema_26 = EMAIndicator(close, window=cfg.ema_slow).ema_indicator().iloc[-1]
        
        # MACD
        macd_ind = MACD(close, window_fast=cfg.ema_fast, window_slow=cfg.ema_slow, window_sign=cfg.macd_signal)
        macd_line = macd_ind.macd().iloc[-1]
        macd_signal = macd_ind.macd_signal().iloc[-1]
        macd_histogram = macd_ind.macd_diff().iloc[-1]
        
        # RSI
        rsi_14 = RSIIndicator(close, window=cfg.rsi_period).rsi().iloc[-1]
        rsi_7 = RSIIndicator(close, window=cfg.rsi_fast_period).rsi().iloc[-1] if len(close) > cfg.rsi_fast_period else None
        
        # Stochastic
        stoch = StochasticOscillator(high, low, close, window=cfg.stoch_period)
        stoch_k = stoch.stoch().iloc[-1]
        stoch_d = stoch.stoch_signal().iloc[-1]
        
        # Williams %R
        williams_r = WilliamsRIndicator(high, low, close, lbp=cfg.stoch_period).williams_r().iloc[-1]
        
        # ADX (trend strength)
        adx = ADXIndicator(high, low, close, window=cfg.adx_period).adx().iloc[-1]
        
        # CCI
        cci = CCIIndicator(high, low, close, window=cfg.cci_period).cci().iloc[-1]
        
        # ATR
        atr_ind = AverageTrueRange(high, low, close, window=cfg.atr_period)
        atr_14 = atr_ind.average_true_range().iloc[-1]
        atr_pct = (atr_14 / current_price) * 100 if current_price > 0 else 0
        
        # Bollinger Bands
        bb = BollingerBands(close, window=cfg.bb_period, window_dev=cfg.bb_std)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        bb_pct_b = bb.bollinger_pband().iloc[-1]
        
        # Volume
        volume_sma = volume.rolling(cfg.volume_period).mean().iloc[-1] if volume.sum() > 0 else 0
        volume_ratio = float(volume.iloc[-1]) / volume_sma if volume_sma > 0 else 1.0
        
        # OBV trend (20-day slope)
        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_trend = self._compute_slope(obv, 20)
        
        # Trend analysis
        above_sma_20 = current_price > sma_20
        above_sma_50 = current_price > sma_50
        above_sma_200 = current_price > sma_200
        golden_cross = sma_50 > sma_200
        death_cross = sma_50 < sma_200
        trend_direction = self._determine_trend(current_price, sma_20, sma_50)
        
        # Derived scores
        momentum_score = self._compute_momentum_score(rsi_14, macd_histogram, bb_pct_b, stoch_k)
        trend_score = self._compute_trend_score(above_sma_20, above_sma_50, above_sma_200, golden_cross, adx)
        volatility_regime = self._determine_volatility_regime(close, atr_pct)
        
        return TechnicalSnapshot(
            current_price=current_price,
            sma_20=float(sma_20) if not pd.isna(sma_20) else 0.0,
            sma_50=float(sma_50) if not pd.isna(sma_50) else 0.0,
            sma_200=float(sma_200) if not pd.isna(sma_200) else 0.0,
            ema_12=float(ema_12) if not pd.isna(ema_12) else 0.0,
            ema_26=float(ema_26) if not pd.isna(ema_26) else 0.0,
            rsi_14=float(rsi_14) if not pd.isna(rsi_14) else 50.0,
            rsi_7=float(rsi_7) if rsi_7 is not None and not pd.isna(rsi_7) else None,
            macd=float(macd_line) if not pd.isna(macd_line) else 0.0,
            macd_signal=float(macd_signal) if not pd.isna(macd_signal) else 0.0,
            macd_histogram=float(macd_histogram) if not pd.isna(macd_histogram) else 0.0,
            atr_14=float(atr_14) if not pd.isna(atr_14) else 0.0,
            atr_pct=float(atr_pct) if not pd.isna(atr_pct) else 0.0,
            bollinger_upper=float(bb_upper) if not pd.isna(bb_upper) else current_price * 1.02,
            bollinger_lower=float(bb_lower) if not pd.isna(bb_lower) else current_price * 0.98,
            bollinger_pct_b=float(np.clip(bb_pct_b, 0, 1)) if not pd.isna(bb_pct_b) else 0.5,
            stoch_k=float(stoch_k) if not pd.isna(stoch_k) else 50.0,
            stoch_d=float(stoch_d) if not pd.isna(stoch_d) else 50.0,
            williams_r=float(williams_r) if not pd.isna(williams_r) else -50.0,
            cci=float(cci) if not pd.isna(cci) else 0.0,
            adx=float(adx) if not pd.isna(adx) else 0.0,
            volume_sma_20=float(volume_sma) if not pd.isna(volume_sma) else 0.0,
            volume_ratio=float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
            obv_trend=float(obv_trend) if not pd.isna(obv_trend) else 0.0,
            trend_direction=trend_direction,
            above_sma_20=above_sma_20,
            above_sma_50=above_sma_50,
            above_sma_200=above_sma_200,
            golden_cross=golden_cross,
            death_cross=death_cross,
            momentum_score=momentum_score,
            trend_score=trend_score,
            volatility_regime=volatility_regime,
        )
    
    def _compute_fallback(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        current_price: float,
    ) -> TechnicalSnapshot:
        """Fallback computation without TA library."""
        cfg = self.config
        
        # Simple calculations
        sma_20 = close.rolling(cfg.sma_short).mean().iloc[-1]
        sma_50 = close.rolling(cfg.sma_medium).mean().iloc[-1]
        sma_200 = close.rolling(cfg.sma_long).mean().iloc[-1] if len(close) >= cfg.sma_long else close.mean()
        
        ema_12 = close.ewm(span=cfg.ema_fast, adjust=False).mean().iloc[-1]
        ema_26 = close.ewm(span=cfg.ema_slow, adjust=False).mean().iloc[-1]
        
        macd_line = ema_12 - ema_26
        macd_signal = pd.Series([ema_12 - ema_26]).ewm(span=cfg.macd_signal, adjust=False).mean().iloc[-1]
        macd_histogram = macd_line - macd_signal
        
        rsi_14 = self._compute_rsi(close, cfg.rsi_period)
        
        above_sma_20 = current_price > sma_20
        above_sma_50 = current_price > sma_50
        above_sma_200 = current_price > sma_200
        golden_cross = sma_50 > sma_200
        death_cross = sma_50 < sma_200
        trend_direction = self._determine_trend(current_price, sma_20, sma_50)
        
        return TechnicalSnapshot(
            current_price=current_price,
            sma_20=float(sma_20) if not pd.isna(sma_20) else 0.0,
            sma_50=float(sma_50) if not pd.isna(sma_50) else 0.0,
            sma_200=float(sma_200) if not pd.isna(sma_200) else 0.0,
            ema_12=float(ema_12) if not pd.isna(ema_12) else 0.0,
            ema_26=float(ema_26) if not pd.isna(ema_26) else 0.0,
            rsi_14=rsi_14,
            macd=float(macd_line),
            macd_signal=float(macd_signal),
            macd_histogram=float(macd_histogram),
            trend_direction=trend_direction,
            above_sma_20=above_sma_20,
            above_sma_50=above_sma_50,
            above_sma_200=above_sma_200,
            golden_cross=golden_cross,
            death_cross=death_cross,
        )
    
    # =========================================================================
    # Series Calculations (for backtesting - returns full series)
    # =========================================================================
    
    def compute_rsi_series(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI for entire series (backtesting use)."""
        if TA_AVAILABLE:
            return RSIIndicator(close, window=period).rsi()
        else:
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.inf)
            return 100 - (100 / (1 + rs))
    
    def compute_sma_series(self, series: pd.Series, period: int) -> pd.Series:
        """Compute SMA for entire series."""
        if TA_AVAILABLE:
            return SMAIndicator(series, window=period).sma_indicator()
        return series.rolling(period).mean()
    
    def compute_ema_series(self, series: pd.Series, period: int) -> pd.Series:
        """Compute EMA for entire series."""
        if TA_AVAILABLE:
            return EMAIndicator(series, window=period).ema_indicator()
        return series.ewm(span=period, adjust=False).mean()
    
    def compute_macd_series(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD line, signal, and histogram series."""
        if TA_AVAILABLE:
            macd = MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
            return macd.macd(), macd.macd_signal(), macd.macd_diff()
        else:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            macd_hist = macd_line - macd_signal_line
            return macd_line, macd_signal_line, macd_hist
    
    def compute_bollinger_series(
        self,
        close: pd.Series,
        period: int = 20,
        std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands: upper, lower, %B."""
        if TA_AVAILABLE:
            bb = BollingerBands(close, window=period, window_dev=std)
            return bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_pband()
        else:
            sma = close.rolling(period).mean()
            std_dev = close.rolling(period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            pct_b = (close - lower) / (upper - lower)
            return upper, lower, pct_b.clip(0, 1)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float:
        """Compute RSI (fallback without TA library)."""
        if len(series) < period + 1:
            return 50.0
        
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        
        if avg_loss == 0 or pd.isna(avg_loss):
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    
    @staticmethod
    def _compute_slope(series: pd.Series, period: int = 20) -> float:
        """Compute normalized slope of a series."""
        if len(series) < period:
            return 0.0
        
        recent = series.iloc[-period:]
        x = np.arange(len(recent))
        
        if recent.std() == 0:
            return 0.0
        
        # Normalized slope
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope / recent.std())
    
    def _determine_trend(
        self,
        current: float,
        sma_20: float,
        sma_50: float,
    ) -> Literal["UP", "DOWN", "SIDEWAYS"]:
        """Determine trend direction based on price and SMAs."""
        # Handle NaN values
        if pd.isna(sma_20) or pd.isna(sma_50):
            return "SIDEWAYS"
        
        if current > sma_20 > sma_50:
            return "UP"
        elif current < sma_20 < sma_50:
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def _compute_momentum_score(
        self,
        rsi: float,
        macd_histogram: float,
        bb_pct_b: float,
        stoch_k: float = 50.0,
    ) -> float:
        """
        Compute momentum score from -1 (bearish) to +1 (bullish).
        
        Components:
        - RSI position (30-70 neutral zone)
        - MACD histogram sign and magnitude
        - Bollinger %B position
        - Stochastic %K position
        """
        # Handle NaN values
        if pd.isna(rsi):
            rsi = 50.0
        if pd.isna(macd_histogram):
            macd_histogram = 0.0
        if pd.isna(bb_pct_b):
            bb_pct_b = 0.5
        if pd.isna(stoch_k):
            stoch_k = 50.0
        
        # RSI contribution (-1 to +1)
        if rsi < 30:
            rsi_score = -0.5 - (30 - rsi) / 60
        elif rsi > 70:
            rsi_score = 0.5 + (rsi - 70) / 60
        else:
            rsi_score = (rsi - 50) / 40
        
        # MACD contribution (normalized)
        macd_score = float(np.clip(macd_histogram / 2.0, -1, 1))
        
        # Bollinger contribution
        bb_score = (bb_pct_b - 0.5) * 2
        
        # Stochastic contribution
        stoch_score = (stoch_k - 50) / 50
        
        # Weighted average
        momentum = 0.35 * rsi_score + 0.30 * macd_score + 0.20 * bb_score + 0.15 * stoch_score
        
        return float(np.clip(momentum, -1, 1))
    
    def _compute_trend_score(
        self,
        above_sma_20: bool,
        above_sma_50: bool,
        above_sma_200: bool,
        golden_cross: bool,
        adx: float = 0.0,
    ) -> float:
        """
        Compute trend score from -1 (bearish) to +1 (bullish).
        """
        score = 0.0
        
        # SMA positions
        if above_sma_200:
            score += 0.35
        else:
            score -= 0.35
        
        if above_sma_50:
            score += 0.25
        else:
            score -= 0.25
        
        if above_sma_20:
            score += 0.15
        else:
            score -= 0.15
        
        # Golden/Death cross
        if golden_cross:
            score += 0.15
        else:
            score -= 0.15
        
        # ADX boost (strong trends get bonus)
        if not pd.isna(adx) and adx > 25:
            # Amplify the score based on trend strength
            multiplier = 1 + (adx - 25) / 50
            score = score * min(multiplier, 1.5)
        
        return float(np.clip(score, -1, 1))
    
    def _determine_volatility_regime(
        self,
        close: pd.Series,
        atr_pct: float,
    ) -> Literal["LOW", "NORMAL", "HIGH", "EXTREME"]:
        """Determine volatility regime based on ATR and historical volatility."""
        if len(close) < 60:
            return "NORMAL"
        
        # Calculate realized volatility
        returns = close.pct_change().dropna()
        
        if len(returns) < 20:
            return "NORMAL"
        
        realized_vol_20d = returns.iloc[-20:].std() * np.sqrt(252) * 100
        
        # Historical volatility for comparison
        if len(returns) >= 252:
            historical_vol = returns.iloc[-252:].std() * np.sqrt(252) * 100
        else:
            historical_vol = returns.std() * np.sqrt(252) * 100
        
        vol_ratio = realized_vol_20d / historical_vol if historical_vol > 0 else 1.0
        
        # Also consider ATR percentage
        if atr_pct > 5.0 or vol_ratio > 2.0:
            return "EXTREME"
        elif atr_pct > 3.0 or vol_ratio > 1.5:
            return "HIGH"
        elif atr_pct < 1.5 and vol_ratio < 0.7:
            return "LOW"
        else:
            return "NORMAL"


# Singleton instance
_technical_service: TechnicalService | None = None


def get_technical_service(config: IndicatorConfig | None = None) -> TechnicalService:
    """Get singleton TechnicalService instance."""
    global _technical_service
    if _technical_service is None or config is not None:
        _technical_service = TechnicalService(config)
    return _technical_service
