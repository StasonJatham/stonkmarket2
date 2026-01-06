"""
Sector Regime Service - Prevents buying good stocks in bad sectors.

This is the "XLF is crashing, don't buy banks" filter.
Uses existing sector_etfs mapping from runtime_settings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.quant_engine.core.technical_service import TechnicalService, get_technical_service

logger = logging.getLogger(__name__)


# Default sector ETF mapping (can be overridden from runtime_settings)
DEFAULT_SECTOR_ETFS = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


@dataclass
class SectorRegimeState:
    """Sector-specific regime information."""
    
    sector: str
    sector_etf: str
    
    # Regime classification
    regime: Literal["STRONG", "NEUTRAL", "WEAK", "CRISIS"]
    
    # Metrics
    price_vs_sma50: float  # % above/below 50-day SMA
    price_vs_sma200: float  # % above/below 200-day SMA
    momentum_20d: float  # 20-day return %
    relative_to_spy: float  # Sector return - SPY return (20d)
    
    # Scoring multiplier (0.5 to 1.15)
    score_multiplier: float
    
    # Reason for the regime
    reason: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "sector": self.sector,
            "sector_etf": self.sector_etf,
            "regime": self.regime,
            "price_vs_sma50": round(self.price_vs_sma50, 2),
            "price_vs_sma200": round(self.price_vs_sma200, 2),
            "momentum_20d": round(self.momentum_20d, 2),
            "relative_to_spy": round(self.relative_to_spy, 2),
            "score_multiplier": round(self.score_multiplier, 2),
            "reason": self.reason,
        }


class SectorRegimeService:
    """
    Determines sector health to adjust stock scores.
    
    A 90/100 quality bank should NOT be bought if XLF is in crisis.
    """
    
    def __init__(
        self,
        technical_service: TechnicalService | None = None,
        sector_etf_map: dict[str, str] | None = None,
    ):
        self.tech = technical_service or get_technical_service()
        self._sector_map = sector_etf_map or DEFAULT_SECTOR_ETFS
        self._cache: dict[str, SectorRegimeState] = {}
    
    def set_sector_map(self, sector_etfs: list[dict[str, str]]) -> None:
        """Update sector mapping from runtime settings format."""
        self._sector_map = {item["sector"]: item["symbol"] for item in sector_etfs}
    
    def get_sector_etf(self, sector: str) -> str:
        """Get the ETF symbol for a sector."""
        return self._sector_map.get(sector, "SPY")
    
    def get_sector_regime(
        self,
        sector: str,
        sector_prices: pd.DataFrame | None = None,
        spy_prices: pd.DataFrame | None = None,
    ) -> SectorRegimeState:
        """
        Get regime state for a sector.
        
        Args:
            sector: Sector name (e.g., "Technology")
            sector_prices: OHLCV data for sector ETF
            spy_prices: OHLCV data for SPY (for relative comparison)
        """
        sector_etf = self.get_sector_etf(sector)
        
        if sector_prices is None or len(sector_prices) < 50:
            return self._default_regime(sector, sector_etf)
        
        # Normalize column names
        prices = sector_prices.copy()
        if hasattr(prices.columns, 'str'):
            prices.columns = prices.columns.str.lower()
        else:
            prices.columns = [str(c).lower() for c in prices.columns]
        
        close = prices.get("close", prices.get("adj close", prices.iloc[:, 0]))
        current_price = float(close.iloc[-1])
        
        # Calculate metrics
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_50
        
        price_vs_sma50 = ((current_price / sma_50) - 1) * 100 if sma_50 > 0 else 0
        price_vs_sma200 = ((current_price / sma_200) - 1) * 100 if sma_200 > 0 else 0
        momentum_20d = float(close.pct_change(20).iloc[-1] * 100) if len(close) > 20 else 0
        
        # Relative to SPY
        relative_to_spy = 0.0
        if spy_prices is not None and len(spy_prices) >= 20:
            spy_close = spy_prices.get("Close", spy_prices.get("close", spy_prices.iloc[:, 0]))
            spy_momentum = float(spy_close.pct_change(20).iloc[-1] * 100)
            relative_to_spy = momentum_20d - spy_momentum
        
        # Classify regime
        regime, multiplier, reason = self._classify_sector_regime(
            price_vs_sma50, price_vs_sma200, momentum_20d, relative_to_spy
        )
        
        return SectorRegimeState(
            sector=sector,
            sector_etf=sector_etf,
            regime=regime,
            price_vs_sma50=price_vs_sma50,
            price_vs_sma200=price_vs_sma200,
            momentum_20d=momentum_20d,
            relative_to_spy=relative_to_spy,
            score_multiplier=multiplier,
            reason=reason,
        )
    
    def _classify_sector_regime(
        self,
        vs_sma50: float,
        vs_sma200: float,
        momentum: float,
        relative: float,
    ) -> tuple[Literal["STRONG", "NEUTRAL", "WEAK", "CRISIS"], float, str]:
        """Classify sector regime and determine score multiplier."""
        
        # CRISIS: Below both SMAs + negative momentum + underperforming SPY
        if vs_sma50 < -5 and vs_sma200 < -10 and momentum < -10 and relative < -5:
            return (
                "CRISIS",
                0.5,  # Halve all scores in this sector
                f"Sector crisis: {momentum:.1f}% down, {relative:.1f}% vs SPY"
            )
        
        # WEAK: Below SMA50 + underperforming
        if vs_sma50 < -2 and momentum < -5:
            return (
                "WEAK",
                0.75,
                f"Sector weakness: Below SMA50, {momentum:.1f}% momentum"
            )
        
        # STRONG: Above both SMAs + outperforming SPY
        if vs_sma50 > 3 and vs_sma200 > 5 and relative > 2:
            return (
                "STRONG",
                1.15,  # Boost scores in strong sectors
                f"Sector strength: +{momentum:.1f}%, outperforming SPY by {relative:.1f}%"
            )
        
        # NEUTRAL: Everything else
        return (
            "NEUTRAL",
            1.0,
            "Sector neutral: No significant trend"
        )
    
    def _default_regime(self, sector: str, sector_etf: str) -> SectorRegimeState:
        """Return neutral regime when data is unavailable."""
        return SectorRegimeState(
            sector=sector,
            sector_etf=sector_etf,
            regime="NEUTRAL",
            price_vs_sma50=0.0,
            price_vs_sma200=0.0,
            momentum_20d=0.0,
            relative_to_spy=0.0,
            score_multiplier=1.0,
            reason="Insufficient sector data",
        )


# Singleton
_sector_regime_service: SectorRegimeService | None = None


def get_sector_regime_service() -> SectorRegimeService:
    """Get singleton SectorRegimeService instance."""
    global _sector_regime_service
    if _sector_regime_service is None:
        _sector_regime_service = SectorRegimeService()
    return _sector_regime_service
