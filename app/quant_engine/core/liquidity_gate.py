"""
Liquidity Gate - Ensures stocks are tradeable.

Prevents recommendations on illiquid stocks that retail can't exit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LiquidityState:
    """Liquidity analysis for a stock."""
    
    # Pass/fail
    passes_gate: bool
    
    # Metrics
    avg_daily_volume: float
    avg_daily_dollar_volume: float
    bid_ask_spread_pct: float | None  # If available
    
    # Classification
    liquidity_tier: Literal["EXCELLENT", "GOOD", "ADEQUATE", "POOR", "ILLIQUID"]
    
    # Position sizing guidance
    max_position_dollars: float  # Based on 1% of daily volume
    
    # Reason
    reason: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passes_gate": self.passes_gate,
            "avg_daily_volume": int(self.avg_daily_volume),
            "avg_daily_dollar_volume": int(self.avg_daily_dollar_volume),
            "bid_ask_spread_pct": round(self.bid_ask_spread_pct, 3) if self.bid_ask_spread_pct else None,
            "liquidity_tier": self.liquidity_tier,
            "max_position_dollars": int(self.max_position_dollars),
            "reason": self.reason,
        }


class LiquidityGate:
    """
    Ensures stocks have adequate liquidity for retail trading.
    
    Rules:
    - Minimum $500K daily dollar volume
    - Position should be <1% of daily volume for easy exit
    """
    
    def __init__(
        self,
        min_dollar_volume: float = 500_000,
        min_share_volume: float = 50_000,
        position_pct_of_volume: float = 0.01,
    ):
        self.min_dollar_volume = min_dollar_volume
        self.min_share_volume = min_share_volume
        self.position_pct_of_volume = position_pct_of_volume
    
    def check_liquidity(
        self,
        volume_series: pd.Series | None,
        price: float,
        bid_ask_spread_pct: float | None = None,
        avg_volume: float | None = None,
    ) -> LiquidityState:
        """
        Check if a stock has adequate liquidity.
        
        Args:
            volume_series: Historical daily volume (optional if avg_volume provided)
            price: Current price
            bid_ask_spread_pct: Bid-ask spread as percentage (if available)
            avg_volume: Pre-calculated average volume (alternative to volume_series)
        """
        # Calculate average volume
        if avg_volume is not None:
            calc_avg_volume = float(avg_volume)
        elif volume_series is not None and len(volume_series) >= 5:
            # Use last 20 days or what we have
            window = min(20, len(volume_series))
            calc_avg_volume = float(volume_series.tail(window).mean())
        else:
            return self._insufficient_data()
        
        # Calculate dollar volume
        avg_dollar_volume = calc_avg_volume * price
        
        # Determine tier
        tier, passes, reason = self._classify_liquidity(
            calc_avg_volume, avg_dollar_volume, bid_ask_spread_pct
        )
        
        # Calculate max position size (1% of daily volume)
        max_position = avg_dollar_volume * self.position_pct_of_volume
        
        return LiquidityState(
            passes_gate=passes,
            avg_daily_volume=calc_avg_volume,
            avg_daily_dollar_volume=avg_dollar_volume,
            bid_ask_spread_pct=bid_ask_spread_pct,
            liquidity_tier=tier,
            max_position_dollars=max_position,
            reason=reason,
        )
    
    def _classify_liquidity(
        self,
        avg_volume: float,
        avg_dollar_volume: float,
        spread_pct: float | None,
    ) -> tuple[Literal["EXCELLENT", "GOOD", "ADEQUATE", "POOR", "ILLIQUID"], bool, str]:
        """Classify liquidity tier."""
        
        # ILLIQUID: Below minimums
        if avg_dollar_volume < self.min_dollar_volume or avg_volume < self.min_share_volume:
            return (
                "ILLIQUID",
                False,
                f"Insufficient liquidity: ${avg_dollar_volume:,.0f} daily volume"
            )
        
        # Check spread if available (>1% spread is problematic)
        if spread_pct and spread_pct > 1.0:
            return (
                "POOR",
                False,
                f"Wide bid-ask spread: {spread_pct:.2f}%"
            )
        
        # EXCELLENT: Very liquid ($50M+)
        if avg_dollar_volume > 50_000_000:
            return (
                "EXCELLENT",
                True,
                f"Highly liquid: ${avg_dollar_volume/1e6:.0f}M daily volume"
            )
        
        # GOOD: Well liquid ($10M+)
        if avg_dollar_volume > 10_000_000:
            return (
                "GOOD",
                True,
                f"Good liquidity: ${avg_dollar_volume/1e6:.0f}M daily volume"
            )
        
        # ADEQUATE: Meets minimums
        if avg_dollar_volume > 1_000_000:
            return (
                "ADEQUATE",
                True,
                f"Adequate liquidity: ${avg_dollar_volume/1e6:.1f}M daily volume"
            )
        
        # POOR: Below $1M but above minimum
        return (
            "POOR",
            True,  # Still passes but with caution
            f"Low liquidity: ${avg_dollar_volume/1e3:.0f}K daily volume - use limit orders"
        )
    
    def _insufficient_data(self) -> LiquidityState:
        """Return when insufficient volume data."""
        return LiquidityState(
            passes_gate=False,
            avg_daily_volume=0,
            avg_daily_dollar_volume=0,
            bid_ask_spread_pct=None,
            liquidity_tier="ILLIQUID",
            max_position_dollars=0,
            reason="Insufficient volume history",
        )


# Singleton
_liquidity_gate: LiquidityGate | None = None


def get_liquidity_gate() -> LiquidityGate:
    """Get singleton LiquidityGate instance."""
    global _liquidity_gate
    if _liquidity_gate is None:
        _liquidity_gate = LiquidityGate()
    return _liquidity_gate
