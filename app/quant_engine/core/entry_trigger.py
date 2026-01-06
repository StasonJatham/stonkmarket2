"""
Entry Trigger Service - The "When to Buy" overlay.

Converts continuous scores into discrete BUY/WAIT signals using
technical indicators from TechnicalSnapshot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from app.quant_engine.core.technical_service import TechnicalSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TriggerCondition:
    """A single entry condition."""
    name: str
    is_active: bool
    value: float
    threshold: float
    description: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_active": self.is_active,
            "value": round(self.value, 2),
            "threshold": round(self.threshold, 2),
            "description": self.description,
        }


@dataclass
class EntryTriggerState:
    """Complete entry trigger analysis."""
    
    # Overall signal
    signal: Literal["BUY_NOW", "BUY_ZONE", "WAIT", "AVOID"]
    signal_strength: float  # 0-100
    
    # Individual conditions
    conditions: list[TriggerCondition] = field(default_factory=list)
    active_count: int = 0
    required_count: int = 2  # How many needed for BUY signal
    
    # Entry zone visualization
    entry_zone_low: float | None = None  # Lower bound of buy zone
    entry_zone_high: float | None = None  # Upper bound of buy zone
    current_price: float = 0.0
    
    # Recommendation
    action_text: str = ""
    wait_for: str | None = None  # What to wait for if not buying
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "signal_strength": round(self.signal_strength, 1),
            "conditions": [c.to_dict() for c in self.conditions],
            "active_count": self.active_count,
            "required_count": self.required_count,
            "entry_zone": {
                "low": round(self.entry_zone_low, 2) if self.entry_zone_low else None,
                "high": round(self.entry_zone_high, 2) if self.entry_zone_high else None,
                "current": round(self.current_price, 2),
            },
            "action_text": self.action_text,
            "wait_for": self.wait_for,
        }


class EntryTriggerService:
    """
    Determines if NOW is the right time to buy.
    
    Converts technical indicators into actionable triggers:
    - RSI oversold
    - Bollinger Band touch
    - Volume capitulation
    - Stochastic oversold
    """
    
    def __init__(
        self,
        rsi_oversold: float = 35,
        rsi_very_oversold: float = 25,
        bb_lower_threshold: float = 0.15,
        volume_spike_threshold: float = 2.0,
        stoch_oversold: float = 25,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_very_oversold = rsi_very_oversold
        self.bb_lower_threshold = bb_lower_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.stoch_oversold = stoch_oversold
    
    def analyze_entry(
        self,
        technicals: TechnicalSnapshot,
        current_drawdown_pct: float,
        support_levels: list[float] | None = None,
    ) -> EntryTriggerState:
        """
        Analyze entry timing and generate trigger signals.
        
        Args:
            technicals: Technical snapshot from TechnicalService
            current_drawdown_pct: Current drawdown from high (negative %)
            support_levels: Optional support price levels
        """
        conditions = []
        current_price = technicals.current_price
        
        # 1. RSI Oversold
        rsi_active = technicals.rsi_14 < self.rsi_oversold
        conditions.append(TriggerCondition(
            name="RSI Oversold",
            is_active=rsi_active,
            value=technicals.rsi_14,
            threshold=self.rsi_oversold,
            description=f"RSI {technicals.rsi_14:.0f} {'<' if rsi_active else '>'} {self.rsi_oversold}",
        ))
        
        # 2. Bollinger Band Lower Touch
        bb_active = technicals.bollinger_pct_b < self.bb_lower_threshold
        conditions.append(TriggerCondition(
            name="Bollinger Lower",
            is_active=bb_active,
            value=technicals.bollinger_pct_b * 100,
            threshold=self.bb_lower_threshold * 100,
            description=f"Price at {technicals.bollinger_pct_b*100:.0f}% of Bollinger range",
        ))
        
        # 3. Volume Spike (capitulation)
        vol_active = technicals.volume_ratio > self.volume_spike_threshold
        conditions.append(TriggerCondition(
            name="Volume Spike",
            is_active=vol_active,
            value=technicals.volume_ratio,
            threshold=self.volume_spike_threshold,
            description=f"Volume {technicals.volume_ratio:.1f}x average",
        ))
        
        # 4. Stochastic Oversold
        stoch_active = technicals.stoch_k < self.stoch_oversold
        conditions.append(TriggerCondition(
            name="Stochastic Oversold",
            is_active=stoch_active,
            value=technicals.stoch_k,
            threshold=self.stoch_oversold,
            description=f"Stochastic %K at {technicals.stoch_k:.0f}",
        ))
        
        # 5. Significant Dip (buying at discount)
        dip_active = current_drawdown_pct < -10
        conditions.append(TriggerCondition(
            name="Significant Dip",
            is_active=dip_active,
            value=current_drawdown_pct,
            threshold=-10,
            description=f"Down {abs(current_drawdown_pct):.0f}% from high",
        ))
        
        # 6. MACD Bullish Divergence (histogram turning up)
        macd_bullish = technicals.macd_histogram > 0 and technicals.macd < 0
        conditions.append(TriggerCondition(
            name="MACD Bullish Turn",
            is_active=macd_bullish,
            value=technicals.macd_histogram,
            threshold=0,
            description="MACD histogram turning positive from below zero",
        ))
        
        # 7. Support Level Test (if provided)
        if support_levels and len(support_levels) > 0:
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
            support_distance_pct = ((current_price / nearest_support) - 1) * 100
            support_active = abs(support_distance_pct) < 3  # Within 3% of support
            conditions.append(TriggerCondition(
                name="At Support",
                is_active=support_active,
                value=support_distance_pct,
                threshold=3.0,
                description=f"{support_distance_pct:.1f}% from support ${nearest_support:.2f}",
            ))
        
        # Count active conditions
        active_count = sum(1 for c in conditions if c.is_active)
        required_count = 2  # Need at least 2 triggers for BUY signal
        
        # Determine signal
        signal, strength, action, wait_for = self._determine_signal(
            conditions, active_count, required_count, technicals
        )
        
        # Calculate entry zone
        entry_zone_low = technicals.bollinger_lower if technicals.bollinger_lower > 0 else None
        entry_zone_high = technicals.sma_20 * 0.98 if technicals.sma_20 > 0 else None
        
        return EntryTriggerState(
            signal=signal,
            signal_strength=strength,
            conditions=conditions,
            active_count=active_count,
            required_count=required_count,
            entry_zone_low=entry_zone_low,
            entry_zone_high=entry_zone_high,
            current_price=current_price,
            action_text=action,
            wait_for=wait_for,
        )
    
    def _determine_signal(
        self,
        conditions: list[TriggerCondition],
        active_count: int,
        required: int,
        technicals: TechnicalSnapshot,
    ) -> tuple[Literal["BUY_NOW", "BUY_ZONE", "WAIT", "AVOID"], float, str, str | None]:
        """Determine overall signal from conditions."""
        
        # Check for AVOID conditions
        if technicals.rsi_14 > 70:
            return (
                "AVOID",
                20.0,
                "Overbought - wait for pullback",
                "RSI below 60"
            )
        
        if technicals.trend_direction == "DOWN" and technicals.death_cross:
            return (
                "AVOID",
                30.0,
                "Downtrend with death cross - high risk entry",
                "SMA50 to cross above SMA200"
            )
        
        # Very strong buy: RSI < 25 + in dip
        rsi_cond = next((c for c in conditions if c.name == "RSI Oversold"), None)
        dip_cond = next((c for c in conditions if c.name == "Significant Dip"), None)
        if rsi_cond and rsi_cond.value < self.rsi_very_oversold and dip_cond and dip_cond.is_active:
            return (
                "BUY_NOW",
                min(100, 75 + active_count * 5),
                f"Strong oversold condition (RSI {rsi_cond.value:.0f}) with significant dip",
                None
            )
        
        # Strong buy: 3+ triggers active
        if active_count >= 3:
            return (
                "BUY_NOW",
                min(100, 60 + active_count * 10),
                f"{active_count} buy triggers active - strong entry signal",
                None
            )
        
        # Buy zone: 2 triggers active
        if active_count >= required:
            return (
                "BUY_ZONE",
                min(80, 40 + active_count * 15),
                f"{active_count} triggers active - good entry zone",
                None
            )
        
        # Wait: Less than required
        inactive = [c for c in conditions if not c.is_active]
        wait_for_text = inactive[0].name if inactive else "better entry conditions"
        
        return (
            "WAIT",
            max(20, active_count * 20),
            f"Only {active_count}/{required} triggers active",
            wait_for_text
        )


# Singleton
_entry_trigger_service: EntryTriggerService | None = None


def get_entry_trigger_service() -> EntryTriggerService:
    """Get singleton EntryTriggerService instance."""
    global _entry_trigger_service
    if _entry_trigger_service is None:
        _entry_trigger_service = EntryTriggerService()
    return _entry_trigger_service
