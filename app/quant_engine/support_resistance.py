"""Support and resistance level detection.

Identifies key price levels where the stock has historically found
support (bounced) or resistance (rejected). Used to assess how far
a dip has to fall vs key support levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger("quant_engine.support_resistance")


@dataclass
class PriceLevel:
    """A significant price level (support or resistance)."""
    
    price: float
    level_type: str  # "support" or "resistance"
    strength: float  # 0-100, how strong is this level
    touches: int  # How many times price touched this level
    last_touch_days_ago: int  # Days since last touch
    volume_at_level: float | None = None  # Average volume when price was at this level
    
    def to_dict(self) -> dict:
        return {
            "price": round(self.price, 2),
            "level_type": self.level_type,
            "strength": round(self.strength, 1),
            "touches": self.touches,
            "last_touch_days_ago": self.last_touch_days_ago,
            "volume_at_level": self.volume_at_level,
        }


@dataclass
class SupportResistanceAnalysis:
    """Complete support/resistance analysis for a stock."""
    
    current_price: float
    
    # Key levels
    nearest_support: PriceLevel | None = None
    nearest_resistance: PriceLevel | None = None
    all_supports: list[PriceLevel] | None = None
    all_resistances: list[PriceLevel] | None = None
    
    # Distance metrics
    distance_to_support_pct: float | None = None  # % above nearest support
    distance_to_resistance_pct: float | None = None  # % below nearest resistance
    
    # Risk/reward from current price
    risk_to_support: float | None = None  # Potential loss to support
    reward_to_resistance: float | None = None  # Potential gain to resistance
    risk_reward_ratio: float | None = None  # reward / risk
    
    # Overall assessment
    price_position: str = "unknown"  # "near_support", "mid_range", "near_resistance", "below_support"
    entry_quality: str = "unknown"  # "excellent", "good", "fair", "poor"
    
    def to_dict(self) -> dict:
        return {
            "current_price": round(self.current_price, 2),
            "nearest_support": self.nearest_support.to_dict() if self.nearest_support else None,
            "nearest_resistance": self.nearest_resistance.to_dict() if self.nearest_resistance else None,
            "all_supports": [s.to_dict() for s in self.all_supports] if self.all_supports else None,
            "all_resistances": [r.to_dict() for r in self.all_resistances] if self.all_resistances else None,
            "distance_to_support_pct": round(self.distance_to_support_pct, 2) if self.distance_to_support_pct else None,
            "distance_to_resistance_pct": round(self.distance_to_resistance_pct, 2) if self.distance_to_resistance_pct else None,
            "risk_to_support": round(self.risk_to_support, 2) if self.risk_to_support else None,
            "reward_to_resistance": round(self.reward_to_resistance, 2) if self.reward_to_resistance else None,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2) if self.risk_reward_ratio else None,
            "price_position": self.price_position,
            "entry_quality": self.entry_quality,
        }


def find_swing_points(
    prices: np.ndarray,
    window: int = 5,
) -> tuple[list[int], list[int]]:
    """
    Find swing highs and swing lows in price data.
    
    A swing high is a local maximum (higher than surrounding points).
    A swing low is a local minimum (lower than surrounding points).
    
    Args:
        prices: Array of prices
        window: Number of bars on each side to check
        
    Returns:
        (swing_high_indices, swing_low_indices)
    """
    n = len(prices)
    swing_highs = []
    swing_lows = []
    
    for i in range(window, n - window):
        is_swing_high = True
        is_swing_low = True
        
        for j in range(1, window + 1):
            if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                is_swing_high = False
            if prices[i] >= prices[i - j] or prices[i] >= prices[i + j]:
                is_swing_low = False
        
        if is_swing_high:
            swing_highs.append(i)
        if is_swing_low:
            swing_lows.append(i)
    
    return swing_highs, swing_lows


def cluster_price_levels(
    prices: list[float],
    tolerance_pct: float = 2.0,
) -> list[tuple[float, int]]:
    """
    Cluster nearby price levels into zones.
    
    Args:
        prices: List of price levels
        tolerance_pct: % tolerance for clustering
        
    Returns:
        List of (cluster_center, count) tuples
    """
    if not prices:
        return []
    
    prices = sorted(prices)
    clusters = []
    current_cluster = [prices[0]]
    
    for price in prices[1:]:
        cluster_center = np.mean(current_cluster)
        if abs(price - cluster_center) / cluster_center <= tolerance_pct / 100:
            current_cluster.append(price)
        else:
            clusters.append((float(np.mean(current_cluster)), len(current_cluster)))
            current_cluster = [price]
    
    if current_cluster:
        clusters.append((float(np.mean(current_cluster)), len(current_cluster)))
    
    return clusters


def analyze_support_resistance(
    df: pd.DataFrame,
    current_price: float | None = None,
    lookback_days: int = 252,
    swing_window: int = 5,
    cluster_tolerance: float = 2.0,
    min_touches: int = 2,
) -> SupportResistanceAnalysis:
    """
    Analyze support and resistance levels from historical price data.
    
    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
        current_price: Current price (uses last close if None)
        lookback_days: Days of history to analyze
        swing_window: Window for swing point detection
        cluster_tolerance: % tolerance for level clustering
        min_touches: Minimum touches to be a valid level
        
    Returns:
        SupportResistanceAnalysis with all levels and metrics
    """
    if df is None or df.empty:
        return SupportResistanceAnalysis(current_price=current_price or 0.0)
    
    # Use recent data
    df = df.tail(lookback_days).copy()
    
    if current_price is None:
        current_price = float(df["Close"].iloc[-1])
    
    result = SupportResistanceAnalysis(current_price=current_price)
    
    # Find swing points
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None
    
    swing_high_idx, swing_low_idx = find_swing_points(highs, swing_window)
    _, swing_low_idx_from_lows = find_swing_points(-lows, swing_window)  # Invert to find lows
    
    # Get prices at swing points
    resistance_prices = [float(highs[i]) for i in swing_high_idx]
    support_prices = [float(lows[i]) for i in swing_low_idx_from_lows]
    
    # Also add swing lows from the low series
    support_prices.extend([float(lows[i]) for i in swing_low_idx])
    
    # Cluster into levels
    resistance_clusters = cluster_price_levels(resistance_prices, cluster_tolerance)
    support_clusters = cluster_price_levels(support_prices, cluster_tolerance)
    
    # Filter by minimum touches
    resistance_clusters = [(p, c) for p, c in resistance_clusters if c >= min_touches]
    support_clusters = [(p, c) for p, c in support_clusters if c >= min_touches]
    
    # Create PriceLevel objects
    n_days = len(df)
    all_supports = []
    all_resistances = []
    
    for price, touches in support_clusters:
        if price < current_price:  # Only consider supports below current price
            # Find when price was last near this level
            tolerance = price * cluster_tolerance / 100
            close_to_level = (np.abs(lows - price) <= tolerance)
            last_touch_idx = np.where(close_to_level)[0]
            last_touch_days = n_days - last_touch_idx[-1] if len(last_touch_idx) > 0 else n_days
            
            # Calculate strength (more touches + recency = stronger)
            recency_factor = max(0, 1 - last_touch_days / 252)  # Decay over a year
            strength = min(100, touches * 20 + recency_factor * 30)
            
            level = PriceLevel(
                price=price,
                level_type="support",
                strength=strength,
                touches=touches,
                last_touch_days_ago=int(last_touch_days),
            )
            all_supports.append(level)
    
    for price, touches in resistance_clusters:
        if price > current_price:  # Only consider resistances above current price
            tolerance = price * cluster_tolerance / 100
            close_to_level = (np.abs(highs - price) <= tolerance)
            last_touch_idx = np.where(close_to_level)[0]
            last_touch_days = n_days - last_touch_idx[-1] if len(last_touch_idx) > 0 else n_days
            
            recency_factor = max(0, 1 - last_touch_days / 252)
            strength = min(100, touches * 20 + recency_factor * 30)
            
            level = PriceLevel(
                price=price,
                level_type="resistance",
                strength=strength,
                touches=touches,
                last_touch_days_ago=int(last_touch_days),
            )
            all_resistances.append(level)
    
    # Sort by proximity to current price
    all_supports.sort(key=lambda x: current_price - x.price)
    all_resistances.sort(key=lambda x: x.price - current_price)
    
    result.all_supports = all_supports if all_supports else None
    result.all_resistances = all_resistances if all_resistances else None
    
    # Set nearest levels
    if all_supports:
        result.nearest_support = all_supports[0]
        result.distance_to_support_pct = ((current_price - all_supports[0].price) / current_price) * 100
        result.risk_to_support = current_price - all_supports[0].price
    
    if all_resistances:
        result.nearest_resistance = all_resistances[0]
        result.distance_to_resistance_pct = ((all_resistances[0].price - current_price) / current_price) * 100
        result.reward_to_resistance = all_resistances[0].price - current_price
    
    # Calculate risk/reward ratio
    if result.risk_to_support and result.reward_to_resistance and result.risk_to_support > 0:
        result.risk_reward_ratio = result.reward_to_resistance / result.risk_to_support
    
    # Determine price position
    if result.distance_to_support_pct is not None:
        if result.distance_to_support_pct <= 3:
            result.price_position = "near_support"
        elif result.distance_to_support_pct <= 10:
            result.price_position = "mid_range"
        else:
            result.price_position = "near_resistance" if result.distance_to_resistance_pct and result.distance_to_resistance_pct <= 5 else "mid_range"
    
    # Check if below all supports (broken support)
    if all_supports and current_price < min(s.price for s in all_supports):
        result.price_position = "below_support"
    
    # Determine entry quality
    if result.price_position == "near_support":
        if result.risk_reward_ratio and result.risk_reward_ratio >= 2:
            result.entry_quality = "excellent"
        else:
            result.entry_quality = "good"
    elif result.price_position == "below_support":
        result.entry_quality = "poor"  # Broken support is concerning
    elif result.risk_reward_ratio and result.risk_reward_ratio >= 2:
        result.entry_quality = "good"
    elif result.risk_reward_ratio and result.risk_reward_ratio >= 1:
        result.entry_quality = "fair"
    else:
        result.entry_quality = "poor"
    
    return result


async def get_support_resistance_analysis(
    symbol: str,
    lookback_days: int = 252,
) -> SupportResistanceAnalysis:
    """
    Get support/resistance analysis for a symbol.
    
    Args:
        symbol: Stock ticker
        lookback_days: Days of history to analyze
        
    Returns:
        SupportResistanceAnalysis
    """
    from app.services.data_providers import get_yfinance_service
    
    svc = get_yfinance_service()
    
    # Fetch enough history
    df = await svc.get_stock_history(symbol, period="2y")
    
    if df is None or df.empty:
        return SupportResistanceAnalysis(current_price=0.0)
    
    return analyze_support_resistance(df, lookback_days=lookback_days)
