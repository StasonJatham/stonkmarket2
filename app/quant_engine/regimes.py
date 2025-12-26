"""
Market regime detection.

Classifies market into trend (bull/bear/neutral) and volatility (low/medium/high) regimes.
Used for conditioning alpha models and dip effectiveness analysis.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.types import RegimeState, RegimeTrend, RegimeVolatility

logger = logging.getLogger(__name__)


def compute_trend_regime(
    market_returns: pd.Series,
    short_window: int = 21,
    long_window: int = 252,
    bull_threshold: float = 0.0,
    bear_threshold: float = 0.0,
) -> tuple[RegimeTrend, float]:
    """
    Classify trend regime based on moving average crossover.
    
    Uses simple short/long MA crossover to determine trend direction.
    
    Parameters
    ----------
    market_returns : pd.Series
        Market index returns.
    short_window : int
        Short MA window in trading days.
    long_window : int
        Long MA window in trading days.
    bull_threshold : float
        Minimum excess for bull classification.
    bear_threshold : float
        Minimum excess for bear classification.
    
    Returns
    -------
    tuple[RegimeTrend, float]
        Regime classification and underlying score.
    """
    if len(market_returns) < long_window:
        return RegimeTrend.NEUTRAL, 0.0
    
    # Compute cumulative returns for MA calculation
    cum_ret = (1 + market_returns).cumprod()
    
    short_ma = cum_ret.rolling(short_window).mean().iloc[-1]
    long_ma = cum_ret.rolling(long_window).mean().iloc[-1]
    
    # Score: (short - long) / long
    if long_ma > 0:
        score = float((short_ma - long_ma) / long_ma)
    else:
        score = 0.0
    
    # Classify
    if score > bull_threshold:
        regime = RegimeTrend.BULL
    elif score < -bear_threshold:
        regime = RegimeTrend.BEAR
    else:
        regime = RegimeTrend.NEUTRAL
    
    return regime, score


def compute_volatility_regime(
    market_returns: pd.Series,
    window: int = 21,
    low_threshold: float = 0.10,
    high_threshold: float = 0.25,
) -> tuple[RegimeVolatility, float]:
    """
    Classify volatility regime based on realized volatility.
    
    Uses annualized rolling volatility vs thresholds.
    
    Parameters
    ----------
    market_returns : pd.Series
        Market index returns.
    window : int
        Volatility estimation window.
    low_threshold : float
        Annualized vol below this = LOW.
    high_threshold : float
        Annualized vol above this = HIGH.
    
    Returns
    -------
    tuple[RegimeVolatility, float]
        Regime classification and underlying volatility.
    """
    if len(market_returns) < window:
        return RegimeVolatility.MEDIUM, 0.15
    
    # Compute annualized volatility
    vol = float(market_returns.iloc[-window:].std() * np.sqrt(252))
    
    # Classify
    if vol < low_threshold:
        regime = RegimeVolatility.LOW
    elif vol > high_threshold:
        regime = RegimeVolatility.HIGH
    else:
        regime = RegimeVolatility.MEDIUM
    
    return regime, vol


def compute_regime_state(
    market_returns: pd.Series,
    as_of: date,
    short_window: int = 21,
    long_window: int = 252,
    vol_window: int = 21,
) -> RegimeState:
    """
    Compute complete regime state.
    
    Parameters
    ----------
    market_returns : pd.Series
        Market index returns (date-indexed).
    as_of : date
        Reference date.
    short_window : int
        Short MA window for trend.
    long_window : int
        Long MA window for trend.
    vol_window : int
        Volatility window.
    
    Returns
    -------
    RegimeState
        Complete regime classification.
    """
    # Filter to as_of date
    ts = pd.Timestamp(as_of)
    returns = market_returns[market_returns.index <= ts]
    
    trend, trend_score = compute_trend_regime(returns, short_window, long_window)
    vol, vol_score = compute_volatility_regime(returns, vol_window)
    
    return RegimeState(
        trend=trend,
        volatility=vol,
        trend_score=trend_score,
        vol_score=vol_score,
        as_of=as_of,
    )


def compute_regime_series(
    market_returns: pd.Series,
    short_window: int = 21,
    long_window: int = 252,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Compute regime classification for each date in series.
    
    Parameters
    ----------
    market_returns : pd.Series
        Market index returns.
    short_window : int
        Short MA window.
    long_window : int
        Long MA window.
    vol_window : int
        Volatility window.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: trend, vol, trend_score, vol_score.
    """
    records = []
    
    for i in range(long_window, len(market_returns)):
        dt = market_returns.index[i]
        returns_to_date = market_returns.iloc[:i+1]
        
        trend, trend_score = compute_trend_regime(
            returns_to_date, short_window, long_window
        )
        vol, vol_score = compute_volatility_regime(
            returns_to_date, vol_window
        )
        
        records.append({
            "date": dt,
            "trend": trend.value,
            "vol": vol.value,
            "trend_score": trend_score,
            "vol_score": vol_score,
        })
    
    return pd.DataFrame(records).set_index("date")


def get_regime_conditional_stats(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute return statistics conditioned on regime.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    regimes : pd.DataFrame
        Regime classifications.
    
    Returns
    -------
    dict[str, Any]
        Regime-conditional statistics.
    """
    # Align
    returns, regimes = returns.align(regimes, join="inner", axis=0)
    
    results = {}
    
    for trend in RegimeTrend:
        for vol in RegimeVolatility:
            mask = (regimes["trend"] == trend.value) & (regimes["vol"] == vol.value)
            regime_returns = returns.loc[mask]
            
            if len(regime_returns) < 10:
                continue
            
            key = f"{trend.value}_{vol.value}"
            results[key] = {
                "mean": float(regime_returns.mean().mean()),
                "std": float(regime_returns.std().mean()),
                "n_days": int(mask.sum()),
                "sharpe": float(
                    regime_returns.mean().mean() / regime_returns.std().mean() * np.sqrt(252)
                ) if regime_returns.std().mean() > 1e-12 else 0.0,
            }
    
    return results
