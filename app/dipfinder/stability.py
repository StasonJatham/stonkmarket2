"""Stability scoring module.

Computes stability metrics from price data and yfinance info:
- Beta (from yfinance)
- Realized volatility (from price history)
- Max drawdown (from price history)
- Typical dip size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import DipFinderConfig, get_dipfinder_config
from .dip import compute_dip_series_windowed, compute_typical_dip


@dataclass
class StabilityMetrics:
    """Stability metrics and score for a stock."""

    ticker: str
    score: float  # 0-100

    # Contributing factors
    beta: Optional[float] = None
    volatility_252d: Optional[float] = None  # Annualized volatility
    max_drawdown_5y: Optional[float] = None  # Maximum drawdown
    typical_dip_365: Optional[float] = None  # Median 365-day dip

    # Sub-scores
    beta_score: float = 50.0
    volatility_score: float = 50.0
    drawdown_score: float = 50.0
    typical_dip_score: float = 50.0
    fundamental_stability_score: float = 50.0

    # Data quality
    has_price_data: bool = False
    price_data_days: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "score": round(self.score, 2),
            "beta": round(self.beta, 4) if self.beta is not None else None,
            "volatility_252d": round(self.volatility_252d, 4)
            if self.volatility_252d is not None
            else None,
            "max_drawdown_5y": round(self.max_drawdown_5y, 4)
            if self.max_drawdown_5y is not None
            else None,
            "typical_dip_365": round(self.typical_dip_365, 4)
            if self.typical_dip_365 is not None
            else None,
            "beta_score": round(self.beta_score, 2),
            "volatility_score": round(self.volatility_score, 2),
            "drawdown_score": round(self.drawdown_score, 2),
            "typical_dip_score": round(self.typical_dip_score, 2),
            "fundamental_stability_score": round(self.fundamental_stability_score, 2),
            "has_price_data": self.has_price_data,
            "price_data_days": self.price_data_days,
        }


def compute_daily_returns(close_prices: np.ndarray) -> np.ndarray:
    """
    Compute daily returns from close prices.

    Args:
        close_prices: Array of closing prices

    Returns:
        Array of daily returns (length n-1)
    """
    if len(close_prices) < 2:
        return np.array([])

    return np.diff(close_prices) / close_prices[:-1]


def compute_volatility(
    close_prices: np.ndarray,
    days: int = 252,
    annualize: bool = True,
) -> Optional[float]:
    """
    Compute realized volatility.

    Args:
        close_prices: Array of closing prices
        days: Number of days to use
        annualize: If True, annualize using sqrt(252)

    Returns:
        Volatility as decimal (0.25 = 25%)
    """
    # Use last N days
    prices = close_prices[-days:] if len(close_prices) > days else close_prices

    if len(prices) < 20:  # Need minimum data
        return None

    returns = compute_daily_returns(prices)
    daily_vol = float(np.std(returns))

    if annualize:
        return daily_vol * np.sqrt(252)
    return daily_vol


def compute_max_drawdown(close_prices: np.ndarray) -> Optional[float]:
    """
    Compute maximum drawdown from peak to trough.

    Args:
        close_prices: Array of closing prices

    Returns:
        Max drawdown as positive fraction (0.35 = 35% drawdown)
    """
    if len(close_prices) < 2:
        return None

    # Running maximum
    running_max = np.maximum.accumulate(close_prices)

    # Drawdown at each point
    drawdowns = (running_max - close_prices) / running_max

    # Maximum drawdown
    max_dd = float(np.max(drawdowns))

    return max_dd


def _score_beta(beta: Optional[float]) -> float:
    """Score beta (lower is more stable, 1.0 is market average)."""
    if beta is None:
        return 50.0  # Neutral

    # Beta < 0.5: very stable (score 90-100)
    # Beta 0.5-1.0: stable (score 70-90)
    # Beta 1.0-1.5: average (score 50-70)
    # Beta 1.5-2.0: volatile (score 30-50)
    # Beta > 2.0: very volatile (score 10-30)

    if beta < 0:
        # Negative beta is unusual, treat as moderately stable
        return 60.0
    elif beta <= 0.5:
        return 90.0 + (0.5 - beta) / 0.5 * 10
    elif beta <= 1.0:
        return 70.0 + (1.0 - beta) / 0.5 * 20
    elif beta <= 1.5:
        return 50.0 + (1.5 - beta) / 0.5 * 20
    elif beta <= 2.0:
        return 30.0 + (2.0 - beta) / 0.5 * 20
    else:
        return max(10.0, 30.0 - (beta - 2.0) * 10)


def _score_volatility(volatility: Optional[float]) -> float:
    """Score annualized volatility (lower is more stable)."""
    if volatility is None:
        return 50.0

    # Volatility < 15%: very stable (85-100)
    # Volatility 15-25%: stable (65-85)
    # Volatility 25-40%: average (45-65)
    # Volatility 40-60%: volatile (25-45)
    # Volatility > 60%: very volatile (10-25)

    if volatility <= 0.15:
        return 85.0 + (0.15 - volatility) / 0.15 * 15
    elif volatility <= 0.25:
        return 65.0 + (0.25 - volatility) / 0.10 * 20
    elif volatility <= 0.40:
        return 45.0 + (0.40 - volatility) / 0.15 * 20
    elif volatility <= 0.60:
        return 25.0 + (0.60 - volatility) / 0.20 * 20
    else:
        return max(10.0, 25.0 - (volatility - 0.60) * 50)


def _score_max_drawdown(max_dd: Optional[float]) -> float:
    """Score max drawdown (lower is better)."""
    if max_dd is None:
        return 50.0

    # MDD < 20%: excellent (85-100)
    # MDD 20-35%: good (65-85)
    # MDD 35-50%: average (45-65)
    # MDD 50-70%: poor (25-45)
    # MDD > 70%: very poor (10-25)

    if max_dd <= 0.20:
        return 85.0 + (0.20 - max_dd) / 0.20 * 15
    elif max_dd <= 0.35:
        return 65.0 + (0.35 - max_dd) / 0.15 * 20
    elif max_dd <= 0.50:
        return 45.0 + (0.50 - max_dd) / 0.15 * 20
    elif max_dd <= 0.70:
        return 25.0 + (0.70 - max_dd) / 0.20 * 20
    else:
        return max(10.0, 25.0 - (max_dd - 0.70) * 50)


def _score_typical_dip(typical_dip: Optional[float]) -> float:
    """Score typical dip size (lower is more stable)."""
    if typical_dip is None:
        return 50.0

    # Typical dip < 5%: very stable (85-100)
    # Typical dip 5-10%: stable (65-85)
    # Typical dip 10-15%: average (45-65)
    # Typical dip 15-25%: volatile (25-45)
    # Typical dip > 25%: very volatile (10-25)

    if typical_dip <= 0.05:
        return 85.0 + (0.05 - typical_dip) / 0.05 * 15
    elif typical_dip <= 0.10:
        return 65.0 + (0.10 - typical_dip) / 0.05 * 20
    elif typical_dip <= 0.15:
        return 45.0 + (0.15 - typical_dip) / 0.05 * 20
    elif typical_dip <= 0.25:
        return 25.0 + (0.25 - typical_dip) / 0.10 * 20
    else:
        return max(10.0, 25.0 - (typical_dip - 0.25) * 100)


def _compute_fundamental_stability_score(info: Dict[str, Any]) -> float:
    """Compute fundamental stability from yfinance info."""
    scores = []

    # Positive free cash flow is stable
    fcf = info.get("freeCashflow")
    if fcf is not None:
        if fcf > 0:
            scores.append(75.0)
        else:
            scores.append(35.0)

    # Non-tiny profit margins are stable
    margin = info.get("profitMargins")
    if margin is not None:
        if margin > 0.10:
            scores.append(80.0)
        elif margin > 0.05:
            scores.append(65.0)
        elif margin > 0:
            scores.append(50.0)
        else:
            scores.append(30.0)

    # Low debt to equity is stable
    de = info.get("debtToEquity")
    if de is not None:
        if de < 50:
            scores.append(80.0)
        elif de < 100:
            scores.append(65.0)
        elif de < 200:
            scores.append(45.0)
        else:
            scores.append(30.0)

    return sum(scores) / len(scores) if scores else 50.0


def compute_stability_score(
    ticker: str,
    close_prices: np.ndarray,
    info: Optional[Dict[str, Any]] = None,
    config: Optional[DipFinderConfig] = None,
) -> StabilityMetrics:
    """
    Compute stability score for a stock.

    Args:
        ticker: Stock ticker symbol
        close_prices: Array of closing prices
        info: yfinance info dictionary (for beta and fundamental data)
        config: Optional config

    Returns:
        StabilityMetrics with score and factors
    """
    if config is None:
        config = get_dipfinder_config()

    has_price_data = len(close_prices) > 0
    price_data_days = len(close_prices)

    # Get beta from yfinance info
    beta = None
    if info:
        raw_beta = info.get("beta")
        if raw_beta is not None:
            try:
                beta = float(raw_beta)
            except (ValueError, TypeError):
                pass

    # Compute volatility
    volatility = compute_volatility(close_prices, days=252)

    # Compute max drawdown (use all available history)
    max_dd = compute_max_drawdown(close_prices)

    # Compute typical dip (365-day window)
    typical_dip = None
    if len(close_prices) >= 365:
        dip_series = compute_dip_series_windowed(close_prices, 365)
        typical_dip = compute_typical_dip(dip_series)
    elif len(close_prices) >= 30:
        # Fall back to 30-day window
        dip_series = compute_dip_series_windowed(close_prices, 30)
        typical_dip = compute_typical_dip(dip_series)

    # Score each component
    beta_score = _score_beta(beta)
    volatility_score = _score_volatility(volatility)
    drawdown_score = _score_max_drawdown(max_dd)
    typical_dip_score = _score_typical_dip(typical_dip)
    fundamental_stability_score = _compute_fundamental_stability_score(info or {})

    # Weighted final score
    # Price-based metrics are more important than yfinance beta
    final_score = (
        beta_score * 0.15
        + volatility_score * 0.25
        + drawdown_score * 0.25
        + typical_dip_score * 0.20
        + fundamental_stability_score * 0.15
    )

    return StabilityMetrics(
        ticker=ticker,
        score=final_score,
        beta=beta,
        volatility_252d=volatility,
        max_drawdown_5y=max_dd,
        typical_dip_365=typical_dip,
        beta_score=beta_score,
        volatility_score=volatility_score,
        drawdown_score=drawdown_score,
        typical_dip_score=typical_dip_score,
        fundamental_stability_score=fundamental_stability_score,
        has_price_data=has_price_data,
        price_data_days=price_data_days,
    )
