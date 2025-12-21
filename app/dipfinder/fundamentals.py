"""Fundamentals module for quality scoring using yfinance.

Fetches stock info from yfinance with caching and rate limiting.
Computes quality score (0-100) from profitability, balance sheet,
cash generation, and growth metrics.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import yfinance as yf

from app.core.logging import get_logger
from app.database.connection import fetch_one, execute

from .config import DipFinderConfig, get_dipfinder_config

logger = get_logger("dipfinder.fundamentals")

# Thread pool for yfinance calls (not async native)
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory cache for yfinance info
_info_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

# Rate limiting state
_last_info_request_time: float = 0.0
_info_request_lock = asyncio.Lock()


@dataclass
class QualityMetrics:
    """Quality metrics and score for a stock."""
    
    ticker: str
    score: float  # 0-100
    
    # Contributing factors
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_to_market_cap: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    market_cap: Optional[float] = None
    avg_volume: Optional[float] = None
    
    # Sub-scores (0-100)
    profitability_score: float = 50.0
    balance_sheet_score: float = 50.0
    cash_generation_score: float = 50.0
    growth_score: float = 50.0
    liquidity_score: float = 50.0
    
    # Data quality
    fields_available: int = 0
    fields_total: int = 10
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON/DB storage."""
        return {
            "ticker": self.ticker,
            "score": round(self.score, 2),
            "profit_margin": self.profit_margin,
            "operating_margin": self.operating_margin,
            "debt_to_equity": self.debt_to_equity,
            "current_ratio": self.current_ratio,
            "free_cash_flow": self.free_cash_flow,
            "fcf_to_market_cap": self.fcf_to_market_cap,
            "revenue_growth": self.revenue_growth,
            "earnings_growth": self.earnings_growth,
            "market_cap": self.market_cap,
            "avg_volume": self.avg_volume,
            "profitability_score": round(self.profitability_score, 2),
            "balance_sheet_score": round(self.balance_sheet_score, 2),
            "cash_generation_score": round(self.cash_generation_score, 2),
            "growth_score": round(self.growth_score, 2),
            "liquidity_score": round(self.liquidity_score, 2),
            "fields_available": self.fields_available,
            "fields_total": self.fields_total,
        }


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float, handling None and invalid values."""
    if value is None:
        return default
    try:
        f = float(value)
        if not (f != f):  # Check for NaN
            return f
        return default
    except (ValueError, TypeError):
        return default


def _normalize_score(
    value: Optional[float],
    optimal: float,
    good_range: tuple[float, float],
    bad_threshold: Optional[float] = None,
    inverse: bool = False,
) -> float:
    """
    Normalize a metric to 0-100 score.
    
    Args:
        value: The metric value
        optimal: Optimal value (gets 100)
        good_range: (low, high) range that gets 60-100
        bad_threshold: Value below which score is 0-40
        inverse: If True, lower is better (e.g., debt ratio)
        
    Returns:
        Score 0-100 (50 if value is None)
    """
    if value is None:
        return 50.0  # Neutral for missing data
    
    if inverse:
        # Flip for metrics where lower is better
        value = -value
        optimal = -optimal
        good_range = (-good_range[1], -good_range[0])
        if bad_threshold is not None:
            bad_threshold = -bad_threshold
    
    low, high = good_range
    
    # Check if in good range
    if low <= value <= high:
        # Linear interpolation 60-100
        if high == low:
            return 80.0
        position = (value - low) / (high - low)
        return 60.0 + position * 40.0
    
    # Above good range (excellent)
    if value > high:
        # Cap at 100, scale bonus
        bonus = min((value - high) / (high - low + 0.001) * 10, 10)
        return min(100.0, 90.0 + bonus)
    
    # Below good range
    if bad_threshold is not None and value < bad_threshold:
        return 20.0  # Very bad
    
    # Between bad threshold and good range
    if low != 0:
        position = max(0, value / low)
        return 40.0 + position * 20.0
    
    return 40.0


def _compute_profitability_score(info: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Compute profitability sub-score."""
    profit_margin = _safe_float(info.get("profitMargins"))
    operating_margin = _safe_float(info.get("operatingMargins"))
    
    factors = {
        "profit_margin": profit_margin,
        "operating_margin": operating_margin,
    }
    
    # Score profit margin (10% optimal, 5-20% good)
    pm_score = _normalize_score(
        profit_margin,
        optimal=0.15,
        good_range=(0.05, 0.25),
        bad_threshold=0.0,
    )
    
    # Score operating margin (15% optimal, 8-30% good)
    om_score = _normalize_score(
        operating_margin,
        optimal=0.18,
        good_range=(0.08, 0.35),
        bad_threshold=0.0,
    )
    
    # Average, but weight profit margin slightly higher
    if profit_margin is not None and operating_margin is not None:
        score = pm_score * 0.55 + om_score * 0.45
    elif profit_margin is not None:
        score = pm_score
    elif operating_margin is not None:
        score = om_score
    else:
        score = 50.0  # Neutral
    
    return score, factors


def _compute_balance_sheet_score(info: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Compute balance sheet sub-score."""
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    current_ratio = _safe_float(info.get("currentRatio"))
    
    factors = {
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
    }
    
    # Score debt to equity (lower is better, 0-50% good, >200% bad)
    de_score = _normalize_score(
        debt_to_equity,
        optimal=20.0,  # 20% D/E is good
        good_range=(0.0, 80.0),
        bad_threshold=250.0,
        inverse=True,
    )
    
    # Score current ratio (1.5-2.5 optimal)
    cr_score = _normalize_score(
        current_ratio,
        optimal=1.8,
        good_range=(1.2, 3.0),
        bad_threshold=0.8,
    )
    
    # Average
    scores = []
    if debt_to_equity is not None:
        scores.append(de_score)
    if current_ratio is not None:
        scores.append(cr_score)
    
    score = sum(scores) / len(scores) if scores else 50.0
    
    return score, factors


def _compute_cash_generation_score(info: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Compute cash generation sub-score."""
    free_cash_flow = _safe_float(info.get("freeCashflow"))
    market_cap = _safe_float(info.get("marketCap"))
    
    factors = {
        "free_cash_flow": free_cash_flow,
    }
    
    # FCF positive is baseline
    if free_cash_flow is None:
        return 50.0, factors
    
    if free_cash_flow <= 0:
        return 30.0, factors  # Negative FCF is concerning
    
    # FCF to market cap ratio (yield)
    fcf_yield = None
    if market_cap and market_cap > 0:
        fcf_yield = free_cash_flow / market_cap
        factors["fcf_to_market_cap"] = fcf_yield
        
        # 5% FCF yield is excellent, 2-8% good
        return _normalize_score(
            fcf_yield,
            optimal=0.06,
            good_range=(0.02, 0.10),
            bad_threshold=0.0,
        ), factors
    
    # Just positive FCF without market cap comparison
    return 60.0, factors


def _compute_growth_score(info: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Compute growth sub-score."""
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))
    
    factors = {
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
    }
    
    # Score revenue growth (10-20% good)
    rg_score = _normalize_score(
        revenue_growth,
        optimal=0.15,
        good_range=(0.05, 0.30),
        bad_threshold=-0.10,
    ) if revenue_growth is not None else 50.0
    
    # Score earnings growth (15-25% good)
    eg_score = _normalize_score(
        earnings_growth,
        optimal=0.20,
        good_range=(0.05, 0.40),
        bad_threshold=-0.20,
    ) if earnings_growth is not None else 50.0
    
    # Average with slight revenue weight
    if revenue_growth is not None and earnings_growth is not None:
        score = rg_score * 0.45 + eg_score * 0.55
    elif revenue_growth is not None:
        score = rg_score
    elif earnings_growth is not None:
        score = eg_score
    else:
        score = 50.0
    
    return score, factors


def _compute_liquidity_score(info: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    """Compute liquidity/size sub-score."""
    market_cap = _safe_float(info.get("marketCap"))
    avg_volume = _safe_float(info.get("averageVolume")) or _safe_float(info.get("averageVolume10days"))
    
    factors = {
        "market_cap": market_cap,
        "avg_volume": avg_volume,
    }
    
    scores = []
    
    # Market cap score ($10B+ is most stable, $1B minimum for quality)
    if market_cap is not None:
        if market_cap >= 200e9:  # >$200B mega cap
            mc_score = 100.0
        elif market_cap >= 10e9:  # >$10B large cap
            mc_score = 80.0 + (market_cap - 10e9) / 190e9 * 20
        elif market_cap >= 2e9:  # >$2B mid cap
            mc_score = 60.0 + (market_cap - 2e9) / 8e9 * 20
        elif market_cap >= 300e6:  # >$300M small cap
            mc_score = 40.0 + (market_cap - 300e6) / 1.7e9 * 20
        else:
            mc_score = 20.0  # Micro cap, higher risk
        scores.append(mc_score)
    
    # Volume score (higher is better for liquidity)
    if avg_volume is not None:
        if avg_volume >= 10e6:  # >10M shares/day
            vol_score = 100.0
        elif avg_volume >= 1e6:
            vol_score = 70.0 + (avg_volume - 1e6) / 9e6 * 30
        elif avg_volume >= 100e3:
            vol_score = 50.0 + (avg_volume - 100e3) / 900e3 * 20
        else:
            vol_score = 30.0
        scores.append(vol_score)
    
    score = sum(scores) / len(scores) if scores else 50.0
    
    return score, factors


def _fetch_info_sync(ticker: str) -> Dict[str, Any]:
    """Synchronously fetch yfinance info (for thread pool)."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception as e:
        logger.warning(f"Failed to fetch info for {ticker}: {e}")
        return {}


async def _get_cached_info_from_db(ticker: str) -> Optional[Dict[str, Any]]:
    """Get cached info from database if not expired."""
    try:
        row = await fetch_one(
            """
            SELECT info_data, expires_at
            FROM yfinance_info_cache
            WHERE symbol = $1 AND expires_at > NOW()
            """,
            ticker.upper(),
        )
        
        if row and row["info_data"]:
            return row["info_data"] if isinstance(row["info_data"], dict) else json.loads(row["info_data"])
        
        return None
    except Exception as e:
        logger.debug(f"Could not load cached info for {ticker}: {e}")
        return None


async def _save_info_to_db(ticker: str, info: Dict[str, Any], ttl_seconds: int) -> None:
    """Save info to database cache."""
    try:
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        await execute(
            """
            INSERT INTO yfinance_info_cache (symbol, info_data, fetched_at, expires_at)
            VALUES ($1, $2, NOW(), $3)
            ON CONFLICT (symbol) DO UPDATE SET
                info_data = $2,
                fetched_at = NOW(),
                expires_at = $3
            """,
            ticker.upper(),
            json.dumps(info),
            expires_at,
        )
    except Exception as e:
        logger.debug(f"Could not save info cache for {ticker}: {e}")


async def fetch_stock_info(
    ticker: str,
    config: Optional[DipFinderConfig] = None,
) -> Dict[str, Any]:
    """
    Fetch stock info with caching and rate limiting.
    
    Args:
        ticker: Stock ticker symbol
        config: Optional config for TTL settings
        
    Returns:
        yfinance info dictionary
    """
    global _last_info_request_time
    
    if config is None:
        config = get_dipfinder_config()
    
    ticker = ticker.upper()
    
    # Check in-memory cache first
    now = time.time()
    cached = _info_cache.get(ticker)
    if cached and now - cached[0] < config.info_cache_ttl:
        return cached[1]
    
    # Check database cache
    db_cached = await _get_cached_info_from_db(ticker)
    if db_cached:
        _info_cache[ticker] = (now, db_cached)
        return db_cached
    
    # Rate limiting
    async with _info_request_lock:
        elapsed = time.time() - _last_info_request_time
        if elapsed < config.yf_info_delay:
            await asyncio.sleep(config.yf_info_delay - elapsed)
        _last_info_request_time = time.time()
    
    # Fetch from yfinance (in thread pool)
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(_executor, _fetch_info_sync, ticker)
    
    if info:
        # Cache in memory
        _info_cache[ticker] = (time.time(), info)
        
        # Cache in database
        await _save_info_to_db(ticker, info, config.info_cache_ttl)
    
    return info


async def compute_quality_score(
    ticker: str,
    info: Optional[Dict[str, Any]] = None,
    config: Optional[DipFinderConfig] = None,
) -> QualityMetrics:
    """
    Compute quality score for a stock.
    
    Args:
        ticker: Stock ticker symbol
        info: Pre-fetched yfinance info (fetches if None)
        config: Optional config
        
    Returns:
        QualityMetrics with score and contributing factors
    """
    if info is None:
        info = await fetch_stock_info(ticker, config)
    
    if not info:
        return QualityMetrics(
            ticker=ticker,
            score=50.0,  # Neutral if no data
            fields_available=0,
        )
    
    # Compute sub-scores
    prof_score, prof_factors = _compute_profitability_score(info)
    bs_score, bs_factors = _compute_balance_sheet_score(info)
    cash_score, cash_factors = _compute_cash_generation_score(info)
    growth_score, growth_factors = _compute_growth_score(info)
    liq_score, liq_factors = _compute_liquidity_score(info)
    
    # Weighted final score
    # Profitability and cash generation are most important for dip buying
    final_score = (
        prof_score * 0.25 +
        bs_score * 0.15 +
        cash_score * 0.25 +
        growth_score * 0.15 +
        liq_score * 0.20
    )
    
    # Count available fields
    all_factors = {**prof_factors, **bs_factors, **cash_factors, **growth_factors, **liq_factors}
    fields_available = sum(1 for v in all_factors.values() if v is not None)
    
    return QualityMetrics(
        ticker=ticker,
        score=final_score,
        profit_margin=prof_factors.get("profit_margin"),
        operating_margin=prof_factors.get("operating_margin"),
        debt_to_equity=bs_factors.get("debt_to_equity"),
        current_ratio=bs_factors.get("current_ratio"),
        free_cash_flow=cash_factors.get("free_cash_flow"),
        fcf_to_market_cap=cash_factors.get("fcf_to_market_cap"),
        revenue_growth=growth_factors.get("revenue_growth"),
        earnings_growth=growth_factors.get("earnings_growth"),
        market_cap=liq_factors.get("market_cap"),
        avg_volume=liq_factors.get("avg_volume"),
        profitability_score=prof_score,
        balance_sheet_score=bs_score,
        cash_generation_score=cash_score,
        growth_score=growth_score,
        liquidity_score=liq_score,
        fields_available=fields_available,
        fields_total=10,
    )


async def batch_fetch_info(
    tickers: list[str],
    config: Optional[DipFinderConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch info for multiple tickers with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        config: Optional config
        
    Returns:
        Dict mapping ticker -> info
    """
    if config is None:
        config = get_dipfinder_config()
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for ticker in tickers:
        info = await fetch_stock_info(ticker, config)
        results[ticker] = info
    
    return results


def clear_info_cache() -> None:
    """Clear the in-memory info cache."""
    global _info_cache
    _info_cache = {}
