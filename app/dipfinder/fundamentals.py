"""Fundamentals module for quality scoring.

MIGRATED: Now uses unified YFinanceService for yfinance calls.
Computes quality score (0-100) from profitability, balance sheet,
cash generation, and growth metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.repositories import yfinance_cache_orm as yfinance_cache_repo
from app.services.data_providers import get_yfinance_service

from .config import DipFinderConfig, get_dipfinder_config


logger = get_logger("dipfinder.fundamentals")

# In-memory cache for yfinance info (kept for backward compat, but unified service has its own cache)
_info_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def normalize_debt_to_equity(de: float | None) -> float | None:
    """
    Normalize debt-to-equity to ratio form (0.5 = 50% debt).
    
    yfinance may return D/E as:
    - Ratio form: 0.5, 1.0, 2.0
    - Percentage form: 50, 100, 200
    
    This function detects and normalizes to ratio form.
    """
    if de is None:
        return None
    if de > 10:  # Likely percentage form (D/E > 1000% is extremely rare)
        return de / 100
    return de


@dataclass
class QualityMetrics:
    """Quality metrics and score for a stock."""

    ticker: str
    score: float  # 0-100

    # Contributing factors
    profit_margin: float | None = None
    operating_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    free_cash_flow: float | None = None
    fcf_to_market_cap: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    market_cap: float | None = None
    avg_volume: float | None = None

    # New factors from stored fundamentals
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    ev_to_ebitda: float | None = None  # NEW: Better valuation metric for mature companies
    return_on_equity: float | None = None
    return_on_assets: float | None = None  # NEW: Less distorted than ROE
    recommendation: str | None = None
    target_upside: float | None = None  # (target - current) / current
    short_percent_of_float: float | None = None  # NEW: Short interest risk indicator
    institutional_ownership: float | None = None  # NEW: Smart money confidence

    # Sub-scores (0-100)
    profitability_score: float = 50.0
    balance_sheet_score: float = 50.0
    cash_generation_score: float = 50.0
    growth_score: float = 50.0
    liquidity_score: float = 50.0
    valuation_score: float = 50.0  # NEW: P/E, PEG, Forward P/E, EV/EBITDA
    analyst_score: float = 50.0     # NEW: Recommendations, target price
    risk_score: float = 50.0        # NEW: Short interest, institutional ownership

    # Data quality
    fields_available: int = 0
    fields_total: int = 16  # Updated to include new fields

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
            "pe_ratio": self.pe_ratio,
            "forward_pe": self.forward_pe,
            "peg_ratio": self.peg_ratio,
            "ev_to_ebitda": self.ev_to_ebitda,
            "return_on_equity": self.return_on_equity,
            "return_on_assets": self.return_on_assets,
            "recommendation": self.recommendation,
            "target_upside": round(self.target_upside, 4) if self.target_upside else None,
            "short_percent_of_float": self.short_percent_of_float,
            "institutional_ownership": self.institutional_ownership,
            "profitability_score": round(self.profitability_score, 2),
            "balance_sheet_score": round(self.balance_sheet_score, 2),
            "cash_generation_score": round(self.cash_generation_score, 2),
            "growth_score": round(self.growth_score, 2),
            "liquidity_score": round(self.liquidity_score, 2),
            "valuation_score": round(self.valuation_score, 2),
            "analyst_score": round(self.analyst_score, 2),
            "risk_score": round(self.risk_score, 2),
            "fields_available": self.fields_available,
            "fields_total": self.fields_total,
        }

    def to_pydantic(self) -> "QualityMetricsPydantic":
        """Convert to Pydantic domain model for type-safe API responses.
        
        Returns:
            app.domain.QualityMetrics Pydantic model instance
        """
        from app.domain import QualityMetrics as QualityMetricsPydantic
        return QualityMetricsPydantic.model_validate(self.to_dict())


def _safe_float(value: Any, default: float | None = None) -> float | None:
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
    value: float | None,
    optimal: float,
    good_range: tuple[float, float],
    bad_threshold: float | None = None,
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


def _compute_profitability_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
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


def _compute_balance_sheet_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute balance sheet sub-score."""
    raw_de = _safe_float(info.get("debtToEquity"))
    debt_to_equity = normalize_debt_to_equity(raw_de)
    current_ratio = _safe_float(info.get("currentRatio"))

    factors = {
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
    }

    # Score debt to equity (lower is better)
    # D/E is normalized to ratio form: 0.5 = 50%, 1.0 = 100%, 2.0 = 200%
    # Scoring: monotonically decreasing - higher D/E = lower score
    if debt_to_equity is not None:
        if debt_to_equity <= 0.2:
            de_score = 95.0  # Excellent: very low leverage
        elif debt_to_equity <= 0.5:
            # Good range: 0.2-0.5 → score 75-95
            de_score = 95.0 - (debt_to_equity - 0.2) * (20 / 0.3)
        elif debt_to_equity <= 0.8:
            # Acceptable: 0.5-0.8 → score 60-75
            de_score = 75.0 - (debt_to_equity - 0.5) * (15 / 0.3)
        elif debt_to_equity <= 1.5:
            # Elevated: 0.8-1.5 → score 40-60
            de_score = 60.0 - (debt_to_equity - 0.8) * (20 / 0.7)
        elif debt_to_equity <= 2.5:
            # High: 1.5-2.5 → score 20-40
            de_score = 40.0 - (debt_to_equity - 1.5) * (20 / 1.0)
        else:
            # Very high: >2.5 → score 20 or less
            de_score = max(10.0, 20.0 - (debt_to_equity - 2.5) * 5)
    else:
        de_score = 50.0  # Neutral for missing data

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


def _compute_cash_generation_score(
    info: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
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


def _compute_growth_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute growth sub-score."""
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))

    factors = {
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
    }

    # Score revenue growth (10-20% good)
    rg_score = (
        _normalize_score(
            revenue_growth,
            optimal=0.15,
            good_range=(0.05, 0.30),
            bad_threshold=-0.10,
        )
        if revenue_growth is not None
        else 50.0
    )

    # Score earnings growth (15-25% good)
    eg_score = (
        _normalize_score(
            earnings_growth,
            optimal=0.20,
            good_range=(0.05, 0.40),
            bad_threshold=-0.20,
        )
        if earnings_growth is not None
        else 50.0
    )

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


def _compute_liquidity_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute liquidity/size sub-score."""
    market_cap = _safe_float(info.get("marketCap"))
    avg_volume = _safe_float(info.get("averageVolume")) or _safe_float(
        info.get("averageVolume10days")
    )

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


def _compute_valuation_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Compute valuation sub-score using P/E, Forward P/E, and PEG ratios.
    
    For dip buying, we want reasonably valued stocks, not overvalued ones.
    Lower P/E generally indicates better value, but very low can signal trouble.
    """
    pe_ratio = _safe_float(info.get("trailingPE") or info.get("pe_ratio"))
    forward_pe = _safe_float(info.get("forwardPE") or info.get("forward_pe"))
    peg_ratio = _safe_float(info.get("trailingPegRatio") or info.get("peg_ratio"))

    ev_ebitda = _safe_float(info.get("enterpriseToEbitda") or info.get("ev_to_ebitda"))

    factors = {
        "pe_ratio": pe_ratio,
        "forward_pe": forward_pe,
        "peg_ratio": peg_ratio,
        "ev_to_ebitda": ev_ebitda,
    }

    scores = []

    # P/E Score (10-20 is ideal, <5 might be troubled, >40 is expensive)
    if pe_ratio is not None and pe_ratio > 0:
        if pe_ratio < 5:
            pe_score = 40.0  # Very low, might be value trap
        elif pe_ratio < 10:
            pe_score = 70.0 + (pe_ratio - 5) * 4  # 70-90
        elif pe_ratio <= 20:
            pe_score = 90.0  # Sweet spot
        elif pe_ratio <= 30:
            pe_score = 90.0 - (pe_ratio - 20) * 2  # 90-70
        elif pe_ratio <= 50:
            pe_score = 70.0 - (pe_ratio - 30) * 1.5  # 70-40
        else:
            pe_score = max(20.0, 40.0 - (pe_ratio - 50) * 0.5)
        scores.append(pe_score)

    # Forward P/E Score (lower is better, indicates expected growth)
    if forward_pe is not None and forward_pe > 0:
        if forward_pe < pe_ratio if pe_ratio else True:
            # Forward P/E lower than trailing = expected growth
            if forward_pe < 15:
                fpe_score = 90.0
            elif forward_pe < 25:
                fpe_score = 75.0
            else:
                fpe_score = 55.0
        else:
            # Forward P/E higher = expected slowdown
            fpe_score = 45.0
        scores.append(fpe_score)

    # PEG Score (1.0 = fair value, <1 = undervalued, >2 = overvalued)
    if peg_ratio is not None and peg_ratio > 0:
        if peg_ratio < 0.5:
            peg_score = 95.0  # Very undervalued
        elif peg_ratio <= 1.0:
            peg_score = 80.0 + (1.0 - peg_ratio) * 30  # 80-95
        elif peg_ratio <= 1.5:
            peg_score = 60.0 + (1.5 - peg_ratio) * 40  # 60-80
        elif peg_ratio <= 2.5:
            peg_score = 40.0 + (2.5 - peg_ratio) * 20  # 40-60
        else:
            peg_score = max(20.0, 40.0 - (peg_ratio - 2.5) * 10)
        scores.append(peg_score)

    # EV/EBITDA Score (8-15 is reasonable, <8 = cheap, >25 = expensive)
    # Better metric for mature companies than P/E (includes debt, ignores accounting)
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 6:
            ev_score = 95.0  # Very cheap
        elif ev_ebitda < 10:
            ev_score = 85.0  # Cheap
        elif ev_ebitda < 15:
            ev_score = 75.0  # Reasonable
        elif ev_ebitda < 20:
            ev_score = 60.0  # Getting expensive
        elif ev_ebitda < 30:
            ev_score = 45.0  # Expensive
        else:
            ev_score = max(20.0, 45.0 - (ev_ebitda - 30) * 0.5)
        scores.append(ev_score)

    score = sum(scores) / len(scores) if scores else 50.0

    return score, factors


def _compute_analyst_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Compute analyst sentiment sub-score from recommendations and target prices.
    
    Strong buy/buy ratings with upside to target = high score
    """
    recommendation = info.get("recommendationKey") or info.get("recommendation")
    target_price = _safe_float(info.get("targetMeanPrice") or info.get("target_mean_price"))
    current_price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice") or info.get("current_price"))
    num_analysts = _safe_float(info.get("numberOfAnalystOpinions") or info.get("num_analyst_opinions"))

    target_upside = None
    if target_price and current_price and current_price > 0:
        target_upside = (target_price - current_price) / current_price

    factors = {
        "recommendation": recommendation,
        "target_upside": target_upside,
        "num_analysts": num_analysts,
    }

    scores = []
    weights = []

    # Recommendation score (weight more heavily)
    if recommendation:
        rec_lower = recommendation.lower()
        if rec_lower in ("strong_buy", "strongbuy"):
            rec_score = 95.0
        elif rec_lower == "buy":
            rec_score = 80.0
        elif rec_lower == "hold":
            rec_score = 55.0
        elif rec_lower in ("underperform", "sell"):
            rec_score = 30.0
        elif rec_lower in ("strong_sell", "strongsell"):
            rec_score = 15.0
        else:
            rec_score = 50.0
        scores.append(rec_score)
        weights.append(0.50)

    # Target upside score
    if target_upside is not None:
        if target_upside > 0.50:  # >50% upside
            upside_score = 95.0
        elif target_upside > 0.25:  # >25% upside
            upside_score = 85.0
        elif target_upside > 0.10:  # >10% upside
            upside_score = 70.0
        elif target_upside > 0:  # Any upside
            upside_score = 55.0
        elif target_upside > -0.10:  # Up to 10% downside
            upside_score = 40.0
        else:  # >10% downside to target
            upside_score = 25.0
        scores.append(upside_score)
        weights.append(0.35)

    # Analyst coverage score (more analysts = more reliable)
    if num_analysts is not None:
        if num_analysts >= 20:
            coverage_score = 80.0
        elif num_analysts >= 10:
            coverage_score = 70.0
        elif num_analysts >= 5:
            coverage_score = 55.0
        elif num_analysts >= 1:
            coverage_score = 40.0
        else:
            coverage_score = 30.0
        scores.append(coverage_score)
        weights.append(0.15)

    if not scores:
        return 50.0, factors

    # Weighted average
    total_weight = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / total_weight

    return score, factors


def _compute_risk_score(info: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Compute risk sub-score from short interest and institutional ownership.
    
    High short interest = bearish sentiment / risk
    High institutional ownership = smart money confidence / stability
    """
    short_pct = _safe_float(info.get("shortPercentOfFloat") or info.get("short_percent_of_float"))
    inst_pct = _safe_float(info.get("heldPercentInstitutions") or info.get("held_percent_institutions"))

    factors = {
        "short_percent_of_float": short_pct,
        "institutional_ownership": inst_pct,
    }

    scores = []
    weights = []

    # Short interest score (lower is better - inverse scoring)
    # High short interest can mean: bearish sentiment, potential squeeze, or troubled company
    if short_pct is not None:
        if short_pct < 0.02:  # <2% = minimal short interest
            short_score = 90.0
        elif short_pct < 0.05:  # <5% = low short interest
            short_score = 80.0
        elif short_pct < 0.10:  # <10% = moderate short interest
            short_score = 65.0
        elif short_pct < 0.20:  # <20% = elevated short interest
            short_score = 45.0
        elif short_pct < 0.30:  # <30% = high short interest (squeeze candidate)
            short_score = 35.0
        else:  # >30% = very high short interest (significant bearish bets)
            short_score = 25.0
        scores.append(short_score)
        weights.append(0.40)

    # Institutional ownership score (higher is better)
    # High institutional ownership = smart money confidence
    if inst_pct is not None:
        if inst_pct > 0.80:  # >80% = very high institutional confidence
            inst_score = 90.0
        elif inst_pct > 0.60:  # >60% = solid institutional backing
            inst_score = 80.0
        elif inst_pct > 0.40:  # >40% = moderate institutional interest
            inst_score = 65.0
        elif inst_pct > 0.20:  # >20% = some institutional presence
            inst_score = 50.0
        else:  # <20% = retail-dominated, potentially more volatile
            inst_score = 35.0
        scores.append(inst_score)
        weights.append(0.60)

    if not scores:
        return 50.0, factors

    # Weighted average
    total_weight = sum(weights)
    score = sum(s * w for s, w in zip(scores, weights)) / total_weight

    return score, factors


async def _fetch_info_async(ticker: str) -> dict[str, Any]:
    """Fetch yfinance info via unified service."""
    service = get_yfinance_service()
    info = await service.get_ticker_info(ticker)
    return info or {}


async def _get_cached_info_from_db(ticker: str) -> dict[str, Any] | None:
    """Get cached info from database if not expired."""
    try:
        return await yfinance_cache_repo.get_cached_info(ticker)
    except Exception as e:
        logger.debug(f"Could not load cached info for {ticker}: {e}")
        return None


async def _save_info_to_db(ticker: str, info: dict[str, Any], ttl_seconds: int) -> None:
    """Save info to database cache."""
    try:
        await yfinance_cache_repo.save_info(ticker, info, ttl_seconds)
    except Exception as e:
        logger.debug(f"Could not save info cache for {ticker}: {e}")


async def fetch_stock_info(
    ticker: str,
    config: DipFinderConfig | None = None,
) -> dict[str, Any]:
    """
    Fetch stock info with caching and rate limiting.

    Args:
        ticker: Stock ticker symbol
        config: Optional config for TTL settings

    Returns:
        yfinance info dictionary
    """
    if config is None:
        config = get_dipfinder_config()

    ticker = ticker.upper()

    # Check in-memory cache first
    now = time.time()
    cached = _info_cache.get(ticker)
    if cached and now - cached[0] < config.info_cache_ttl:
        return cached[1]

    # Use unified YFinanceService (handles caching, rate limiting, and DB persistence)
    info = await _fetch_info_async(ticker)

    if info:
        # Cache in local memory for this module
        _info_cache[ticker] = (time.time(), info)

    return info


async def compute_quality_score(
    ticker: str,
    info: dict[str, Any] | None = None,
    config: DipFinderConfig | None = None,
    fundamentals: dict[str, Any] | None = None,
) -> QualityMetrics:
    """
    Compute quality score for a stock.

    If domain-specific scoring is enabled (default), uses specialized adapters
    for banks, REITs, ETFs, etc. Otherwise falls back to generic scoring.

    Args:
        ticker: Stock ticker symbol
        info: Pre-fetched yfinance info (fetches if None)
        config: Optional config
        fundamentals: Pre-fetched fundamentals from stock_fundamentals table

    Returns:
        QualityMetrics with score and contributing factors
    """
    if config is None:
        config = get_dipfinder_config()

    if info is None:
        info = await fetch_stock_info(ticker, config)

    if not info:
        info = {}

    # Merge stored fundamentals into info for scoring (stored fundamentals take precedence)
    if fundamentals:
        merged_info = {**info, **fundamentals}
    else:
        merged_info = info

    if not merged_info:
        return QualityMetrics(
            ticker=ticker,
            score=50.0,  # Neutral if no data
            fields_available=0,
        )

    # Domain-specific scoring path
    if config.domain_scoring_enabled:
        return await _compute_domain_quality_score(ticker, merged_info, config)

    # Legacy generic scoring path
    return await _compute_generic_quality_score(ticker, merged_info)


async def _compute_generic_quality_score(
    ticker: str,
    merged_info: dict[str, Any],
) -> QualityMetrics:
    """Compute quality score using legacy generic scoring (all securities same weights)."""
    # Compute sub-scores
    prof_score, prof_factors = _compute_profitability_score(merged_info)
    bs_score, bs_factors = _compute_balance_sheet_score(merged_info)
    cash_score, cash_factors = _compute_cash_generation_score(merged_info)
    growth_score, growth_factors = _compute_growth_score(merged_info)
    liq_score, liq_factors = _compute_liquidity_score(merged_info)
    val_score, val_factors = _compute_valuation_score(merged_info)
    analyst_score, analyst_factors = _compute_analyst_score(merged_info)
    risk_score, risk_factors = _compute_risk_score(merged_info)

    # Weighted final score - adjusted weights based on professional review
    # For dip buying: FCF is king, valuation matters, risk indicators important
    # Reduced analyst weight (lagging indicator), added risk score
    final_score = (
        prof_score * 0.20       # Profitability
        + bs_score * 0.10       # Balance sheet
        + cash_score * 0.20     # Cash generation (FCF is key for dip buying)
        + growth_score * 0.10   # Growth
        + liq_score * 0.05      # Liquidity (less important for large caps)
        + val_score * 0.15      # Valuation (P/E, PEG, EV/EBITDA)
        + analyst_score * 0.10  # Analyst consensus (reduced - lagging indicator)
        + risk_score * 0.10     # NEW: Short interest + institutional ownership
    )

    # Count available fields
    all_factors = {
        **prof_factors,
        **bs_factors,
        **cash_factors,
        **growth_factors,
        **liq_factors,
        **val_factors,
        **analyst_factors,
        **risk_factors,
    }
    fields_available = sum(1 for v in all_factors.values() if v is not None)

    # Calculate target upside for the return object
    target_upside = analyst_factors.get("target_upside")

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
        pe_ratio=val_factors.get("pe_ratio"),
        forward_pe=val_factors.get("forward_pe"),
        peg_ratio=val_factors.get("peg_ratio"),
        ev_to_ebitda=val_factors.get("ev_to_ebitda"),
        return_on_equity=_safe_float(merged_info.get("returnOnEquity") or merged_info.get("return_on_equity")),
        return_on_assets=_safe_float(merged_info.get("returnOnAssets") or merged_info.get("return_on_assets")),
        recommendation=analyst_factors.get("recommendation"),
        target_upside=target_upside,
        short_percent_of_float=risk_factors.get("short_percent_of_float"),
        institutional_ownership=risk_factors.get("institutional_ownership"),
        profitability_score=prof_score,
        balance_sheet_score=bs_score,
        cash_generation_score=cash_score,
        growth_score=growth_score,
        liquidity_score=liq_score,
        valuation_score=val_score,
        analyst_score=analyst_score,
        risk_score=risk_score,
        fields_available=fields_available,
        fields_total=16,
    )


async def _compute_domain_quality_score(
    ticker: str,
    merged_info: dict[str, Any],
    config: DipFinderConfig,
) -> QualityMetrics:
    """Compute quality score using domain-specific scoring adapters."""
    from app.dipfinder.domain import get_domain_from_info
    from app.dipfinder.domain_scoring import compute_domain_score

    # Classify the security's domain
    classification = get_domain_from_info(merged_info)

    if config.domain_scoring_log_enabled:
        logger.info(f"Domain classification for {ticker}: {classification}")

    # Compute domain-specific score
    domain_result = compute_domain_score(classification, merged_info)

    if config.domain_scoring_log_enabled:
        logger.info(
            f"Domain score for {ticker}: {domain_result.final_score:.1f} "
            f"(domain={domain_result.domain.value}, confidence={domain_result.domain_confidence:.0%}, "
            f"data_completeness={domain_result.data_completeness:.0%})"
        )

    # Map domain result to QualityMetrics
    # We still extract the standard factors for compatibility with existing code
    prof_score, prof_factors = _compute_profitability_score(merged_info)
    bs_score, bs_factors = _compute_balance_sheet_score(merged_info)
    cash_score, cash_factors = _compute_cash_generation_score(merged_info)
    growth_score, growth_factors = _compute_growth_score(merged_info)
    liq_score, liq_factors = _compute_liquidity_score(merged_info)
    val_score, val_factors = _compute_valuation_score(merged_info)
    analyst_score_val, analyst_factors = _compute_analyst_score(merged_info)
    risk_score_val, risk_factors = _compute_risk_score(merged_info)

    # Count available fields
    all_factors = {
        **prof_factors,
        **bs_factors,
        **cash_factors,
        **growth_factors,
        **liq_factors,
        **val_factors,
        **analyst_factors,
        **risk_factors,
    }
    fields_available = sum(1 for v in all_factors.values() if v is not None)
    target_upside = analyst_factors.get("target_upside")

    # Use domain-specific final score but keep legacy sub-scores for backward compat
    return QualityMetrics(
        ticker=ticker,
        score=domain_result.final_score,  # Domain-specific score!
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
        pe_ratio=val_factors.get("pe_ratio"),
        forward_pe=val_factors.get("forward_pe"),
        peg_ratio=val_factors.get("peg_ratio"),
        ev_to_ebitda=val_factors.get("ev_to_ebitda"),
        return_on_equity=_safe_float(merged_info.get("returnOnEquity") or merged_info.get("return_on_equity")),
        return_on_assets=_safe_float(merged_info.get("returnOnAssets") or merged_info.get("return_on_assets")),
        recommendation=analyst_factors.get("recommendation"),
        target_upside=target_upside,
        short_percent_of_float=risk_factors.get("short_percent_of_float"),
        institutional_ownership=risk_factors.get("institutional_ownership"),
        # Sub-scores are domain-specific where applicable
        profitability_score=prof_score,
        balance_sheet_score=bs_score,
        cash_generation_score=cash_score,
        growth_score=growth_score,
        liquidity_score=liq_score,
        valuation_score=val_score,
        analyst_score=analyst_score_val,
        risk_score=risk_score_val,
        fields_available=fields_available,
        fields_total=16,
    )


async def batch_fetch_info(
    tickers: list[str],
    config: DipFinderConfig | None = None,
) -> dict[str, dict[str, Any]]:
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

    results: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        info = await fetch_stock_info(ticker, config)
        results[ticker] = info

    return results


def clear_info_cache() -> None:
    """Clear the in-memory info cache."""
    global _info_cache
    _info_cache = {}
