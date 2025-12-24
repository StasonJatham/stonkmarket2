"""Stock info service with caching and rate limiting.

NOTE: This module uses synchronous functions because yfinance is blocking.
These functions are designed to be called from ThreadPoolExecutor.
For Valkey caching, we use the sync cache wrapper.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any

import yfinance as yf

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.schemas.dips import StockInfo

logger = get_logger("services.stock_info")

# Process-local cache with short TTL (5 min) for reducing yfinance calls.
# This is acceptable because:
# 1. Stock info changes infrequently
# 2. The primary cache is in Valkey (see dips.py routes)
# 3. These functions run in sync thread pool, can't easily use async Valkey
_INFO_CACHE: Dict[str, tuple[float, StockInfo]] = {}
_PRICE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 300  # 5 minutes
_executor = ThreadPoolExecutor(max_workers=4)

# Known ETF symbols that should skip fundamentals
_KNOWN_ETFS = {"SPY", "QQQ", "IWM", "DIA", "URTH", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM"}


def is_index_or_etf(symbol: str) -> bool:
    """Check if a symbol is an index (^) or known ETF."""
    return symbol.startswith("^") or symbol.upper() in _KNOWN_ETFS


def get_stock_info(symbol: str) -> Optional[StockInfo]:
    """Fetch detailed stock info from Yahoo Finance with caching and rate limiting."""
    now = time.time()
    cached = _INFO_CACHE.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    # Acquire rate limit token
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for {symbol}")
        return None

    is_etf_or_index = is_index_or_etf(symbol)

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # yfinance dividendYield is in percentage format (e.g., 0.17 = 0.17%)
        # We store as decimal (0.0017) so frontend can multiply by 100
        raw_div_yield = info.get("dividendYield")
        dividend_yield = raw_div_yield / 100 if raw_div_yield else None

        stock_info = StockInfo(
            symbol=symbol,
            name=info.get("shortName") or info.get("longName"),
            # Skip fundamentals for ETFs/indexes
            sector=None if is_etf_or_index else info.get("sector"),
            industry=None if is_etf_or_index else info.get("industry"),
            market_cap=info.get("totalAssets") if is_etf_or_index else info.get("marketCap"),
            pe_ratio=None if is_etf_or_index else info.get("trailingPE"),
            forward_pe=None if is_etf_or_index else info.get("forwardPE"),
            peg_ratio=None if is_etf_or_index else info.get("trailingPegRatio"),
            dividend_yield=dividend_yield,  # ETFs can have dividend yield
            beta=info.get("beta"),
            avg_volume=info.get("averageVolume"),
            summary=info.get("longBusinessSummary"),
            website=info.get("website"),
            recommendation=None if is_etf_or_index else info.get("recommendationKey"),
            # Extended fundamentals
            profit_margin=None if is_etf_or_index else info.get("profitMargins"),
            gross_margin=None if is_etf_or_index else info.get("grossMargins"),
            return_on_equity=None if is_etf_or_index else info.get("returnOnEquity"),
            debt_to_equity=None if is_etf_or_index else info.get("debtToEquity"),
            current_ratio=None if is_etf_or_index else info.get("currentRatio"),
            revenue_growth=None if is_etf_or_index else info.get("revenueGrowth"),
            free_cash_flow=None if is_etf_or_index else info.get("freeCashflow"),
            target_mean_price=None if is_etf_or_index else info.get("targetMeanPrice"),
            num_analyst_opinions=None if is_etf_or_index else info.get("numberOfAnalystOpinions"),
        )
        _INFO_CACHE[symbol] = (now, stock_info)
        return stock_info
    except Exception as e:
        logger.warning(f"Failed to get stock info for {symbol}: {e}")
        return None


def _get_stock_info_with_prices(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch stock info including current price and ATH from Yahoo Finance."""
    # Check cache first
    now = time.time()
    cached = _PRICE_CACHE.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]
    
    # Acquire rate limit token
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for {symbol}")
        return None
    
    is_etf_or_index = is_index_or_etf(symbol)
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Get current price and previous close for change calculation
        current_price = info.get("regularMarketPrice") or info.get("previousClose") or 0
        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or 0
        
        # Calculate change percent
        change_percent = None
        if current_price and previous_close and previous_close > 0:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        
        # Get 52-week high as proxy for ATH
        ath_price = info.get("fiftyTwoWeekHigh") or 0
        
        # Get IPO year from first trade date
        ipo_year = None
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            from datetime import datetime
            ipo_year = datetime.utcfromtimestamp(first_trade_ms / 1000).year
        
        result = {
            "symbol": symbol,
            "name": info.get("shortName") or info.get("longName"),
            # Skip fundamentals for ETFs/indexes - they don't have meaningful values
            "sector": None if is_etf_or_index else info.get("sector"),
            "industry": None if is_etf_or_index else info.get("industry"),
            "market_cap": info.get("totalAssets") if is_etf_or_index else info.get("marketCap"),  # ETFs use totalAssets
            "current_price": float(current_price) if current_price else 0,
            "previous_close": float(previous_close) if previous_close else None,
            "change_percent": round(change_percent, 4) if change_percent is not None else None,
            "ath_price": float(ath_price) if ath_price else 0,
            "fifty_two_week_high": float(ath_price) if ath_price else 0,
            "fifty_two_week_low": float(info.get("fiftyTwoWeekLow", 0)),
            # Skip P/E and recommendation for ETFs/indexes
            "pe_ratio": None if is_etf_or_index else info.get("trailingPE"),
            "avg_volume": info.get("averageVolume"),
            "summary": info.get("longBusinessSummary"),
            "website": info.get("website"),  # For logo URLs
            "ipo_year": ipo_year,
            "recommendation": None if is_etf_or_index else info.get("recommendationKey"),
            "is_etf_or_index": is_etf_or_index,  # Flag for frontend
        }
        _PRICE_CACHE[symbol] = (now, result)
        return result
    except Exception as e:
        logger.warning(f"Failed to get stock info with prices for {symbol}: {e}")
        return None


async def get_stock_info_async(symbol: str) -> Optional[Dict[str, Any]]:
    """Async wrapper that returns stock info with prices."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _get_stock_info_with_prices, symbol)
    return result


def clear_info_cache() -> None:
    """Clear the stock info cache."""
    global _INFO_CACHE, _PRICE_CACHE
    _INFO_CACHE = {}
    _PRICE_CACHE = {}
