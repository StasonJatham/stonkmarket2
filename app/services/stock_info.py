"""Stock info service with caching and rate limiting."""

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

_INFO_CACHE: Dict[str, tuple[float, StockInfo]] = {}
_PRICE_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 300  # 5 minutes
_executor = ThreadPoolExecutor(max_workers=4)


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

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        stock_info = StockInfo(
            symbol=symbol,
            name=info.get("shortName") or info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            dividend_yield=info.get("dividendYield"),
            beta=info.get("beta"),
            avg_volume=info.get("averageVolume"),
            summary=info.get("longBusinessSummary"),
            website=info.get("website"),
            recommendation=info.get("recommendationKey"),
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
        
        result = {
            "symbol": symbol,
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "current_price": float(current_price) if current_price else 0,
            "previous_close": float(previous_close) if previous_close else None,
            "change_percent": round(change_percent, 4) if change_percent is not None else None,
            "ath_price": float(ath_price) if ath_price else 0,
            "fifty_two_week_high": float(ath_price) if ath_price else 0,
            "fifty_two_week_low": float(info.get("fiftyTwoWeekLow", 0)),
            "pe_ratio": info.get("trailingPE"),
            "avg_volume": info.get("averageVolume"),
            "summary": info.get("longBusinessSummary"),
            "recommendation": info.get("recommendationKey"),
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
