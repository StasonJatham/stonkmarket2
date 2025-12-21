"""Stock info service with caching."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any

import yfinance as yf

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.dips import StockInfo

logger = get_logger("services.stock_info")

_INFO_CACHE: Dict[str, tuple[float, StockInfo]] = {}
_CACHE_TTL = 300  # 5 minutes
_executor = ThreadPoolExecutor(max_workers=4)


def get_stock_info(symbol: str) -> Optional[StockInfo]:
    """Fetch detailed stock info from Yahoo Finance with caching."""
    now = time.time()
    cached = _INFO_CACHE.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

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


async def get_stock_info_async(symbol: str) -> Optional[Dict[str, Any]]:
    """Async wrapper for get_stock_info that returns a dict."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, get_stock_info, symbol)
    
    if result:
        return {
            "symbol": result.symbol,
            "name": result.name,
            "sector": result.sector,
            "industry": result.industry,
            "market_cap": result.market_cap,
            "pe_ratio": result.pe_ratio,
            "forward_pe": result.forward_pe,
            "dividend_yield": result.dividend_yield,
            "beta": result.beta,
            "avg_volume": result.avg_volume,
            "summary": result.summary,
            "website": result.website,
            "recommendation": result.recommendation,
        }
    return None


def clear_info_cache() -> None:
    """Clear the stock info cache."""
    global _INFO_CACHE
    _INFO_CACHE = {}
