"""Stock info service with caching."""

from __future__ import annotations

import time
from typing import Dict, Optional

import yfinance as yf

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.dips import StockInfo

logger = get_logger("services.stock_info")

_INFO_CACHE: Dict[str, tuple[float, StockInfo]] = {}
_CACHE_TTL = 300  # 5 minutes


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


def clear_info_cache() -> None:
    """Clear the stock info cache."""
    global _INFO_CACHE
    _INFO_CACHE = {}
