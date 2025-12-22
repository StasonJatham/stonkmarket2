"""Suggestion stock info fetching service.

This module extracts yfinance-related blocking I/O from the suggestions routes
into a dedicated service for better separation of concerns.
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import yfinance as yf

from app.core.exceptions import ValidationError
from app.core.logging import get_logger

logger = get_logger("services.suggestion_stock_info")

# Thread pool for blocking yfinance calls (4 workers for better concurrency)
_executor = ThreadPoolExecutor(max_workers=4)

# Symbol validation pattern: 1-10 chars, alphanumeric + dot only
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.]{1,10}$')

# Rate limit error messages to detect
RATE_LIMIT_INDICATORS = ["rate limit", "too many requests", "429"]


def validate_symbol_format(symbol: str) -> str:
    """Validate and normalize symbol format.
    
    Args:
        symbol: Raw symbol input
        
    Returns:
        Normalized uppercase symbol
        
    Raises:
        ValidationError: If symbol format is invalid
    """
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValidationError(
            message="Symbol cannot be empty",
            details={"symbol": symbol}
        )
    if len(normalized) > 10:
        raise ValidationError(
            message="Symbol must be 10 characters or less",
            details={"symbol": symbol, "max_length": 10}
        )
    if not SYMBOL_PATTERN.match(normalized):
        raise ValidationError(
            message="Symbol can only contain letters, numbers, and dots",
            details={"symbol": symbol}
        )
    return normalized


def get_ipo_year(symbol: str) -> Optional[int]:
    """Get IPO/first trade year for a symbol from Yahoo Finance.
    
    This is a blocking function - use in executor or thread pool.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Year of IPO/first trade, or None if unavailable
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            first_trade_date = datetime.utcfromtimestamp(first_trade_ms / 1000)
            return first_trade_date.year
    except Exception as e:
        logger.debug(f"Failed to get IPO year for {symbol}: {e}")
    return None


def get_stock_info_basic(symbol: str) -> dict:
    """Get IPO year and website for a symbol from Yahoo Finance.
    
    This is a blocking function - use in executor or thread pool.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        dict with ipo_year and website keys
    """
    result = {"ipo_year": None, "website": None}
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Get IPO year
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            first_trade_date = datetime.utcfromtimestamp(first_trade_ms / 1000)
            result["ipo_year"] = first_trade_date.year
        
        # Get website
        result["website"] = info.get("website")
    except Exception as e:
        logger.debug(f"Failed to get stock info for {symbol}: {e}")
    return result


def get_stock_info_full(symbol: str) -> dict:
    """Get comprehensive stock info from Yahoo Finance for suggestions.
    
    This is a blocking function - use in executor or thread pool.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        dict with keys:
        - valid: bool - whether symbol is valid
        - name: str | None
        - sector: str | None
        - summary: str | None
        - website: str | None
        - ipo_year: int | None
        - current_price: float | None
        - ath_price: float | None (52-week high as proxy)
        - fetch_status: 'fetched' | 'rate_limited' | 'error' | 'invalid'
        - fetch_error: str | None
    """
    result = {
        "valid": False,
        "name": None,
        "sector": None,
        "summary": None,
        "website": None,
        "ipo_year": None,
        "current_price": None,
        "ath_price": None,
        "fetch_status": "error",
        "fetch_error": None,
    }
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Check if valid symbol (must have at least a name or price)
        name = info.get("shortName") or info.get("longName")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        
        if not name and not current_price:
            result["fetch_status"] = "invalid"
            result["fetch_error"] = "Symbol not found on Yahoo Finance"
            return result
        
        result["valid"] = True
        result["fetch_status"] = "fetched"
        result["name"] = name
        result["sector"] = info.get("sector")
        result["summary"] = info.get("longBusinessSummary")
        result["website"] = info.get("website")
        result["current_price"] = current_price
        result["ath_price"] = info.get("fiftyTwoWeekHigh")
        
        # Get IPO year
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            first_trade_date = datetime.utcfromtimestamp(first_trade_ms / 1000)
            result["ipo_year"] = first_trade_date.year
            
    except Exception as e:
        error_str = str(e).lower()
        # Check if rate limited
        if any(indicator in error_str for indicator in RATE_LIMIT_INDICATORS):
            result["fetch_status"] = "rate_limited"
            result["fetch_error"] = "Yahoo Finance rate limit reached. Will retry automatically."
            logger.warning(f"Rate limited fetching {symbol}: {e}")
        else:
            result["fetch_status"] = "error"
            result["fetch_error"] = str(e) if str(e) else "Failed to fetch stock info"
            logger.debug(f"Failed to get stock info for {symbol}: {e}")
    
    return result


async def get_stock_info_full_async(symbol: str) -> dict:
    """Async wrapper for get_stock_info_full.
    
    Runs the blocking yfinance call in a thread pool.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Stock info dict (see get_stock_info_full for keys)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_stock_info_full, symbol)


async def get_stock_info_basic_async(symbol: str) -> dict:
    """Async wrapper for get_stock_info_basic.
    
    Runs the blocking yfinance call in a thread pool.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        dict with ipo_year and website
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_stock_info_basic, symbol)


def is_rate_limited(error_message: str) -> bool:
    """Check if an error message indicates rate limiting.
    
    Args:
        error_message: Error message string
        
    Returns:
        True if this appears to be a rate limit error
    """
    error_lower = error_message.lower()
    return any(indicator in error_lower for indicator in RATE_LIMIT_INDICATORS)
