"""Suggestion stock info fetching service.

This module provides stock info specifically for the suggestions feature.
Uses the unified yfinance service for all API calls.
"""

from __future__ import annotations

import re
from typing import Optional

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.services.data_providers import get_yfinance_service

logger = get_logger("services.suggestion_stock_info")

# Get singleton service instance
_yf_service = get_yfinance_service()

# Symbol validation pattern: 1-10 chars, alphanumeric + dot only
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.]{1,10}$')


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


async def get_ipo_year(symbol: str) -> Optional[int]:
    """Get IPO/first trade year for a symbol from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Year of IPO/first trade, or None if unavailable
    """
    info = await _yf_service.get_ticker_info(symbol)
    return info.get("ipo_year") if info else None


async def get_stock_info_basic(symbol: str) -> dict:
    """Get IPO year and website for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        dict with ipo_year and website keys
    """
    info = await _yf_service.get_ticker_info(symbol)
    if not info:
        return {"ipo_year": None, "website": None}
    
    return {
        "ipo_year": info.get("ipo_year"),
        "website": info.get("website"),
    }


async def get_stock_info_full(symbol: str) -> dict:
    """Get comprehensive stock info for suggestions.
    
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
    
    info = await _yf_service.get_ticker_info(symbol)
    
    if not info:
        result["fetch_status"] = "invalid"
        result["fetch_error"] = "Symbol not found on Yahoo Finance"
        return result
    
    # Check if valid symbol (must have at least a name or price)
    name = info.get("name")
    current_price = info.get("current_price")
    
    if not name and not current_price:
        result["fetch_status"] = "invalid"
        result["fetch_error"] = "Symbol not found on Yahoo Finance"
        return result
    
    result["valid"] = True
    result["fetch_status"] = "fetched"
    result["name"] = name
    result["sector"] = info.get("sector")
    result["summary"] = info.get("summary")
    result["website"] = info.get("website")
    result["current_price"] = current_price
    result["ath_price"] = info.get("fifty_two_week_high")
    result["ipo_year"] = info.get("ipo_year")
    
    return result


# Async aliases for backwards compatibility
async def get_stock_info_full_async(symbol: str) -> dict:
    """Async wrapper for get_stock_info_full."""
    return await get_stock_info_full(symbol)


async def get_stock_info_basic_async(symbol: str) -> dict:
    """Async wrapper for get_stock_info_basic."""
    return await get_stock_info_basic(symbol)
