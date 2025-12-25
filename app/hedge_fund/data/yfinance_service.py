"""
yfinance data service for fetching market data.

Provides a clean interface to yfinance with caching and error handling
for the hedge_fund analysis module. Returns typed Pydantic models.

Uses the global rate limiter to avoid hitting yfinance API limits.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Optional

import yfinance as yf

from app.core.rate_limiter import get_yfinance_limiter
from app.hedge_fund.schemas import (
    CalendarEvents,
    Fundamentals,
    MarketData,
    PricePoint,
    PriceSeries,
)

logger = logging.getLogger(__name__)

# Thread pool for yfinance calls (yfinance is not async)
_executor = ThreadPoolExecutor(max_workers=10)

# TTL cache for ticker info (5 minutes)
_INFO_CACHE: dict[str, tuple[dict, float]] = {}
_INFO_CACHE_TTL = 300  # seconds

# TTL cache for price data (1 minute)
_PRICE_CACHE: dict[str, tuple[PriceSeries, float]] = {}
_PRICE_CACHE_TTL = 60  # seconds


# =============================================================================
# Low-level yfinance helpers
# =============================================================================


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN check
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _get_ticker_info_sync(symbol: str) -> dict[str, Any]:
    """Get ticker info synchronously (for thread pool)."""
    now = datetime.now().timestamp()
    
    # Check cache
    if symbol in _INFO_CACHE:
        cached, cached_at = _INFO_CACHE[symbol]
        if now - cached_at < _INFO_CACHE_TTL:
            return cached
    
    # Rate limit before hitting yfinance
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for {symbol}")
        return {}
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Cache the result
        _INFO_CACHE[symbol] = (info, now)
        return info
    except Exception as e:
        logger.warning(f"Failed to fetch info for {symbol}: {e}")
        return {}


def _get_price_history_sync(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> list[dict]:
    """Get price history synchronously (for thread pool)."""
    # Rate limit before hitting yfinance
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for price history {symbol}")
        return []
    
    try:
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)
        
        if df.empty:
            return []
        
        prices = []
        for idx, row in df.iterrows():
            prices.append({
                "date": idx.date() if hasattr(idx, "date") else idx,
                "open": _safe_float(row.get("Open"), 0.0),
                "high": _safe_float(row.get("High"), 0.0),
                "low": _safe_float(row.get("Low"), 0.0),
                "close": _safe_float(row.get("Close"), 0.0),
                "volume": _safe_int(row.get("Volume"), 0),
                "adj_close": _safe_float(row.get("Adj Close")),
            })
        
        return prices
    except Exception as e:
        logger.warning(f"Failed to fetch price history for {symbol}: {e}")
        return []


def _get_calendar_sync(symbol: str) -> dict[str, Any]:
    """Get calendar events synchronously."""
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        
        if calendar is None:
            return {}
        
        # yfinance returns either a DataFrame or dict
        if hasattr(calendar, "to_dict"):
            return calendar.to_dict()
        return calendar if isinstance(calendar, dict) else {}
    except Exception as e:
        logger.debug(f"No calendar data for {symbol}: {e}")
        return {}


# =============================================================================
# Async wrappers
# =============================================================================


async def get_ticker_info(symbol: str) -> dict[str, Any]:
    """Get ticker info asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _get_ticker_info_sync, symbol)


async def get_price_history(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> PriceSeries:
    """Get price history as PriceSeries."""
    now = datetime.now().timestamp()
    cache_key = f"{symbol}:{start_date}:{end_date}:{period}"
    
    # Check cache
    if cache_key in _PRICE_CACHE:
        cached, cached_at = _PRICE_CACHE[cache_key]
        if now - cached_at < _PRICE_CACHE_TTL:
            return cached
    
    loop = asyncio.get_event_loop()
    prices_raw = await loop.run_in_executor(
        _executor,
        _get_price_history_sync,
        symbol,
        start_date,
        end_date,
        period,
    )
    
    prices = [
        PricePoint(
            date=p["date"],
            open=p["open"],
            high=p["high"],
            low=p["low"],
            close=p["close"],
            volume=p["volume"],
            adj_close=p.get("adj_close"),
        )
        for p in prices_raw
    ]
    
    series = PriceSeries(symbol=symbol, prices=prices)
    _PRICE_CACHE[cache_key] = (series, now)
    return series


async def get_calendar_events(symbol: str) -> CalendarEvents:
    """Get upcoming calendar events."""
    loop = asyncio.get_event_loop()
    calendar = await loop.run_in_executor(_executor, _get_calendar_sync, symbol)
    
    # Parse calendar data
    next_earnings = None
    ex_dividend = None
    dividend_date = None
    
    # Handle different yfinance calendar formats
    if isinstance(calendar, dict):
        # Try to extract earnings date
        earnings = calendar.get("Earnings Date")
        if earnings:
            if isinstance(earnings, list) and len(earnings) > 0:
                next_earnings = earnings[0]
            elif hasattr(earnings, "date"):
                next_earnings = earnings.date() if callable(getattr(earnings, "date", None)) else earnings
        
        # Dividend dates
        ex_div = calendar.get("Ex-Dividend Date")
        if ex_div and hasattr(ex_div, "date"):
            ex_dividend = ex_div.date() if callable(getattr(ex_div, "date", None)) else ex_div
        
        div_date = calendar.get("Dividend Date")
        if div_date and hasattr(div_date, "date"):
            dividend_date = div_date.date() if callable(getattr(div_date, "date", None)) else div_date
    
    return CalendarEvents(
        symbol=symbol,
        next_earnings_date=next_earnings,
        ex_dividend_date=ex_dividend,
        dividend_date=dividend_date,
    )


async def get_fundamentals(symbol: str) -> Fundamentals:
    """Get fundamental data for a symbol."""
    info = await get_ticker_info(symbol)
    
    if not info:
        return Fundamentals(symbol=symbol, name=symbol)
    
    return Fundamentals(
        symbol=symbol,
        name=info.get("shortName") or info.get("longName") or symbol,
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=_safe_float(info.get("marketCap")),
        enterprise_value=_safe_float(info.get("enterpriseValue")),
        
        # Valuation
        pe_ratio=_safe_float(info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forwardPE")),
        peg_ratio=_safe_float(info.get("pegRatio")),
        price_to_book=_safe_float(info.get("priceToBook")),
        price_to_sales=_safe_float(info.get("priceToSalesTrailing12Months")),
        ev_to_ebitda=_safe_float(info.get("enterpriseToEbitda")),
        ev_to_revenue=_safe_float(info.get("enterpriseToRevenue")),
        
        # Profitability
        profit_margin=_safe_float(info.get("profitMargins")),
        operating_margin=_safe_float(info.get("operatingMargins")),
        gross_margin=_safe_float(info.get("grossMargins")),
        roe=_safe_float(info.get("returnOnEquity")),
        roa=_safe_float(info.get("returnOnAssets")),
        
        # Growth
        revenue_growth=_safe_float(info.get("revenueGrowth")),
        earnings_growth=_safe_float(info.get("earningsGrowth")),
        
        # Financial health
        current_ratio=_safe_float(info.get("currentRatio")),
        quick_ratio=_safe_float(info.get("quickRatio")),
        debt_to_equity=_safe_float(info.get("debtToEquity")),
        free_cash_flow=_safe_float(info.get("freeCashflow")),
        
        # Dividends
        dividend_yield=_safe_float(info.get("dividendYield")),
        payout_ratio=_safe_float(info.get("payoutRatio")),
        
        # Per-share
        eps=_safe_float(info.get("trailingEps")),
        eps_forward=_safe_float(info.get("forwardEps")),
        book_value_per_share=_safe_float(info.get("bookValue")),
        
        # Risk
        beta=_safe_float(info.get("beta")),
        shares_outstanding=_safe_float(info.get("sharesOutstanding")),
        float_shares=_safe_float(info.get("floatShares")),
        short_ratio=_safe_float(info.get("shortRatio")),
        short_percent_of_float=_safe_float(info.get("shortPercentOfFloat")),
        
        raw_info=info,
    )


async def get_market_data(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> MarketData:
    """Get complete market data for a symbol."""
    # Fetch all data concurrently
    prices_task = get_price_history(symbol, start_date, end_date, period)
    fundamentals_task = get_fundamentals(symbol)
    calendar_task = get_calendar_events(symbol)
    
    prices, fundamentals, calendar = await asyncio.gather(
        prices_task,
        fundamentals_task,
        calendar_task,
    )
    
    return MarketData(
        symbol=symbol,
        prices=prices,
        fundamentals=fundamentals,
        calendar=calendar,
    )


async def get_market_data_batch(
    symbols: list[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> dict[str, MarketData]:
    """Get market data for multiple symbols concurrently."""
    tasks = [
        get_market_data(symbol, start_date, end_date, period)
        for symbol in symbols
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    data = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch data for {symbol}: {result}")
            # Return minimal data
            data[symbol] = MarketData(
                symbol=symbol,
                prices=PriceSeries(symbol=symbol, prices=[]),
                fundamentals=Fundamentals(symbol=symbol, name=symbol),
            )
        else:
            data[symbol] = result
    
    return data


# =============================================================================
# Cache management
# =============================================================================


def clear_cache():
    """Clear all caches."""
    global _INFO_CACHE, _PRICE_CACHE
    _INFO_CACHE.clear()
    _PRICE_CACHE.clear()


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics."""
    return {
        "info_cache_size": len(_INFO_CACHE),
        "price_cache_size": len(_PRICE_CACHE),
    }
