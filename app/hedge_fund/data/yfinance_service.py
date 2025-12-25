"""
yfinance data service for fetching market data.

MIGRATED: Now uses unified YFinanceService for all yfinance calls.
Provides a clean interface with typed Pydantic models for the hedge_fund module.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

from app.services.data_providers import get_yfinance_service
from app.hedge_fund.schemas import (
    CalendarEvents,
    Fundamentals,
    MarketData,
    PricePoint,
    PriceSeries,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Async wrappers using unified YFinanceService
# =============================================================================


async def get_ticker_info(symbol: str) -> dict[str, Any]:
    """Get ticker info asynchronously via unified service."""
    service = get_yfinance_service()
    info = await service.get_ticker_info(symbol)
    return info or {}


async def get_price_history(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> PriceSeries:
    """Get price history as PriceSeries."""
    service = get_yfinance_service()
    
    df, _ = await service.get_price_history(
        symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
    )
    
    prices = []
    if df is not None and not df.empty:
        for idx, row in df.iterrows():
            prices.append(PricePoint(
                date=idx.date() if hasattr(idx, "date") else idx,
                open=_safe_float(row.get("Open"), 0.0),
                high=_safe_float(row.get("High"), 0.0),
                low=_safe_float(row.get("Low"), 0.0),
                close=_safe_float(row.get("Close"), 0.0),
                volume=_safe_int(row.get("Volume"), 0),
                adj_close=_safe_float(row.get("Adj Close")),
            ))
    
    return PriceSeries(symbol=symbol, prices=prices)


async def get_calendar_events(symbol: str) -> CalendarEvents:
    """Get upcoming calendar events."""
    service = get_yfinance_service()
    calendar, _ = await service.get_calendar(symbol)
    
    # Parse calendar data
    next_earnings = None
    ex_dividend = None
    dividend_date = None
    
    if calendar:
        # Try to extract earnings date
        earnings = calendar.get("next_earnings_date")
        if earnings:
            if isinstance(earnings, list) and len(earnings) > 0:
                next_earnings = earnings[0]
            elif hasattr(earnings, "date"):
                next_earnings = earnings.date() if callable(getattr(earnings, "date", None)) else earnings
            else:
                next_earnings = earnings
        
        # Dividend dates
        ex_div = calendar.get("ex_dividend_date")
        if ex_div:
            if hasattr(ex_div, "date"):
                ex_dividend = ex_div.date() if callable(getattr(ex_div, "date", None)) else ex_div
            else:
                ex_dividend = ex_div
        
        div_date = calendar.get("dividend_date")
        if div_date:
            if hasattr(div_date, "date"):
                dividend_date = div_date.date() if callable(getattr(div_date, "date", None)) else div_date
            else:
                dividend_date = div_date
    
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
        name=info.get("name") or info.get("shortName") or info.get("longName") or symbol,
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=_safe_float(info.get("market_cap") or info.get("marketCap")),
        enterprise_value=_safe_float(info.get("enterprise_value") or info.get("enterpriseValue")),
        
        # Valuation
        pe_ratio=_safe_float(info.get("pe_ratio") or info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forward_pe") or info.get("forwardPE")),
        peg_ratio=_safe_float(info.get("peg_ratio") or info.get("pegRatio")),
        price_to_book=_safe_float(info.get("price_to_book") or info.get("priceToBook")),
        price_to_sales=_safe_float(info.get("price_to_sales") or info.get("priceToSalesTrailing12Months")),
        ev_to_ebitda=_safe_float(info.get("ev_to_ebitda") or info.get("enterpriseToEbitda")),
        ev_to_revenue=_safe_float(info.get("ev_to_revenue") or info.get("enterpriseToRevenue")),
        
        # Profitability
        profit_margin=_safe_float(info.get("profit_margin") or info.get("profitMargins")),
        operating_margin=_safe_float(info.get("operating_margin") or info.get("operatingMargins")),
        gross_margin=_safe_float(info.get("gross_margin") or info.get("grossMargins")),
        roe=_safe_float(info.get("return_on_equity") or info.get("returnOnEquity")),
        roa=_safe_float(info.get("return_on_assets") or info.get("returnOnAssets")),
        
        # Growth
        revenue_growth=_safe_float(info.get("revenue_growth") or info.get("revenueGrowth")),
        earnings_growth=_safe_float(info.get("earnings_growth") or info.get("earningsGrowth")),
        
        # Financial health
        current_ratio=_safe_float(info.get("current_ratio") or info.get("currentRatio")),
        quick_ratio=_safe_float(info.get("quick_ratio") or info.get("quickRatio")),
        debt_to_equity=_safe_float(info.get("debt_to_equity") or info.get("debtToEquity")),
        free_cash_flow=_safe_float(info.get("free_cash_flow") or info.get("freeCashflow")),
        
        # Dividends
        dividend_yield=_safe_float(info.get("dividend_yield") or info.get("dividendYield")),
        payout_ratio=_safe_float(info.get("payoutRatio")),
        
        # Per-share
        eps=_safe_float(info.get("eps_trailing") or info.get("trailingEps")),
        eps_forward=_safe_float(info.get("eps_forward") or info.get("forwardEps")),
        book_value_per_share=_safe_float(info.get("book_value") or info.get("bookValue")),
        
        # Risk
        beta=_safe_float(info.get("beta")),
        shares_outstanding=_safe_float(info.get("shares_outstanding") or info.get("sharesOutstanding")),
        float_shares=_safe_float(info.get("float_shares") or info.get("floatShares")),
        short_ratio=_safe_float(info.get("short_ratio") or info.get("shortRatio")),
        short_percent_of_float=_safe_float(info.get("short_percent_of_float") or info.get("shortPercentOfFloat")),
        
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
# Helpers
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


# =============================================================================
# Cache management (delegated to unified service)
# =============================================================================


def clear_cache():
    """Clear cache - unified service manages its own cache."""
    # The unified service handles cache management
    pass


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics."""
    # Return empty stats - unified service manages cache
    return {
        "info_cache_size": 0,
        "price_cache_size": 0,
        "note": "Cache managed by unified YFinanceService",
    }
