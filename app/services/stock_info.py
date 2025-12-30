"""Stock info service - wrapper around unified yfinance service.

Provides stock info in the format expected by the StockInfo schema.
Uses the unified yfinance service for all API calls.
Uses FinancialUniverse as fallback for sector/country when yfinance data is missing.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.schemas.dips import StockInfo
from app.services.data_providers import get_yfinance_service
from app.services import financedatabase_service


logger = get_logger("services.stock_info")

# Get singleton service instance
_yf_service = get_yfinance_service()


def is_index_or_etf(symbol: str, quote_type: str | None = None) -> bool:
    """Check if a symbol is an index, ETF, or fund - detected dynamically from quote_type."""
    if symbol.startswith("^"):
        return True
    if quote_type:
        return quote_type.upper() in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    return False


async def _enrich_from_universe(symbol: str, info: dict[str, Any]) -> dict[str, Any]:
    """Enrich stock info with sector/country from FinancialUniverse if missing.
    
    Falls back to local universe data when yfinance doesn't return sector/country.
    This is common for non-US stocks or when yfinance has incomplete data.
    """
    # Only look up universe if sector or country is missing
    if info.get("sector") and info.get("country"):
        return info
    
    try:
        universe_data = await financedatabase_service.get_by_symbol(symbol)
        if universe_data:
            if not info.get("sector") and universe_data.get("sector"):
                info["sector"] = universe_data["sector"]
                logger.debug(f"Enriched {symbol} sector from universe: {universe_data['sector']}")
            if not info.get("country") and universe_data.get("country"):
                info["country"] = universe_data["country"]
                logger.debug(f"Enriched {symbol} country from universe: {universe_data['country']}")
            if not info.get("industry") and universe_data.get("industry"):
                info["industry"] = universe_data["industry"]
    except Exception as e:
        logger.warning(f"Failed to enrich {symbol} from universe: {e}")
    
    return info


async def get_stock_info(symbol: str) -> StockInfo | None:
    """Fetch detailed stock info using unified yfinance service.
    
    Enriches with FinancialUniverse data as fallback for sector/country.
    """
    info = await _yf_service.get_ticker_info(symbol)
    if not info:
        return None

    # Use is_etf flag from unified service (detected from quote_type)
    is_etf = info.get("is_etf", False)
    
    # Enrich with universe data for sector/country/industry if missing (only for non-ETFs)
    if not is_etf:
        info = await _enrich_from_universe(symbol, info)

    return StockInfo(
        symbol=info["symbol"],
        name=info["name"],
        sector=None if is_etf else info.get("sector"),
        industry=None if is_etf else info.get("industry"),
        country=info.get("country"),
        market_cap=info.get("market_cap"),
        current_price=info.get("current_price"),
        pe_ratio=None if is_etf else info.get("pe_ratio"),
        forward_pe=None if is_etf else info.get("forward_pe"),
        peg_ratio=None if is_etf else info.get("peg_ratio"),
        dividend_yield=info.get("dividend_yield"),
        beta=info.get("beta"),
        avg_volume=info.get("avg_volume"),
        summary=info.get("summary"),
        website=info.get("website"),
        recommendation=None if is_etf else info.get("recommendation"),
        profit_margin=None if is_etf else info.get("profit_margin"),
        gross_margin=None if is_etf else info.get("gross_margin"),
        return_on_equity=None if is_etf else info.get("return_on_equity"),
        debt_to_equity=None if is_etf else info.get("debt_to_equity"),
        current_ratio=None if is_etf else info.get("current_ratio"),
        revenue_growth=None if is_etf else info.get("revenue_growth"),
        free_cash_flow=None if is_etf else info.get("free_cash_flow"),
        target_mean_price=None if is_etf else info.get("target_mean_price"),
        num_analyst_opinions=None if is_etf else info.get("num_analyst_opinions"),
    )

async def get_stock_info_with_prices(symbol: str) -> dict[str, Any] | None:
    """Fetch stock info including current price and ATH.
    
    Enriches with FinancialUniverse data as fallback for sector/country.
    """
    info = await _yf_service.get_ticker_info(symbol)
    if not info:
        return None

    # Use is_etf flag from unified service (detected from quote_type)
    is_etf = info.get("is_etf", False)
    
    # Enrich with universe data for sector/country/industry if missing (only for non-ETFs)
    if not is_etf:
        info = await _enrich_from_universe(symbol, info)
    
    current_price = info.get("current_price") or 0
    previous_close = info.get("previous_close") or 0
    ath_price = info.get("fifty_two_week_high") or 0

    # Calculate change percent
    change_percent = None
    if current_price and previous_close and previous_close > 0:
        change_percent = ((current_price - previous_close) / previous_close) * 100

    return {
        "symbol": info["symbol"],
        "name": info["name"],
        "sector": None if is_etf else info.get("sector"),
        "industry": None if is_etf else info.get("industry"),
        "country": info.get("country"),  # Added country field
        "market_cap": info.get("market_cap"),
        "current_price": float(current_price),
        "previous_close": float(previous_close) if previous_close else None,
        "change_percent": round(change_percent, 4) if change_percent is not None else None,
        "ath_price": float(ath_price),
        "fifty_two_week_high": float(ath_price),
        "fifty_two_week_low": float(info.get("fifty_two_week_low") or 0),
        "pe_ratio": None if is_etf else info.get("pe_ratio"),
        "avg_volume": info.get("avg_volume"),
        "summary": info.get("summary"),
        "website": info.get("website"),
        "ipo_year": info.get("ipo_year"),
        "recommendation": None if is_etf else info.get("recommendation"),
        "is_etf_or_index": is_etf,
    }


# Alias for backwards compatibility
async def get_stock_info_async(symbol: str) -> dict[str, Any] | None:
    """Async wrapper for get_stock_info_with_prices."""
    return await get_stock_info_with_prices(symbol)


async def get_stock_info_batch_async(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch stock info for multiple symbols in parallel.
    
    Uses asyncio.gather to fetch info for all symbols concurrently,
    which is more efficient than sequential calls.
    
    Args:
        symbols: List of stock symbols to fetch
        
    Returns:
        Dictionary mapping symbol to its info dict (empty dict for failures)
    """
    import asyncio

    async def fetch_one(symbol: str) -> tuple[str, dict[str, Any]]:
        info = await get_stock_info_with_prices(symbol)
        return symbol, info or {}

    results = await asyncio.gather(*[fetch_one(s) for s in symbols], return_exceptions=True)

    output = {}
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Failed to fetch stock info: {result}")
            continue
        symbol, info = result
        output[symbol] = info

    return output


def clear_info_cache() -> None:
    """Clear the stock info cache (no-op - service manages its own cache)."""
    pass
