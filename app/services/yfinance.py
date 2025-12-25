"""
DEPRECATED: Legacy yfinance module - Façade delegating to unified YFinanceService.

This module maintains backward compatibility for existing imports.
All new code should import directly from app.services.data_providers.

Usage (old - still works):
    from app.services.yfinance import get_ticker_info, search_tickers
    
Usage (new - preferred):
    from app.services.data_providers import get_yfinance_service
    service = get_yfinance_service()
    info = await service.get_ticker_info("AAPL")
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from app.core.logging import get_logger
from app.database.connection import fetch_one, fetch_all, execute
from app.services.data_providers import get_yfinance_service

logger = get_logger("services.yfinance")

# Emit deprecation warning once per session
_WARNED = False


def _emit_deprecation_warning():
    """Emit deprecation warning once."""
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "app.services.yfinance is deprecated. Use app.services.data_providers.get_yfinance_service() instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _WARNED = True


# Re-export utility functions for compatibility
def is_etf_or_index(symbol: str, quote_type: Optional[str] = None) -> bool:
    """Check if symbol is an ETF, index, or fund."""
    if symbol.startswith("^"):
        return True
    if quote_type:
        return quote_type.upper() in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    return False


# =============================================================================
# Async Public API - Delegates to YFinanceService
# =============================================================================


async def get_ticker_info(
    symbol: str,
    use_cache: bool = True,
) -> Optional[dict[str, Any]]:
    """
    Get complete ticker info for a symbol.
    
    DEPRECATED: Use get_yfinance_service().get_ticker_info() instead.
    """
    _emit_deprecation_warning()
    service = get_yfinance_service()
    return await service.get_ticker_info(symbol, skip_cache=not use_cache)


async def get_price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
) -> Optional[list[dict]]:
    """
    Get price history for a symbol.
    
    DEPRECATED: Use get_yfinance_service().get_price_history() instead.
    
    Returns list of OHLCV dicts for backward compatibility.
    """
    _emit_deprecation_warning()
    service = get_yfinance_service()
    df, _ = await service.get_price_history(symbol, period=period, interval=interval)
    
    if df is None or df.empty:
        return None
    
    # Convert DataFrame to list of dicts for backward compatibility
    prices = []
    for idx, row in df.iterrows():
        prices.append({
            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10],
            "open": round(float(row.get("Open", 0)), 2),
            "high": round(float(row.get("High", 0)), 2),
            "low": round(float(row.get("Low", 0)), 2),
            "close": round(float(row.get("Close", 0)), 2),
            "volume": int(row.get("Volume", 0)),
        })
    
    return prices


async def search_tickers(
    query: str,
    max_results: int = 10,
    save_to_db: bool = True,
) -> list[dict[str, Any]]:
    """
    Search for tickers matching query.
    
    DEPRECATED: Use get_yfinance_service().search_tickers() instead.
    
    Search strategy:
    1. Search local symbols table first (instant)
    2. Check search cache for previous API results
    3. Query yfinance API if needed
    4. Save ALL results to local DB for future queries
    """
    _emit_deprecation_warning()
    
    if len(query.strip()) < 2:
        return []
    
    normalized = query.strip().upper()
    results = []
    seen_symbols = set()
    
    # 1. Search local symbols table first (fast)
    local_results = await _search_local_db(normalized, max_results)
    for r in local_results:
        if r["symbol"] not in seen_symbols:
            seen_symbols.add(r["symbol"])
            r["source"] = "local"
            results.append(r)
    
    # If we have enough results from local, return early
    if len(results) >= max_results:
        return results[:max_results]
    
    # 2. Check search cache
    cached = await _get_search_cache(normalized)
    if cached:
        for r in cached:
            if r["symbol"] not in seen_symbols:
                seen_symbols.add(r["symbol"])
                r["source"] = "cache"
                results.append(r)
        return results[:max_results]
    
    # 3. Query via unified service
    service = get_yfinance_service()
    api_results = await service.search_tickers(query, max_results=max_results, save_to_db=save_to_db)
    
    # 4. Save to cache
    if api_results:
        await _save_search_cache(normalized, api_results)
        
        for r in api_results:
            if r["symbol"] not in seen_symbols:
                seen_symbols.add(r["symbol"])
                r["source"] = "api"
                results.append(r)
    
    return results[:max_results]


async def validate_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """
    Validate a symbol exists and get basic info.
    
    DEPRECATED: Use get_yfinance_service().validate_symbol() instead.
    """
    _emit_deprecation_warning()
    symbol = symbol.strip().upper()
    
    # Check local first
    row = await fetch_one(
        """
        SELECT symbol, name, sector, symbol_type, market_cap
        FROM symbols 
        WHERE symbol = $1 AND is_active = TRUE
        """,
        symbol,
    )
    
    if row:
        return {
            "symbol": row["symbol"],
            "name": row["name"],
            "sector": row["sector"],
            "quote_type": row["symbol_type"] or "EQUITY",
            "market_cap": float(row["market_cap"]) if row["market_cap"] else None,
            "valid": True,
            "source": "local",
        }
    
    # Fall back to unified service
    service = get_yfinance_service()
    info = await service.get_ticker_info(symbol)
    if info and info.get("current_price"):
        return {
            "symbol": info["symbol"],
            "name": info["name"],
            "sector": info["sector"],
            "quote_type": info["quote_type"],
            "market_cap": info["market_cap"],
            "valid": True,
            "source": "api",
        }
    
    return None


async def get_fundamentals_data(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get fundamentals data for a symbol.
    
    DEPRECATED: Use get_yfinance_service().get_ticker_info() instead.
    """
    _emit_deprecation_warning()
    service = get_yfinance_service()
    info = await service.get_ticker_info(symbol)
    if not info:
        return None
    
    # Return subset focused on fundamentals
    return {
        "symbol": info["symbol"],
        "name": info.get("name"),
        # Valuation
        "pe_ratio": info.get("pe_ratio"),
        "forward_pe": info.get("forward_pe"),
        "peg_ratio": info.get("peg_ratio"),
        "price_to_book": info.get("price_to_book"),
        "price_to_sales": info.get("price_to_sales"),
        "ev_to_ebitda": info.get("ev_to_ebitda"),
        # Profitability
        "profit_margin": info.get("profit_margin"),
        "operating_margin": info.get("operating_margin"),
        "gross_margin": info.get("gross_margin"),
        "return_on_equity": info.get("return_on_equity"),
        "return_on_assets": info.get("return_on_assets"),
        # Health
        "debt_to_equity": info.get("debt_to_equity"),
        "current_ratio": info.get("current_ratio"),
        "free_cash_flow": info.get("free_cash_flow"),
        # Growth
        "revenue_growth": info.get("revenue_growth"),
        "earnings_growth": info.get("earnings_growth"),
        # Analyst
        "recommendation": info.get("recommendation"),
        "target_mean_price": info.get("target_mean_price"),
        "num_analyst_opinions": info.get("num_analyst_opinions"),
        # Risk
        "beta": info.get("beta"),
        "short_percent_of_float": info.get("short_percent_of_float"),
    }


# =============================================================================
# Database Helpers (kept for local search which is still useful)
# =============================================================================


async def _search_local_db(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search local symbols table and search results."""
    rows = await fetch_all(
        """
        SELECT 
            symbol, name, sector, symbol_type, market_cap, pe_ratio,
            CASE 
                WHEN UPPER(symbol) = $1 THEN 100
                WHEN UPPER(symbol) LIKE $1 || '%' THEN 90
                WHEN UPPER(name) LIKE '%' || $1 || '%' THEN 70
                ELSE 50
            END as score
        FROM symbols
        WHERE is_active = TRUE
          AND (UPPER(symbol) LIKE $1 || '%' OR UPPER(name) LIKE '%' || $1 || '%')
        
        UNION ALL
        
        SELECT 
            symbol, name, sector, quote_type as symbol_type, market_cap, NULL as pe_ratio,
            CASE 
                WHEN UPPER(symbol) = $1 THEN 85
                WHEN UPPER(symbol) LIKE $1 || '%' THEN 75
                WHEN UPPER(name) LIKE '%' || $1 || '%' THEN 60
                ELSE 40
            END as score
        FROM symbol_search_results
        WHERE symbol NOT IN (SELECT symbol FROM symbols WHERE is_active = TRUE)
          AND (UPPER(symbol) LIKE $1 || '%' OR UPPER(name) LIKE '%' || $1 || '%')
          AND expires_at > NOW()
        
        ORDER BY score DESC, market_cap DESC NULLS LAST
        LIMIT $2
        """,
        query,
        limit,
    )
    
    return [
        {
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "quote_type": (r["symbol_type"] or "EQUITY").upper(),
            "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
            "pe_ratio": float(r["pe_ratio"]) if r["pe_ratio"] else None,
            "score": r["score"],
        }
        for r in rows
    ]


async def _get_search_cache(query: str) -> Optional[list[dict]]:
    """Get cached search results."""
    import json
    row = await fetch_one(
        """
        SELECT results FROM symbol_search_cache
        WHERE query = $1 AND expires_at > NOW()
        """,
        query,
    )
    
    if row and row["results"]:
        results = row["results"]
        if isinstance(results, str):
            results = json.loads(results)
        return results
    
    return None


async def _save_search_cache(query: str, results: list[dict]) -> None:
    """Cache search results."""
    import json
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    try:
        await execute(
            """
            INSERT INTO symbol_search_cache (query, results, expires_at)
            VALUES ($1, $2::jsonb, $3)
            ON CONFLICT (query) DO UPDATE SET
                results = EXCLUDED.results,
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW()
            """,
            query,
            json.dumps(results),
            expires_at,
        )
    except Exception as e:
        logger.warning(f"Failed to cache search results: {e}")


# =============================================================================
# Cache Management - Delegates to unified service's cache
# =============================================================================


def clear_info_cache(symbol: Optional[str] = None) -> None:
    """Clear in-memory info cache (no-op, unified service manages its own cache)."""
    _emit_deprecation_warning()
    # The unified service handles its own cache management
    pass


def clear_price_cache(symbol: Optional[str] = None) -> None:
    """Clear in-memory price cache (no-op, unified service manages its own cache)."""
    _emit_deprecation_warning()
    # The unified service handles its own cache management
    pass
