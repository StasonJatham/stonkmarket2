"""Symbol search service - Local-first search with yfinance fallback.

SEARCH STRATEGY:
1. Search local symbols table first (instant results)
2. Search symbol_search_results table (previously cached API results)
3. If not enough results, suggest fresh API search (user-initiated)

All yfinance API calls are delegated to the unified yfinance service.
This module focuses on the search strategy and user experience.

Usage:
    from app.services.symbol_search import search_symbols, lookup_symbol
    
    # Search (local-first, with option to search fresh)
    results = await search_symbols("apple")
    # Returns: { "results": [...], "has_more": True, "suggest_fresh_search": True }
    
    # Force fresh API search (user requested)
    results = await search_symbols("apple", force_api=True)
    
    # Direct symbol lookup
    info = await lookup_symbol("AAPL")
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.logging import get_logger
from app.database.connection import fetch_all, fetch_one
from app.services import yfinance as yf_service

logger = get_logger("services.symbol_search")

# Minimum query length
MIN_QUERY_LENGTH = 2

# Threshold for suggesting fresh search
MIN_LOCAL_RESULTS_THRESHOLD = 3


def _normalize_query(query: str) -> str:
    """Normalize query for consistent searching."""
    return query.strip().upper()


async def _search_local_db(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search local database for symbols.
    
    Searches both symbols table and symbol_search_results table.
    Returns combined results sorted by relevance.
    """
    normalized = _normalize_query(query)
    
    rows = await fetch_all(
        """
        WITH combined AS (
            -- Active symbols (highest priority)
            SELECT 
                s.symbol,
                s.name,
                s.sector,
                COALESCE(s.symbol_type, 'EQUITY') as quote_type,
                s.market_cap,
                NULL as pe_ratio,
                'local' as source,
                CASE 
                    WHEN UPPER(s.symbol) = $1 THEN 100
                    WHEN UPPER(s.symbol) LIKE $1 || '%' THEN 90
                    WHEN UPPER(s.name) LIKE '%' || $1 || '%' THEN 70
                    ELSE 50
                END as score
            FROM symbols s
            WHERE s.is_active = TRUE
              AND (
                  UPPER(s.symbol) LIKE $1 || '%'
                  OR UPPER(s.name) LIKE '%' || $1 || '%'
              )
              
            UNION ALL
            
            -- Cached search results (not in symbols table)
            SELECT 
                r.symbol,
                r.name,
                r.sector,
                COALESCE(r.quote_type, 'EQUITY') as quote_type,
                r.market_cap,
                NULL as pe_ratio,
                'cached' as source,
                CASE 
                    WHEN UPPER(r.symbol) = $1 THEN 85
                    WHEN UPPER(r.symbol) LIKE $1 || '%' THEN 75
                    WHEN UPPER(r.name) LIKE '%' || $1 || '%' THEN 60
                    ELSE 40
                END as score
            FROM symbol_search_results r
            WHERE r.expires_at > NOW()
              AND NOT EXISTS (SELECT 1 FROM symbols s WHERE s.symbol = r.symbol AND s.is_active = TRUE)
              AND (
                  UPPER(r.symbol) LIKE $1 || '%'
                  OR UPPER(r.name) LIKE '%' || $1 || '%'
              )
        )
        SELECT DISTINCT ON (symbol) *
        FROM combined
        ORDER BY symbol, score DESC
        """,
        normalized,
    )
    
    # Re-sort by score DESC, then market_cap DESC
    results = [
        {
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "quote_type": (r["quote_type"] or "EQUITY").upper(),
            "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
            "pe_ratio": float(r["pe_ratio"]) if r["pe_ratio"] else None,
            "source": r["source"],
            "score": r["score"],
        }
        for r in rows
    ]
    
    # Sort by score DESC, market_cap DESC
    results.sort(key=lambda x: (-x["score"], -(x["market_cap"] or 0)))
    
    return results[:limit]


async def search_symbols(
    query: str,
    max_results: int = 10,
    force_api: bool = False,
) -> dict[str, Any]:
    """
    Search for symbols matching query.
    
    LOCAL-FIRST STRATEGY:
    1. Search local DB first (symbols + cached search results)
    2. If enough results, return immediately
    3. If not enough, indicate that fresh API search is available
    
    When force_api=True:
    - Query yfinance API directly
    - Save ALL results to DB for future local searches
    
    Args:
        query: Search query (symbol or company name)
        max_results: Maximum results to return
        force_api: Force fresh search from yfinance API
        
    Returns:
        Dict with:
        - results: List of matching symbols
        - count: Number of results
        - suggest_fresh_search: Whether to suggest API search
        - search_type: "local" or "api"
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return {
            "results": [],
            "count": 0,
            "suggest_fresh_search": False,
            "search_type": "local",
        }
    
    # If force_api, query yfinance directly
    if force_api:
        api_results = await yf_service.search_tickers(
            query, 
            max_results=max_results * 2,  # Fetch more to filter
            save_to_db=True,  # IMPORTANT: Save all results for future local search
        )
        
        return {
            "results": api_results[:max_results],
            "count": len(api_results),
            "suggest_fresh_search": False,
            "search_type": "api",
        }
    
    # Local-first search
    local_results = await _search_local_db(query, limit=max_results)
    
    # Determine if we should suggest fresh search
    suggest_fresh = len(local_results) < MIN_LOCAL_RESULTS_THRESHOLD
    
    return {
        "results": local_results,
        "count": len(local_results),
        "suggest_fresh_search": suggest_fresh,
        "search_type": "local",
    }


async def lookup_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """
    Lookup a specific symbol to validate and get info.
    
    Uses unified yfinance service for API calls.
    
    Args:
        symbol: Stock symbol to lookup
        
    Returns:
        Symbol info dict or None if invalid/not found
    """
    symbol = symbol.strip().upper()
    
    # Check local symbols with fundamentals
    row = await fetch_one(
        """
        SELECT s.symbol, s.name, s.sector, s.symbol_type, s.market_cap,
               f.pe_ratio, f.forward_pe, f.recommendation, f.target_mean_price,
               f.current_price, f.previous_close
        FROM symbols s
        LEFT JOIN stock_fundamentals f ON s.symbol = f.symbol
        WHERE s.symbol = $1 AND s.is_active = TRUE
        """,
        symbol,
    )
    
    if row:
        return {
            "symbol": row["symbol"],
            "name": row["name"],
            "sector": row["sector"],
            "quote_type": (row["symbol_type"] or "EQUITY").upper(),
            "market_cap": float(row["market_cap"]) if row["market_cap"] else None,
            "pe_ratio": float(row["pe_ratio"]) if row["pe_ratio"] else None,
            "forward_pe": float(row["forward_pe"]) if row["forward_pe"] else None,
            "recommendation": row["recommendation"],
            "target_mean_price": float(row["target_mean_price"]) if row["target_mean_price"] else None,
            "current_price": float(row["current_price"]) if row["current_price"] else None,
            "source": "local",
            "valid": True,
        }
    
    # Check cached search results
    cached_row = await fetch_one(
        """
        SELECT symbol, name, sector, quote_type, market_cap
        FROM symbol_search_results
        WHERE symbol = $1 AND expires_at > NOW()
        """,
        symbol,
    )
    
    if cached_row:
        return {
            "symbol": cached_row["symbol"],
            "name": cached_row["name"],
            "sector": cached_row["sector"],
            "quote_type": (cached_row["quote_type"] or "EQUITY").upper(),
            "market_cap": float(cached_row["market_cap"]) if cached_row["market_cap"] else None,
            "source": "cached",
            "valid": True,
        }
    
    # Fall back to yfinance validation
    return await yf_service.validate_symbol(symbol)


async def get_symbol_suggestions(
    partial: str,
    limit: int = 5,
) -> list[dict[str, str]]:
    """
    Get autocomplete suggestions for partial symbol/name.
    
    Local-only for speed. Does not trigger API calls.
    
    Args:
        partial: Partial input from user
        limit: Max suggestions
        
    Returns:
        List of {symbol, name} dicts
    """
    if len(partial.strip()) < 1:
        return []
    
    results = await _search_local_db(partial, limit=limit)
    return [{"symbol": r["symbol"], "name": r["name"]} for r in results]


async def get_known_symbols(limit: int = 100) -> list[str]:
    """Get list of all known active symbols."""
    rows = await fetch_all(
        "SELECT symbol FROM symbols WHERE is_active = TRUE ORDER BY symbol LIMIT $1",
        limit,
    )
    return [r["symbol"] for r in rows]
