"""Symbol search service with intelligent caching.

Uses yfinance Search/Lookup for ticker discovery with local DB caching
to minimize API calls. Searches local cache first before hitting yfinance.

Usage:
    from app.services.symbol_search import search_symbols, lookup_symbol
    
    # Search by name or symbol
    results = await search_symbols("apple")  # Returns list of matches
    
    # Direct symbol lookup
    info = await lookup_symbol("AAPL")  # Returns symbol info or None
"""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import yfinance as yf

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.database.connection import fetch_all, fetch_one, execute

logger = get_logger("services.symbol_search")

_executor = ThreadPoolExecutor(max_workers=4)

# Cache TTL for search results (7 days - company names don't change often)
SEARCH_CACHE_TTL_DAYS = 7

# Minimum query length
MIN_QUERY_LENGTH = 2


def _normalize_query(query: str) -> str:
    """Normalize query for consistent caching."""
    return query.strip().upper()


def _search_yfinance_sync(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search yfinance for symbols matching query (sync, runs in thread pool).
    
    Uses yfinance.Search for comprehensive results including quotes and related items.
    """
    # Rate limit
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for search: {query}")
        return []
    
    try:
        search = yf.Search(
            query,
            max_results=max_results,
            news_count=0,  # Skip news for efficiency
            enable_fuzzy_query=True,
        )
        
        results = []
        
        # Extract quotes from search results
        if hasattr(search, 'quotes') and search.quotes:
            for quote in search.quotes[:max_results]:
                # Filter to stocks/ETFs only (skip options, futures, etc.)
                quote_type = quote.get("quoteType", "").upper()
                if quote_type not in ("EQUITY", "ETF", "INDEX"):
                    continue
                
                results.append({
                    "symbol": quote.get("symbol", ""),
                    "name": quote.get("shortname") or quote.get("longname") or quote.get("symbol"),
                    "exchange": quote.get("exchange", ""),
                    "quote_type": quote_type,
                    "score": quote.get("score", 0),  # Relevance score from yfinance
                })
        
        return results
        
    except Exception as e:
        logger.error(f"yfinance search failed for '{query}': {e}")
        return []


def _lookup_symbol_sync(symbol: str) -> Optional[dict[str, Any]]:
    """Lookup a specific symbol using yfinance.Lookup (sync).
    
    Uses fast_info for efficiency when we just need basic validation.
    """
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for lookup: {symbol}")
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        # Use fast_info for quick validation
        fast_info = ticker.fast_info
        
        if not fast_info or fast_info.last_price is None:
            return None
        
        # Get basic info
        info = ticker.info or {}
        
        return {
            "symbol": symbol.upper(),
            "name": info.get("shortName") or info.get("longName") or symbol,
            "exchange": info.get("exchange", ""),
            "quote_type": info.get("quoteType", "EQUITY"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "current_price": fast_info.last_price,
            "valid": True,
        }
        
    except Exception as e:
        logger.debug(f"Symbol lookup failed for {symbol}: {e}")
        return None


async def _get_cached_search_results(query: str) -> Optional[list[dict[str, Any]]]:
    """Get cached search results from database."""
    import json
    normalized = _normalize_query(query)
    
    row = await fetch_one(
        """
        SELECT results, expires_at
        FROM symbol_search_cache
        WHERE query = $1 AND expires_at > NOW()
        """,
        normalized,
    )
    
    if row and row["results"]:
        results = row["results"]
        # Handle case where asyncpg returns JSONB as string
        if isinstance(results, str):
            results = json.loads(results)
        return results
    
    return None


async def _cache_search_results(query: str, results: list[dict[str, Any]]) -> None:
    """Cache search results in database."""
    import json
    normalized = _normalize_query(query)
    expires_at = datetime.now(timezone.utc) + timedelta(days=SEARCH_CACHE_TTL_DAYS)
    
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
            normalized,
            json.dumps(results),
            expires_at,
        )
    except Exception as e:
        logger.warning(f"Failed to cache search results: {e}")


async def _search_local_symbols(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search local symbols table first (much faster than API)."""
    normalized = _normalize_query(query)
    
    # Search by symbol prefix or name contains
    rows = await fetch_all(
        """
        SELECT 
            s.symbol,
            s.name,
            s.sector,
            s.symbol_type,
            s.market_cap,
            s.pe_ratio,
            CASE 
                WHEN UPPER(s.symbol) = $1 THEN 100
                WHEN UPPER(s.symbol) LIKE $1 || '%' THEN 90
                WHEN UPPER(s.name) LIKE '%' || $1 || '%' THEN 70
                ELSE 50
            END as relevance_score
        FROM symbols s
        WHERE s.is_active = TRUE
          AND (
              UPPER(s.symbol) LIKE $1 || '%'
              OR UPPER(s.name) LIKE '%' || $1 || '%'
          )
        ORDER BY relevance_score DESC, s.market_cap DESC NULLS LAST
        LIMIT $2
        """,
        normalized,
        limit,
    )
    
    return [
        {
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "quote_type": r["symbol_type"].upper() if r["symbol_type"] else "EQUITY",
            "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
            "pe_ratio": float(r["pe_ratio"]) if r["pe_ratio"] else None,
            "source": "local",
            "score": r["relevance_score"],
        }
        for r in rows
    ]


async def search_symbols(
    query: str,
    max_results: int = 10,
    local_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Search for symbols matching query.
    
    Strategy:
    1. Search local symbols table first (instant)
    2. Check search cache for API results
    3. If not cached, query yfinance and cache results
    4. Merge and dedupe results, prioritizing local matches
    
    Args:
        query: Search query (symbol or company name)
        max_results: Maximum results to return
        local_only: If True, only search local database
        
    Returns:
        List of matching symbols with metadata
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return []
    
    results = []
    seen_symbols = set()
    
    # 1. Search local symbols first (fast, authoritative for tracked stocks)
    local_results = await _search_local_symbols(query, limit=max_results)
    for r in local_results:
        if r["symbol"] not in seen_symbols:
            seen_symbols.add(r["symbol"])
            results.append(r)
    
    if local_only:
        return results[:max_results]
    
    # 2. Check search cache
    cached = await _get_cached_search_results(query)
    if cached:
        for r in cached:
            if r["symbol"] not in seen_symbols:
                seen_symbols.add(r["symbol"])
                r["source"] = "cache"
                results.append(r)
        return results[:max_results]
    
    # 3. Query yfinance API
    loop = asyncio.get_event_loop()
    api_results = await loop.run_in_executor(
        _executor, _search_yfinance_sync, query, max_results
    )
    
    # 4. Cache API results (even if empty, to avoid repeated calls)
    if api_results:
        await _cache_search_results(query, api_results)
        for r in api_results:
            if r["symbol"] not in seen_symbols:
                seen_symbols.add(r["symbol"])
                r["source"] = "api"
                results.append(r)
    
    return results[:max_results]


async def lookup_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """
    Lookup a specific symbol to validate and get basic info.
    
    Strategy:
    1. Check local symbols table first
    2. Check fundamentals table for stored data
    3. Fall back to yfinance API with fast_info
    
    Args:
        symbol: Stock symbol to lookup
        
    Returns:
        Symbol info dict or None if invalid/not found
    """
    symbol = symbol.strip().upper()
    
    # 1. Check local symbols
    row = await fetch_one(
        """
        SELECT s.symbol, s.name, s.sector, s.symbol_type, s.market_cap, s.pe_ratio,
               f.forward_pe, f.recommendation, f.target_mean_price
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
            "quote_type": row["symbol_type"].upper() if row["symbol_type"] else "EQUITY",
            "market_cap": float(row["market_cap"]) if row["market_cap"] else None,
            "pe_ratio": float(row["pe_ratio"]) if row["pe_ratio"] else None,
            "forward_pe": float(row["forward_pe"]) if row["forward_pe"] else None,
            "recommendation": row["recommendation"],
            "target_mean_price": float(row["target_mean_price"]) if row["target_mean_price"] else None,
            "source": "local",
            "valid": True,
        }
    
    # 2. Check if in fundamentals table (might be pending activation)
    fund_row = await fetch_one(
        "SELECT symbol FROM stock_fundamentals WHERE symbol = $1",
        symbol,
    )
    
    # 3. Fall back to yfinance
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _lookup_symbol_sync, symbol)
    
    if result:
        result["source"] = "api"
    
    return result


async def get_symbol_suggestions(
    partial: str,
    limit: int = 5,
) -> list[dict[str, str]]:
    """
    Get autocomplete suggestions for a partial symbol/name.
    
    Optimized for speed - only returns symbol and name.
    
    Args:
        partial: Partial input from user
        limit: Max suggestions
        
    Returns:
        List of {symbol, name} dicts
    """
    if len(partial.strip()) < 1:
        return []
    
    # Fast local-only search
    results = await search_symbols(partial, max_results=limit, local_only=True)
    
    # If not enough local results, try API
    if len(results) < limit:
        all_results = await search_symbols(partial, max_results=limit, local_only=False)
        results = all_results
    
    return [{"symbol": r["symbol"], "name": r["name"]} for r in results[:limit]]
