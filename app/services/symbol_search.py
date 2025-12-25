"""Symbol search service - Local-first search with yfinance fallback.

SEARCH STRATEGY:
1. Search local symbols table first (instant results)
2. Search symbol_search_results table (previously cached API results)
3. Use trigram similarity for fuzzy name matching
4. If not enough results, suggest fresh API search (user-initiated)

PAGINATION:
Uses cursor-based pagination for stable results during scrolling.
Cursor format: "score:id" (e.g., "0.850:1234")

All yfinance API calls are delegated to the unified yfinance service.
This module focuses on the search strategy and user experience.

Usage:
    from app.services.symbol_search import search_symbols, lookup_symbol
    
    # Search (local-first, with option to search fresh)
    results = await search_symbols("apple")
    # Returns: { "results": [...], "has_more": True, "suggest_fresh_search": True }
    
    # Paginate with cursor
    page2 = await search_symbols("apple", cursor=results["next_cursor"])
    
    # Force fresh API search (user requested)
    results = await search_symbols("apple", force_api=True)
    
    # Direct symbol lookup
    info = await lookup_symbol("AAPL")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from app.core.logging import get_logger
from app.database.connection import execute, fetch_all, fetch_one
from app.services import yfinance as yf_service

logger = get_logger("services.symbol_search")

# Minimum query length
MIN_QUERY_LENGTH = 2

# Threshold for suggesting fresh search
MIN_LOCAL_RESULTS_THRESHOLD = 3

# Default page size
DEFAULT_PAGE_SIZE = 10


@dataclass
class SearchCursor:
    """Cursor for pagination - encoded as base64 string."""
    score: Decimal
    id: int
    
    def encode(self) -> str:
        """Encode cursor to URL-safe string."""
        raw = f"{self.score}:{self.id}"
        return base64.urlsafe_b64encode(raw.encode()).decode()
    
    @classmethod
    def decode(cls, cursor_str: str) -> Optional["SearchCursor"]:
        """Decode cursor from string, returns None if invalid."""
        try:
            raw = base64.urlsafe_b64decode(cursor_str.encode()).decode()
            score_str, id_str = raw.split(":")
            return cls(score=Decimal(score_str), id=int(id_str))
        except (ValueError, TypeError):
            return None


def _normalize_query(query: str) -> str:
    """Normalize query for consistent searching."""
    return query.strip().upper()


async def _search_local_db(
    query: str, 
    limit: int = 10,
    cursor: Optional[SearchCursor] = None,
    use_trigram: bool = True,
) -> tuple[list[dict[str, Any]], Optional[SearchCursor]]:
    """
    Search local database for symbols with cursor pagination.
    
    Searches both symbols table and symbol_search_results table.
    Returns combined results sorted by relevance, with pagination support.
    
    Args:
        query: Normalized search query
        limit: Max results to return
        cursor: Pagination cursor (score:id)
        use_trigram: Whether to use trigram similarity for name matching
        
    Returns:
        Tuple of (results, next_cursor)
    """
    normalized = _normalize_query(query)
    
    # Build cursor WHERE clause
    cursor_clause = ""
    cursor_params: list[Any] = [normalized]
    param_idx = 2
    
    if cursor:
        cursor_clause = f"""
            AND (
                score < ${param_idx}
                OR (score = ${param_idx} AND id > ${param_idx + 1})
            )
        """
        cursor_params.extend([cursor.score, cursor.id])
        param_idx += 2
    
    # Trigram similarity clause for fuzzy name matching
    trigram_select = ""
    trigram_join = ""
    if use_trigram:
        # Add similarity score as a factor
        trigram_select = f", similarity(name, $1) as name_similarity"
    
    # Fetch one extra to detect if there are more results
    fetch_limit = limit + 1
    cursor_params.append(fetch_limit)
    
    rows = await fetch_all(
        f"""
        WITH combined AS (
            -- Active symbols (highest priority)
            SELECT 
                s.id::bigint as id,
                s.symbol,
                s.name,
                s.sector,
                COALESCE(s.symbol_type, 'EQUITY') as quote_type,
                s.market_cap,
                NULL::numeric as pe_ratio,
                'local' as source,
                CASE 
                    WHEN UPPER(s.symbol) = $1 THEN 1.00
                    WHEN UPPER(s.symbol) LIKE $1 || '%' THEN 0.90
                    WHEN UPPER(s.name) LIKE '%' || $1 || '%' THEN 0.70
                    ELSE 0.50
                END::numeric as score
                {trigram_select.replace('name', 's.name') if use_trigram else ', 0 as name_similarity'}
            FROM symbols s
            WHERE s.is_active = TRUE
              AND (
                  UPPER(s.symbol) LIKE $1 || '%'
                  OR UPPER(s.name) LIKE '%' || $1 || '%'
                  {"OR similarity(s.name, $1) > 0.3" if use_trigram else ""}
              )
              
            UNION ALL
            
            -- Cached search results (not in symbols table)
            SELECT 
                r.id::bigint as id,
                r.symbol,
                r.name,
                r.sector,
                COALESCE(r.quote_type, 'EQUITY') as quote_type,
                r.market_cap,
                NULL::numeric as pe_ratio,
                'cached' as source,
                COALESCE(r.confidence_score, 
                    CASE 
                        WHEN UPPER(r.symbol) = $1 THEN 0.85
                        WHEN UPPER(r.symbol) LIKE $1 || '%' THEN 0.75
                        WHEN UPPER(r.name) LIKE '%' || $1 || '%' THEN 0.60
                        ELSE 0.40
                    END
                )::numeric as score
                {trigram_select.replace('name', 'r.name') if use_trigram else ', 0 as name_similarity'}
            FROM symbol_search_results r
            WHERE r.expires_at > NOW()
              AND NOT EXISTS (SELECT 1 FROM symbols s WHERE s.symbol = r.symbol AND s.is_active = TRUE)
              AND (
                  UPPER(r.symbol) LIKE $1 || '%'
                  OR UPPER(r.name) LIKE '%' || $1 || '%'
                  {"OR similarity(r.name, $1) > 0.3" if use_trigram else ""}
              )
        )
        SELECT DISTINCT ON (symbol) *
        FROM combined
        WHERE 1=1 {cursor_clause}
        ORDER BY symbol, score DESC, id ASC
        LIMIT ${param_idx}
        """,
        *cursor_params,
    )
    
    # Check if there are more results
    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]
    
    # Build results list
    results = []
    for r in rows:
        # Boost score with trigram similarity if available
        base_score = float(r["score"])
        if use_trigram and r.get("name_similarity"):
            # Blend: 70% base score + 30% trigram similarity
            final_score = base_score * 0.7 + float(r["name_similarity"]) * 0.3
        else:
            final_score = base_score
            
        results.append({
            "id": r["id"],
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "quote_type": (r["quote_type"] or "EQUITY").upper(),
            "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
            "pe_ratio": float(r["pe_ratio"]) if r["pe_ratio"] else None,
            "source": r["source"],
            "score": round(final_score, 3),
        })
    
    # Sort by score DESC, then market_cap DESC for final ordering
    results.sort(key=lambda x: (-x["score"], -(x["market_cap"] or 0)))
    
    # Build next cursor from last result
    next_cursor = None
    if has_more and results:
        last = results[-1]
        next_cursor = SearchCursor(
            score=Decimal(str(last["score"])),
            id=last["id"],
        )
    
    return results, next_cursor


async def _update_last_seen(symbols: list[str]) -> None:
    """Update last_seen_at for search results that were returned."""
    if not symbols:
        return
    
    try:
        await execute(
            """
            UPDATE symbol_search_results 
            SET last_seen_at = NOW()
            WHERE symbol = ANY($1)
            """,
            symbols,
        )
    except Exception as e:
        logger.warning(f"Failed to update last_seen_at: {e}")


async def search_symbols(
    query: str,
    max_results: int = DEFAULT_PAGE_SIZE,
    force_api: bool = False,
    cursor: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search for symbols matching query.
    
    LOCAL-FIRST STRATEGY:
    1. Search local DB first (symbols + cached search results)
    2. Use trigram similarity for fuzzy matching
    3. If enough results, return immediately
    4. If not enough, indicate that fresh API search is available
    
    PAGINATION:
    Uses cursor-based pagination for stable scrolling.
    Pass the `next_cursor` from previous response to get next page.
    
    When force_api=True:
    - Query yfinance API directly
    - Save ALL results to DB for future local searches
    
    Args:
        query: Search query (symbol or company name)
        max_results: Maximum results per page (default 10)
        force_api: Force fresh search from yfinance API
        cursor: Pagination cursor from previous response
        
    Returns:
        Dict with:
        - results: List of matching symbols
        - count: Number of results
        - suggest_fresh_search: Whether to suggest API search
        - search_type: "local" or "api"
        - next_cursor: Cursor for next page (None if no more results)
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return {
            "results": [],
            "count": 0,
            "suggest_fresh_search": False,
            "search_type": "local",
            "next_cursor": None,
        }
    
    # If force_api, query yfinance directly (no pagination for API search)
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
            "next_cursor": None,
        }
    
    # Parse cursor if provided
    parsed_cursor = SearchCursor.decode(cursor) if cursor else None
    
    # Local-first search with pagination
    local_results, next_cursor = await _search_local_db(
        query, 
        limit=max_results,
        cursor=parsed_cursor,
    )
    
    # Update last_seen_at for returned results (async, fire-and-forget)
    cached_symbols = [r["symbol"] for r in local_results if r["source"] == "cached"]
    if cached_symbols:
        await _update_last_seen(cached_symbols)
    
    # Determine if we should suggest fresh search
    # Only suggest on first page when results are sparse
    suggest_fresh = (
        parsed_cursor is None 
        and len(local_results) < MIN_LOCAL_RESULTS_THRESHOLD
    )
    
    return {
        "results": local_results,
        "count": len(local_results),
        "suggest_fresh_search": suggest_fresh,
        "search_type": "local",
        "next_cursor": next_cursor.encode() if next_cursor else None,
    }
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
