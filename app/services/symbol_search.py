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
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, literal, or_, select, update
from sqlalchemy.exc import ProgrammingError

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import StockFundamentals, Symbol, SymbolSearchResult
from app.services.data_providers import get_yfinance_service


logger = get_logger("services.symbol_search")

# Get singleton service instance
_yf_service = get_yfinance_service()

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
    def decode(cls, cursor_str: str) -> SearchCursor | None:
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
    cursor: SearchCursor | None = None,
    use_trigram: bool = True,
) -> tuple[list[dict[str, Any]], SearchCursor | None]:
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
    fetch_limit = limit * 5

    def _base_score(symbol: str, name: str | None) -> float:
        symbol_upper = (symbol or "").upper()
        name_upper = (name or "").upper()
        if symbol_upper == normalized:
            return 1.00
        if symbol_upper.startswith(normalized):
            return 0.90
        if normalized in name_upper:
            return 0.70
        return 0.50

    def _build_symbol_stmt(use_similarity: bool):
        columns = [
            Symbol.id.label("id"),
            Symbol.symbol.label("symbol"),
            Symbol.name.label("name"),
            Symbol.sector.label("sector"),
            Symbol.symbol_type.label("quote_type"),
            Symbol.market_cap.label("market_cap"),
        ]
        if use_similarity:
            columns.append(func.similarity(Symbol.name, normalized).label("name_similarity"))
        else:
            columns.append(literal(0).label("name_similarity"))

        conditions = [
            Symbol.symbol.ilike(f"{normalized}%"),
            Symbol.name.ilike(f"%{normalized}%"),
        ]
        if use_similarity:
            conditions.append(func.similarity(Symbol.name, normalized) > 0.3)

        return (
            select(*columns)
            .where(Symbol.is_active == True)
            .where(or_(*conditions))
            .limit(fetch_limit)
        )

    def _build_cached_stmt(use_similarity: bool):
        columns = [
            SymbolSearchResult.id.label("id"),
            SymbolSearchResult.symbol.label("symbol"),
            SymbolSearchResult.name.label("name"),
            SymbolSearchResult.sector.label("sector"),
            SymbolSearchResult.quote_type.label("quote_type"),
            SymbolSearchResult.market_cap.label("market_cap"),
            SymbolSearchResult.confidence_score.label("confidence_score"),
        ]
        if use_similarity:
            columns.append(func.similarity(SymbolSearchResult.name, normalized).label("name_similarity"))
        else:
            columns.append(literal(0).label("name_similarity"))

        conditions = [
            SymbolSearchResult.symbol.ilike(f"{normalized}%"),
            SymbolSearchResult.name.ilike(f"%{normalized}%"),
        ]
        if use_similarity:
            conditions.append(func.similarity(SymbolSearchResult.name, normalized) > 0.3)

        active_symbols = select(Symbol.symbol).where(Symbol.is_active == True)

        return (
            select(*columns)
            .where(SymbolSearchResult.expires_at > datetime.now(UTC))
            .where(~SymbolSearchResult.symbol.in_(active_symbols))
            .where(or_(*conditions))
            .limit(fetch_limit)
        )

    async with get_session() as session:
        try:
            symbol_rows = (await session.execute(_build_symbol_stmt(use_trigram))).mappings().all()
            cached_rows = (await session.execute(_build_cached_stmt(use_trigram))).mappings().all()
        except ProgrammingError as exc:
            if use_trigram:
                logger.warning(f"pg_trgm unavailable, falling back to basic search: {exc}")
                use_trigram = False
                symbol_rows = (await session.execute(_build_symbol_stmt(False))).mappings().all()
                cached_rows = (await session.execute(_build_cached_stmt(False))).mappings().all()
            else:
                raise

    results = []
    seen_symbols = set()

    for r in symbol_rows:
        base = _base_score(r["symbol"], r["name"])
        name_similarity = float(r.get("name_similarity") or 0)
        final_score = base * 0.7 + name_similarity * 0.3 if use_trigram else base
        results.append(
            {
                "id": r["id"],
                "symbol": r["symbol"],
                "name": r["name"],
                "sector": r["sector"],
                "quote_type": (r["quote_type"] or "EQUITY").upper(),
                "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
                "pe_ratio": None,
                "source": "local",
                "score": round(final_score, 3),
            }
        )
        seen_symbols.add(r["symbol"])

    for r in cached_rows:
        if r["symbol"] in seen_symbols:
            continue
        base = float(r.get("confidence_score") or _base_score(r["symbol"], r["name"]))
        name_similarity = float(r.get("name_similarity") or 0)
        final_score = base * 0.7 + name_similarity * 0.3 if use_trigram else base
        results.append(
            {
                "id": r["id"],
                "symbol": r["symbol"],
                "name": r["name"],
                "sector": r["sector"],
                "quote_type": (r["quote_type"] or "EQUITY").upper(),
                "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
                "pe_ratio": None,
                "source": "cached",
                "score": round(final_score, 3),
            }
        )

    # Sort by score DESC, then market_cap DESC for final ordering
    results.sort(key=lambda x: (-x["score"], -(x["market_cap"] or 0), x["id"]))

    # Apply cursor filtering after sorting
    if cursor:
        results = [
            r for r in results
            if r["score"] < float(cursor.score)
            or (r["score"] == float(cursor.score) and r["id"] > cursor.id)
        ]

    # Fetch one extra to detect if there are more results
    has_more = len(results) > limit
    if has_more:
        results = results[:limit]

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
        async with get_session() as session:
            await session.execute(
                update(SymbolSearchResult)
                .where(SymbolSearchResult.symbol.in_(symbols))
                .values(last_seen_at=datetime.now(UTC))
            )
            await session.commit()
    except Exception as e:
        logger.warning(f"Failed to update last_seen_at: {e}")


async def search_symbols(
    query: str,
    max_results: int = DEFAULT_PAGE_SIZE,
    force_api: bool = False,
    cursor: str | None = None,
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
        api_results = await _yf_service.search_tickers(
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


async def lookup_symbol(symbol: str) -> dict[str, Any] | None:
    """
    Lookup a specific symbol to validate and get info.
    
    Uses unified yfinance service for API calls.
    
    Args:
        symbol: Stock symbol to lookup
        
    Returns:
        Symbol info dict or None if invalid/not found
    """
    symbol = symbol.strip().upper()

    # Check local symbols with fundamentals using JOIN
    async with get_session() as session:
        result = await session.execute(
            select(
                Symbol.symbol,
                Symbol.name,
                Symbol.sector,
                Symbol.symbol_type,
                Symbol.market_cap,
                StockFundamentals.pe_ratio,
                StockFundamentals.forward_pe,
                StockFundamentals.recommendation,
                StockFundamentals.target_mean_price,
            )
            .outerjoin(StockFundamentals, Symbol.symbol == StockFundamentals.symbol)
            .where(Symbol.symbol == symbol, Symbol.is_active == True)
        )
        row = result.one_or_none()

    if row:
        return {
            "symbol": row.symbol,
            "name": row.name,
            "sector": row.sector,
            "quote_type": (row.symbol_type or "EQUITY").upper(),
            "market_cap": float(row.market_cap) if row.market_cap else None,
            "pe_ratio": float(row.pe_ratio) if row.pe_ratio else None,
            "forward_pe": float(row.forward_pe) if row.forward_pe else None,
            "recommendation": row.recommendation,
            "target_mean_price": float(row.target_mean_price) if row.target_mean_price else None,
            "source": "local",
            "valid": True,
        }

    # Check cached search results
    async with get_session() as session:
        result = await session.execute(
            select(SymbolSearchResult).where(
                SymbolSearchResult.symbol == symbol,
                SymbolSearchResult.expires_at > datetime.now(UTC),
            )
        )
        cached_row = result.scalar_one_or_none()

    if cached_row:
        return {
            "symbol": cached_row.symbol,
            "name": cached_row.name,
            "sector": cached_row.sector,
            "quote_type": (cached_row.quote_type or "EQUITY").upper(),
            "market_cap": float(cached_row.market_cap) if cached_row.market_cap else None,
            "source": "cached",
            "valid": True,
        }

    # Fall back to yfinance validation
    return await _yf_service.validate_symbol(symbol)


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

    # _search_local_db returns (results, next_cursor) tuple
    results, _ = await _search_local_db(partial, limit=limit)
    return [{"symbol": r["symbol"], "name": r["name"]} for r in results]


async def get_known_symbols(limit: int = 100) -> list[str]:
    """Get list of all known active symbols."""
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.symbol)
            .where(Symbol.is_active == True)
            .order_by(Symbol.symbol)
            .limit(limit)
        )
        return [r[0] for r in result.all()]
