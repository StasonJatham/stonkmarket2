"""FinanceDatabase integration service.

Provides local financial universe search and symbol validation using
the FinanceDatabase package (~130K symbols across asset classes).

FEATURES:
- Ingest from financedatabase package to local PostgreSQL
- Fast local search with trigram fuzzy matching
- Multi-asset search (equities, ETFs, indices, funds, crypto)
- Sector/industry/country faceting
- ISIN/CUSIP/FIGI identifier resolution
- Symbol validation without external API

USAGE:
    from app.services.financedatabase_service import (
        ingest_universe,
        search_universe,
        get_by_symbol,
        get_facet_options,
    )
    
    # Weekly sync (called by job)
    stats = await ingest_universe()
    
    # Fast local search
    results = await search_universe("apple", asset_classes=["equity", "etf"])
    
    # Exact symbol lookup
    info = await get_by_symbol("AAPL")
    
    # Get available sectors for faceting
    sectors = await get_facet_options("sector")
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, literal, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import ProgrammingError

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import FinancialUniverse


logger = get_logger("services.financedatabase")

# Supported asset classes (yfinance/yahooquery compatible)
SUPPORTED_ASSET_CLASSES = ["equity", "etf", "fund", "index"]


# =============================================================================
# FALLBACK SEARCH (direct package access when DB is empty)
# =============================================================================


async def _fallback_get_by_symbol(symbol: str) -> dict[str, Any] | None:
    """Fallback lookup from financedatabase package when DB is empty.
    
    Searches across equity, etf, index, fund asset classes.
    """
    import financedatabase as fd
    
    symbol_upper = symbol.strip().upper()
    
    # Check each supported asset class
    for asset_class, loader_cls in [
        ("equity", fd.Equities),
        ("etf", fd.ETFs),
        ("index", fd.Indices),
        ("fund", fd.Funds),
    ]:
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, lambda: loader_cls().select())
            
            if symbol_upper in df.index:
                row = df.loc[symbol_upper]
                return {
                    "symbol": symbol_upper,
                    "name": row.get("name"),
                    "asset_class": asset_class,
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                    "industry_group": row.get("industry_group"),
                    "country": row.get("country"),
                    "exchange": row.get("exchange"),
                    "market": row.get("market"),
                    "currency": row.get("currency"),
                    "category": row.get("category"),
                    "category_group": row.get("category_group"),
                    "family": row.get("family"),
                    "market_cap_category": row.get("market_cap"),
                    "isin": row.get("isin"),
                    "cusip": row.get("cusip"),
                    "figi": row.get("figi"),
                    "summary": row.get("summary"),
                    "source": "financedatabase_package",
                }
        except Exception as e:
            logger.debug(f"Fallback search failed for {asset_class}: {e}")
            continue
    
    return None


async def _fallback_search(
    query: str,
    asset_classes: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fallback search in financedatabase package when DB is empty.
    
    Uses direct package access as fallback.
    """
    import financedatabase as fd
    
    if asset_classes is None:
        asset_classes = ["equity", "etf", "index"]
    
    # Filter to supported only
    asset_classes = [ac for ac in asset_classes if ac in SUPPORTED_ASSET_CLASSES]
    
    query_upper = query.strip().upper()
    results: list[dict[str, Any]] = []
    
    loader_map = {
        "equity": fd.Equities,
        "etf": fd.ETFs,
        "index": fd.Indices,
        "fund": fd.Funds,
    }
    
    for asset_class in asset_classes:
        if asset_class not in loader_map:
            continue
            
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, lambda ac=asset_class: loader_map[ac]().select())
            df = df.reset_index().rename(columns={"index": "symbol"})
            
            # Search by symbol prefix or name contains
            matches = df[
                df["symbol"].str.upper().str.startswith(query_upper) |
                df["name"].fillna("").str.upper().str.contains(query_upper, regex=False)
            ].head(limit)
            
            for _, row in matches.iterrows():
                symbol = str(row["symbol"]).upper()
                name = row.get("name")
                
                # Score: exact match > prefix > contains
                if symbol == query_upper:
                    score = 1.0
                elif symbol.startswith(query_upper):
                    score = 0.9
                else:
                    score = 0.6
                
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "asset_class": asset_class,
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                    "country": row.get("country"),
                    "exchange": row.get("exchange"),
                    "category": row.get("category"),
                    "market_cap_category": row.get("market_cap"),
                    "score": score,
                    "source": "financedatabase_package",
                })
        except Exception as e:
            logger.debug(f"Fallback search failed for {asset_class}: {e}")
            continue
    
    # Sort by score descending
    results.sort(key=lambda x: (-x["score"], len(x["symbol"])))
    return results[:limit]


# =============================================================================
# INGESTION
# =============================================================================


def _load_financedatabase_class(asset_class: str) -> pd.DataFrame:
    """Load data from financedatabase package for a given asset class.
    
    Runs synchronously - call from thread pool for async.
    """
    import financedatabase as fd
    
    class_map = {
        "equity": fd.Equities,
        "etf": fd.ETFs,
        "fund": fd.Funds,
        "index": fd.Indices,
        "crypto": fd.Cryptos,
        "currency": fd.Currencies,
        "moneymarket": fd.Moneymarkets,
    }
    
    if asset_class not in class_map:
        raise ValueError(f"Unknown asset class: {asset_class}")
    
    loader = class_map[asset_class]()
    df = loader.select()
    
    # The index is the symbol
    df = df.reset_index()
    df = df.rename(columns={"index": "symbol"})
    
    return df


def _normalize_dataframe(df: pd.DataFrame, asset_class: str) -> list[dict[str, Any]]:
    """Normalize DataFrame columns to match FinancialUniverse schema.
    
    Different asset classes have different column schemas:
    - Equities: sector, industry_group, industry, country, market, isin, cusip, figi, etc.
    - ETFs: category_group, category, family
    - Funds: category_group, category, family
    - Indices: category_group, category
    - Cryptos/Currencies/Moneymarkets: minimal fields
    """
    records = []
    
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "")).strip()
        if not symbol or len(symbol) > 30:
            continue
        
        record = {
            "symbol": symbol,
            "name": _clean_str(row.get("name"), 255),
            "asset_class": asset_class,
            "currency": _clean_str(row.get("currency"), 10),
            "exchange": _clean_str(row.get("exchange"), 50),
            "summary": _clean_str(row.get("summary"), 5000),  # Truncate long summaries
            "is_active": True,
            "source_updated_at": datetime.now(UTC),
        }
        
        # Asset-class specific fields
        if asset_class == "equity":
            record.update({
                "sector": _clean_str(row.get("sector"), 100),
                "industry_group": _clean_str(row.get("industry_group"), 150),
                "industry": _clean_str(row.get("industry"), 150),
                "country": _clean_str(row.get("country"), 100),
                "market": _clean_str(row.get("market"), 100),
                "market_cap_category": _clean_str(row.get("market_cap"), 20),  # "Large Cap", etc.
                "isin": _clean_str(row.get("isin"), 12),
                "cusip": _clean_str(row.get("cusip"), 9),
                "figi": _clean_str(row.get("figi"), 12),
                "composite_figi": _clean_str(row.get("composite_figi"), 12),
            })
        elif asset_class in ("etf", "fund"):
            record.update({
                "category_group": _clean_str(row.get("category_group"), 100),
                "category": _clean_str(row.get("category"), 100),
                "family": _clean_str(row.get("family"), 100),
            })
        elif asset_class == "index":
            record.update({
                "category_group": _clean_str(row.get("category_group"), 100),
                "category": _clean_str(row.get("category"), 100),
            })
        
        records.append(record)
    
    return records


def _clean_str(value: Any, max_len: int) -> str | None:
    """Clean and truncate string value."""
    if pd.isna(value) or value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return None
    return s[:max_len] if len(s) > max_len else s


async def ingest_asset_class(asset_class: str) -> int:
    """Ingest a single asset class from financedatabase.
    
    Uses upsert (INSERT ON CONFLICT UPDATE) to handle existing symbols.
    
    Args:
        asset_class: One of equity, etf, fund, index, crypto, currency, moneymarket
        
    Returns:
        Number of records processed
    """
    logger.info(f"Loading {asset_class} data from financedatabase...")
    
    # Load data in thread pool (synchronous pandas I/O)
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _load_financedatabase_class, asset_class)
    
    logger.info(f"Loaded {len(df)} {asset_class} records, normalizing...")
    records = _normalize_dataframe(df, asset_class)
    
    if not records:
        logger.warning(f"No valid records for {asset_class}")
        return 0
    
    # Deduplicate by symbol (keep first occurrence)
    # financedatabase can have duplicates for same ticker on different exchanges
    seen_symbols: set[str] = set()
    unique_records: list[dict[str, Any]] = []
    for rec in records:
        sym = rec["symbol"]
        if sym not in seen_symbols:
            seen_symbols.add(sym)
            unique_records.append(rec)
    
    logger.info(
        f"Deduplicated {len(records)} -> {len(unique_records)} unique symbols for {asset_class}"
    )
    records = unique_records

    # Batch upsert - smaller batch for equity (has long summaries)
    batch_size = 100 if asset_class == "equity" else 500
    processed = 0

    async with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            stmt = pg_insert(FinancialUniverse).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={
                    "name": stmt.excluded.name,
                    "asset_class": stmt.excluded.asset_class,
                    "sector": stmt.excluded.sector,
                    "industry_group": stmt.excluded.industry_group,
                    "industry": stmt.excluded.industry,
                    "category_group": stmt.excluded.category_group,
                    "category": stmt.excluded.category,
                    "family": stmt.excluded.family,
                    "exchange": stmt.excluded.exchange,
                    "market": stmt.excluded.market,
                    "country": stmt.excluded.country,
                    "currency": stmt.excluded.currency,
                    "isin": stmt.excluded.isin,
                    "cusip": stmt.excluded.cusip,
                    "figi": stmt.excluded.figi,
                    "composite_figi": stmt.excluded.composite_figi,
                    "market_cap_category": stmt.excluded.market_cap_category,
                    "summary": stmt.excluded.summary,
                    "is_active": stmt.excluded.is_active,
                    "source_updated_at": stmt.excluded.source_updated_at,
                    "updated_at": func.now(),
                },
            )
            
            await session.execute(stmt)
            processed += len(batch)
            
            if processed % 5000 == 0:
                logger.info(f"  {asset_class}: processed {processed}/{len(records)}")
        
        await session.commit()
    
    logger.info(f"Ingested {processed} {asset_class} records")
    return processed


async def ingest_universe(
    asset_classes: list[str] | None = None,
    mark_missing_inactive: bool = True,
) -> dict[str, int]:
    """Ingest all asset classes from financedatabase.
    
    Args:
        asset_classes: List of asset classes to ingest (default: supported only)
        mark_missing_inactive: Mark symbols not in new data as inactive
        
    Returns:
        Dict of {asset_class: count} ingested
        
    Note: Only equity, etf, fund, index are supported (yfinance/yahooquery compatible).
    Crypto, currency, moneymarket are NOT supported.
    """
    # Only ingest asset classes that yfinance/yahooquery can provide data for
    SUPPORTED_ASSET_CLASSES = ["equity", "etf", "fund", "index"]
    
    if asset_classes is None:
        asset_classes = SUPPORTED_ASSET_CLASSES
    else:
        # Filter to only supported classes
        asset_classes = [ac for ac in asset_classes if ac in SUPPORTED_ASSET_CLASSES]
        
    if not asset_classes:
        logger.warning("No supported asset classes requested")
        return {}
    
    logger.info(f"Starting universe ingestion for: {asset_classes}")
    
    # Record start time for marking stale entries
    ingestion_start = datetime.now(UTC)
    
    stats: dict[str, int] = {}
    
    for asset_class in asset_classes:
        try:
            count = await ingest_asset_class(asset_class)
            stats[asset_class] = count
        except Exception as e:
            logger.error(f"Failed to ingest {asset_class}: {e}")
            stats[asset_class] = 0
    
    # Mark symbols not updated during this ingestion as inactive
    # Uses date comparison instead of IN clause to avoid parameter limits
    if mark_missing_inactive:
        from sqlalchemy import update
        
        async with get_session() as session:
            # Symbols updated before ingestion started are stale
            result = await session.execute(
                update(FinancialUniverse)
                .where(FinancialUniverse.source_updated_at < ingestion_start)
                .values(is_active=False)
            )
            stale_count = result.rowcount
            await session.commit()
            
            if stale_count > 0:
                logger.info(f"Marked {stale_count} stale symbols as inactive")
    
    total = sum(stats.values())
    logger.info(f"Universe ingestion complete: {total} total records across {len(asset_classes)} asset classes")
    
    return stats


# =============================================================================
# SEARCH
# =============================================================================


async def search_universe(
    query: str,
    asset_classes: list[str] | None = None,
    country: str | None = None,
    sector: str | None = None,
    limit: int = 20,
    use_trigram: bool = True,
) -> list[dict[str, Any]]:
    """Search the financial universe for symbols matching query.
    
    Uses multiple search strategies:
    1. Exact symbol match (highest score)
    2. Symbol prefix match
    3. Name contains (case-insensitive)
    4. Trigram similarity on name (fuzzy match)
    
    Args:
        query: Search query (symbol or name)
        asset_classes: Filter to specific asset classes (default: equity, etf, index)
        country: Filter by country
        sector: Filter by sector
        limit: Max results to return
        use_trigram: Whether to use trigram similarity (requires pg_trgm extension)
        
    Returns:
        List of matching records with relevance scores
    """
    if not query or len(query.strip()) < 1:
        return []
    
    normalized = query.strip().upper()
    
    if asset_classes is None:
        # Default to most useful asset classes for stock search
        asset_classes = ["equity", "etf", "index"]
    
    def _base_score(symbol: str, name: str | None) -> float:
        """Calculate base relevance score."""
        symbol_upper = (symbol or "").upper()
        name_upper = (name or "").upper()
        
        if symbol_upper == normalized:
            return 1.00  # Exact symbol match
        if symbol_upper.startswith(normalized):
            return 0.90  # Symbol prefix
        if normalized in symbol_upper:
            return 0.80  # Symbol contains
        if name_upper.startswith(normalized):
            return 0.75  # Name starts with
        if normalized in name_upper:
            return 0.60  # Name contains
        return 0.40  # Trigram/fuzzy match only
    
    async with get_session() as session:
        # Build query with optional trigram
        columns = [
            FinancialUniverse.id,
            FinancialUniverse.symbol,
            FinancialUniverse.name,
            FinancialUniverse.asset_class,
            FinancialUniverse.sector,
            FinancialUniverse.industry,
            FinancialUniverse.country,
            FinancialUniverse.exchange,
            FinancialUniverse.category,
            FinancialUniverse.market_cap_category,
        ]
        
        if use_trigram:
            columns.append(func.similarity(FinancialUniverse.name, normalized).label("name_similarity"))
        else:
            columns.append(literal(0).label("name_similarity"))
        
        # Build WHERE conditions
        conditions = [
            FinancialUniverse.is_active == True,
            FinancialUniverse.asset_class.in_(asset_classes),
        ]
        
        # Search conditions (OR)
        search_conditions = [
            FinancialUniverse.symbol.ilike(f"{normalized}%"),
            FinancialUniverse.name.ilike(f"%{normalized}%"),
        ]
        
        if use_trigram:
            search_conditions.append(func.similarity(FinancialUniverse.name, normalized) > 0.2)
        
        conditions.append(or_(*search_conditions))
        
        # Optional filters
        if country:
            conditions.append(FinancialUniverse.country == country)
        if sector:
            conditions.append(FinancialUniverse.sector == sector)
        
        try:
            stmt = (
                select(*columns)
                .where(*conditions)
                .limit(limit * 3)  # Fetch extra for scoring/sorting
            )
            result = await session.execute(stmt)
            rows = result.mappings().all()
        except ProgrammingError as e:
            if use_trigram and "similarity" in str(e).lower():
                # Trigram extension not available, fall back
                logger.warning("pg_trgm extension not available, using basic search")
                return await search_universe(
                    query, asset_classes, country, sector, limit, use_trigram=False
                )
            raise
    
    # Score and sort results
    results = []
    for row in rows:
        base_score = _base_score(row["symbol"], row["name"])
        name_sim = float(row.get("name_similarity") or 0)
        final_score = base_score * 0.7 + name_sim * 0.3 if use_trigram else base_score
        
        results.append({
            "symbol": row["symbol"],
            "name": row["name"],
            "asset_class": row["asset_class"],
            "sector": row["sector"],
            "industry": row["industry"],
            "country": row["country"],
            "exchange": row["exchange"],
            "category": row["category"],
            "market_cap_category": row["market_cap_category"],
            "score": round(final_score, 3),
            "source": "universe",
        })
    
    # Sort by score descending, then by symbol length (shorter = more relevant)
    results.sort(key=lambda x: (-x["score"], len(x["symbol"])))
    
    final_results = results[:limit]
    
    # Fallback to package search if DB returned no results
    if not final_results:
        logger.debug(f"No DB results for '{query}', falling back to package search")
        final_results = await _fallback_search(query, asset_classes, limit)
    
    return final_results


async def get_by_symbol(symbol: str) -> dict[str, Any] | None:
    """Get a single symbol from the universe.
    
    Args:
        symbol: Symbol to look up (case-insensitive)
        
    Returns:
        Symbol record or None if not found
    """
    symbol_upper = symbol.strip().upper()
    
    async with get_session() as session:
        result = await session.execute(
            select(FinancialUniverse).where(
                FinancialUniverse.symbol == symbol_upper,
                FinancialUniverse.is_active == True,
            )
        )
        row = result.scalar_one_or_none()
    
    if not row:
        # Fallback to financedatabase package directly
        return await _fallback_get_by_symbol(symbol_upper)
    
    return {
        "symbol": row.symbol,
        "name": row.name,
        "asset_class": row.asset_class,
        "sector": row.sector,
        "industry": row.industry,
        "industry_group": row.industry_group,
        "country": row.country,
        "exchange": row.exchange,
        "market": row.market,
        "currency": row.currency,
        "category": row.category,
        "category_group": row.category_group,
        "family": row.family,
        "market_cap_category": row.market_cap_category,
        "isin": row.isin,
        "cusip": row.cusip,
        "figi": row.figi,
        "summary": row.summary,
        "source": "universe",
    }


async def get_by_isin(isin: str) -> dict[str, Any] | None:
    """Look up a symbol by ISIN.
    
    Args:
        isin: ISIN identifier (e.g., "US0378331005" for Apple)
        
    Returns:
        Symbol record or None if not found
    """
    isin_upper = isin.strip().upper()
    
    async with get_session() as session:
        result = await session.execute(
            select(FinancialUniverse).where(
                FinancialUniverse.isin == isin_upper,
                FinancialUniverse.is_active == True,
            )
        )
        row = result.scalar_one_or_none()
    
    if not row:
        return None
    
    return await get_by_symbol(row.symbol)


# =============================================================================
# FACETING / OPTIONS
# =============================================================================


async def get_facet_options(
    field: str,
    asset_classes: list[str] | None = None,
) -> list[str]:
    """Get distinct values for a facet field.
    
    Args:
        field: Field name (sector, industry, country, exchange, category, etc.)
        asset_classes: Optional filter by asset class
        
    Returns:
        List of distinct values
    """
    field_map = {
        "sector": FinancialUniverse.sector,
        "industry": FinancialUniverse.industry,
        "industry_group": FinancialUniverse.industry_group,
        "country": FinancialUniverse.country,
        "exchange": FinancialUniverse.exchange,
        "category": FinancialUniverse.category,
        "category_group": FinancialUniverse.category_group,
        "family": FinancialUniverse.family,
        "market_cap_category": FinancialUniverse.market_cap_category,
        "asset_class": FinancialUniverse.asset_class,
    }
    
    if field not in field_map:
        raise ValueError(f"Unknown facet field: {field}. Valid: {list(field_map.keys())}")
    
    column = field_map[field]
    
    async with get_session() as session:
        conditions = [
            FinancialUniverse.is_active == True,
            column.isnot(None),
        ]
        
        if asset_classes:
            conditions.append(FinancialUniverse.asset_class.in_(asset_classes))
        
        result = await session.execute(
            select(column)
            .where(*conditions)
            .distinct()
            .order_by(column)
        )
        
        return [r[0] for r in result.all()]


async def get_universe_stats() -> dict[str, Any]:
    """Get statistics about the financial universe.
    
    Returns:
        Dict with counts by asset class and last update time
    """
    async with get_session() as session:
        # Count by asset class
        result = await session.execute(
            select(
                FinancialUniverse.asset_class,
                func.count(FinancialUniverse.id).label("count"),
            )
            .where(FinancialUniverse.is_active == True)
            .group_by(FinancialUniverse.asset_class)
        )
        counts = {r[0]: r[1] for r in result.all()}
        
        # Last update
        result = await session.execute(
            select(func.max(FinancialUniverse.updated_at))
        )
        last_updated = result.scalar()
    
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "last_updated": last_updated.isoformat() if last_updated else None,
    }
