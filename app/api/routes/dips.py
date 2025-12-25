"""Dip analysis routes - PostgreSQL async version."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import List, Tuple

from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import BaseModel

from app.api.dependencies import require_admin, require_user
from app.cache.cache import Cache
from app.cache.http_cache import (
    CacheableResponse,
    CachePresets,
    check_if_none_match,
    generate_etag,
    NotModifiedResponse,
)
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.security import TokenData
from app.database.connection import fetch_all, fetch_one
from app.dipfinder.service import get_dipfinder_service  # For chart price data
from app.repositories import symbols as symbol_repo
from app.schemas.dips import ChartPoint, DipStateResponse, RankingEntry, StockInfo
from app.services.stock_info import get_stock_info, get_stock_info_async
from app.services.runtime_settings import get_runtime_setting, get_cache_ttl


router = APIRouter()

# Caches - TTLs are applied dynamically from runtime settings when setting values
_ranking_cache = Cache(prefix="ranking", default_ttl=300)
_chart_cache = Cache(prefix="chart", default_ttl=600)
_info_cache = Cache(prefix="stockinfo", default_ttl=300)


async def invalidate_stock_info_cache(symbol: str) -> None:
    """Invalidate the stock info cache for a symbol."""
    await _info_cache.delete(symbol.upper())


def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


class BenchmarkInfo(BaseModel):
    """Public benchmark information."""
    id: str
    symbol: str
    name: str
    description: str | None = None


async def _build_ranking(
    force_refresh: bool = False,
) -> Tuple[List[RankingEntry], List[RankingEntry]]:
    """
    Build ranking data from dip_state table and stock info.
    
    Uses ATH-based dip calculations from dip_state table.
    
    Returns:
        Tuple of (all_entries, filtered_entries) where filtered_entries
        only includes stocks meeting their individual dip thresholds.
    """
    # Get all dip states with symbol info in one query
    rows = await fetch_all(
        """
        SELECT ds.symbol, ds.current_price, ds.ath_price, ds.dip_percentage,
               ds.dip_start_date, ds.first_seen, ds.last_updated,
               s.name, s.sector, s.min_dip_pct, s.symbol_type
        FROM dip_state ds
        JOIN symbols s ON s.symbol = ds.symbol
        WHERE s.is_active = true
        ORDER BY ds.dip_percentage DESC
        """
    )
    
    if not rows:
        return [], []

    # Fetch stock info for enrichment (52w low, market cap, etc)
    symbols_list = [r["symbol"] for r in rows]
    
    async def get_info(symbol: str):
        return symbol, await get_stock_info_async(symbol)
    
    info_tasks = [get_info(s) for s in symbols_list]
    info_results = await asyncio.gather(*info_tasks, return_exceptions=True)
    stock_info_map = {}
    for result in info_results:
        if isinstance(result, tuple):
            sym, info = result
            if info:
                stock_info_map[sym] = info

    # Build ranking entries from dip_state
    all_entries = []
    filtered_entries = []
    
    for row in rows:
        symbol = row["symbol"]
        info = stock_info_map.get(symbol, {})
        
        # Calculate days in dip - prefer dip_start_date, fall back to first_seen
        days_in_dip = 0
        from datetime import datetime, timezone, date as date_type
        
        if row["dip_start_date"]:
            dip_start = row["dip_start_date"]
            if isinstance(dip_start, date_type):
                days_in_dip = (date_type.today() - dip_start).days
        elif row["first_seen"]:
            first_seen = row["first_seen"]
            if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=timezone.utc)
            days_in_dip = (datetime.now(timezone.utc) - first_seen).days
        
        dip_pct = float(row["dip_percentage"]) if row["dip_percentage"] else 0
        
        entry = RankingEntry(
            symbol=symbol,
            name=row["name"] or info.get("name") or symbol,
            depth=dip_pct / 100,  # Convert percentage to decimal
            days_since_dip=days_in_dip,
            last_price=float(row["current_price"]) if row["current_price"] else None,
            previous_close=info.get("previous_close"),
            change_percent=info.get("change_percent"),
            high_52w=float(row["ath_price"]) if row["ath_price"] else info.get("fifty_two_week_high"),
            low_52w=info.get("fifty_two_week_low"),
            market_cap=info.get("market_cap"),
            pe_ratio=info.get("pe_ratio"),
            volume=info.get("avg_volume"),
            sector=row["sector"] or info.get("sector"),
            symbol_type=row.get("symbol_type", "stock"),
            updated_at=row["last_updated"].isoformat() if row["last_updated"] else None,
        )
        all_entries.append(entry)
        
        # Check if this stock meets its individual dip threshold
        min_dip_threshold = float(row["min_dip_pct"]) if row["min_dip_pct"] else 0.10
        if dip_pct / 100 >= min_dip_threshold:
            filtered_entries.append(entry)

    # Already sorted by dip_percentage DESC in query, but ensure consistency
    all_entries.sort(key=lambda x: x.depth, reverse=True)
    filtered_entries.sort(key=lambda x: x.depth, reverse=True)

    return all_entries, filtered_entries


@router.get(
    "/benchmarks",
    response_model=List[BenchmarkInfo],
    summary="Get available benchmarks",
    description="Get list of available benchmarks for comparison. Public endpoint.",
)
async def get_benchmarks() -> CacheableResponse:
    """Get available benchmarks from runtime settings.
    
    Returns with short cache headers since benchmarks rarely change
    but users expect immediate updates when they do.
    """
    benchmarks = get_runtime_setting("benchmarks", [])
    data = [BenchmarkInfo(**b).model_dump() for b in benchmarks]
    etag = generate_etag(data)
    
    return CacheableResponse(
        data,
        etag=etag,
        max_age=60,  # 1 minute cache
        stale_while_revalidate=60,  # Allow stale for 1 more minute while revalidating
    )


@router.get(
    "/ranking",
    summary="Get dip ranking",
    description="Get stocks ranked by dip depth. Use show_all=true to include stocks not in active dip.",
)
async def get_ranking(
    request: Request,
    show_all: bool = False,
) -> CacheableResponse:
    """Get ranked list of stocks. By default only shows stocks in meaningful dip (>10%).
    
    Returns with HTTP caching headers for browser caching.
    """
    cache_key = f"all:{show_all}"

    # Try cache first
    cached = await _ranking_cache.get(cache_key)
    if cached:
        ranking = [RankingEntry(**item) for item in cached]
        data = [r.model_dump(mode="json") for r in ranking]
        etag = generate_etag(data)
        
        # Check for conditional request (If-None-Match)
        if check_if_none_match(request, etag):
            return NotModifiedResponse(etag=etag)
        
        return CacheableResponse(
            data,
            etag=etag,
            **CachePresets.RANKING,
        )

    # Build ranking (use shared helper)
    all_entries, filtered_entries = await _build_ranking(force_refresh=False)
    
    # Select appropriate result based on show_all
    ranking = all_entries if show_all else filtered_entries
    data = [r.model_dump(mode="json") for r in ranking]

    # Cache the result with dynamic TTL
    await _ranking_cache.set(cache_key, data, ttl=get_cache_ttl("ranking"))
    
    etag = generate_etag(data)
    return CacheableResponse(
        data,
        etag=etag,
        **CachePresets.RANKING,
    )


@router.post(
    "/ranking/refresh",
    response_model=List[RankingEntry],
    summary="Refresh dip ranking",
    description="Force refresh of dip ranking (admin only).",
    responses={
        403: {"description": "Admin required"},
    },
)
async def refresh_ranking(
    admin: TokenData = Depends(require_admin),
) -> List[RankingEntry]:
    """Force refresh of dip ranking (fetches new data)."""
    # Invalidate both cache keys
    await _ranking_cache.delete("all:True")
    await _ranking_cache.delete("all:False")

    # Build ranking with force refresh (use shared helper)
    all_entries, filtered_entries = await _build_ranking(force_refresh=True)

    # Cache both filtered and unfiltered results with dynamic TTL
    ttl = get_cache_ttl("ranking")
    await _ranking_cache.set("all:True", [r.model_dump() for r in all_entries], ttl=ttl)
    await _ranking_cache.set("all:False", [r.model_dump() for r in filtered_entries], ttl=ttl)

    return all_entries


@router.get(
    "/states",
    response_model=List[DipStateResponse],
    summary="Get all dip states",
    description="Get current dip state for all tracked symbols.",
)
async def get_states() -> List[DipStateResponse]:
    """Get current dip state for all tracked symbols."""
    # Get dip states from database
    rows = await fetch_all(
        """
        SELECT symbol, ref_high, days_below, last_price, updated_at
        FROM dip_state
        ORDER BY symbol
        """
    )

    return [
        DipStateResponse(
            symbol=row["symbol"],
            ref_high=float(row["ref_high"]) if row["ref_high"] else 0.0,
            days_below=row["days_below"] or 0,
            last_price=float(row["last_price"]) if row["last_price"] else 0.0,
            dip_depth=(
                (float(row["last_price"]) - float(row["ref_high"]))
                / float(row["ref_high"])
            )
            if row["ref_high"] and float(row["ref_high"]) > 0
            else 0.0,
            updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
        )
        for row in rows
    ]


@router.get(
    "/{symbol}/state",
    response_model=DipStateResponse,
    summary="Get symbol dip state",
    description="Get current dip state for a specific symbol.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def get_symbol_state(
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
) -> DipStateResponse:
    """Get dip state for a specific symbol."""
    # Get dip state from database
    row = await fetch_one(
        "SELECT symbol, ref_high, days_below, last_price, updated_at FROM dip_state WHERE symbol = $1",
        symbol,
    )

    if row is None:
        raise NotFoundError(
            message=f"No dip state available for '{symbol}'",
            details={"symbol": symbol},
        )

    ref_high = float(row["ref_high"]) if row["ref_high"] else 0.0
    last_price = float(row["last_price"]) if row["last_price"] else 0.0
    dip_depth = ((last_price - ref_high) / ref_high) if ref_high > 0 else 0.0

    return DipStateResponse(
        symbol=row["symbol"],
        ref_high=ref_high,
        days_below=row["days_below"] or 0,
        last_price=last_price,
        dip_depth=dip_depth,
        updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
    )


@router.get(
    "/{symbol}/chart",
    summary="Get chart data",
    description="Get historical price chart data for a symbol. Works with both tracked and untracked symbols.",
    responses={
        503: {"description": "Could not fetch data for symbol"},
    },
)
async def get_chart(
    request: Request,
    symbol: str = Depends(_validate_symbol_path),
    days: int = Query(
        default=180, ge=7, le=1825, description="Number of days of history (max 5 years)"
    ),
) -> CacheableResponse:
    """Get chart data for a symbol (works for tracked symbols and benchmarks).
    
    Returns with HTTP caching headers for browser caching.
    """
    cache_key = f"{symbol}:{days}"

    # Try cache first
    cached = await _chart_cache.get(cache_key)
    if cached:
        data = [ChartPoint(**item).model_dump(mode="json") for item in cached]
        etag = generate_etag(data)
        
        # Check for conditional request (If-None-Match)
        if check_if_none_match(request, etag):
            return NotModifiedResponse(etag=etag)
        
        return CacheableResponse(
            data,
            etag=etag,
            **CachePresets.CHART,
        )

    # Get config if tracked
    row = await fetch_one("SELECT min_dip_pct FROM symbols WHERE symbol = $1", symbol)
    min_dip_pct = float(row["min_dip_pct"]) if row and row["min_dip_pct"] else 0.10

    try:
        # Use dipfinder service for chart data
        service = get_dipfinder_service()
        prices = await service.price_provider.get_prices(
            symbol,
            start_date=date.today() - timedelta(days=days),
            end_date=date.today(),
        )

        if prices is None or prices.empty:
            raise ExternalServiceError(
                message=f"Could not fetch chart data for '{symbol}'",
                details={"symbol": symbol},
            )

        # Convert to chart points
        chart_points = []
        ref_high = float(prices["Close"].max()) if "Close" in prices.columns else 0.0
        threshold = ref_high * (1.0 - min_dip_pct)

        # Calculate ref_high date and dip low date from the prices data
        ref_high_date = None
        dip_start_date = None
        if "Close" in prices.columns and not prices.empty:
            # Find the index of the highest price (52-week or period high)
            ref_high_idx = prices["Close"].idxmax()
            ref_high_date = str(ref_high_idx.date()) if hasattr(ref_high_idx, "date") else str(ref_high_idx)
            
            # Get prices after the peak to find the lowest point (actual dip bottom)
            prices_after_peak = prices.loc[ref_high_idx:]
            if len(prices_after_peak) > 1:
                # Find the lowest point after the peak - this is the actual dip
                dip_low_idx = prices_after_peak["Close"].idxmin()
                dip_start_date = str(dip_low_idx.date()) if hasattr(dip_low_idx, "date") else str(dip_low_idx)

        for idx, row_data in prices.iterrows():
            close = float(row_data["Close"]) if "Close" in row_data else 0.0
            drawdown = (close - ref_high) / ref_high if ref_high > 0 else 0.0

            chart_points.append(
                ChartPoint(
                    date=str(idx.date()) if hasattr(idx, "date") else str(idx),
                    close=close,
                    ref_high=float(ref_high),
                    threshold=float(threshold),
                    drawdown=float(drawdown),
                    since_dip=None,  # Would need dip start calculation
                    dip_start_date=dip_start_date,
                    ref_high_date=ref_high_date,
                )
            )

        # Cache the result with dynamic TTL
        data = [p.model_dump(mode="json") for p in chart_points]
        await _chart_cache.set(cache_key, data, ttl=get_cache_ttl("charts"))

        etag = generate_etag(data)
        return CacheableResponse(
            data,
            etag=etag,
            **CachePresets.CHART,
        )
    except ExternalServiceError:
        raise
    except Exception as e:
        raise ExternalServiceError(
            message=f"Could not fetch chart data for '{symbol}'",
            details={"symbol": symbol, "error": str(e)},
        )


@router.get(
    "/{symbol}/info",
    response_model=StockInfo,
    summary="Get stock info",
    description="Get detailed stock information from external source. Public endpoint.",
    responses={
        404: {"description": "Symbol not found"},
        503: {"description": "External service unavailable"},
    },
)
async def get_stock_info_endpoint(
    symbol: str = Depends(_validate_symbol_path),
) -> StockInfo:
    """Get detailed stock information, including AI-generated summary if available."""
    # Try cache first
    cached = await _info_cache.get(symbol)
    if cached:
        return StockInfo(**cached)

    info = await get_stock_info(symbol)
    if info is None:
        raise ExternalServiceError(
            message="Could not fetch stock information",
            details={"symbol": symbol},
        )

    # Check if we have an AI summary in the database
    symbol_data = await fetch_one(
        "SELECT summary_ai FROM symbols WHERE symbol = $1",
        symbol.upper(),
    )
    if symbol_data and symbol_data.get("summary_ai"):
        info.summary_ai = symbol_data["summary_ai"]

    # Cache the result with dynamic TTL (skip cache if TTL is 0)
    ttl = get_cache_ttl("ai_content")
    if ttl > 0:
        await _info_cache.set(symbol, info.model_dump(), ttl=ttl)

    return info
