"""Dip analysis routes - PostgreSQL async version."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import List

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel

from app.api.dependencies import require_admin, require_user
from app.cache.cache import Cache
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.security import TokenData
from app.database.connection import fetch_all, fetch_one
from app.dipfinder.service import get_dipfinder_service
from app.repositories import symbols as symbol_repo
from app.schemas.dips import ChartPoint, DipStateResponse, RankingEntry, StockInfo
from app.services.stock_info import get_stock_info, get_stock_info_async
from app.services.runtime_settings import get_runtime_setting


router = APIRouter()

# Caches with appropriate TTLs - data only changes when scheduled job runs
_ranking_cache = Cache(
    prefix="ranking", default_ttl=3600
)  # 1 hour - invalidated by jobs
_chart_cache = Cache(prefix="chart", default_ttl=3600)  # 1 hour
_info_cache = Cache(prefix="stockinfo", default_ttl=3600)  # 1 hour


def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


class BenchmarkInfo(BaseModel):
    """Public benchmark information."""
    id: str
    symbol: str
    name: str
    description: str | None = None


@router.get(
    "/benchmarks",
    response_model=List[BenchmarkInfo],
    summary="Get available benchmarks",
    description="Get list of available benchmarks for comparison. Public endpoint.",
)
async def get_benchmarks() -> List[BenchmarkInfo]:
    """Get available benchmarks from runtime settings."""
    benchmarks = get_runtime_setting("benchmarks", [])
    return [BenchmarkInfo(**b) for b in benchmarks]


@router.get(
    "/ranking",
    response_model=List[RankingEntry],
    summary="Get dip ranking",
    description="Get stocks ranked by dip depth. Use show_all=true to include stocks not in active dip.",
)
async def get_ranking(show_all: bool = False) -> List[RankingEntry]:
    """Get ranked list of stocks. By default only shows stocks in meaningful dip (>10%)."""
    cache_key = f"all:{show_all}"

    # Try cache first
    cached = await _ranking_cache.get(cache_key)
    if cached:
        return [RankingEntry(**item) for item in cached]

    # Get all symbols from database (includes per-symbol min_dip_pct thresholds)
    symbols = await symbol_repo.list_symbols()
    if not symbols:
        return []

    # Build a map of symbol -> min_dip_pct threshold
    symbol_thresholds = {s.symbol: s.min_dip_pct for s in symbols}
    tickers = [s.symbol for s in symbols]

    # Get signals from dipfinder service
    service = get_dipfinder_service()
    signals = await service.get_signals(tickers)

    # Fetch stock info for all symbols in parallel (for enrichment)
    async def get_info(symbol: str):
        return symbol, await get_stock_info_async(symbol)
    
    info_tasks = [get_info(t) for t in tickers]
    info_results = await asyncio.gather(*info_tasks, return_exceptions=True)
    stock_info_map = {}
    for result in info_results:
        if isinstance(result, tuple):
            symbol, info = result
            if info:
                stock_info_map[symbol] = info

    # Convert signals to ranking entries with enriched data
    # Filter by each stock's individual dip threshold unless show_all is True
    ranking = []
    for signal in signals:
        if signal.dip_metrics:
            # Get this stock's minimum dip threshold (default 10% if not set)
            min_dip_threshold = symbol_thresholds.get(signal.ticker, 0.10)
            dip_pct = signal.dip_metrics.dip_pct
            
            # Skip stocks not meeting their individual dip threshold unless show_all is True
            if not show_all and dip_pct < min_dip_threshold:
                continue
                
            # Get enriched info if available
            info = stock_info_map.get(signal.ticker, {})
            
            entry = RankingEntry(
                symbol=signal.ticker,
                name=info.get("name") or signal.ticker,
                depth=signal.dip_metrics.dip_pct,
                days_since_dip=signal.dip_metrics.days_since_peak,
                last_price=signal.dip_metrics.current_price,
                previous_close=info.get("previous_close"),
                change_percent=info.get("change_percent"),
                high_52w=signal.dip_metrics.peak_price,
                low_52w=info.get("fifty_two_week_low"),
                market_cap=info.get("market_cap"),
                pe_ratio=info.get("pe_ratio"),
                volume=info.get("avg_volume"),
                sector=info.get("sector"),
                updated_at=signal.as_of_date if signal.as_of_date else None,
            )
            ranking.append(entry)

    # Sort by depth (deepest dip first - higher value = deeper dip)
    ranking.sort(key=lambda x: x.depth, reverse=True)

    # Cache the result
    await _ranking_cache.set(cache_key, [r.model_dump() for r in ranking])

    return ranking


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

    # Get all symbols from database (includes per-symbol min_dip_pct thresholds)
    symbols = await symbol_repo.list_symbols()
    if not symbols:
        return []

    # Build a map of symbol -> min_dip_pct threshold
    symbol_thresholds = {s.symbol: s.min_dip_pct for s in symbols}
    tickers = [s.symbol for s in symbols]

    # Get signals with force refresh
    service = get_dipfinder_service()
    signals = await service.get_signals(tickers, force_refresh=True)

    # Fetch stock info for all symbols in parallel (for enrichment)
    async def get_info(symbol: str):
        return symbol, await get_stock_info_async(symbol)
    
    info_tasks = [get_info(t) for t in tickers]
    info_results = await asyncio.gather(*info_tasks, return_exceptions=True)
    stock_info_map = {}
    for result in info_results:
        if isinstance(result, tuple):
            symbol, info = result
            if info:
                stock_info_map[symbol] = info

    # Convert signals to ranking entries with enriched data
    # Include all stocks in refresh (show_all behavior)
    ranking = []
    filtered_ranking = []
    for signal in signals:
        if signal.dip_metrics:
            # Get enriched info if available
            info = stock_info_map.get(signal.ticker, {})
            
            entry = RankingEntry(
                symbol=signal.ticker,
                name=info.get("name") or signal.ticker,
                depth=signal.dip_metrics.dip_pct,
                days_since_dip=signal.dip_metrics.days_since_peak,
                last_price=signal.dip_metrics.current_price,
                previous_close=info.get("previous_close"),
                change_percent=info.get("change_percent"),
                high_52w=signal.dip_metrics.peak_price,
                low_52w=info.get("fifty_two_week_low"),
                market_cap=info.get("market_cap"),
                pe_ratio=info.get("pe_ratio"),
                volume=info.get("avg_volume"),
                sector=info.get("sector"),
                updated_at=signal.as_of_date if signal.as_of_date else None,
            )
            ranking.append(entry)
            
            # Check if this stock meets its individual dip threshold
            min_dip_threshold = symbol_thresholds.get(signal.ticker, 0.10)
            if signal.dip_metrics.dip_pct >= min_dip_threshold:
                filtered_ranking.append(entry)

    # Sort by depth (deepest dip first)
    ranking.sort(key=lambda x: x.depth, reverse=True)
    filtered_ranking.sort(key=lambda x: x.depth, reverse=True)

    # Cache both filtered and unfiltered results
    await _ranking_cache.set("all:True", [r.model_dump() for r in ranking])
    await _ranking_cache.set("all:False", [r.model_dump() for r in filtered_ranking])

    return ranking


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
    response_model=List[ChartPoint],
    summary="Get chart data",
    description="Get historical price chart data for a symbol. Works with both tracked and untracked symbols.",
    responses={
        503: {"description": "Could not fetch data for symbol"},
    },
)
async def get_chart(
    symbol: str = Depends(_validate_symbol_path),
    days: int = Query(
        default=180, ge=7, le=365, description="Number of days of history"
    ),
) -> List[ChartPoint]:
    """Get chart data for a symbol (works for tracked symbols and benchmarks)."""
    cache_key = f"{symbol}:{days}"

    # Try cache first
    cached = await _chart_cache.get(cache_key)
    if cached:
        return [ChartPoint(**item) for item in cached]

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

        # Cache the result
        await _chart_cache.set(cache_key, [p.model_dump() for p in chart_points])

        return chart_points
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
    """Get detailed stock information."""
    # Try cache first
    cached = await _info_cache.get(symbol)
    if cached:
        return StockInfo(**cached)

    info = get_stock_info(symbol)
    if info is None:
        raise ExternalServiceError(
            message="Could not fetch stock information",
            details={"symbol": symbol},
        )

    # Cache the result
    await _info_cache.set(symbol, info.model_dump())

    return info
