"""Dip analysis routes - PostgreSQL async version."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, date, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import BaseModel

from app.api.dependencies import require_admin, require_user
from app.cache.cache import Cache
from app.cache.data_version import get_data_version
from app.cache.http_cache import (
    CacheableResponse,
    CachePresets,
    NotModifiedResponse,
    check_if_none_match,
    generate_etag,
)
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.security import TokenData
from app.dipfinder.service import get_dipfinder_service  # For chart price data
from app.repositories import dips_orm as dips_repo
from app.schemas.dips import ChartPoint, DipStateResponse, RankingEntry, StockInfo
from app.services.runtime_settings import get_cache_ttl, get_runtime_setting
from app.services.stock_info import get_stock_info


router = APIRouter()


def _safe_float(value: float, default: float = 0.0) -> float:
    """Convert value to a JSON-safe float (replace NaN/Inf/NA with default)."""
    if value is None:
        return default
    # Handle pandas NA/NaT types
    if pd.isna(value):
        return default
    try:
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return default
        return fval
    except (ValueError, TypeError):
        return default


def _get_close_col(df: pd.DataFrame) -> str:
    """Get the best close column, preferring Adjusted Close for split-adjusted prices."""
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        return "Adj Close"
    return "Close"


def _validate_chart_data(chart_points: list) -> list:
    """
    Validate chart data and skip points with impossible daily changes.
    
    Flags points where daily change exceeds 75% (impossible in real markets,
    likely indicates corrupted/unadjusted split data).
    """
    if len(chart_points) < 2:
        return chart_points
    
    validated = [chart_points[0]]  # Keep first point
    for i in range(1, len(chart_points)):
        prev_close = chart_points[i - 1].close
        curr_close = chart_points[i].close
        
        if prev_close <= 0 or curr_close <= 0:
            validated.append(chart_points[i])
            continue
        
        change_pct = abs((curr_close - prev_close) / prev_close)
        if change_pct <= 0.75:  # Accept changes up to 75%
            validated.append(chart_points[i])
        # else: skip this corrupted data point
    
    return validated

# Caches - TTLs are applied dynamically from runtime settings when setting values
_ranking_cache = Cache(prefix="ranking", default_ttl=300)
_chart_cache = Cache(prefix="chart", default_ttl=600)
_info_cache = Cache(prefix="stockinfo", default_ttl=300)



def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


class BenchmarkInfo(BaseModel):
    """Public benchmark information."""
    id: str
    symbol: str


class SectorETFInfo(BaseModel):
    """Public sector ETF information with optional price data."""
    sector: str
    symbol: str
    name: str
    # Price fields (populated if price history exists)
    current_price: float | None = None
    change_1d_pct: float | None = None
    change_7d_pct: float | None = None
    ytd_pct: float | None = None


class BatchChartRequest(BaseModel):
    """Request model for batch chart data."""
    symbols: list[str]
    days: int = 60


class MiniChartData(BaseModel):
    """Minimal chart data for card backgrounds."""
    symbol: str
    points: list[dict]  # [{date, close}]


class BatchChartResponse(BaseModel):
    """Response for batch chart data."""
    charts: list[MiniChartData]


async def _build_ranking(
    force_refresh: bool = False,
) -> tuple[list[RankingEntry], list[RankingEntry]]:
    """
    Build ranking data from dip_state table and stock info.
    
    Uses ATH-based dip calculations from dip_state table.
    
    Returns:
        Tuple of (all_entries, filtered_entries) where filtered_entries
        only includes stocks meeting their individual dip thresholds.
    """
    # Get all dip states with symbol info in one query
    # Now includes cached stock info (52w low/high, previous_close, pe_ratio, etc.)
    rows = await dips_repo.get_ranking_data()

    if not rows:
        return [], []

    # Build ranking entries from dip_state - using cached stock info from symbols table
    all_entries = []
    filtered_entries = []

    for row in rows:
        symbol = row["symbol"]

        # Calculate days in dip - prefer dip_start_date, fall back to first_seen
        days_in_dip = 0
        from datetime import date as date_type
        from datetime import datetime

        if row["dip_start_date"]:
            dip_start = row["dip_start_date"]
            if isinstance(dip_start, date_type):
                days_in_dip = (date_type.today() - dip_start).days
        elif row["first_seen"]:
            first_seen = row["first_seen"]
            if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=UTC)
            days_in_dip = (datetime.now(UTC) - first_seen).days

        dip_pct = float(row["dip_percentage"]) if row["dip_percentage"] else 0
        
        # Use cached stock info from symbols table (populated by prices_daily_job)
        previous_close = float(row["previous_close"]) if row["previous_close"] else None
        current_price = float(row["current_price"]) if row["current_price"] else None
        
        # Calculate change_percent from cached data if available
        change_percent = None
        if previous_close and current_price and previous_close > 0:
            change_percent = ((current_price - previous_close) / previous_close) * 100

        entry = RankingEntry(
            symbol=symbol,
            name=row["name"] or symbol,
            depth=dip_pct / 100,  # Convert percentage to decimal
            days_since_dip=days_in_dip,
            last_price=current_price,
            previous_close=previous_close,
            change_percent=change_percent,
            high_52w=float(row["ath_price"]) if row["ath_price"] else (float(row["fifty_two_week_high"]) if row["fifty_two_week_high"] else None),
            low_52w=float(row["fifty_two_week_low"]) if row["fifty_two_week_low"] else None,
            market_cap=int(row["market_cap"]) if row["market_cap"] else None,
            pe_ratio=float(row["pe_ratio"]) if row["pe_ratio"] else None,
            volume=int(row["avg_volume"]) if row["avg_volume"] else None,
            sector=row["sector"],
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
    response_model=list[BenchmarkInfo],
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
    "/sector-etfs",
    response_model=list[SectorETFInfo],
    summary="Get sector ETF mappings",
    description="Get list of sector ETF mappings with current prices. Public endpoint.",
)
async def get_sector_etfs() -> CacheableResponse:
    """Get sector ETF mappings from runtime settings with price data.
    
    Includes current price and change percentages when price history is available.
    Returns with short cache headers since prices change but not too frequently.
    """
    from datetime import date, timedelta
    from app.repositories import price_history_orm as price_history_repo
    
    sector_etfs = get_runtime_setting("sector_etfs", [])
    if not sector_etfs:
        return CacheableResponse([], etag=generate_etag([]), max_age=60)
    
    # Get ETF symbols
    symbols = [etf["symbol"] for etf in sector_etfs if isinstance(etf, dict) and "symbol" in etf]
    
    # Fetch price data for all ETFs in one batch query
    today = date.today()
    start_date = date(today.year, 1, 1)  # YTD start
    seven_days_ago = today - timedelta(days=7)
    
    # Batch fetch all prices at once
    all_prices = await price_history_repo.get_prices_batch(symbols, start_date, today)
    
    # Build result with price data
    results = []
    for etf in sector_etfs:
        if not isinstance(etf, dict) or "symbol" not in etf:
            continue
        
        symbol = etf["symbol"]
        info = SectorETFInfo(
            sector=etf.get("sector", ""),
            symbol=symbol,
            name=etf.get("name", ""),
        )
        
        # Get prices from batch result
        prices = all_prices.get(symbol.upper(), [])
        if prices:
            # Sort by date descending
            sorted_prices = sorted(prices, key=lambda p: p.date, reverse=True)
            
            # Current price (most recent)
            if sorted_prices:
                info.current_price = float(sorted_prices[0].close) if sorted_prices[0].close else None
            
            # YTD return (first price of year vs current)
            if len(sorted_prices) > 1:
                ytd_prices = [p for p in sorted_prices if p.date >= start_date]
                if ytd_prices:
                    oldest_ytd = min(ytd_prices, key=lambda p: p.date)
                    if oldest_ytd.close and sorted_prices[0].close:
                        info.ytd_pct = (float(sorted_prices[0].close) - float(oldest_ytd.close)) / float(oldest_ytd.close) * 100
            
            # 7-day change
            week_prices = [p for p in sorted_prices if p.date >= seven_days_ago]
            if len(week_prices) >= 2:
                oldest_week = min(week_prices, key=lambda p: p.date)
                if oldest_week.close and sorted_prices[0].close:
                    info.change_7d_pct = (float(sorted_prices[0].close) - float(oldest_week.close)) / float(oldest_week.close) * 100
            
            # 1-day change
            if len(sorted_prices) >= 2:
                prev_day = sorted_prices[1]
                if prev_day.close and sorted_prices[0].close:
                    info.change_1d_pct = (float(sorted_prices[0].close) - float(prev_day.close)) / float(prev_day.close) * 100
        
        results.append(info.model_dump())
    
    etag = generate_etag(results)
    return CacheableResponse(
        results,
        etag=etag,
        max_age=300,  # 5 minute cache (prices don't change that often)
        stale_while_revalidate=60,
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
    response_model=list[RankingEntry],
    summary="Refresh dip ranking",
    description="Force refresh of dip ranking (admin only).",
    responses={
        403: {"description": "Admin required"},
    },
)
async def refresh_ranking(
    admin: TokenData = Depends(require_admin),
) -> list[RankingEntry]:
    """Force refresh of dip ranking (fetches new data)."""
    from app.cache.data_version import bump_data_version
    
    # Invalidate both cache keys
    await _ranking_cache.delete("all:True")
    await _ranking_cache.delete("all:False")

    # Build ranking with force refresh (use shared helper)
    all_entries, filtered_entries = await _build_ranking(force_refresh=True)

    # Cache both filtered and unfiltered results with dynamic TTL
    ttl = get_cache_ttl("ranking")
    await _ranking_cache.set("all:True", [r.model_dump() for r in all_entries], ttl=ttl)
    await _ranking_cache.set("all:False", [r.model_dump() for r in filtered_entries], ttl=ttl)

    # Bump data version so frontend caches invalidate
    await bump_data_version()

    return all_entries


@router.get(
    "/states",
    response_model=list[DipStateResponse],
    summary="Get all dip states",
    description="Get current dip state for all tracked symbols.",
)
async def get_states() -> list[DipStateResponse]:
    """Get current dip state for all tracked symbols."""
    # Get dip states from database
    rows = await dips_repo.get_all_dip_states()

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
    row = await dips_repo.get_dip_state(symbol)

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
    min_dip_pct = await dips_repo.get_symbol_min_dip_pct(symbol)

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

        # Use Adjusted Close for split-adjusted prices (prevents chart corruption)
        close_col = _get_close_col(prices)
        
        # Convert to chart points
        chart_points = []
        ref_high = _safe_float(prices[close_col].max())
        threshold = ref_high * (1.0 - min_dip_pct)

        # Calculate ref_high date and dip low date from the prices data
        ref_high_date = None
        dip_start_date = None
        if close_col in prices.columns and not prices.empty:
            # Find the index of the highest price (52-week or period high)
            ref_high_idx = prices[close_col].idxmax()
            ref_high_date = str(ref_high_idx.date()) if hasattr(ref_high_idx, "date") else str(ref_high_idx)

            # Get prices after the peak to find the lowest point (actual dip bottom)
            prices_after_peak = prices.loc[ref_high_idx:]
            if len(prices_after_peak) > 1:
                # Find the lowest point after the peak - this is the actual dip
                dip_low_idx = prices_after_peak[close_col].idxmin()
                dip_start_date = str(dip_low_idx.date()) if hasattr(dip_low_idx, "date") else str(dip_low_idx)

        for idx, row_data in prices.iterrows():
            close = _safe_float(row_data.get(close_col, 0.0))
            # Skip rows with zero/invalid close price
            if close <= 0:
                continue
            drawdown = (close - ref_high) / ref_high if ref_high > 0 else 0.0

            chart_points.append(
                ChartPoint(
                    date=str(idx.date()) if hasattr(idx, "date") else str(idx),
                    close=_safe_float(close),
                    ref_high=_safe_float(ref_high),
                    threshold=_safe_float(threshold),
                    drawdown=_safe_float(drawdown),
                    since_dip=None,  # Would need dip start calculation
                    dip_start_date=dip_start_date,
                    ref_high_date=ref_high_date,
                )
            )

        # Validate and filter out corrupted data points (e.g., unadjusted splits)
        chart_points = _validate_chart_data(chart_points)

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


@router.post(
    "/batch/charts",
    response_model=BatchChartResponse,
    summary="Get batch chart data",
    description="Get mini chart data for multiple symbols at once. Used for card backgrounds.",
)
async def get_batch_charts(
    request: BatchChartRequest,
) -> BatchChartResponse:
    """Get minimal chart data for multiple symbols (for card sparklines).
    
    Returns simplified chart data (just date and close price) optimized for card backgrounds.
    Limits to 50 data points per symbol and max 30 symbols per request.
    """
    # Limit symbols to prevent abuse
    symbols = request.symbols[:30]
    days = min(request.days, 120)
    
    async def fetch_mini_chart(symbol: str) -> MiniChartData | None:
        """Fetch mini chart for a single symbol."""
        try:
            symbol = symbol.strip().upper()
            cache_key = f"{symbol}:{days}"
            
            # Try cache first
            cached = await _chart_cache.get(cache_key)
            if cached:
                # Extract just date and close for mini chart (last 50 points)
                points = [{"date": p["date"], "close": _safe_float(p["close"])} for p in cached[-50:] if _safe_float(p["close"]) > 0]
                return MiniChartData(symbol=symbol, points=points)
            
            # Fetch from service
            service = get_dipfinder_service()
            prices = await service.price_provider.get_prices(
                symbol,
                start_date=date.today() - timedelta(days=days),
                end_date=date.today(),
            )
            
            if prices is None or prices.empty:
                return None
            
            # Use Adjusted Close for split-adjusted prices
            close_col = _get_close_col(prices)
            
            # Convert to mini chart points (last 50)
            points = []
            for idx, row_data in list(prices.iterrows())[-50:]:
                close = _safe_float(row_data.get(close_col, 0.0))
                if close <= 0:
                    continue
                points.append({
                    "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                    "close": close,
                })
            
            return MiniChartData(symbol=symbol, points=points)
        except Exception:
            return None
    
    # Fetch all in parallel
    tasks = [fetch_mini_chart(s) for s in symbols]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    charts = [r for r in results if r is not None]
    
    return BatchChartResponse(charts=charts)


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
    summary_ai = await dips_repo.get_symbol_summary_ai(symbol)
    if summary_ai:
        info.summary_ai = summary_ai

    # Cache the result with dynamic TTL (skip cache if TTL is 0)
    ttl = get_cache_ttl("ai_content")
    if ttl > 0:
        await _info_cache.set(symbol, info.model_dump(), ttl=ttl)

    return info
