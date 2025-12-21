"""Dip analysis routes."""

from __future__ import annotations

import sqlite3
from typing import List

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import get_db, require_admin, require_user
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.security import TokenData
from app.repositories import dips as dip_repo
from app.repositories import symbols as symbol_repo
from app.schemas.dips import ChartPoint, DipStateResponse, RankingEntry, StockInfo
from app.services import dip_service
from app.services.stock_info import get_stock_info

router = APIRouter()


def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


@router.get(
    "/ranking",
    response_model=List[RankingEntry],
    summary="Get dip ranking",
    description="Get stocks ranked by dip depth (most negative first).",
)
async def get_ranking(
    conn: sqlite3.Connection = Depends(get_db),
) -> List[RankingEntry]:
    """Get ranked list of stocks in dip territory."""
    return dip_service.compute_ranking_details(conn)


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
    conn: sqlite3.Connection = Depends(get_db),
) -> List[RankingEntry]:
    """Force refresh of dip ranking (fetches new data)."""
    return dip_service.compute_ranking_details(conn, force_refresh=True)


@router.get(
    "/states",
    response_model=List[DipStateResponse],
    summary="Get all dip states",
    description="Get current dip state for all tracked symbols.",
)
async def get_states(
    conn: sqlite3.Connection = Depends(get_db),
) -> List[DipStateResponse]:
    """Get current dip state for all tracked symbols."""
    states = dip_repo.load_states(conn)
    symbols = symbol_repo.list_symbols(conn)
    symbol_set = {s.symbol for s in symbols}

    return [
        DipStateResponse(
            symbol=sym,
            ref_high=state.ref_high,
            days_below=state.days_below,
            last_price=state.last_price,
            dip_depth=dip_service.dip_depth(state),
            updated_at=state.updated_at,
        )
        for sym, state in states.items()
        if sym in symbol_set
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
    conn: sqlite3.Connection = Depends(get_db),
) -> DipStateResponse:
    """Get dip state for a specific symbol."""
    # Verify symbol exists
    config = symbol_repo.get_symbol(conn, symbol)
    if config is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Load state
    states = dip_repo.load_states(conn)
    state = states.get(symbol)

    if state is None:
        # Try to refresh
        dip_service.refresh_symbol(conn, symbol)
        states = dip_repo.load_states(conn)
        state = states.get(symbol)

    if state is None:
        raise NotFoundError(
            message=f"No dip state available for '{symbol}'",
            details={"symbol": symbol},
        )

    return DipStateResponse(
        symbol=symbol,
        ref_high=state.ref_high,
        days_below=state.days_below,
        last_price=state.last_price,
        dip_depth=dip_service.dip_depth(state),
        updated_at=state.updated_at,
    )


@router.get(
    "/{symbol}/chart",
    response_model=List[ChartPoint],
    summary="Get chart data",
    description="Get historical price chart data for a symbol.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def get_chart(
    symbol: str = Depends(_validate_symbol_path),
    days: int = Query(default=180, ge=7, le=365, description="Number of days of history"),
    conn: sqlite3.Connection = Depends(get_db),
) -> List[ChartPoint]:
    """Get chart data for a symbol."""
    # Verify symbol exists
    config = symbol_repo.get_symbol(conn, symbol)
    if config is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    return dip_service.get_chart_points(symbol, config.min_dip_pct, days=days)


@router.get(
    "/{symbol}/info",
    response_model=StockInfo,
    summary="Get stock info",
    description="Get detailed stock information from external source.",
    responses={
        404: {"description": "Symbol not found"},
        503: {"description": "External service unavailable"},
    },
)
async def get_stock_info_endpoint(
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> StockInfo:
    """Get detailed stock information."""
    # Verify symbol exists
    config = symbol_repo.get_symbol(conn, symbol)
    if config is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    info = get_stock_info(symbol)
    if info is None:
        raise ExternalServiceError(
            message="Could not fetch stock information",
            details={"symbol": symbol},
        )

    return info
