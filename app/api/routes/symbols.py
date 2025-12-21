"""Symbol CRUD routes - strict REST endpoints."""

from __future__ import annotations

import sqlite3
from typing import List

from fastapi import APIRouter, Depends, Path, status

from app.api.dependencies import get_db, require_user
from app.core.exceptions import ConflictError, NotFoundError
from app.core.security import TokenData
from app.repositories import dips as dip_repo
from app.repositories import symbols as symbol_repo
from app.schemas.symbols import SymbolCreate, SymbolResponse, SymbolUpdate
from app.services.dip_service import refresh_symbol

router = APIRouter()


def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


@router.get(
    "",
    response_model=List[SymbolResponse],
    summary="List all symbols",
    description="Get all tracked stock symbols.",
)
async def list_symbols(
    user: TokenData = Depends(require_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> List[SymbolResponse]:
    """List all tracked symbols."""
    symbols = symbol_repo.list_symbols(conn)
    return [
        SymbolResponse(
            symbol=s.symbol,
            min_dip_pct=s.min_dip_pct,
            min_days=s.min_days,
        )
        for s in symbols
    ]


@router.get(
    "/{symbol}",
    response_model=SymbolResponse,
    summary="Get symbol",
    description="Get a specific symbol's configuration.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def get_symbol(
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> SymbolResponse:
    """Get a specific symbol's configuration."""
    config = symbol_repo.get_symbol(conn, symbol)
    if config is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )
    return SymbolResponse(
        symbol=config.symbol,
        min_dip_pct=config.min_dip_pct,
        min_days=config.min_days,
    )


@router.post(
    "",
    response_model=SymbolResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create symbol",
    description="Add a new stock symbol to track.",
    responses={
        409: {"description": "Symbol already exists"},
    },
)
async def create_symbol(
    payload: SymbolCreate,
    user: TokenData = Depends(require_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> SymbolResponse:
    """
    Create a new tracked symbol.

    After creation, fetches initial dip state data.
    """
    # Check if already exists
    existing = symbol_repo.get_symbol(conn, payload.symbol)
    if existing is not None:
        raise ConflictError(
            message=f"Symbol '{payload.symbol}' already exists",
            details={"symbol": payload.symbol},
        )

    # Create symbol
    created = symbol_repo.upsert_symbol(
        conn,
        payload.symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

    # Initialize dip state
    try:
        refresh_symbol(conn, created.symbol)
    except Exception:
        # Don't fail creation if refresh fails
        pass

    return SymbolResponse(
        symbol=created.symbol,
        min_dip_pct=created.min_dip_pct,
        min_days=created.min_days,
    )


@router.put(
    "/{symbol}",
    response_model=SymbolResponse,
    summary="Update symbol",
    description="Update a symbol's configuration.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def update_symbol(
    payload: SymbolUpdate,
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> SymbolResponse:
    """
    Update a symbol's configuration.

    Recalculates dip state after update.
    """
    # Check if exists
    existing = symbol_repo.get_symbol(conn, symbol)
    if existing is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Update symbol
    updated = symbol_repo.upsert_symbol(
        conn,
        symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

    # Recalculate dip state
    dip_repo.delete_state(conn, symbol)
    try:
        refresh_symbol(conn, symbol)
    except Exception:
        pass

    return SymbolResponse(
        symbol=updated.symbol,
        min_dip_pct=updated.min_dip_pct,
        min_days=updated.min_days,
    )


@router.delete(
    "/{symbol}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete symbol",
    description="Remove a symbol from tracking.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def delete_symbol(
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
    conn: sqlite3.Connection = Depends(get_db),
) -> None:
    """Delete a tracked symbol and its state."""
    # Check if exists
    existing = symbol_repo.get_symbol(conn, symbol)
    if existing is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Delete symbol and state
    symbol_repo.delete_symbol(conn, symbol)
    dip_repo.delete_state(conn, symbol)
