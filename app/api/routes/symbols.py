"""Symbol CRUD routes - PostgreSQL async."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Path, status
from pydantic import BaseModel

from app.api.dependencies import require_user
from app.cache.cache import Cache
from app.core.exceptions import ConflictError, NotFoundError
from app.core.security import TokenData
from app.repositories import symbols as symbol_repo
from app.schemas.symbols import SymbolCreate, SymbolResponse, SymbolUpdate
from app.services.stock_info import get_stock_info

router = APIRouter()

# Cache for symbol validations - 30 days TTL
_validation_cache = Cache(prefix="validation", default_ttl=30 * 24 * 60 * 60)  # 30 days


class SymbolValidationResponse(BaseModel):
    """Response for symbol validation."""

    valid: bool
    symbol: str
    name: str | None = None
    sector: str | None = None
    error: str | None = None


def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


@router.get(
    "/validate/{symbol}",
    response_model=SymbolValidationResponse,
    summary="Validate a stock symbol",
    description="Check if a stock symbol exists on Yahoo Finance without requiring it to be in the database. Results are cached for 30 days.",
)
async def validate_symbol(
    symbol: str = Path(
        ..., description="Stock symbol to validate", min_length=1, max_length=10
    ),
) -> SymbolValidationResponse:
    """Validate a stock symbol against Yahoo Finance (cached for 30 days)."""
    symbol_upper = symbol.upper().strip()
    cache_key = f"symbol:{symbol_upper}"

    # Try to get from cache first
    cached_result = await _validation_cache.get(cache_key)
    if cached_result is not None:
        return SymbolValidationResponse(**cached_result)

    # Not in cache, validate against Yahoo Finance
    try:
        # get_stock_info is synchronous, no await needed
        info = get_stock_info(symbol_upper)
        if info and info.name:
            result = SymbolValidationResponse(
                valid=True,
                symbol=symbol_upper,
                name=info.name,
                sector=info.sector,
            )
        else:
            result = SymbolValidationResponse(
                valid=False,
                symbol=symbol_upper,
                error="Symbol not found on Yahoo Finance",
            )
    except Exception as e:
        result = SymbolValidationResponse(
            valid=False,
            symbol=symbol_upper,
            error=str(e) if str(e) else "Failed to validate symbol",
        )

    # Cache the result (valid results for 30 days, invalid for 1 day)
    ttl = 30 * 24 * 60 * 60 if result.valid else 24 * 60 * 60
    await _validation_cache.set(cache_key, result.model_dump(), ttl=ttl)

    return result


@router.get(
    "",
    response_model=List[SymbolResponse],
    summary="List all symbols",
    description="Get all tracked stock symbols.",
)
async def list_symbols(
    user: TokenData = Depends(require_user),
) -> List[SymbolResponse]:
    """List all tracked symbols."""
    symbols = await symbol_repo.list_symbols()
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
) -> SymbolResponse:
    """Get a specific symbol's configuration."""
    config = await symbol_repo.get_symbol(symbol)
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
) -> SymbolResponse:
    """
    Create a new tracked symbol.

    After creation, fetches initial dip state data.
    """
    # Check if already exists
    existing = await symbol_repo.get_symbol(payload.symbol)
    if existing is not None:
        raise ConflictError(
            message=f"Symbol '{payload.symbol}' already exists",
            details={"symbol": payload.symbol},
        )

    # Create symbol
    created = await symbol_repo.upsert_symbol(
        payload.symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

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
) -> SymbolResponse:
    """
    Update a symbol's configuration.
    """
    # Check if exists
    existing = await symbol_repo.get_symbol(symbol)
    if existing is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Update symbol
    updated = await symbol_repo.upsert_symbol(
        symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

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
) -> None:
    """Delete a tracked symbol and its state."""
    # Check if exists
    existing = await symbol_repo.get_symbol(symbol)
    if existing is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Delete symbol (cascade will delete dip_state)
    await symbol_repo.delete_symbol(symbol)
