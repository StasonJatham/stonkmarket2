"""Watchlist API routes.

CRUD operations for user watchlists and watchlist items.
"""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import TokenData
from app.repositories import auth_user_orm as auth_repo
from app.repositories import watchlists_orm as watchlists_repo
from app.schemas.watchlist import (
    WatchlistCreateRequest,
    WatchlistDippingResponse,
    WatchlistDippingStock,
    WatchlistItemBulkAddRequest,
    WatchlistItemBulkAddResponse,
    WatchlistItemCreateRequest,
    WatchlistItemResponse,
    WatchlistItemUpdateRequest,
    WatchlistListResponse,
    WatchlistOpportunitiesResponse,
    WatchlistOpportunity,
    WatchlistResponse,
    WatchlistUpdateRequest,
)


router = APIRouter(prefix="/watchlists", tags=["Watchlists"])


async def _get_user_id(user: TokenData) -> int:
    """Get database user ID from token."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


# =============================================================================
# WATCHLIST ENDPOINTS
# =============================================================================


@router.get("", response_model=WatchlistListResponse)
async def list_watchlists(
    include_items: bool = Query(False, description="Include items in response"),
    user: TokenData = Depends(require_user),
) -> WatchlistListResponse:
    """List all watchlists for the current user."""
    user_id = await _get_user_id(user)
    watchlists = await watchlists_repo.list_watchlists(user_id, include_items=include_items)
    return WatchlistListResponse(
        watchlists=[WatchlistResponse(**w) for w in watchlists],
        total_count=len(watchlists),
    )


@router.post("", response_model=WatchlistResponse, status_code=status.HTTP_201_CREATED)
async def create_watchlist(
    payload: WatchlistCreateRequest,
    user: TokenData = Depends(require_user),
) -> WatchlistResponse:
    """Create a new watchlist."""
    user_id = await _get_user_id(user)
    watchlist = await watchlists_repo.create_watchlist(
        user_id=user_id,
        name=payload.name,
        description=payload.description,
        is_default=payload.is_default,
    )
    return WatchlistResponse(**watchlist)


@router.get("/default", response_model=WatchlistResponse)
async def get_default_watchlist(
    user: TokenData = Depends(require_user),
) -> WatchlistResponse:
    """Get the user's default watchlist, creating one if it doesn't exist."""
    user_id = await _get_user_id(user)
    watchlist = await watchlists_repo.get_default_watchlist(user_id)
    return WatchlistResponse(**watchlist)


@router.get("/dipping", response_model=WatchlistDippingResponse)
async def get_dipping_stocks(
    min_dip_pct: float = Query(10.0, ge=0, description="Minimum dip percentage"),
    user: TokenData = Depends(require_user),
) -> WatchlistDippingResponse:
    """Get watchlist stocks currently in a significant dip."""
    user_id = await _get_user_id(user)
    stocks = await watchlists_repo.get_dipping_watchlist_stocks(user_id, min_dip_pct)
    return WatchlistDippingResponse(
        stocks=[WatchlistDippingStock(**s) for s in stocks],
        total_count=len(stocks),
    )


@router.get("/opportunities", response_model=WatchlistOpportunitiesResponse)
async def get_opportunities(
    user: TokenData = Depends(require_user),
) -> WatchlistOpportunitiesResponse:
    """Get watchlist stocks that have hit their target price."""
    user_id = await _get_user_id(user)
    opportunities = await watchlists_repo.get_watchlist_opportunities(user_id)
    return WatchlistOpportunitiesResponse(
        opportunities=[WatchlistOpportunity(**o) for o in opportunities],
        total_count=len(opportunities),
    )


@router.get("/{watchlist_id}", response_model=WatchlistResponse)
async def get_watchlist(
    watchlist_id: int,
    include_items: bool = Query(True, description="Include items in response"),
    user: TokenData = Depends(require_user),
) -> WatchlistResponse:
    """Get a specific watchlist by ID."""
    user_id = await _get_user_id(user)
    watchlist = await watchlists_repo.get_watchlist(
        watchlist_id, user_id=user_id, include_items=include_items
    )
    if not watchlist:
        raise NotFoundError(message="Watchlist not found")
    return WatchlistResponse(**watchlist)


@router.patch("/{watchlist_id}", response_model=WatchlistResponse)
async def update_watchlist(
    watchlist_id: int,
    payload: WatchlistUpdateRequest,
    user: TokenData = Depends(require_user),
) -> WatchlistResponse:
    """Update a watchlist."""
    user_id = await _get_user_id(user)
    watchlist = await watchlists_repo.update_watchlist(
        watchlist_id,
        user_id,
        name=payload.name,
        description=payload.description,
        is_default=payload.is_default,
    )
    if not watchlist:
        raise NotFoundError(message="Watchlist not found")
    return WatchlistResponse(**watchlist)


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watchlist(
    watchlist_id: int,
    user: TokenData = Depends(require_user),
) -> None:
    """Delete a watchlist."""
    user_id = await _get_user_id(user)
    deleted = await watchlists_repo.delete_watchlist(watchlist_id, user_id)
    if not deleted:
        raise NotFoundError(message="Watchlist not found")


# =============================================================================
# WATCHLIST ITEM ENDPOINTS
# =============================================================================


@router.get("/{watchlist_id}/items", response_model=list[WatchlistItemResponse])
async def list_items(
    watchlist_id: int,
    user: TokenData = Depends(require_user),
) -> list[WatchlistItemResponse]:
    """List items in a watchlist with current market data."""
    user_id = await _get_user_id(user)
    items = await watchlists_repo.list_items_with_dip_data(watchlist_id, user_id)
    if items is None:
        raise NotFoundError(message="Watchlist not found")
    return [WatchlistItemResponse(**item) for item in items]


@router.post(
    "/{watchlist_id}/items",
    response_model=WatchlistItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_item(
    watchlist_id: int,
    payload: WatchlistItemCreateRequest,
    user: TokenData = Depends(require_user),
) -> WatchlistItemResponse:
    """Add an item to a watchlist."""
    user_id = await _get_user_id(user)
    target_price = Decimal(str(payload.target_price)) if payload.target_price else None
    item = await watchlists_repo.add_item(
        watchlist_id=watchlist_id,
        user_id=user_id,
        symbol=payload.symbol.upper(),
        notes=payload.notes,
        target_price=target_price,
        alert_on_dip=payload.alert_on_dip,
    )
    if not item:
        raise NotFoundError(message="Watchlist not found")
    return WatchlistItemResponse(**item)


@router.post(
    "/{watchlist_id}/items/bulk",
    response_model=WatchlistItemBulkAddResponse,
)
async def bulk_add_items(
    watchlist_id: int,
    payload: WatchlistItemBulkAddRequest,
    user: TokenData = Depends(require_user),
) -> WatchlistItemBulkAddResponse:
    """Add multiple items to a watchlist at once."""
    user_id = await _get_user_id(user)
    
    # Verify watchlist exists
    watchlist = await watchlists_repo.get_watchlist(watchlist_id, user_id)
    if not watchlist:
        raise NotFoundError(message="Watchlist not found")
    
    added = []
    already_exists = []
    invalid = []
    
    for symbol in payload.symbols:
        symbol_upper = symbol.strip().upper()
        if not symbol_upper or len(symbol_upper) > 20:
            invalid.append(symbol)
            continue
        
        item = await watchlists_repo.add_item(
            watchlist_id=watchlist_id,
            user_id=user_id,
            symbol=symbol_upper,
        )
        if item:
            # Check if it was newly created or updated (we can't tell easily, assume added)
            added.append(symbol_upper)
    
    return WatchlistItemBulkAddResponse(
        added=added,
        already_exists=already_exists,
        invalid=invalid,
    )


@router.patch("/{watchlist_id}/items/{item_id}", response_model=WatchlistItemResponse)
async def update_item(
    watchlist_id: int,
    item_id: int,
    payload: WatchlistItemUpdateRequest,
    user: TokenData = Depends(require_user),
) -> WatchlistItemResponse:
    """Update a watchlist item."""
    user_id = await _get_user_id(user)
    target_price = Decimal(str(payload.target_price)) if payload.target_price else None
    item = await watchlists_repo.update_item(
        item_id,
        user_id,
        notes=payload.notes,
        target_price=target_price,
        alert_on_dip=payload.alert_on_dip,
    )
    if not item:
        raise NotFoundError(message="Item not found")
    return WatchlistItemResponse(**item)


@router.delete("/{watchlist_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_item_by_id(
    watchlist_id: int,
    item_id: int,
    user: TokenData = Depends(require_user),
) -> None:
    """Remove an item from a watchlist by ID."""
    user_id = await _get_user_id(user)
    removed = await watchlists_repo.remove_item_by_id(item_id, user_id)
    if not removed:
        raise NotFoundError(message="Item not found")


@router.delete(
    "/{watchlist_id}/items/symbol/{symbol}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_item_by_symbol(
    watchlist_id: int,
    symbol: str,
    user: TokenData = Depends(require_user),
) -> None:
    """Remove an item from a watchlist by symbol."""
    user_id = await _get_user_id(user)
    removed = await watchlists_repo.remove_item(watchlist_id, user_id, symbol)
    if not removed:
        raise NotFoundError(message="Item not found")
