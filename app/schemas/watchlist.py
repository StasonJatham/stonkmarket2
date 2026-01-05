"""Watchlist schemas for API validation.

Pydantic models for watchlist-related requests and responses.

Usage:
    from app.schemas.watchlist import (
        WatchlistCreateRequest,
        WatchlistResponse,
        WatchlistItemCreateRequest,
        WatchlistItemResponse,
    )
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# WATCHLIST SCHEMAS
# =============================================================================


class WatchlistCreateRequest(BaseModel):
    """Request to create a new watchlist."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    is_default: bool = False


class WatchlistUpdateRequest(BaseModel):
    """Request to update a watchlist."""
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    is_default: bool | None = None


class WatchlistItemResponse(BaseModel):
    """Response for a watchlist item."""
    id: int
    watchlist_id: int
    symbol: str
    notes: str | None = None
    target_price: float | None = None
    alert_on_dip: bool = True
    created_at: str | None = None
    updated_at: str | None = None
    
    # Enriched fields (when joined with DipState)
    current_price: float | None = None
    dip_percent: float | None = None
    days_below: int | None = None
    is_tail_event: bool | None = None
    ath_price: float | None = None


class WatchlistResponse(BaseModel):
    """Response for a watchlist."""
    id: int
    user_id: int
    name: str
    description: str | None = None
    is_default: bool = False
    item_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    items: list[WatchlistItemResponse] | None = None


class WatchlistListResponse(BaseModel):
    """Response for listing watchlists."""
    watchlists: list[WatchlistResponse]
    total_count: int


# =============================================================================
# WATCHLIST ITEM SCHEMAS
# =============================================================================


class WatchlistItemCreateRequest(BaseModel):
    """Request to add an item to a watchlist."""
    symbol: str = Field(..., min_length=1, max_length=20)
    notes: str | None = Field(None, max_length=1000)
    target_price: float | None = Field(None, ge=0)
    alert_on_dip: bool = True


class WatchlistItemUpdateRequest(BaseModel):
    """Request to update a watchlist item."""
    notes: str | None = Field(None, max_length=1000)
    target_price: float | None = Field(None, ge=0)
    alert_on_dip: bool | None = None


class WatchlistItemBulkAddRequest(BaseModel):
    """Request to add multiple items to a watchlist."""
    symbols: list[str] = Field(..., min_length=1, max_length=100)


class WatchlistItemBulkAddResponse(BaseModel):
    """Response for bulk adding items."""
    added: list[str]
    already_exists: list[str]
    invalid: list[str]


# =============================================================================
# OPPORTUNITY & ALERT SCHEMAS
# =============================================================================


class WatchlistDippingStock(BaseModel):
    """A stock from a watchlist that is currently dipping."""
    symbol: str
    watchlist_id: int
    notes: str | None = None
    target_price: float | None = None
    current_price: float | None = None
    dip_percent: float | None = None
    days_below: int | None = None
    is_tail_event: bool = False


class WatchlistOpportunity(BaseModel):
    """A stock from a watchlist that hit target price."""
    symbol: str
    watchlist_id: int
    notes: str | None = None
    target_price: float
    current_price: float
    discount_percent: float


class WatchlistDippingResponse(BaseModel):
    """Response for dipping stocks query."""
    stocks: list[WatchlistDippingStock]
    total_count: int


class WatchlistOpportunitiesResponse(BaseModel):
    """Response for opportunities query."""
    opportunities: list[WatchlistOpportunity]
    total_count: int
