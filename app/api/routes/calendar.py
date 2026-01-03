"""Calendar API routes - earnings, IPOs, splits, economic events."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel

from app.api.dependencies import require_admin
from app.core.logging import get_logger


logger = get_logger("api.calendar")

router = APIRouter(prefix="/calendar")


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


class EarningsEvent(BaseModel):
    """Earnings calendar event."""
    id: int
    symbol: str
    company_name: str | None
    earnings_date: str
    earnings_time: str | None
    eps_estimate: float | None
    eps_actual: float | None
    surprise_percent: float | None
    is_past: bool | None = None


class IPOEvent(BaseModel):
    """IPO calendar event."""
    id: int
    symbol: str | None
    company_name: str
    ipo_date: str
    price_range_low: float | None
    price_range_high: float | None
    offer_price: float | None
    shares_offered: int | None
    exchange: str | None
    status: str | None


class SplitEvent(BaseModel):
    """Stock split calendar event."""
    id: int
    symbol: str
    company_name: str | None
    split_date: str
    split_ratio: str
    split_from: int | None
    split_to: int | None


class EconomicEvent(BaseModel):
    """Economic calendar event."""
    id: int
    event_name: str
    event_date: str
    country: str | None
    estimate: str | None
    actual: str | None
    prior: str | None
    importance: str | None


class CalendarSummary(BaseModel):
    """Summary of upcoming calendar events."""
    period_days: int
    earnings_count: int
    splits_count: int
    ipos_count: int
    economic_events_count: int
    earnings: list[EarningsEvent]
    splits: list[SplitEvent]
    ipos: list[IPOEvent]
    economic_events: list[EconomicEvent]


class SyncResult(BaseModel):
    """Result of calendar sync operation."""
    earnings_synced: int
    ipos_synced: int
    splits_synced: int
    economic_events_synced: int
    errors: list[str]
    duration_seconds: float
    date_range: dict[str, str]


# =============================================================================
# ROUTES
# =============================================================================


@router.get(
    "/summary",
    response_model=CalendarSummary,
    summary="Get calendar summary",
    description="Get a summary of all upcoming calendar events for the next N days.",
)
async def get_calendar_summary(
    days: Annotated[int, Query(ge=1, le=60)] = 7,
):
    """Get calendar summary for the next N days."""
    from app.services.calendar_data import get_calendar_summary
    
    summary = await get_calendar_summary(days=days)
    return CalendarSummary(**summary)


@router.get(
    "/earnings",
    response_model=list[EarningsEvent],
    summary="Get upcoming earnings",
    description="Get upcoming earnings announcements for the next N days.",
)
async def get_upcoming_earnings(
    days: Annotated[int, Query(ge=1, le=60)] = 7,
    symbol: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
):
    """Get upcoming earnings events."""
    from app.services.calendar_data import get_upcoming_earnings
    
    earnings = await get_upcoming_earnings(days=days, symbol=symbol, limit=limit)
    return [EarningsEvent(**e) for e in earnings]


@router.get(
    "/earnings/{symbol}",
    response_model=list[EarningsEvent],
    summary="Get symbol earnings history",
    description="Get earnings history and upcoming for a specific symbol.",
)
async def get_symbol_earnings(
    symbol: str,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """Get earnings for a specific symbol."""
    from app.services.calendar_data import get_symbol_earnings
    
    earnings = await get_symbol_earnings(symbol.upper(), limit=limit)
    return [EarningsEvent(**e) for e in earnings]


@router.get(
    "/splits",
    response_model=list[SplitEvent],
    summary="Get upcoming splits",
    description="Get upcoming stock splits for the next N days.",
)
async def get_upcoming_splits(
    days: Annotated[int, Query(ge=1, le=90)] = 30,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    """Get upcoming stock splits."""
    from app.services.calendar_data import get_upcoming_splits
    
    splits = await get_upcoming_splits(days=days, limit=limit)
    return [SplitEvent(**s) for s in splits]


@router.get(
    "/splits/{symbol}",
    response_model=list[SplitEvent],
    summary="Get symbol split history",
    description="Get split history for a specific symbol.",
)
async def get_symbol_splits(symbol: str):
    """Get splits for a specific symbol."""
    from app.services.calendar_data import get_symbol_splits
    
    splits = await get_symbol_splits(symbol.upper())
    return [SplitEvent(**s) for s in splits]


@router.get(
    "/ipos",
    response_model=list[IPOEvent],
    summary="Get upcoming IPOs",
    description="Get upcoming IPO listings for the next N days.",
)
async def get_upcoming_ipos(
    days: Annotated[int, Query(ge=1, le=90)] = 30,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    """Get upcoming IPOs."""
    from app.services.calendar_data import get_upcoming_ipos
    
    ipos = await get_upcoming_ipos(days=days, limit=limit)
    return [IPOEvent(**i) for i in ipos]


@router.get(
    "/economic",
    response_model=list[EconomicEvent],
    summary="Get upcoming economic events",
    description="Get upcoming economic events (Fed meetings, jobs reports, etc.) for the next N days.",
)
async def get_upcoming_economic_events(
    days: Annotated[int, Query(ge=1, le=60)] = 14,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    """Get upcoming economic events."""
    from app.services.calendar_data import get_upcoming_economic_events
    
    events = await get_upcoming_economic_events(days=days, limit=limit)
    return [EconomicEvent(**e) for e in events]


# =============================================================================
# ADMIN ROUTES
# =============================================================================


@router.post(
    "/sync",
    response_model=SyncResult,
    summary="Trigger calendar sync",
    description="Manually trigger a sync of all calendar data from yfinance.",
    dependencies=[Depends(require_admin)],
)
async def sync_calendar_data(
    weeks_ahead: Annotated[int, Query(ge=1, le=12)] = 5,
):
    """
    Trigger full calendar sync (admin only).
    
    This runs the same job as the weekly scheduled sync.
    """
    from app.services.calendar_data import sync_all_calendar_data
    
    result = await sync_all_calendar_data(weeks_ahead=weeks_ahead)
    return SyncResult(**result)
