"""
Calendar Data Service - Earnings, IPOs, Splits, Economic Events from yfinance.

Provides weekly-updated calendar data for:
- Earnings announcements with EPS estimates
- IPO listings and pricing
- Stock splits
- Economic events (Fed meetings, jobs reports, etc.)

Usage:
    from app.services.calendar_data import (
        sync_all_calendar_data,
        get_upcoming_earnings,
        get_upcoming_splits,
        get_symbol_earnings,
    )
    
    # Sync all calendar data (run weekly on Saturday)
    await sync_all_calendar_data()
    
    # Get upcoming earnings for next 7 days
    earnings = await get_upcoming_earnings(days=7)
    
    # Get earnings for a specific symbol
    symbol_earnings = await get_symbol_earnings("AAPL")
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import yfinance as yf

from app.cache.cache import Cache
from app.core.data_helpers import run_in_executor, safe_int
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import (
    CalendarEarnings,
    CalendarEconomicEvent,
    CalendarIPO,
    CalendarSplit,
)


logger = get_logger("services.calendar_data")

# Module-level cache instance for calendar data
_calendar_cache = Cache(prefix="calendar", default_ttl=60 * 60 * 24)  # 24 hours


# Cache TTL: 1 day (data is updated weekly but can be queried frequently)
CACHE_TTL = 60 * 60 * 24  # 24 hours

# How many weeks ahead to fetch
WEEKS_AHEAD = 5


# =============================================================================
# DATA FETCHING FROM YFINANCE
# =============================================================================


def _safe_decimal(value: Any) -> Decimal | None:
    """Safely convert a value to Decimal."""
    if value is None or pd.isna(value):
        return None
    try:
        return Decimal(str(value))
    except (ValueError, TypeError):
        return None


# Use centralized safe_int (handles pd.isna)
_safe_int = safe_int


def _df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert DataFrame to list of dicts, handling NaN values."""
    if df is None or df.empty:
        return []
    # Convert to records and handle NaN
    records = df.reset_index().to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
    return records


def _fetch_earnings_calendar(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch earnings calendar from yfinance (blocking call)."""
    try:
        calendar = yf.Calendars(start=start, end=end)
        # Get up to 100 results per call, with market cap filter
        df = calendar.get_earnings_calendar(
            limit=100,
            filter_most_active=False,  # Get all, not just most active
        )
        return _df_to_records(df)
    except Exception as e:
        logger.error(f"Failed to fetch earnings calendar: {e}")
        return []


def _fetch_ipo_calendar(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch IPO calendar from yfinance (blocking call)."""
    try:
        calendar = yf.Calendars(start=start, end=end)
        df = calendar.get_ipo_info_calendar(limit=100)
        return _df_to_records(df)
    except Exception as e:
        logger.error(f"Failed to fetch IPO calendar: {e}")
        return []


def _fetch_splits_calendar(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch splits calendar from yfinance (blocking call)."""
    try:
        calendar = yf.Calendars(start=start, end=end)
        df = calendar.get_splits_calendar(limit=100)
        return _df_to_records(df)
    except Exception as e:
        logger.error(f"Failed to fetch splits calendar: {e}")
        return []


def _fetch_economic_events_calendar(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Fetch economic events calendar from yfinance (blocking call)."""
    try:
        calendar = yf.Calendars(start=start, end=end)
        df = calendar.get_economic_events_calendar(limit=100)
        return _df_to_records(df)
    except Exception as e:
        logger.error(f"Failed to fetch economic events calendar: {e}")
        return []


# Alias for backward compatibility
_run_in_executor = run_in_executor


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================


async def _upsert_earnings(records: list[dict[str, Any]]) -> int:
    """Insert or update earnings records. Returns count of upserted records."""
    if not records:
        return 0
    
    async with get_session() as session:
        from sqlalchemy.dialects.postgresql import insert
        
        count = 0
        for record in records:
            try:
                # Extract fields with various possible column names
                symbol = record.get("Symbol") or record.get("symbol") or record.get("Ticker")
                if not symbol:
                    continue
                
                # Parse earnings date
                earnings_date = record.get("Earnings Date") or record.get("earnings_date") or record.get("index")
                if earnings_date is None:
                    continue
                if isinstance(earnings_date, str):
                    earnings_date = pd.to_datetime(earnings_date)
                if isinstance(earnings_date, pd.Timestamp):
                    earnings_date = earnings_date.to_pydatetime()
                if earnings_date.tzinfo is None:
                    earnings_date = earnings_date.replace(tzinfo=UTC)
                
                stmt = insert(CalendarEarnings).values(
                    symbol=symbol,
                    company_name=record.get("Company") or record.get("company_name") or record.get("Name"),
                    earnings_date=earnings_date,
                    earnings_time=record.get("Earnings Call Time") or record.get("Time"),
                    eps_estimate=_safe_decimal(record.get("EPS Estimate") or record.get("eps_estimate")),
                    eps_actual=_safe_decimal(record.get("Reported EPS") or record.get("eps_actual")),
                    surprise_percent=_safe_decimal(record.get("Surprise(%)") or record.get("surprise_percent")),
                    raw_data=record,
                    updated_at=datetime.now(UTC),
                ).on_conflict_do_update(
                    constraint="uq_calendar_earnings_symbol_date",
                    set_={
                        "company_name": record.get("Company") or record.get("company_name") or record.get("Name"),
                        "earnings_time": record.get("Earnings Call Time") or record.get("Time"),
                        "eps_estimate": _safe_decimal(record.get("EPS Estimate") or record.get("eps_estimate")),
                        "eps_actual": _safe_decimal(record.get("Reported EPS") or record.get("eps_actual")),
                        "surprise_percent": _safe_decimal(record.get("Surprise(%)") or record.get("surprise_percent")),
                        "raw_data": record,
                        "updated_at": datetime.now(UTC),
                    },
                )
                await session.execute(stmt)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert earnings record: {e}")
                continue
        
        await session.commit()
        return count


async def _upsert_ipos(records: list[dict[str, Any]]) -> int:
    """Insert or update IPO records. Returns count of upserted records."""
    if not records:
        return 0
    
    async with get_session() as session:
        from sqlalchemy.dialects.postgresql import insert
        
        count = 0
        for record in records:
            try:
                company_name = record.get("Company") or record.get("company_name") or record.get("Name")
                if not company_name:
                    continue
                
                # Parse IPO date
                ipo_date = record.get("Date") or record.get("ipo_date") or record.get("index")
                if ipo_date is None:
                    continue
                if isinstance(ipo_date, str):
                    ipo_date = pd.to_datetime(ipo_date).date()
                elif isinstance(ipo_date, pd.Timestamp):
                    ipo_date = ipo_date.date()
                elif isinstance(ipo_date, datetime):
                    ipo_date = ipo_date.date()
                
                stmt = insert(CalendarIPO).values(
                    symbol=record.get("Symbol") or record.get("symbol") or record.get("Ticker"),
                    company_name=company_name,
                    ipo_date=ipo_date,
                    price_range_low=_safe_decimal(record.get("Price Range Low")),
                    price_range_high=_safe_decimal(record.get("Price Range High")),
                    offer_price=_safe_decimal(record.get("Offer Price") or record.get("Price")),
                    shares_offered=_safe_int(record.get("Shares Offered") or record.get("Shares")),
                    deal_size=_safe_decimal(record.get("Deal Size")),
                    exchange=record.get("Exchange"),
                    status=record.get("Status") or record.get("status"),
                    raw_data=record,
                    updated_at=datetime.now(UTC),
                ).on_conflict_do_update(
                    constraint="uq_calendar_ipos_company_date",
                    set_={
                        "symbol": record.get("Symbol") or record.get("symbol") or record.get("Ticker"),
                        "price_range_low": _safe_decimal(record.get("Price Range Low")),
                        "price_range_high": _safe_decimal(record.get("Price Range High")),
                        "offer_price": _safe_decimal(record.get("Offer Price") or record.get("Price")),
                        "shares_offered": _safe_int(record.get("Shares Offered") or record.get("Shares")),
                        "deal_size": _safe_decimal(record.get("Deal Size")),
                        "exchange": record.get("Exchange"),
                        "status": record.get("Status") or record.get("status"),
                        "raw_data": record,
                        "updated_at": datetime.now(UTC),
                    },
                )
                await session.execute(stmt)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert IPO record: {e}")
                continue
        
        await session.commit()
        return count


async def _upsert_splits(records: list[dict[str, Any]]) -> int:
    """Insert or update split records. Returns count of upserted records."""
    if not records:
        return 0
    
    async with get_session() as session:
        from sqlalchemy.dialects.postgresql import insert
        
        count = 0
        for record in records:
            try:
                symbol = record.get("Symbol") or record.get("symbol") or record.get("Ticker")
                if not symbol:
                    continue
                
                # Parse split date
                split_date = record.get("Date") or record.get("split_date") or record.get("index")
                if split_date is None:
                    continue
                if isinstance(split_date, str):
                    split_date = pd.to_datetime(split_date).date()
                elif isinstance(split_date, pd.Timestamp):
                    split_date = split_date.date()
                elif isinstance(split_date, datetime):
                    split_date = split_date.date()
                
                # Get split ratio
                split_ratio = record.get("Split Ratio") or record.get("Ratio") or record.get("split_ratio") or "1:1"
                
                # Parse split ratio (e.g., "4:1" -> from=4, to=1)
                split_from, split_to = None, None
                if isinstance(split_ratio, str) and ":" in split_ratio:
                    parts = split_ratio.split(":")
                    if len(parts) == 2:
                        split_from = _safe_int(parts[0])
                        split_to = _safe_int(parts[1])
                
                stmt = insert(CalendarSplit).values(
                    symbol=symbol,
                    company_name=record.get("Company") or record.get("company_name") or record.get("Name"),
                    split_date=split_date,
                    split_ratio=str(split_ratio),
                    split_from=split_from,
                    split_to=split_to,
                    raw_data=record,
                    updated_at=datetime.now(UTC),
                ).on_conflict_do_update(
                    constraint="uq_calendar_splits_symbol_date",
                    set_={
                        "company_name": record.get("Company") or record.get("company_name") or record.get("Name"),
                        "split_ratio": str(split_ratio),
                        "split_from": split_from,
                        "split_to": split_to,
                        "raw_data": record,
                        "updated_at": datetime.now(UTC),
                    },
                )
                await session.execute(stmt)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert split record: {e}")
                continue
        
        await session.commit()
        return count


async def _upsert_economic_events(records: list[dict[str, Any]]) -> int:
    """Insert or update economic event records. Returns count of upserted records."""
    if not records:
        return 0
    
    async with get_session() as session:
        from sqlalchemy.dialects.postgresql import insert
        
        count = 0
        for record in records:
            try:
                event_name = record.get("Event") or record.get("event_name") or record.get("Name")
                if not event_name:
                    continue
                
                # Parse event date
                event_date = record.get("Date") or record.get("event_date") or record.get("index")
                if event_date is None:
                    continue
                if isinstance(event_date, str):
                    event_date = pd.to_datetime(event_date)
                if isinstance(event_date, pd.Timestamp):
                    event_date = event_date.to_pydatetime()
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=UTC)
                
                stmt = insert(CalendarEconomicEvent).values(
                    event_name=event_name,
                    event_date=event_date,
                    country=record.get("Country") or record.get("country"),
                    estimate=str(record.get("Estimate") or record.get("estimate") or "") if record.get("Estimate") else None,
                    actual=str(record.get("Actual") or record.get("actual") or "") if record.get("Actual") else None,
                    prior=str(record.get("Prior") or record.get("prior") or "") if record.get("Prior") else None,
                    importance=record.get("Importance") or record.get("importance"),
                    raw_data=record,
                    updated_at=datetime.now(UTC),
                ).on_conflict_do_update(
                    constraint="uq_calendar_econ_event_date",
                    set_={
                        "country": record.get("Country") or record.get("country"),
                        "estimate": str(record.get("Estimate") or record.get("estimate") or "") if record.get("Estimate") else None,
                        "actual": str(record.get("Actual") or record.get("actual") or "") if record.get("Actual") else None,
                        "prior": str(record.get("Prior") or record.get("prior") or "") if record.get("Prior") else None,
                        "importance": record.get("Importance") or record.get("importance"),
                        "raw_data": record,
                        "updated_at": datetime.now(UTC),
                    },
                )
                await session.execute(stmt)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert economic event record: {e}")
                continue
        
        await session.commit()
        return count


# =============================================================================
# PUBLIC API - SYNC FUNCTIONS
# =============================================================================


async def sync_all_calendar_data(weeks_ahead: int = WEEKS_AHEAD) -> dict[str, Any]:
    """
    Sync all calendar data from yfinance for the next N weeks.
    
    This is the main job function, should be run weekly on Saturday.
    
    Args:
        weeks_ahead: Number of weeks ahead to fetch (default 5)
    
    Returns:
        Dict with sync results
    """
    logger.info(f"[CALENDAR] Starting full calendar sync for next {weeks_ahead} weeks")
    start_time = datetime.now(UTC)
    
    # Calculate date range
    today = datetime.now(UTC)
    end_date = today + timedelta(weeks=weeks_ahead)
    
    results = {
        "earnings_synced": 0,
        "ipos_synced": 0,
        "splits_synced": 0,
        "economic_events_synced": 0,
        "errors": [],
        "duration_seconds": 0,
        "date_range": {
            "start": today.isoformat(),
            "end": end_date.isoformat(),
        },
    }
    
    # Fetch earnings
    try:
        logger.info("[CALENDAR] Fetching earnings calendar...")
        earnings_records = await _run_in_executor(_fetch_earnings_calendar, today, end_date)
        results["earnings_synced"] = await _upsert_earnings(earnings_records)
        logger.info(f"[CALENDAR] Synced {results['earnings_synced']} earnings events")
        await Cache.delete("calendar:earnings:*")
    except Exception as e:
        logger.error(f"Failed to sync earnings: {e}")
        results["errors"].append(f"Earnings: {str(e)}")
    
    await asyncio.sleep(0.5)  # Small delay between API calls
    
    # Fetch IPOs
    try:
        logger.info("[CALENDAR] Fetching IPO calendar...")
        ipo_records = await _run_in_executor(_fetch_ipo_calendar, today, end_date)
        results["ipos_synced"] = await _upsert_ipos(ipo_records)
        logger.info(f"[CALENDAR] Synced {results['ipos_synced']} IPO events")
        await Cache.delete("calendar:ipos:*")
    except Exception as e:
        logger.error(f"Failed to sync IPOs: {e}")
        results["errors"].append(f"IPOs: {str(e)}")
    
    await asyncio.sleep(0.5)
    
    # Fetch splits
    try:
        logger.info("[CALENDAR] Fetching splits calendar...")
        splits_records = await _run_in_executor(_fetch_splits_calendar, today, end_date)
        results["splits_synced"] = await _upsert_splits(splits_records)
        logger.info(f"[CALENDAR] Synced {results['splits_synced']} split events")
        await Cache.delete("calendar:splits:*")
    except Exception as e:
        logger.error(f"Failed to sync splits: {e}")
        results["errors"].append(f"Splits: {str(e)}")
    
    await asyncio.sleep(0.5)
    
    # Fetch economic events
    try:
        logger.info("[CALENDAR] Fetching economic events calendar...")
        econ_records = await _run_in_executor(_fetch_economic_events_calendar, today, end_date)
        results["economic_events_synced"] = await _upsert_economic_events(econ_records)
        logger.info(f"[CALENDAR] Synced {results['economic_events_synced']} economic events")
        await Cache.delete("calendar:economic:*")
    except Exception as e:
        logger.error(f"Failed to sync economic events: {e}")
        results["errors"].append(f"Economic Events: {str(e)}")
    
    duration = (datetime.now(UTC) - start_time).total_seconds()
    results["duration_seconds"] = round(duration, 2)
    
    total_synced = (
        results["earnings_synced"] + 
        results["ipos_synced"] + 
        results["splits_synced"] + 
        results["economic_events_synced"]
    )
    
    logger.info(
        f"[CALENDAR] Sync complete: "
        f"{total_synced} total events, "
        f"{len(results['errors'])} errors, "
        f"{results['duration_seconds']}s"
    )
    
    return results


# =============================================================================
# PUBLIC API - READ FUNCTIONS
# =============================================================================


async def get_upcoming_earnings(
    days: int = 7,
    symbol: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Get upcoming earnings events.
    
    Args:
        days: Number of days ahead to look
        symbol: Optional filter by symbol
        limit: Max results to return
    
    Returns:
        List of earnings events
    """
    cache_key = f"calendar:earnings:upcoming:{days}:{symbol or 'all'}:{limit}"
    cached = await _calendar_cache.get(cache_key)
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        today = datetime.now(UTC)
        end_date = today + timedelta(days=days)
        
        query = (
            select(CalendarEarnings)
            .where(CalendarEarnings.earnings_date >= today)
            .where(CalendarEarnings.earnings_date <= end_date)
            .order_by(CalendarEarnings.earnings_date)
            .limit(limit)
        )
        
        if symbol:
            query = query.where(CalendarEarnings.symbol == symbol.upper())
        
        result = await session.execute(query)
        earnings = result.scalars().all()
        
        data = [
            {
                "id": e.id,
                "symbol": e.symbol,
                "company_name": e.company_name,
                "earnings_date": e.earnings_date.isoformat(),
                "earnings_time": e.earnings_time,
                "eps_estimate": float(e.eps_estimate) if e.eps_estimate else None,
                "eps_actual": float(e.eps_actual) if e.eps_actual else None,
                "surprise_percent": float(e.surprise_percent) if e.surprise_percent else None,
            }
            for e in earnings
        ]
        
        await _calendar_cache.set(cache_key, data, ttl=CACHE_TTL)
        return data


async def get_symbol_earnings(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get earnings history and upcoming for a specific symbol."""
    cache_key = f"calendar:earnings:symbol:{symbol.upper()}:{limit}"
    cached = await _calendar_cache.get(cache_key)
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        query = (
            select(CalendarEarnings)
            .where(CalendarEarnings.symbol == symbol.upper())
            .order_by(CalendarEarnings.earnings_date.desc())
            .limit(limit)
        )
        
        result = await session.execute(query)
        earnings = result.scalars().all()
        
        data = [
            {
                "id": e.id,
                "symbol": e.symbol,
                "company_name": e.company_name,
                "earnings_date": e.earnings_date.isoformat(),
                "earnings_time": e.earnings_time,
                "eps_estimate": float(e.eps_estimate) if e.eps_estimate else None,
                "eps_actual": float(e.eps_actual) if e.eps_actual else None,
                "surprise_percent": float(e.surprise_percent) if e.surprise_percent else None,
                "is_past": e.earnings_date < datetime.now(UTC),
            }
            for e in earnings
        ]
        
        await _calendar_cache.set(cache_key, data, ttl=CACHE_TTL)
        return data


async def get_upcoming_splits(days: int = 30, limit: int = 50) -> list[dict[str, Any]]:
    """Get upcoming stock splits."""
    cache_key = f"calendar:splits:upcoming:{days}:{limit}"
    cached = await _calendar_cache.get(cache_key)
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        today = datetime.now(UTC).date()
        end_date = today + timedelta(days=days)
        
        query = (
            select(CalendarSplit)
            .where(CalendarSplit.split_date >= today)
            .where(CalendarSplit.split_date <= end_date)
            .order_by(CalendarSplit.split_date)
            .limit(limit)
        )
        
        result = await session.execute(query)
        splits = result.scalars().all()
        
        data = [
            {
                "id": s.id,
                "symbol": s.symbol,
                "company_name": s.company_name,
                "split_date": s.split_date.isoformat(),
                "split_ratio": s.split_ratio,
                "split_from": s.split_from,
                "split_to": s.split_to,
            }
            for s in splits
        ]
        
        await _calendar_cache.set(cache_key, data, ttl=CACHE_TTL)
        return data


async def get_symbol_splits(symbol: str) -> list[dict[str, Any]]:
    """Get split history for a specific symbol."""
    async with get_session() as session:
        from sqlalchemy import select
        
        query = (
            select(CalendarSplit)
            .where(CalendarSplit.symbol == symbol.upper())
            .order_by(CalendarSplit.split_date.desc())
        )
        
        result = await session.execute(query)
        splits = result.scalars().all()
        
        return [
            {
                "id": s.id,
                "symbol": s.symbol,
                "company_name": s.company_name,
                "split_date": s.split_date.isoformat(),
                "split_ratio": s.split_ratio,
                "split_from": s.split_from,
                "split_to": s.split_to,
            }
            for s in splits
        ]


async def get_upcoming_ipos(days: int = 30, limit: int = 50) -> list[dict[str, Any]]:
    """Get upcoming IPOs."""
    cache_key = f"calendar:ipos:upcoming:{days}:{limit}"
    cached = await _calendar_cache.get(cache_key)
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        today = datetime.now(UTC).date()
        end_date = today + timedelta(days=days)
        
        query = (
            select(CalendarIPO)
            .where(CalendarIPO.ipo_date >= today)
            .where(CalendarIPO.ipo_date <= end_date)
            .order_by(CalendarIPO.ipo_date)
            .limit(limit)
        )
        
        result = await session.execute(query)
        ipos = result.scalars().all()
        
        data = [
            {
                "id": i.id,
                "symbol": i.symbol,
                "company_name": i.company_name,
                "ipo_date": i.ipo_date.isoformat(),
                "price_range_low": float(i.price_range_low) if i.price_range_low else None,
                "price_range_high": float(i.price_range_high) if i.price_range_high else None,
                "offer_price": float(i.offer_price) if i.offer_price else None,
                "shares_offered": i.shares_offered,
                "exchange": i.exchange,
                "status": i.status,
            }
            for i in ipos
        ]
        
        await _calendar_cache.set(cache_key, data, ttl=CACHE_TTL)
        return data


async def get_upcoming_economic_events(days: int = 14, limit: int = 50) -> list[dict[str, Any]]:
    """Get upcoming economic events."""
    cache_key = f"calendar:economic:upcoming:{days}:{limit}"
    cached = await _calendar_cache.get(cache_key)
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        today = datetime.now(UTC)
        end_date = today + timedelta(days=days)
        
        query = (
            select(CalendarEconomicEvent)
            .where(CalendarEconomicEvent.event_date >= today)
            .where(CalendarEconomicEvent.event_date <= end_date)
            .order_by(CalendarEconomicEvent.event_date)
            .limit(limit)
        )
        
        result = await session.execute(query)
        events = result.scalars().all()
        
        data = [
            {
                "id": e.id,
                "event_name": e.event_name,
                "event_date": e.event_date.isoformat(),
                "country": e.country,
                "estimate": e.estimate,
                "actual": e.actual,
                "prior": e.prior,
                "importance": e.importance,
            }
            for e in events
        ]
        
        await _calendar_cache.set(cache_key, data, ttl=CACHE_TTL)
        return data


async def get_calendar_summary(days: int = 7) -> dict[str, Any]:
    """
    Get a summary of all upcoming calendar events.
    
    Useful for the calendar widget in the UI.
    """
    cache_key = f"calendar:summary:{days}"
    cached = await _calendar_cache.get(cache_key)
    if cached:
        return cached
    
    earnings = await get_upcoming_earnings(days=days, limit=20)
    splits = await get_upcoming_splits(days=days, limit=10)
    ipos = await get_upcoming_ipos(days=days, limit=10)
    economic_events = await get_upcoming_economic_events(days=days, limit=10)
    
    summary = {
        "period_days": days,
        "earnings_count": len(earnings),
        "splits_count": len(splits),
        "ipos_count": len(ipos),
        "economic_events_count": len(economic_events),
        "earnings": earnings[:5],  # Top 5 for preview
        "splits": splits[:3],
        "ipos": ipos[:3],
        "economic_events": economic_events[:5],
    }
    
    await _calendar_cache.set(cache_key, summary, ttl=CACHE_TTL)
    return summary
