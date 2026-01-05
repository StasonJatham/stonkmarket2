"""Watchlists repository using SQLAlchemy ORM.

CRUD operations for user watchlists and watchlist items.

Usage:
    from app.repositories.watchlists_orm import (
        create_watchlist, get_watchlist, list_watchlists,
        add_item, remove_item, list_items,
    )
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select, func, delete
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import Watchlist, WatchlistItem, DipState


logger = get_logger("repositories.watchlists_orm")


# =============================================================================
# WATCHLIST OPERATIONS
# =============================================================================


async def create_watchlist(
    user_id: int,
    name: str,
    description: str | None = None,
    is_default: bool = False,
) -> dict[str, Any]:
    """Create a new watchlist.
    
    Args:
        user_id: Owner of the watchlist
        name: Display name for the watchlist
        description: Optional description
        is_default: Whether this is the user's default watchlist
        
    Returns:
        Created watchlist as dict
    """
    async with get_session() as session:
        # If setting as default, unset any existing default
        if is_default:
            existing = await session.execute(
                select(Watchlist).where(
                    Watchlist.user_id == user_id,
                    Watchlist.is_default == True  # noqa: E712
                )
            )
            for wl in existing.scalars():
                wl.is_default = False
        
        watchlist = Watchlist(
            user_id=user_id,
            name=name,
            description=description,
            is_default=is_default,
        )
        session.add(watchlist)
        await session.commit()
        await session.refresh(watchlist)
        
        return _watchlist_to_dict(watchlist)


async def get_watchlist(
    watchlist_id: int,
    user_id: int | None = None,
    include_items: bool = True,
) -> dict[str, Any] | None:
    """Get a watchlist by ID, optionally filtered by user.
    
    Args:
        watchlist_id: The watchlist ID
        user_id: Optional user ID for access control
        include_items: Whether to include watchlist items
        
    Returns:
        Watchlist dict or None if not found
    """
    async with get_session() as session:
        stmt = select(Watchlist).where(Watchlist.id == watchlist_id)
        if user_id is not None:
            stmt = stmt.where(Watchlist.user_id == user_id)
        if include_items:
            stmt = stmt.options(selectinload(Watchlist.items))
            
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if watchlist:
            return _watchlist_to_dict(watchlist, include_items=include_items)
        return None


async def get_default_watchlist(user_id: int) -> dict[str, Any] | None:
    """Get the user's default watchlist, creating one if none exists.
    
    Args:
        user_id: The user ID
        
    Returns:
        Default watchlist dict
    """
    async with get_session() as session:
        stmt = select(Watchlist).where(
            Watchlist.user_id == user_id,
            Watchlist.is_default == True  # noqa: E712
        ).options(selectinload(Watchlist.items))
        
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if watchlist:
            return _watchlist_to_dict(watchlist, include_items=True)
    
    # Create default watchlist if none exists
    return await create_watchlist(user_id, "My Watchlist", is_default=True)


async def list_watchlists(
    user_id: int,
    include_items: bool = False,
) -> list[dict[str, Any]]:
    """List all watchlists for a user.
    
    Args:
        user_id: The user ID
        include_items: If True, include items in each watchlist
        
    Returns:
        List of watchlist dicts
    """
    async with get_session() as session:
        stmt = select(Watchlist).where(Watchlist.user_id == user_id)
        
        if include_items:
            stmt = stmt.options(selectinload(Watchlist.items))
            
        stmt = stmt.order_by(Watchlist.is_default.desc(), Watchlist.name)
        
        result = await session.execute(stmt)
        watchlists = result.scalars().all()
        
        return [
            _watchlist_to_dict(wl, include_items=include_items)
            for wl in watchlists
        ]


async def update_watchlist(
    watchlist_id: int,
    user_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    is_default: bool | None = None,
) -> dict[str, Any] | None:
    """Update a watchlist.
    
    Args:
        watchlist_id: The watchlist to update
        user_id: Owner (for access control)
        **kwargs: Fields to update
        
    Returns:
        Updated watchlist dict or None if not found
    """
    async with get_session() as session:
        stmt = select(Watchlist).where(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == user_id,
        ).options(selectinload(Watchlist.items))
        
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if not watchlist:
            return None
        
        # If setting as default, unset any existing default
        if is_default is True:
            existing = await session.execute(
                select(Watchlist).where(
                    Watchlist.user_id == user_id,
                    Watchlist.is_default == True,  # noqa: E712
                    Watchlist.id != watchlist_id,
                )
            )
            for wl in existing.scalars():
                wl.is_default = False
        
        if name is not None:
            watchlist.name = name
        if description is not None:
            watchlist.description = description
        if is_default is not None:
            watchlist.is_default = is_default
            
        await session.commit()
        await session.refresh(watchlist)
        
        return _watchlist_to_dict(watchlist, include_items=True)


async def delete_watchlist(watchlist_id: int, user_id: int) -> bool:
    """Delete a watchlist.
    
    Args:
        watchlist_id: The watchlist to delete
        user_id: Owner (for access control)
        
    Returns:
        True if deleted, False if not found
    """
    async with get_session() as session:
        stmt = select(Watchlist).where(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == user_id,
        )
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if not watchlist:
            return False
        
        await session.delete(watchlist)
        await session.commit()
        return True


# =============================================================================
# WATCHLIST ITEM OPERATIONS
# =============================================================================


async def add_item(
    watchlist_id: int,
    user_id: int,
    symbol: str,
    notes: str | None = None,
    target_price: Decimal | None = None,
    alert_on_dip: bool = True,
) -> dict[str, Any] | None:
    """Add an item to a watchlist.
    
    Args:
        watchlist_id: The watchlist to add to
        user_id: Owner (for access control)
        symbol: Stock symbol to add
        notes: Optional notes
        target_price: Optional target buy price
        alert_on_dip: Whether to alert when stock dips
        
    Returns:
        Created item dict or None if watchlist not found
    """
    async with get_session() as session:
        # Verify ownership
        stmt = select(Watchlist).where(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == user_id,
        )
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if not watchlist:
            return None
        
        # Check if symbol already exists
        stmt = select(WatchlistItem).where(
            WatchlistItem.watchlist_id == watchlist_id,
            WatchlistItem.symbol == symbol.upper(),
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing item
            if notes is not None:
                existing.notes = notes
            if target_price is not None:
                existing.target_price = target_price
            existing.alert_on_dip = alert_on_dip
            await session.commit()
            await session.refresh(existing)
            return _item_to_dict(existing)
        
        # Create new item
        item = WatchlistItem(
            watchlist_id=watchlist_id,
            symbol=symbol.upper(),
            notes=notes,
            target_price=target_price,
            alert_on_dip=alert_on_dip,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        
        return _item_to_dict(item)


async def update_item(
    item_id: int,
    user_id: int,
    *,
    notes: str | None = None,
    target_price: Decimal | None = None,
    alert_on_dip: bool | None = None,
) -> dict[str, Any] | None:
    """Update a watchlist item.
    
    Args:
        item_id: The item to update
        user_id: Owner (for access control via watchlist)
        **kwargs: Fields to update
        
    Returns:
        Updated item dict or None if not found
    """
    async with get_session() as session:
        stmt = select(WatchlistItem).join(Watchlist).where(
            WatchlistItem.id == item_id,
            Watchlist.user_id == user_id,
        )
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()
        
        if not item:
            return None
        
        if notes is not None:
            item.notes = notes
        if target_price is not None:
            item.target_price = target_price
        if alert_on_dip is not None:
            item.alert_on_dip = alert_on_dip
            
        await session.commit()
        await session.refresh(item)
        
        return _item_to_dict(item)


async def remove_item(
    watchlist_id: int,
    user_id: int,
    symbol: str,
) -> bool:
    """Remove an item from a watchlist by symbol.
    
    Args:
        watchlist_id: The watchlist
        user_id: Owner (for access control)
        symbol: Symbol to remove
        
    Returns:
        True if removed, False if not found
    """
    async with get_session() as session:
        # Verify ownership
        stmt = select(Watchlist).where(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == user_id,
        )
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if not watchlist:
            return False
        
        stmt = delete(WatchlistItem).where(
            WatchlistItem.watchlist_id == watchlist_id,
            WatchlistItem.symbol == symbol.upper(),
        )
        result = await session.execute(stmt)
        await session.commit()
        
        return result.rowcount > 0


async def remove_item_by_id(item_id: int, user_id: int) -> bool:
    """Remove a watchlist item by ID.
    
    Args:
        item_id: The item ID to remove
        user_id: Owner (for access control)
        
    Returns:
        True if removed, False if not found
    """
    async with get_session() as session:
        stmt = select(WatchlistItem).join(Watchlist).where(
            WatchlistItem.id == item_id,
            Watchlist.user_id == user_id,
        )
        result = await session.execute(stmt)
        item = result.scalar_one_or_none()
        
        if not item:
            return False
        
        await session.delete(item)
        await session.commit()
        return True


async def list_items_with_dip_data(
    watchlist_id: int,
    user_id: int,
) -> list[dict[str, Any]]:
    """List items in a watchlist with current dip data.
    
    Args:
        watchlist_id: The watchlist
        user_id: Owner (for access control)
        
    Returns:
        List of items with dip data joined
    """
    async with get_session() as session:
        # Verify ownership
        stmt = select(Watchlist).where(
            Watchlist.id == watchlist_id,
            Watchlist.user_id == user_id,
        )
        result = await session.execute(stmt)
        watchlist = result.scalar_one_or_none()
        
        if not watchlist:
            return []
        
        # Get items with dip data
        stmt = (
            select(WatchlistItem, DipState)
            .outerjoin(DipState, WatchlistItem.symbol == DipState.symbol)
            .where(WatchlistItem.watchlist_id == watchlist_id)
            .order_by(WatchlistItem.created_at.desc())
        )
        
        result = await session.execute(stmt)
        rows = result.all()
        
        items = []
        for item, dip in rows:
            item_dict = _item_to_dict(item)
            if dip:
                item_dict["current_price"] = float(dip.current_price) if dip.current_price else None
                item_dict["dip_percent"] = float(dip.dip_percentage) if dip.dip_percentage else None
                item_dict["days_below"] = dip.days_below
                item_dict["is_tail_event"] = dip.is_tail_event
                item_dict["ath_price"] = float(dip.ref_high) if dip.ref_high else None
            items.append(item_dict)
        
        return items


async def get_watchlist_symbols(user_id: int) -> list[str]:
    """Get all unique symbols across all user's watchlists.
    
    Args:
        user_id: The user ID
        
    Returns:
        List of unique symbols
    """
    async with get_session() as session:
        stmt = (
            select(WatchlistItem.symbol)
            .distinct()
            .join(Watchlist)
            .where(Watchlist.user_id == user_id)
        )
        result = await session.execute(stmt)
        return [row[0] for row in result.all()]


async def get_dipping_watchlist_stocks(
    user_id: int,
    min_dip_pct: float = 10.0,
) -> list[dict[str, Any]]:
    """Get watchlist stocks currently in a significant dip.
    
    Args:
        user_id: The user ID
        min_dip_pct: Minimum dip percentage to include
        
    Returns:
        List of dipping stock dicts
    """
    async with get_session() as session:
        stmt = (
            select(WatchlistItem, DipState)
            .join(Watchlist, WatchlistItem.watchlist_id == Watchlist.id)
            .join(DipState, WatchlistItem.symbol == DipState.symbol)
            .where(
                Watchlist.user_id == user_id,
                WatchlistItem.alert_on_dip == True,  # noqa: E712
                DipState.dip_percentage >= min_dip_pct,
            )
            .order_by(DipState.dip_percentage.desc())
        )
        
        result = await session.execute(stmt)
        rows = result.all()
        
        return [
            {
                "symbol": item.symbol,
                "watchlist_id": item.watchlist_id,
                "notes": item.notes,
                "target_price": float(item.target_price) if item.target_price else None,
                "current_price": float(dip.current_price) if dip.current_price else None,
                "dip_percent": float(dip.dip_percentage) if dip.dip_percentage else None,
                "days_below": dip.days_below,
                "is_tail_event": dip.is_tail_event,
            }
            for item, dip in rows
        ]


async def get_watchlist_opportunities(user_id: int) -> list[dict[str, Any]]:
    """Get watchlist stocks that have hit their target price.
    
    Args:
        user_id: The user ID
        
    Returns:
        List of opportunity dicts
    """
    async with get_session() as session:
        stmt = (
            select(WatchlistItem, DipState)
            .join(Watchlist, WatchlistItem.watchlist_id == Watchlist.id)
            .join(DipState, WatchlistItem.symbol == DipState.symbol)
            .where(
                Watchlist.user_id == user_id,
                WatchlistItem.target_price.isnot(None),
            )
        )
        
        result = await session.execute(stmt)
        rows = result.all()
        
        opportunities = []
        for item, dip in rows:
            if dip.current_price and item.target_price:
                if dip.current_price <= item.target_price:
                    opportunities.append({
                        "symbol": item.symbol,
                        "watchlist_id": item.watchlist_id,
                        "notes": item.notes,
                        "target_price": float(item.target_price),
                        "current_price": float(dip.current_price),
                        "discount_percent": float(
                            (item.target_price - dip.current_price) / item.target_price * 100
                        ),
                    })
        
        return sorted(opportunities, key=lambda x: x["discount_percent"], reverse=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _watchlist_to_dict(
    watchlist: Watchlist,
    include_items: bool = False,
) -> dict[str, Any]:
    """Convert Watchlist ORM model to dict."""
    result = {
        "id": watchlist.id,
        "user_id": watchlist.user_id,
        "name": watchlist.name,
        "description": watchlist.description,
        "is_default": watchlist.is_default,
        "created_at": watchlist.created_at.isoformat() if watchlist.created_at else None,
        "updated_at": watchlist.updated_at.isoformat() if watchlist.updated_at else None,
    }
    
    if include_items and hasattr(watchlist, "items") and watchlist.items is not None:
        result["items"] = [_item_to_dict(item) for item in watchlist.items]
        result["item_count"] = len(watchlist.items)
    elif hasattr(watchlist, "items") and watchlist.items is not None:
        result["item_count"] = len(watchlist.items)
    else:
        result["item_count"] = 0
    
    return result


def _item_to_dict(item: WatchlistItem) -> dict[str, Any]:
    """Convert WatchlistItem ORM model to dict."""
    return {
        "id": item.id,
        "watchlist_id": item.watchlist_id,
        "symbol": item.symbol,
        "notes": item.notes,
        "target_price": float(item.target_price) if item.target_price else None,
        "alert_on_dip": item.alert_on_dip,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }
