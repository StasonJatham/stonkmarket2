"""DipState repository using SQLAlchemy ORM.

Repository for managing dip state records - current price, ATH, dip percentage
for tracked symbols.

Usage:
    from app.repositories import dip_state_orm as dip_state_repo
    
    await dip_state_repo.upsert_dip_state(
        symbol="AAPL",
        current_price=150.0,
        ath_price=180.0,
        dip_percentage=16.67,
    )
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import DipState


logger = get_logger("repositories.dip_state_orm")


async def upsert_dip_state(
    symbol: str,
    current_price: float,
    ath_price: float,
    dip_percentage: float,
) -> bool:
    """Create or update a dip state record.
    
    Args:
        symbol: Symbol ticker (will be uppercased)
        current_price: Current stock price
        ath_price: All-time high price
        dip_percentage: Percentage dip from ATH
    
    Returns:
        True if upsert succeeded
    """
    async with get_session() as session:
        now = datetime.utcnow()

        stmt = insert(DipState).values(
            symbol=symbol.upper(),
            current_price=Decimal(str(current_price)),
            ath_price=Decimal(str(ath_price)),
            dip_percentage=Decimal(str(dip_percentage)),
            first_seen=now,
            last_updated=now,
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "current_price": Decimal(str(current_price)),
                "ath_price": Decimal(str(ath_price)),
                "dip_percentage": Decimal(str(dip_percentage)),
                "last_updated": now,
            }
        )

        await session.execute(stmt)
        await session.commit()
        return True


async def get_dip_state(symbol: str) -> DipState | None:
    """Get dip state for a symbol.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        DipState ORM object or None if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(DipState).where(DipState.symbol == symbol.upper())
        )
        return result.scalar_one_or_none()


async def delete_dip_state(symbol: str) -> bool:
    """Delete dip state for a symbol.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        True if deleted, False if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(DipState).where(DipState.symbol == symbol.upper())
        )
        dip = result.scalar_one_or_none()

        if dip:
            await session.delete(dip)
            await session.commit()
            return True
        return False


async def get_dip_states_for_symbols(symbols: list[str]) -> dict[str, DipState]:
    """Get dip states for multiple symbols.
    
    Args:
        symbols: List of symbol tickers
    
    Returns:
        Dict mapping symbol to DipState ORM object
    """
    if not symbols:
        return {}

    async with get_session() as session:
        upper_symbols = [s.upper() for s in symbols]
        result = await session.execute(
            select(DipState).where(DipState.symbol.in_(upper_symbols))
        )
        states = result.scalars().all()
        return {state.symbol: state for state in states}
