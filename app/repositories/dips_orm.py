"""Dips repository using SQLAlchemy ORM.

Repository for dip state and ranking queries.

Usage:
    from app.repositories import dips_orm as dips_repo
    
    ranking = await dips_repo.get_ranking_data()
    state = await dips_repo.get_dip_state("AAPL")
"""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select

from app.database.connection import get_session
from app.database.orm import DipState, Symbol
from app.core.logging import get_logger

logger = get_logger("repositories.dips_orm")


async def get_ranking_data() -> list[dict]:
    """Get dip states with symbol info for ranking.
    
    Returns:
        List of dicts with dip state and symbol data
    """
    async with get_session() as session:
        result = await session.execute(
            select(
                DipState.symbol,
                DipState.current_price,
                DipState.ath_price,
                DipState.dip_percentage,
                DipState.dip_start_date,
                DipState.first_seen,
                DipState.last_updated,
                Symbol.name,
                Symbol.sector,
                Symbol.min_dip_pct,
                Symbol.symbol_type,
            )
            .join(Symbol, Symbol.symbol == DipState.symbol)
            .where(Symbol.is_active == True)
            .order_by(DipState.dip_percentage.desc())
        )
        rows = result.all()
        
        return [
            {
                "symbol": row.symbol,
                "current_price": row.current_price,
                "ath_price": row.ath_price,
                "dip_percentage": row.dip_percentage,
                "dip_start_date": row.dip_start_date,
                "first_seen": row.first_seen,
                "last_updated": row.last_updated,
                "name": row.name,
                "sector": row.sector,
                "min_dip_pct": row.min_dip_pct,
                "symbol_type": row.symbol_type,
            }
            for row in rows
        ]


async def get_all_dip_states() -> list[dict]:
    """Get all dip states.
    
    Returns:
        List of dicts with dip state data (legacy columns)
    """
    async with get_session() as session:
        # Note: This uses the new column names since the old ones don't exist
        result = await session.execute(
            select(DipState)
            .order_by(DipState.symbol)
        )
        states = result.scalars().all()
        
        return [
            {
                "symbol": s.symbol,
                # Map new columns to expected legacy names for API compatibility
                "ref_high": s.ath_price,
                "days_below": None,  # Not tracked in new schema
                "last_price": s.current_price,
                "updated_at": s.last_updated,
            }
            for s in states
        ]


async def get_dip_state(symbol: str) -> Optional[dict]:
    """Get dip state for a specific symbol.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        Dict with dip state data or None
    """
    async with get_session() as session:
        result = await session.execute(
            select(DipState)
            .where(DipState.symbol == symbol.upper())
        )
        state = result.scalar_one_or_none()
        
        if not state:
            return None
        
        return {
            "symbol": state.symbol,
            # Map new columns to expected legacy names for API compatibility
            "ref_high": state.ath_price,
            "days_below": None,  # Not tracked in new schema
            "last_price": state.current_price,
            "updated_at": state.last_updated,
        }


async def get_symbol_min_dip_pct(symbol: str) -> float:
    """Get min_dip_pct for a symbol.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        Min dip percentage threshold (default 0.10)
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.min_dip_pct)
            .where(Symbol.symbol == symbol.upper())
        )
        val = result.scalar()
        return float(val) if val else 0.10


async def get_symbol_summary_ai(symbol: str) -> Optional[str]:
    """Get AI summary for a symbol.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        AI summary string or None
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.summary_ai)
            .where(Symbol.symbol == symbol.upper())
        )
        return result.scalar()
