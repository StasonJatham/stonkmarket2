"""Repository for managing quant scores."""

from __future__ import annotations

from sqlalchemy import select

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import QuantScore

logger = get_logger("repositories.quant_scores_orm")


async def get_quant_score(symbol: str) -> QuantScore | None:
    """Get the latest quant score for a symbol."""
    if not symbol:
        return None

    async with get_session() as session:
        result = await session.execute(
            select(QuantScore).where(QuantScore.symbol == symbol.upper()).limit(1)
        )
        return result.scalar_one_or_none()


async def get_quant_scores_for_symbols(
    symbols: list[str],
) -> dict[str, QuantScore]:
    """Get quant scores for multiple symbols."""
    if not symbols:
        return {}

    symbols_upper = [s.upper() for s in symbols if s]
    if not symbols_upper:
        return {}

    async with get_session() as session:
        result = await session.execute(
            select(QuantScore).where(QuantScore.symbol.in_(symbols_upper))
        )
        scores = result.scalars().all()
        return {score.symbol: score for score in scores}
