"""DipFinder signals repository using SQLAlchemy ORM.

Repository for managing dipfinder signals and history.

Usage:
    from app.repositories import dipfinder_orm as dipfinder_repo
    
    signals = await dipfinder_repo.get_latest_signals(limit=50)
    await dipfinder_repo.save_signal(signal_data)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import DipfinderHistory, DipfinderSignal, DipState


logger = get_logger("repositories.dipfinder_orm")


async def get_latest_signals(
    limit: int = 50,
    min_final_score: float | None = None,
    only_alerts: bool = False,
) -> Sequence[DipfinderSignal]:
    """Get latest dipfinder signals.
    
    Args:
        limit: Maximum number of signals to return
        min_final_score: Minimum final score filter
        only_alerts: If True, only return alerts
    
    Returns:
        Sequence of DipfinderSignal objects
    """
    async with get_session() as session:
        query = select(DipfinderSignal)

        conditions = []
        if min_final_score is not None:
            conditions.append(DipfinderSignal.final_score >= min_final_score)
        if only_alerts:
            conditions.append(DipfinderSignal.should_alert == True)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(DipfinderSignal.final_score.desc()).limit(limit)

        result = await session.execute(query)
        return result.scalars().all()


async def get_latest_signals_for_tickers(
    tickers: Sequence[str],
) -> Sequence[DipfinderSignal]:
    """Get the latest dipfinder signal per ticker.

    Returns the most recent as_of_date per ticker, then picks the highest
    final_score if multiple windows exist for that date.
    """
    if not tickers:
        return []

    normalized = [t.upper() for t in tickers]

    async with get_session() as session:
        latest_subq = (
            select(
                DipfinderSignal.ticker.label("ticker"),
                func.max(DipfinderSignal.as_of_date).label("max_date"),
            )
            .where(DipfinderSignal.ticker.in_(normalized))
            .group_by(DipfinderSignal.ticker)
            .subquery()
        )

        query = (
            select(DipfinderSignal)
            .join(
                latest_subq,
                and_(
                    DipfinderSignal.ticker == latest_subq.c.ticker,
                    DipfinderSignal.as_of_date == latest_subq.c.max_date,
                ),
            )
            .where(DipfinderSignal.ticker.in_(normalized))
            .order_by(
                DipfinderSignal.ticker.asc(),
                DipfinderSignal.final_score.desc().nullslast(),
                DipfinderSignal.window_days.desc(),
            )
        )

        result = await session.execute(query)
        return result.scalars().all()


async def get_dip_history(
    ticker: str,
    days: int = 90,
) -> Sequence[DipfinderHistory]:
    """Get dip history for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of history
    
    Returns:
        Sequence of DipfinderHistory objects
    """
    since = datetime.now(UTC) - timedelta(days=days)

    async with get_session() as session:
        result = await session.execute(
            select(DipfinderHistory)
            .where(
                and_(
                    DipfinderHistory.ticker == ticker.upper(),
                    DipfinderHistory.recorded_at >= since,
                )
            )
            .order_by(DipfinderHistory.recorded_at.desc())
        )
        return result.scalars().all()


async def get_dip_state(ticker: str) -> DipState | None:
    """Get dip state for a ticker (ATH-based source of truth).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        DipState object or None
    """
    async with get_session() as session:
        result = await session.execute(
            select(DipState).where(DipState.symbol == ticker.upper())
        )
        return result.scalar_one_or_none()


async def get_previous_signal(
    ticker: str,
    window_days: int,
    before_date: date,
) -> DipfinderSignal | None:
    """Get the previous signal for a ticker/window before a given date.
    
    Args:
        ticker: Stock ticker symbol
        window_days: Window days
        before_date: Get signal before this date
    
    Returns:
        DipfinderSignal object or None
    """
    async with get_session() as session:
        result = await session.execute(
            select(DipfinderSignal)
            .where(
                and_(
                    DipfinderSignal.ticker == ticker.upper(),
                    DipfinderSignal.window_days == window_days,
                    DipfinderSignal.as_of_date < before_date,
                )
            )
            .order_by(DipfinderSignal.as_of_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


async def save_signal(
    ticker: str,
    benchmark: str,
    window_days: int,
    as_of_date: date,
    dip_stock: float,
    peak_stock: float,
    dip_pctl: float,
    dip_vs_typical: float,
    persist_days: int,
    dip_mkt: float,
    excess_dip: float,
    dip_class: str,
    quality_score: float,
    stability_score: float,
    dip_score: float,
    final_score: float,
    alert_level: str,
    should_alert: bool,
    reason: str,
    quality_factors: dict,
    stability_factors: dict,
    expires_at: datetime | None = None,
) -> bool:
    """Save a dipfinder signal to database.
    
    Returns:
        True if saved successfully
    """
    if expires_at is None:
        expires_at = datetime.now(UTC) + timedelta(days=7)

    async with get_session() as session:
        stmt = insert(DipfinderSignal).values(
            ticker=ticker.upper(),
            benchmark=benchmark,
            window_days=window_days,
            as_of_date=as_of_date,
            dip_stock=Decimal(str(dip_stock)),
            peak_stock=Decimal(str(peak_stock)),
            dip_pctl=Decimal(str(dip_pctl)),
            dip_vs_typical=Decimal(str(dip_vs_typical)),
            persist_days=persist_days,
            dip_mkt=Decimal(str(dip_mkt)),
            excess_dip=Decimal(str(excess_dip)),
            dip_class=dip_class,
            quality_score=Decimal(str(quality_score)),
            stability_score=Decimal(str(stability_score)),
            dip_score=Decimal(str(dip_score)),
            final_score=Decimal(str(final_score)),
            alert_level=alert_level,
            should_alert=should_alert,
            reason=reason,
            quality_factors=json.dumps(quality_factors),
            stability_factors=json.dumps(stability_factors),
            expires_at=expires_at,
        ).on_conflict_do_update(
            index_elements=["ticker", "benchmark", "window_days", "as_of_date"],
            set_={
                "dip_stock": Decimal(str(dip_stock)),
                "peak_stock": Decimal(str(peak_stock)),
                "dip_pctl": Decimal(str(dip_pctl)),
                "dip_vs_typical": Decimal(str(dip_vs_typical)),
                "persist_days": persist_days,
                "dip_mkt": Decimal(str(dip_mkt)),
                "excess_dip": Decimal(str(excess_dip)),
                "dip_class": dip_class,
                "quality_score": Decimal(str(quality_score)),
                "stability_score": Decimal(str(stability_score)),
                "dip_score": Decimal(str(dip_score)),
                "final_score": Decimal(str(final_score)),
                "alert_level": alert_level,
                "should_alert": should_alert,
                "reason": reason,
                "quality_factors": json.dumps(quality_factors),
                "stability_factors": json.dumps(stability_factors),
                "expires_at": expires_at,
            }
        )

        await session.execute(stmt)
        await session.commit()
        return True


async def log_history_event(
    ticker: str,
    event_type: str,
    window_days: int,
    dip_pct: float,
    final_score: float,
    dip_class: str,
) -> bool:
    """Log a dip history event.
    
    Args:
        ticker: Stock ticker symbol
        event_type: Type of event (entered_dip, exited_dip, etc.)
        window_days: Window days
        dip_pct: Dip percentage
        final_score: Final score
        dip_class: Dip classification
    
    Returns:
        True if logged successfully
    """
    async with get_session() as session:
        history = DipfinderHistory(
            ticker=ticker.upper(),
            event_type=event_type,
            window_days=window_days,
            dip_pct=Decimal(str(dip_pct)),
            final_score=Decimal(str(final_score)),
            dip_class=dip_class,
        )
        session.add(history)
        await session.commit()
        return True
