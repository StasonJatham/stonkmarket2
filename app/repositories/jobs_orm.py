"""Jobs repository using SQLAlchemy ORM.

Repository for job-related database operations including:
- Symbol ingest queue
- Price history
- Dip state management
- Cleanup operations

Usage:
    from app.repositories import jobs_orm as jobs_repo
    
    # Ingest queue
    pending = await jobs_repo.get_pending_ingest_symbols(limit=20)
    await jobs_repo.mark_ingest_processing(queue_id)
    await jobs_repo.mark_ingest_completed(queue_id)
    
    # Price history
    ath_price = await jobs_repo.get_ath_price("AAPL", fallback=100.0)
    dip_start = await jobs_repo.calculate_dip_start_date("AAPL", 180.0, 0.15)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import (
    AiAgentAnalysis,
    AnalysisVersion,
    DipAIAnalysis,
    DipState,
    PriceHistory,
    StockFundamentals,
    StockSuggestion,
    Symbol,
    SymbolIngestQueue,
    SymbolSearchResult,
    UserApiKey,
)


logger = get_logger("repositories.jobs_orm")


# =============================================================================
# SYMBOL INGEST QUEUE
# =============================================================================

async def get_pending_ingest_symbols(limit: int = 20) -> Sequence[SymbolIngestQueue]:
    """Get pending symbols from the ingest queue.
    
    Args:
        limit: Maximum number of symbols to return
    
    Returns:
        Sequence of SymbolIngestQueue objects
    """
    async with get_session() as session:
        result = await session.execute(
            select(SymbolIngestQueue)
            .where(SymbolIngestQueue.status == "pending")
            .order_by(SymbolIngestQueue.priority.desc(), SymbolIngestQueue.queued_at.asc())
            .limit(limit)
        )
        return result.scalars().all()


async def mark_ingest_processing(queue_id: int, attempts: int) -> None:
    """Mark an ingest queue item as processing.
    
    Args:
        queue_id: Queue item ID
        attempts: New attempt count
    """
    async with get_session() as session:
        await session.execute(
            update(SymbolIngestQueue)
            .where(SymbolIngestQueue.id == queue_id)
            .values(
                status="processing",
                processing_started_at=datetime.now(UTC),
                attempts=attempts,
            )
        )
        await session.commit()


async def mark_ingest_completed(queue_id: int) -> None:
    """Mark an ingest queue item as completed.
    
    Args:
        queue_id: Queue item ID
    """
    async with get_session() as session:
        await session.execute(
            update(SymbolIngestQueue)
            .where(SymbolIngestQueue.id == queue_id)
            .values(
                status="completed",
                completed_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def mark_ingest_failed(queue_id: int, error: str) -> None:
    """Mark an ingest queue item as failed.
    
    Args:
        queue_id: Queue item ID
        error: Error message
    """
    async with get_session() as session:
        await session.execute(
            update(SymbolIngestQueue)
            .where(SymbolIngestQueue.id == queue_id)
            .values(
                status="failed",
                last_error=error[:500],  # Truncate to fit column
            )
        )
        await session.commit()


async def mark_ingest_pending_retry(queue_id: int, error: str) -> None:
    """Reset an ingest queue item to pending for retry.
    
    Args:
        queue_id: Queue item ID
        error: Error message
    """
    async with get_session() as session:
        await session.execute(
            update(SymbolIngestQueue)
            .where(SymbolIngestQueue.id == queue_id)
            .values(
                status="pending",
                last_error=error[:500],
            )
        )
        await session.commit()


async def get_ingest_queue_count() -> int:
    """Get count of pending symbols in the ingest queue.
    
    Returns:
        Number of pending items
    """
    async with get_session() as session:
        result = await session.execute(
            select(func.count())
            .select_from(SymbolIngestQueue)
            .where(SymbolIngestQueue.status == "pending")
        )
        return result.scalar() or 0


async def add_to_ingest_queue(symbol: str, priority: int = 0) -> bool:
    """Add a symbol to the ingest queue.
    
    Args:
        symbol: Symbol ticker
        priority: Priority level (higher = processed first)
    
    Returns:
        True if added, False if already in queue
    """
    symbol_upper = symbol.upper()

    async with get_session() as session:
        # Check if already in queue
        result = await session.execute(
            select(SymbolIngestQueue.id)
            .where(SymbolIngestQueue.symbol == symbol_upper)
        )
        if result.scalar():
            logger.debug(f"Symbol {symbol_upper} already in ingest queue")
            return False

        # Insert using ON CONFLICT DO NOTHING for race condition safety
        stmt = insert(SymbolIngestQueue).values(
            symbol=symbol_upper,
            status="pending",
            priority=priority,
            attempts=0,
            max_attempts=3,
            queued_at=datetime.now(UTC),
        ).on_conflict_do_nothing(index_elements=["symbol"])

        await session.execute(stmt)
        await session.commit()
        logger.info(f"Added {symbol_upper} to ingest queue (priority={priority})")
        return True


# =============================================================================
# PRICE HISTORY OPERATIONS
# =============================================================================

async def get_price_history_chronological(symbol: str) -> Sequence[PriceHistory]:
    """Get all price history for a symbol in chronological order.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        Sequence of PriceHistory objects ordered by date ASC
    """
    async with get_session() as session:
        result = await session.execute(
            select(PriceHistory)
            .where(PriceHistory.symbol == symbol.upper())
            .order_by(PriceHistory.date.asc())
        )
        return result.scalars().all()


async def get_ath_price(symbol: str, fallback: float = 0.0) -> float:
    """Get all-time high price for a symbol.
    
    Args:
        symbol: Symbol ticker
        fallback: Fallback value if no data found
    
    Returns:
        ATH price or fallback
    """
    async with get_session() as session:
        result = await session.execute(
            select(func.max(PriceHistory.close))
            .where(PriceHistory.symbol == symbol.upper())
        )
        ath = result.scalar()
        return float(ath) if ath else fallback


async def calculate_dip_start_date(
    symbol: str,
    ath_price: float,
    dip_threshold: float,
) -> date | None:
    """Calculate when a stock first entered the current dip period.
    
    Uses price history to find the first date where the stock dropped
    below the dip threshold (from ATH) and stayed there.
    
    Args:
        symbol: Stock symbol
        ath_price: All-time high price
        dip_threshold: Minimum dip percentage (e.g., 0.15 for 15%)
    
    Returns:
        Date when dip started, or None if not in dip
    """
    if ath_price <= 0:
        return None

    prices = await get_price_history_chronological(symbol)

    if not prices:
        return None

    # Calculate the threshold price
    dip_threshold_price = ath_price * (1 - dip_threshold)

    # Walk through prices chronologically
    dip_start = None
    currently_in_dip = False

    for price in prices:
        price_val = float(price.close) if price.close else 0
        is_dip = price_val <= dip_threshold_price

        if is_dip and not currently_in_dip:
            dip_start = price.date
            currently_in_dip = True
        elif not is_dip and currently_in_dip:
            dip_start = None
            currently_in_dip = False

    return dip_start if currently_in_dip else None


# =============================================================================
# SYMBOLS AND DIP STATE
# =============================================================================

async def get_active_symbols() -> Sequence[Symbol]:
    """Get all active symbols.
    
    Returns:
        Sequence of Symbol objects
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol).where(Symbol.is_active == True)
        )
        return result.scalars().all()


async def get_active_symbol_tickers() -> list[str]:
    """Get list of active symbol tickers.
    
    Returns:
        List of symbol tickers
    """
    symbols = await get_active_symbols()
    return [s.symbol for s in symbols]


async def get_symbol_dip_thresholds() -> dict[str, float]:
    """Get min_dip_pct for all active symbols.
    
    Returns:
        Dict mapping symbol to dip threshold percentage
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.symbol, Symbol.min_dip_pct)
            .where(Symbol.is_active == True)
        )
        rows = result.all()
        return {
            row.symbol: float(row.min_dip_pct) if row.min_dip_pct else 0.15
            for row in rows
        }


async def upsert_dip_state_with_dates(
    symbol: str,
    current_price: float,
    ath_price: float,
    dip_percentage: float,
    dip_start_date: date | None = None,
) -> None:
    """Create or update a dip state record.
    
    Args:
        symbol: Symbol ticker
        current_price: Current stock price
        ath_price: All-time high price
        dip_percentage: Percentage dip from ATH
        dip_start_date: When the dip started
    """
    async with get_session() as session:
        now = datetime.now(UTC)

        stmt = insert(DipState).values(
            symbol=symbol.upper(),
            current_price=Decimal(str(current_price)),
            ath_price=Decimal(str(ath_price)),
            dip_percentage=Decimal(str(dip_percentage)),
            dip_start_date=dip_start_date,
            first_seen=now,
            last_updated=now,
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "current_price": Decimal(str(current_price)),
                "ath_price": Decimal(str(ath_price)),
                "dip_percentage": Decimal(str(dip_percentage)),
                # COALESCE equivalent - only update if new value is provided
                "dip_start_date": func.coalesce(dip_start_date, DipState.dip_start_date),
                "last_updated": now,
            }
        )

        await session.execute(stmt)
        await session.commit()


async def get_top_dip_symbols(limit: int = 20) -> list[str]:
    """Get top symbols ordered by dip percentage.
    
    Args:
        limit: Maximum number of symbols
    
    Returns:
        List of symbol tickers
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.symbol)
            .outerjoin(DipState, Symbol.symbol == DipState.symbol)
            .where(Symbol.is_active == True)
            .order_by(func.coalesce(DipState.dip_percentage, 0).desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return list(rows)


async def get_symbol_min_dip_pct(symbol: str) -> float:
    """Get min_dip_pct for a specific symbol.
    
    Args:
        symbol: Symbol ticker
    
    Returns:
        Minimum dip percentage threshold (default 0.10)
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.min_dip_pct)
            .where(Symbol.symbol == symbol.upper())
        )
        val = result.scalar()
        return float(val) if val else 0.10


# =============================================================================
# FUNDAMENTALS
# =============================================================================

async def get_stocks_needing_fundamentals_refresh() -> list[dict]:
    """Get stocks that need fundamentals refresh.
    
    Returns:
        List of dicts with symbol and refresh metadata
    """
    async with get_session() as session:
        result = await session.execute(
            select(
                Symbol.symbol,
                StockFundamentals.fetched_at,
                StockFundamentals.earnings_date,
                StockFundamentals.next_earnings_date,
                StockFundamentals.financials_fetched_at,
                StockFundamentals.domain,
            )
            .outerjoin(StockFundamentals, Symbol.symbol == StockFundamentals.symbol)
            .where(
                and_(
                    Symbol.symbol_type == "stock",
                    Symbol.is_active == True,
                )
            )
        )
        rows = result.all()
        return [
            {
                "symbol": row.symbol,
                "fetched_at": row.fetched_at,
                "earnings_date": row.earnings_date,
                "next_earnings_date": row.next_earnings_date,
                "financials_fetched_at": row.financials_fetched_at,
                "domain": row.domain,
            }
            for row in rows
        ]


# =============================================================================
# ANALYSIS VERSIONS (for batch collection)
# =============================================================================

async def get_pending_batch_jobs() -> list[str]:
    """Get pending batch job IDs from analysis versions.
    
    Returns:
        List of unique batch job IDs
    """
    async with get_session() as session:
        result = await session.execute(
            select(AnalysisVersion.batch_job_id)
            .where(
                and_(
                    AnalysisVersion.batch_job_id.isnot(None),
                    AnalysisVersion.analysis_type == "agent_analysis",
                    AnalysisVersion.generated_at > datetime.now(UTC) - timedelta(hours=24),
                )
            )
            .distinct()
        )
        return [row[0] for row in result.all()]


async def clear_batch_job_references(batch_id: str) -> None:
    """Clear batch_job_id references for a processed batch.
    
    Args:
        batch_id: Batch job ID to clear
    """
    async with get_session() as session:
        await session.execute(
            update(AnalysisVersion)
            .where(AnalysisVersion.batch_job_id == batch_id)
            .values(batch_job_id=None)
        )
        await session.commit()


# =============================================================================
# CLEANUP
# =============================================================================

async def cleanup_expired_suggestions() -> int:
    """Delete rejected suggestions older than 7 days.
    
    Returns:
        Number of rows deleted
    """
    cutoff = datetime.now(UTC) - timedelta(days=7)
    async with get_session() as session:
        result = await session.execute(
            delete(StockSuggestion)
            .where(
                and_(
                    StockSuggestion.status == "rejected",
                    StockSuggestion.reviewed_at < cutoff,
                )
            )
        )
        await session.commit()
        return result.rowcount


async def cleanup_stale_pending_suggestions() -> int:
    """Delete pending suggestions older than 30 days.
    
    Returns:
        Number of rows deleted
    """
    cutoff = datetime.now(UTC) - timedelta(days=30)
    async with get_session() as session:
        result = await session.execute(
            delete(StockSuggestion)
            .where(
                and_(
                    StockSuggestion.status == "pending",
                    StockSuggestion.created_at < cutoff,
                )
            )
        )
        await session.commit()
        return result.rowcount


async def cleanup_expired_ai_analyses() -> int:
    """Delete expired AI analyses.
    
    Returns:
        Number of rows deleted
    """
    now = datetime.now(UTC)
    async with get_session() as session:
        result = await session.execute(
            delete(DipAIAnalysis)
            .where(
                and_(
                    DipAIAnalysis.expires_at.isnot(None),
                    DipAIAnalysis.expires_at < now,
                )
            )
        )
        await session.commit()
        return result.rowcount


async def cleanup_expired_agent_analyses() -> int:
    """Delete expired AI agent analyses.
    
    Returns:
        Number of rows deleted
    """
    now = datetime.now(UTC)
    async with get_session() as session:
        result = await session.execute(
            delete(AiAgentAnalysis)
            .where(
                and_(
                    AiAgentAnalysis.expires_at.isnot(None),
                    AiAgentAnalysis.expires_at < now,
                )
            )
        )
        await session.commit()
        return result.rowcount


async def cleanup_expired_api_keys() -> int:
    """Delete expired user API keys.
    
    Returns:
        Number of rows deleted
    """
    now = datetime.now(UTC)
    async with get_session() as session:
        result = await session.execute(
            delete(UserApiKey)
            .where(
                and_(
                    UserApiKey.expires_at.isnot(None),
                    UserApiKey.expires_at < now,
                )
            )
        )
        await session.commit()
        return result.rowcount


async def cleanup_expired_symbol_search_results() -> int:
    """Delete expired symbol search results.

    Returns:
        Number of rows deleted
    """
    now = datetime.now(UTC)
    async with get_session() as session:
        result = await session.execute(
            delete(SymbolSearchResult)
            .where(SymbolSearchResult.expires_at < now)
        )
        await session.commit()
        return result.rowcount
