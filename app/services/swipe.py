"""Swipe service - combines dips with AI analysis and voting (PostgreSQL)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, or_, select

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import DipAIAnalysis, DipState, Symbol
from app.repositories import dip_votes_orm as dip_votes_repo
from app.schemas.swipe import DipCard, DipStats, VoteCounts
from app.services import stock_info
from app.services.fundamentals import get_fundamentals_for_analysis
from app.services.statistical_rating import calculate_rating


logger = get_logger("swipe")


async def get_dip_card(symbol: str) -> DipCard | None:
    """
    Get a complete dip card with AI analysis for a symbol.

    Returns:
        DipCard with symbol, prices, AI analysis, and vote counts.
        None if symbol not found in dip state.
    """
    # Get dip state and symbol info from PostgreSQL
    async with get_session() as session:
        result = await session.execute(
            select(DipState, Symbol)
            .outerjoin(Symbol, Symbol.symbol == DipState.symbol)
            .where(DipState.symbol == symbol.upper())
        )
        row = result.one_or_none()

    if not row:
        return None

    dip_state, sym = row

    # Get cached AI analysis
    ai_analysis = await dip_votes_repo.get_ai_analysis(symbol)

    # Get vote counts
    vote_counts_dict = await dip_votes_repo.get_vote_counts(symbol)
    vote_counts = VoteCounts(**vote_counts_dict)

    # Calculate days in dip from first_seen
    days_below = 0
    if dip_state.first_seen:
        first_seen = dip_state.first_seen
        if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=UTC)
        days_below = (datetime.now(UTC) - first_seen).days

    # Get opportunity_type - prefer cached value from DipState, fallback to computing
    opportunity_type = dip_state.opportunity_type or "NONE"
    if opportunity_type == "NONE":
        # Try to compute if not cached (lazy import to avoid circular dependency)
        try:
            from app.dipfinder.service import get_dipfinder_service
            dipfinder = get_dipfinder_service()
            signal = await dipfinder.get_signal(symbol)
            if signal:
                opportunity_type = signal.opportunity_type.value
        except Exception as e:
            logger.debug(f"Could not compute opportunity_type for {symbol}: {e}")

    # Build card with Pydantic model
    card = DipCard(
        symbol=symbol.upper(),
        name=sym.name if sym else None,
        sector=sym.sector if sym else None,
        current_price=float(dip_state.current_price) if dip_state.current_price else 0,
        ref_high=float(dip_state.ath_price) if dip_state.ath_price else 0,
        dip_pct=float(dip_state.dip_percentage) if dip_state.dip_percentage else 0,
        days_below=days_below,
        opportunity_type=opportunity_type,
        is_tail_event=dip_state.is_tail_event,
        return_period_years=float(dip_state.return_period_years) if dip_state.return_period_years else None,
        regime_dip_percentile=float(dip_state.regime_dip_percentile) if dip_state.regime_dip_percentile else None,
        vote_counts=vote_counts,
        summary_ai=sym.summary_ai if sym else None,
        swipe_bio=ai_analysis.get("swipe_bio") if ai_analysis else None,
        ai_rating=ai_analysis.get("ai_rating") if ai_analysis else None,
        ai_reasoning=ai_analysis.get("ai_reasoning") if ai_analysis else None,
    )

    return card


async def get_dip_card_with_fresh_ai(symbol: str, force_refresh: bool = False) -> DipCard | None:
    """
    Get dip card, queueing AI generation via batch if needed.

    If AI content is missing, queues for batch processing and returns
    the card with ai_pending=True. UI should show pending state.
    
    Args:
        symbol: Stock symbol
        force_refresh: If True, re-queue AI for batch processing
        
    Returns:
        DipCard with AI content if available, or ai_pending=True if queued.
        None if symbol not found in dip state.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None

    # Check if AI is pending
    ai_pending = False
    async with get_session() as session:
        result = await session.execute(
            select(DipAIAnalysis.ai_pending).where(DipAIAnalysis.symbol == symbol.upper())
        )
        row = result.scalar_one_or_none()
        ai_pending = row is True

    # If AI analysis already cached and not forcing refresh, return as-is
    if card.swipe_bio and card.ai_rating and not force_refresh:
        return card

    # If AI is already pending (queued for batch), return card with pending status
    if ai_pending and not force_refresh:
        return card.model_copy(update={"ai_pending": True})

    # Queue for batch processing if AI is missing or forced
    if force_refresh or not card.swipe_bio or not card.ai_rating:
        from app.services.batch_scheduler import queue_ai_for_symbols
        await queue_ai_for_symbols([symbol])
        return card.model_copy(update={"ai_pending": True})

    return card


async def regenerate_ai_field(
    symbol: str,
    field: str,
) -> DipCard | None:
    """
    Queue regeneration of a swipe-specific AI field via batch API.

    Args:
        symbol: Stock symbol
        field: Field to regenerate: 'rating' or 'bio'
               Note: 'summary' should use /symbols/{symbol}/ai/summary endpoint

    Returns:
        DipCard with ai_pending=True (queued for batch processing).
        None if symbol not found in dip state.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None

    # Queue for batch processing
    from app.services.batch_scheduler import queue_ai_for_symbols
    await queue_ai_for_symbols([symbol])

    # Return card with pending status
    return card.model_copy(update={"ai_pending": True})


async def get_dip_cards_page(
    limit: int | None = None,
    offset: int = 0,
    search: str | None = None,
) -> tuple[list[DipCard], int]:
    """Get dip cards with optional pagination and search."""
    async with get_session() as session:
        stmt = (
            select(DipState, Symbol)
            .join(Symbol, Symbol.symbol == DipState.symbol)
            .order_by(DipState.dip_percentage.desc())
        )

        if search:
            term = f"%{search.strip()}%"
            stmt = stmt.where(
                or_(
                    Symbol.symbol.ilike(term),
                    Symbol.name.ilike(term),
                    Symbol.sector.ilike(term),
                )
            )

        total_result = await session.execute(
            select(func.count()).select_from(stmt.subquery())
        )
        total = total_result.scalar() or 0

        page_stmt = stmt.offset(offset)
        if limit is not None:
            page_stmt = page_stmt.limit(limit)

        result = await session.execute(page_stmt)
        dip_rows = result.all()

        symbols = [dip_state.symbol for dip_state, _ in dip_rows]
        vote_counts_by_symbol = await dip_votes_repo.get_vote_counts_for_symbols_with_session(
            session, symbols
        )

        ai_by_symbol: dict[str, DipAIAnalysis] = {}
        if symbols:
            ai_result = await session.execute(
                select(DipAIAnalysis).where(
                    DipAIAnalysis.symbol.in_(symbols),
                    (DipAIAnalysis.expires_at == None) | (DipAIAnalysis.expires_at > datetime.now(UTC)),
                )
            )
            ai_by_symbol = {row.symbol: row for row in ai_result.scalars().all()}

        cards: list[DipCard] = []
        now = datetime.now(UTC)
        for dip_state, sym in dip_rows:
            symbol = dip_state.symbol

            days_below = 0
            if dip_state.first_seen:
                first_seen = dip_state.first_seen
                if hasattr(first_seen, "tzinfo") and first_seen.tzinfo is None:
                    first_seen = first_seen.replace(tzinfo=UTC)
                days_below = (now - first_seen).days

            vote_counts_dict = vote_counts_by_symbol.get(
                symbol,
                {"buy": 0, "sell": 0, "buy_weighted": 0, "sell_weighted": 0, "net_score": 0},
            )
            vote_counts = VoteCounts(**vote_counts_dict)

            ai = ai_by_symbol.get(symbol)

            card = DipCard(
                symbol=symbol,
                name=sym.name if sym else None,
                sector=sym.sector if sym else None,
                current_price=float(dip_state.current_price) if dip_state.current_price else 0,
                ref_high=float(dip_state.ath_price) if dip_state.ath_price else 0,
                dip_pct=float(dip_state.dip_percentage) if dip_state.dip_percentage else 0,
                days_below=days_below,
                opportunity_type=dip_state.opportunity_type or "NONE",
                is_tail_event=dip_state.is_tail_event,
                return_period_years=float(dip_state.return_period_years) if dip_state.return_period_years else None,
                regime_dip_percentile=float(dip_state.regime_dip_percentile) if dip_state.regime_dip_percentile else None,
                vote_counts=vote_counts,
                summary_ai=sym.summary_ai if sym else None,
                swipe_bio=ai.swipe_bio if ai else None,
                ai_rating=ai.ai_rating if ai else None,
                ai_reasoning=ai.rating_reasoning if ai else None,
            )
            cards.append(card)

        return cards, total


async def get_all_dip_cards(include_ai: bool = False) -> list[DipCard]:
    """
    Get all current dips as cards.

    Args:
        include_ai: Kept for backward compatibility (AI refresh is queued at API layer)

    Returns:
        List of DipCard objects sorted by dip percentage.
    """
    cards, _ = await get_dip_cards_page()
    return cards


async def vote_on_dip(
    symbol: str,
    voter_identifier: str,
    vote_type: str,
    vote_weight: int = 1,
    api_key_id: int | None = None,
    skip_cooldown: bool = False,
) -> tuple[bool, str | None]:
    """
    Record a vote on a dip.

    Args:
        symbol: Stock symbol
        voter_identifier: Hashed voter ID (fingerprint)
        vote_type: 'buy' or 'sell'
        vote_weight: Vote weight multiplier (default 1, API key users get 10)
        api_key_id: Optional API key ID if using authenticated voting
        skip_cooldown: If True, skip cooldown check (for admins)

    Returns:
        Tuple of (success, error_message)
    """
    return await dip_votes_repo.add_vote(
        symbol=symbol,
        fingerprint=voter_identifier,
        vote_type=vote_type,
        vote_weight=vote_weight,
        api_key_id=api_key_id,
        skip_cooldown=skip_cooldown,
    )


async def get_vote_stats(symbol: str) -> DipStats:
    """Get voting statistics for a symbol."""
    counts_dict = await dip_votes_repo.get_vote_counts(symbol)
    vote_counts = VoteCounts(**counts_dict)

    total = counts_dict["buy"] + counts_dict["sell"]
    weighted_total = counts_dict["buy_weighted"] + counts_dict["sell_weighted"]

    return DipStats(
        symbol=symbol,
        vote_counts=vote_counts,
        total_votes=total,
        weighted_total=weighted_total,
        buy_pct=round(counts_dict["buy"] / total * 100, 1) if total > 0 else 0,
        sell_pct=round(counts_dict["sell"] / total * 100, 1) if total > 0 else 0,
        sentiment=_calculate_sentiment(counts_dict),
    )


def _calculate_sentiment(counts: dict) -> str:
    """Calculate overall sentiment from weighted vote counts."""
    buy = counts.get("buy_weighted", counts.get("buy", 0))
    sell = counts.get("sell_weighted", counts.get("sell", 0))

    if buy == 0 and sell == 0:
        return "neutral"

    ratio = buy / (buy + sell) if (buy + sell) > 0 else 0.5

    if ratio >= 0.7:
        return "very_bullish"
    elif ratio >= 0.55:
        return "bullish"
    elif ratio >= 0.45:
        return "neutral"
    elif ratio >= 0.3:
        return "bearish"
    else:
        return "very_bearish"
