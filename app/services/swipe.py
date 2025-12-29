"""Swipe service - combines dips with AI analysis and voting (PostgreSQL)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, or_, select

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import DipAIAnalysis, DipState, Symbol
from app.dipfinder.service import get_dipfinder_service
from app.repositories import dip_votes_orm as dip_votes_repo
from app.schemas.swipe import DipCard, DipStats, VoteCounts
from app.services import stock_info
from app.services.fundamentals import get_fundamentals_for_analysis
from app.services.openai_client import generate_bio, rate_dip
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
        # Try to compute if not cached
        try:
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
        vote_counts=vote_counts,
        summary_ai=sym.summary_ai if sym else None,
        swipe_bio=ai_analysis.get("swipe_bio") if ai_analysis else None,
        ai_rating=ai_analysis.get("ai_rating") if ai_analysis else None,
        ai_reasoning=ai_analysis.get("ai_reasoning") if ai_analysis else None,
    )

    return card


async def get_dip_card_with_fresh_ai(symbol: str, force_refresh: bool = False) -> DipCard | None:
    """
    Get dip card with fresh AI analysis (generates if needed or forced).

    This fetches stock info and generates AI content if not cached.
    
    Args:
        symbol: Stock symbol
        force_refresh: If True, regenerate AI even if cached
        
    Returns:
        DipCard with AI content populated.
        None if symbol not found in dip state.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None

    # If AI analysis already cached and not forcing refresh, return as-is
    if card.swipe_bio and card.ai_rating and not force_refresh:
        return card

    # Prefer local symbol data to avoid extra yfinance calls
    info = None
    name = card.name
    sector = card.sector
    summary = card.summary_ai
    industry = card.industry
    website = card.website
    ipo_year = card.ipo_year

    if not (name and sector and summary):
        info = await stock_info.get_stock_info_async(symbol)
        if info:
            name = info.get("name") or name
            sector = info.get("sector") or sector
            summary = info.get("summary") or summary
            industry = info.get("industry") or industry
            website = info.get("website") or website
            ipo_year = info.get("ipo_year") or ipo_year

    # Generate AI content only if missing or forced
    bio = card.swipe_bio
    rating_result = None
    generated_ai = False

    if force_refresh or not card.swipe_bio:
        bio = await generate_bio(
            symbol=symbol,
            name=name,
            sector=sector,
            summary=summary,
            dip_pct=card.dip_pct,
        )
        if bio:
            generated_ai = True

    if force_refresh or not card.ai_rating:
        fundamentals = await get_fundamentals_for_analysis(symbol)
        rating_result = await rate_dip(
            symbol=symbol,
            current_price=card.current_price,
            ref_high=card.ref_high,
            dip_pct=card.dip_pct,
            days_below=card.days_below,
            name=name,
            sector=sector,
            summary=summary,
            **fundamentals,
        )
        if rating_result:
            generated_ai = True

    if generated_ai:
        await dip_votes_repo.upsert_ai_analysis(
            symbol=symbol,
            swipe_bio=bio,
            ai_rating=rating_result.get("rating") if rating_result else card.ai_rating,
            ai_reasoning=rating_result.get("reasoning") if rating_result else card.ai_reasoning,
            is_batch=False,
        )

    # Update card with fresh data
    card = card.model_copy(update={
        "name": name,
        "sector": sector,
        "industry": industry,
        "website": website,
        "ipo_year": ipo_year,
        "swipe_bio": bio,
        "ai_rating": rating_result.get("rating") if rating_result else card.ai_rating,
        "ai_reasoning": rating_result.get("reasoning") if rating_result else card.ai_reasoning,
        "ai_confidence": rating_result.get("confidence") if rating_result else card.ai_confidence,
    })

    return card


async def regenerate_ai_field(
    symbol: str,
    field: str,
) -> DipCard | None:
    """
    Regenerate a swipe-specific AI field for a dip card.

    Args:
        symbol: Stock symbol
        field: Field to regenerate: 'rating' or 'bio'
               Note: 'summary' should use /symbols/{symbol}/ai/summary endpoint

    Returns:
        Updated DipCard with the regenerated field.
        None if symbol not found in dip state.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None

    existing_ai = await dip_votes_repo.get_ai_analysis(symbol)
    existing_bio = existing_ai.get("swipe_bio") if existing_ai else card.swipe_bio
    existing_rating = existing_ai.get("ai_rating") if existing_ai else card.ai_rating
    existing_reasoning = existing_ai.get("ai_reasoning") if existing_ai else card.ai_reasoning

    info = None
    name = card.name
    sector = card.sector
    summary = card.summary_ai
    industry = card.industry
    website = card.website
    ipo_year = card.ipo_year

    if not (name and sector and summary):
        info = await stock_info.get_stock_info_async(symbol)
        if info:
            name = info.get("name") or name
            sector = info.get("sector") or sector
            summary = info.get("summary") or summary
            industry = info.get("industry") or industry
            website = info.get("website") or website
            ipo_year = info.get("ipo_year") or ipo_year

    if field == "bio":
        # Regenerate Swipe bio
        bio = await generate_bio(
            symbol=symbol,
            name=name,
            sector=sector,
            summary=summary,
            dip_pct=card.dip_pct,
        )
        if bio:
            await dip_votes_repo.upsert_ai_analysis(
                symbol=symbol,
                swipe_bio=bio,
                ai_rating=existing_rating,
                ai_reasoning=existing_reasoning,
                is_batch=False,
            )
            card = card.model_copy(update={
                "swipe_bio": bio,
                "ai_rating": existing_rating,
                "ai_reasoning": existing_reasoning,
            })

    elif field == "rating":
        # Get fundamentals for richer AI analysis
        fundamentals = await get_fundamentals_for_analysis(symbol)

        # Regenerate AI rating
        rating_result = await rate_dip(
            symbol=symbol,
            current_price=card.current_price,
            ref_high=card.ref_high,
            dip_pct=card.dip_pct,
            days_below=card.days_below,
            name=name,
            sector=sector,
            summary=summary,
            **fundamentals,  # Include all fundamental metrics
        )
        if rating_result:
            await dip_votes_repo.upsert_ai_analysis(
                symbol=symbol,
                swipe_bio=existing_bio,
                ai_rating=rating_result.get("rating"),
                ai_reasoning=rating_result.get("reasoning"),
                is_batch=False,
            )
            card = card.model_copy(update={
                "swipe_bio": existing_bio,
                "ai_rating": rating_result.get("rating"),
                "ai_reasoning": rating_result.get("reasoning"),
                "ai_confidence": rating_result.get("confidence"),
            })

    # Update card with stock info
    card = card.model_copy(update={
        "name": name,
        "sector": sector,
        "industry": industry,
        "website": website,
        "ipo_year": ipo_year,
    })

    return card


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
