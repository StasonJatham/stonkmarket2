"""Stock suggestions repository using SQLAlchemy ORM.

Repository for managing stock suggestions and votes.

Usage:
    from app.repositories import suggestions_orm as suggestions_repo
    
    suggestion = await suggestions_repo.get_suggestion_by_symbol("AAPL")
    await suggestions_repo.vote_on_suggestion(suggestion_id=1, fingerprint="...", vote_type="up")
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from sqlalchemy import select, func, delete, distinct, and_, or_, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_session
from app.database.orm import StockSuggestion, SuggestionVote, Symbol
from app.core.logging import get_logger

logger = get_logger("repositories.suggestions_orm")


# =============================================================================
# SUGGESTION QUERIES
# =============================================================================


async def get_suggestion_by_id(suggestion_id: int) -> Optional[StockSuggestion]:
    """Get a suggestion by ID."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        return result.scalar_one_or_none()


async def get_suggestion_by_symbol(symbol: str) -> Optional[StockSuggestion]:
    """Get a suggestion by symbol."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.symbol == symbol.upper())
        )
        return result.scalar_one_or_none()


async def get_unique_voter_count(suggestion_id: int) -> int:
    """Get count of unique voters (by fingerprint) for a suggestion."""
    async with get_session() as session:
        result = await session.execute(
            select(func.count(distinct(SuggestionVote.fingerprint)))
            .where(SuggestionVote.suggestion_id == suggestion_id)
        )
        return result.scalar() or 0


async def get_suggestion_age_hours(suggestion_id: int) -> float:
    """Get age of suggestion in hours."""
    async with get_session() as session:
        result = await session.execute(
            select(
                func.extract("epoch", func.now() - StockSuggestion.created_at) / 3600
            ).where(StockSuggestion.id == suggestion_id)
        )
        return float(result.scalar() or 0)


async def auto_approve_suggestion(suggestion_id: int) -> bool:
    """Mark suggestion as auto-approved."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(
                and_(
                    StockSuggestion.id == suggestion_id,
                    StockSuggestion.status == "pending"
                )
            )
        )
        suggestion = result.scalar_one_or_none()
        
        if suggestion:
            suggestion.status = "approved"
            suggestion.approved_by_id = None
            suggestion.reviewed_at = datetime.utcnow()
            await session.commit()
            return True
        return False


# =============================================================================
# VOTING
# =============================================================================


async def get_existing_vote(
    suggestion_id: int,
    fingerprint: str,
) -> Optional[SuggestionVote]:
    """Get existing vote by a user on a suggestion."""
    async with get_session() as session:
        result = await session.execute(
            select(SuggestionVote).where(
                and_(
                    SuggestionVote.suggestion_id == suggestion_id,
                    SuggestionVote.fingerprint == fingerprint
                )
            )
        )
        return result.scalar_one_or_none()


async def upsert_vote(
    suggestion_id: int,
    fingerprint: str,
    vote_type: str,
    vote_weight: int = 1,
    api_key_id: Optional[int] = None,
) -> Tuple[int, bool]:
    """Create or update a vote on a suggestion.
    
    Returns:
        Tuple of (new_score, was_changed)
        - new_score: Updated vote_score for the suggestion
        - was_changed: True if vote was created/changed, False if same vote exists
    """
    async with get_session() as session:
        # Check for existing vote
        existing = await session.execute(
            select(SuggestionVote).where(
                and_(
                    SuggestionVote.suggestion_id == suggestion_id,
                    SuggestionVote.fingerprint == fingerprint
                )
            )
        )
        existing_vote = existing.scalar_one_or_none()
        
        if existing_vote:
            if existing_vote.vote_type == vote_type:
                # Same vote exists, get current score
                score_result = await session.execute(
                    select(StockSuggestion.vote_score).where(
                        StockSuggestion.id == suggestion_id
                    )
                )
                current_score = score_result.scalar() or 0
                return (current_score, False)
            
            # Change vote - update existing
            old_weight = existing_vote.vote_weight if existing_vote.vote_type == "up" else -existing_vote.vote_weight
            new_weight = vote_weight if vote_type == "up" else -vote_weight
            score_delta = new_weight - old_weight
            
            existing_vote.vote_type = vote_type
            existing_vote.vote_weight = vote_weight
            existing_vote.api_key_id = api_key_id
        else:
            # New vote
            new_vote = SuggestionVote(
                suggestion_id=suggestion_id,
                fingerprint=fingerprint,
                vote_type=vote_type,
                vote_weight=vote_weight,
                api_key_id=api_key_id,
            )
            session.add(new_vote)
            score_delta = vote_weight if vote_type == "up" else -vote_weight
        
        # Update suggestion score
        suggestion_result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        suggestion = suggestion_result.scalar_one_or_none()
        
        if suggestion:
            suggestion.vote_score = (suggestion.vote_score or 0) + score_delta
            new_score = suggestion.vote_score
        else:
            new_score = 0
        
        await session.commit()
        return (new_score, True)


async def delete_vote(suggestion_id: int, fingerprint: str) -> int:
    """Delete a vote and return updated score.
    
    Returns:
        Updated vote_score for the suggestion
    """
    async with get_session() as session:
        # Get existing vote
        existing = await session.execute(
            select(SuggestionVote).where(
                and_(
                    SuggestionVote.suggestion_id == suggestion_id,
                    SuggestionVote.fingerprint == fingerprint
                )
            )
        )
        vote = existing.scalar_one_or_none()
        
        if not vote:
            # No vote to delete
            score_result = await session.execute(
                select(StockSuggestion.vote_score).where(
                    StockSuggestion.id == suggestion_id
                )
            )
            return score_result.scalar() or 0
        
        # Calculate score change
        score_delta = -vote.vote_weight if vote.vote_type == "up" else vote.vote_weight
        
        # Delete vote
        await session.delete(vote)
        
        # Update suggestion score
        suggestion_result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        suggestion = suggestion_result.scalar_one_or_none()
        
        if suggestion:
            suggestion.vote_score = (suggestion.vote_score or 0) + score_delta
            new_score = suggestion.vote_score
        else:
            new_score = 0
        
        await session.commit()
        return new_score


# =============================================================================
# ADMIN ACTIONS
# =============================================================================


async def approve_suggestion(
    suggestion_id: int,
    admin_id: int,
) -> bool:
    """Approve a pending suggestion."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(
                and_(
                    StockSuggestion.id == suggestion_id,
                    StockSuggestion.status == "pending"
                )
            )
        )
        suggestion = result.scalar_one_or_none()
        
        if suggestion:
            suggestion.status = "approved"
            suggestion.approved_by_id = admin_id
            suggestion.reviewed_at = datetime.utcnow()
            await session.commit()
            return True
        return False


async def reject_suggestion(
    suggestion_id: int,
    admin_id: int,
) -> bool:
    """Reject a pending suggestion."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(
                and_(
                    StockSuggestion.id == suggestion_id,
                    StockSuggestion.status == "pending"
                )
            )
        )
        suggestion = result.scalar_one_or_none()
        
        if suggestion:
            suggestion.status = "rejected"
            suggestion.approved_by_id = admin_id
            suggestion.reviewed_at = datetime.utcnow()
            await session.commit()
            return True
        return False


async def delete_suggestion(suggestion_id: int) -> bool:
    """Delete a suggestion and its votes."""
    async with get_session() as session:
        result = await session.execute(
            delete(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        await session.commit()
        return result.rowcount > 0


# =============================================================================
# LIST QUERIES
# =============================================================================


async def list_pending_suggestions(
    limit: int = 50,
    offset: int = 0,
) -> Sequence[StockSuggestion]:
    """List pending suggestions ordered by vote score."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion)
            .where(StockSuggestion.status == "pending")
            .order_by(StockSuggestion.vote_score.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()


async def count_pending_suggestions() -> int:
    """Count pending suggestions."""
    async with get_session() as session:
        result = await session.execute(
            select(func.count()).select_from(StockSuggestion)
            .where(StockSuggestion.status == "pending")
        )
        return result.scalar() or 0


async def list_all_suggestions(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[Sequence[StockSuggestion], int]:
    """List suggestions with optional status filter.
    
    Returns:
        Tuple of (suggestions, total_count)
    """
    async with get_session() as session:
        # Base query
        query = select(StockSuggestion)
        count_query = select(func.count()).select_from(StockSuggestion)
        
        if status:
            query = query.where(StockSuggestion.status == status)
            count_query = count_query.where(StockSuggestion.status == status)
        
        # Get count
        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0
        
        # Get suggestions
        result = await session.execute(
            query
            .order_by(StockSuggestion.vote_score.desc(), StockSuggestion.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        suggestions = result.scalars().all()
        
        return (suggestions, total)


async def get_user_voted_suggestions(fingerprint: str) -> Sequence[int]:
    """Get list of suggestion IDs that a user has voted on."""
    async with get_session() as session:
        result = await session.execute(
            select(SuggestionVote.suggestion_id)
            .where(SuggestionVote.fingerprint == fingerprint)
        )
        return [row[0] for row in result.all()]


# =============================================================================
# CREATION
# =============================================================================


async def create_suggestion(
    symbol: str,
    fingerprint: str,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    summary: Optional[str] = None,
    website: Optional[str] = None,
    ipo_year: Optional[int] = None,
    reason: Optional[str] = None,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    fetch_status: Optional[str] = None,
    fetch_error: Optional[str] = None,
) -> StockSuggestion:
    """Create a new stock suggestion."""
    async with get_session() as session:
        suggestion = StockSuggestion(
            symbol=symbol.upper(),
            fingerprint=fingerprint,
            company_name=company_name,
            sector=sector,
            summary=summary,
            website=website,
            ipo_year=ipo_year,
            reason=reason,
            current_price=Decimal(str(current_price)) if current_price else None,
            ath_price=Decimal(str(ath_price)) if ath_price else None,
            status="pending",
            vote_score=0,
            fetch_status=fetch_status or "pending",
            fetch_error=fetch_error,
        )
        session.add(suggestion)
        await session.commit()
        await session.refresh(suggestion)
        return suggestion


# =============================================================================
# SYMBOL CHECKS
# =============================================================================


async def get_tracked_symbols() -> Sequence[str]:
    """Get list of symbols that are already tracked."""
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.symbol)
        )
        return [row[0] for row in result.all()]


async def get_pending_suggestion_symbols() -> Sequence[str]:
    """Get list of symbols with pending suggestions."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion.symbol)
            .where(StockSuggestion.status == "pending")
        )
        return [row[0] for row in result.all()]


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================


async def search_pending_suggestions(
    query: str,
    limit: int = 10,
) -> Sequence[dict]:
    """Search pending suggestions by symbol or company name.
    
    Returns list of dicts with symbol, name, sector, vote_score.
    """
    async with get_session() as session:
        search_pattern = f"%{query.upper()}%"
        
        result = await session.execute(
            select(
                StockSuggestion.symbol,
                StockSuggestion.company_name,
                StockSuggestion.sector,
                StockSuggestion.vote_score,
            )
            .where(
                and_(
                    StockSuggestion.status == "pending",
                    or_(
                        StockSuggestion.symbol.ilike(search_pattern),
                        StockSuggestion.company_name.ilike(search_pattern),
                    )
                )
            )
            .order_by(StockSuggestion.vote_score.desc())
            .limit(limit)
        )
        
        return [
            {
                "symbol": row[0],
                "name": row[1],
                "sector": row[2],
                "vote_score": row[3],
            }
            for row in result.all()
        ]


async def search_tracked_symbols(
    query: str,
    limit: int = 10,
) -> Sequence[dict]:
    """Search tracked symbols by symbol or name.
    
    Returns list of dicts with symbol, name, sector.
    """
    async with get_session() as session:
        search_pattern = f"%{query.upper()}%"
        
        result = await session.execute(
            select(
                Symbol.symbol,
                Symbol.name,
                Symbol.sector,
            )
            .where(
                or_(
                    Symbol.symbol.ilike(search_pattern),
                    Symbol.name.ilike(search_pattern),
                )
            )
            .order_by(Symbol.symbol)
            .limit(limit)
        )
        
        return [
            {
                "symbol": row[0],
                "name": row[1],
                "sector": row[2],
            }
            for row in result.all()
        ]


# =============================================================================
# UPDATE FUNCTIONS
# =============================================================================


async def update_suggestion_fetch_status(
    suggestion_id: int,
    fetch_status: str,
    fetch_error: Optional[str] = None,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    summary: Optional[str] = None,
    website: Optional[str] = None,
    ipo_year: Optional[int] = None,
) -> bool:
    """Update suggestion fetch status and optional stock info."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        suggestion.fetch_status = fetch_status
        suggestion.fetch_error = fetch_error
        suggestion.fetched_at = datetime.utcnow()
        
        if current_price is not None:
            suggestion.current_price = Decimal(str(current_price))
        if ath_price is not None:
            suggestion.ath_price = Decimal(str(ath_price))
        if company_name is not None:
            suggestion.company_name = company_name
        if sector is not None:
            suggestion.sector = sector
        if summary is not None:
            suggestion.summary = summary
        if website is not None:
            suggestion.website = website
        if ipo_year is not None:
            suggestion.ipo_year = ipo_year
        
        await session.commit()
        return True


async def add_symbol_from_suggestion(
    symbol: str,
    name: Optional[str] = None,
    sector: Optional[str] = None,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> bool:
    """Add a symbol to the tracked symbols list (when approving suggestion)."""
    async with get_session() as session:
        # Check if already exists
        existing = await session.execute(
            select(Symbol).where(Symbol.symbol == symbol.upper())
        )
        if existing.scalar_one_or_none():
            return True  # Already exists, consider it success
        
        new_symbol = Symbol(
            symbol=symbol.upper(),
            name=name,
            sector=sector,
            min_dip_pct=Decimal(str(min_dip_pct)),
            min_days=min_days,
            is_active=True,
        )
        session.add(new_symbol)
        await session.commit()
        return True


async def update_suggestion_symbol(
    suggestion_id: int,
    new_symbol: str,
) -> bool:
    """Update a suggestion's symbol (e.g., to correct .F to .DE)."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        suggestion.symbol = new_symbol.upper()
        await session.commit()
        return True


async def check_symbol_exists_in_other_suggestion(
    symbol: str,
    exclude_id: int,
) -> bool:
    """Check if symbol exists in another suggestion."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion.id).where(
                and_(
                    StockSuggestion.symbol == symbol.upper(),
                    StockSuggestion.id != exclude_id
                )
            )
        )
        return result.scalar_one_or_none() is not None


async def update_suggestion_stock_info(
    symbol: str,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    summary: Optional[str] = None,
    website: Optional[str] = None,
    ipo_year: Optional[int] = None,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    fetch_status: Optional[str] = None,
    fetch_error: Optional[str] = None,
) -> bool:
    """Update suggestion stock info by symbol."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.symbol == symbol.upper())
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        if company_name is not None:
            suggestion.company_name = company_name
        if sector is not None:
            suggestion.sector = sector
        if summary is not None:
            suggestion.summary = summary
        if website is not None:
            suggestion.website = website
        if ipo_year is not None:
            suggestion.ipo_year = ipo_year
        if current_price is not None:
            suggestion.current_price = Decimal(str(current_price))
        if ath_price is not None:
            suggestion.ath_price = Decimal(str(ath_price))
        if fetch_status is not None:
            suggestion.fetch_status = fetch_status
            suggestion.fetched_at = datetime.utcnow()
        if fetch_error is not None:
            suggestion.fetch_error = fetch_error
        
        await session.commit()
        return True


async def set_suggestion_fetching(symbol: str) -> bool:
    """Set suggestion fetch_status to 'fetching' (for processing)."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.symbol == symbol.upper())
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        suggestion.fetch_status = "fetching"
        suggestion.fetch_error = None
        await session.commit()
        return True


async def set_suggestion_fetched(symbol: str) -> bool:
    """Mark suggestion as successfully fetched."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.symbol == symbol.upper())
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        suggestion.fetch_status = "fetched"
        suggestion.fetched_at = datetime.utcnow()
        await session.commit()
        return True


async def set_suggestion_error(symbol: str, error: str) -> bool:
    """Mark suggestion with error status."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.symbol == symbol.upper())
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        suggestion.fetch_status = "error"
        suggestion.fetch_error = error[:500] if error else None
        await session.commit()
        return True


async def reject_suggestion_with_reason(
    suggestion_id: int,
    admin_id: Optional[int],
    reason: Optional[str] = None,
) -> bool:
    """Reject a suggestion with reason."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion).where(StockSuggestion.id == suggestion_id)
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            return False
        
        suggestion.status = "rejected"
        suggestion.approved_by_id = admin_id
        suggestion.reviewed_at = datetime.utcnow()
        suggestion.reason = reason
        await session.commit()
        return True


async def list_suggestions_needing_backfill(limit: int = 10) -> Sequence[StockSuggestion]:
    """Get suggestions with pending or null fetch_status."""
    async with get_session() as session:
        result = await session.execute(
            select(StockSuggestion)
            .where(
                or_(
                    StockSuggestion.fetch_status == "pending",
                    StockSuggestion.fetch_status.is_(None)
                )
            )
            .order_by(StockSuggestion.vote_score.desc())
            .limit(limit)
        )
        return result.scalars().all()
