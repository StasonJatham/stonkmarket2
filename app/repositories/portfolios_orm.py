"""Portfolio repository - SQLAlchemy ORM async."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, desc, select
from sqlalchemy.dialects.postgresql import insert

from app.database.connection import get_session
from app.database.orm import (
    Portfolio,
    PortfolioAnalytics,
    PortfolioHolding,
    PortfolioTransaction,
)


# ───────────────────────────────────────────────────────────────────────────────
# Portfolio CRUD
# ───────────────────────────────────────────────────────────────────────────────


def _portfolio_to_dict(p: Portfolio) -> dict[str, Any]:
    """Convert Portfolio ORM object to dictionary."""
    return {
        "id": p.id,
        "user_id": p.user_id,
        "name": p.name,
        "description": p.description,
        "base_currency": p.base_currency,
        "is_active": p.is_active,
        "visibility": p.visibility or "private",
        "share_token": p.share_token,
        "shared_at": p.shared_at,
        "ai_analysis_summary": p.ai_analysis_summary,
        "ai_analysis_hash": p.ai_analysis_hash,
        "ai_analysis_at": p.ai_analysis_at,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
    }


async def create_portfolio(
    user_id: int,
    name: str,
    *,
    description: str | None = None,
    base_currency: str = "USD",
    visibility: str = "private",
) -> dict[str, Any]:
    """Create a new portfolio."""
    async with get_session() as session:
        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description,
            base_currency=base_currency,
            is_active=True,
            visibility=visibility,
        )
        session.add(portfolio)
        await session.commit()
        await session.refresh(portfolio)
        return _portfolio_to_dict(portfolio)


async def list_portfolios(user_id: int) -> list[dict[str, Any]]:
    """List active portfolios for a user."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio)
            .where(Portfolio.user_id == user_id, Portfolio.is_active == True)
            .order_by(desc(Portfolio.created_at))
        )
        return [_portfolio_to_dict(p) for p in result.scalars().all()]


async def list_all_active_portfolios() -> list[dict[str, Any]]:
    """List all active portfolios across all users (for batch jobs)."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio)
            .where(Portfolio.is_active == True)
            .order_by(Portfolio.id)
        )
        return [_portfolio_to_dict(p) for p in result.scalars().all()]


async def get_portfolio(portfolio_id: int, user_id: int) -> dict[str, Any] | None:
    """Get a portfolio by id."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == user_id,
                Portfolio.is_active == True,
            )
        )
        portfolio = result.scalar_one_or_none()
        return _portfolio_to_dict(portfolio) if portfolio else None


async def update_portfolio(
    portfolio_id: int,
    user_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    base_currency: str | None = None,
    is_active: bool | None = None,
) -> dict[str, Any] | None:
    """Update a portfolio."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == user_id,
            )
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return None

        if name is not None:
            portfolio.name = name
        if description is not None:
            portfolio.description = description
        if base_currency is not None:
            portfolio.base_currency = base_currency
        if is_active is not None:
            portfolio.is_active = is_active

        portfolio.updated_at = datetime.now()
        await session.commit()
        await session.refresh(portfolio)
        return _portfolio_to_dict(portfolio)


async def archive_portfolio(portfolio_id: int, user_id: int) -> bool:
    """Soft-delete a portfolio."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == user_id,
            )
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return False

        portfolio.is_active = False
        portfolio.updated_at = datetime.now()
        await session.commit()
        return True


# ───────────────────────────────────────────────────────────────────────────────
# Visibility & Sharing (Multi-Tenant)
# ───────────────────────────────────────────────────────────────────────────────


async def get_portfolio_with_visibility(portfolio_id: int) -> dict[str, Any] | None:
    """
    Get portfolio with visibility info for authorization checks.
    
    This function does NOT check ownership - use for authorization layer.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(
                Portfolio.id == portfolio_id,
                Portfolio.is_active == True,
            )
        )
        portfolio = result.scalar_one_or_none()
        return _portfolio_to_dict(portfolio) if portfolio else None


async def get_portfolio_by_share_token(share_token: str) -> dict[str, Any] | None:
    """Get a portfolio by its share token (for public/shared access)."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(
                Portfolio.share_token == share_token,
                Portfolio.is_active == True,
            )
        )
        portfolio = result.scalar_one_or_none()
        return _portfolio_to_dict(portfolio) if portfolio else None


async def set_share_token(
    portfolio_id: int,
    share_token: str | None,
    visibility: str,
) -> bool:
    """Set or revoke share token and update visibility."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return False
        
        portfolio.share_token = share_token
        portfolio.visibility = visibility
        portfolio.shared_at = datetime.now() if share_token else None
        portfolio.updated_at = datetime.now()
        await session.commit()
        return True


async def update_portfolio_visibility(
    portfolio_id: int,
    visibility: str,
) -> bool:
    """Update portfolio visibility."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return False
        
        portfolio.visibility = visibility
        portfolio.updated_at = datetime.now()
        await session.commit()
        return True


async def list_public_portfolios(
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List public portfolios for discovery."""
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio)
            .where(
                Portfolio.visibility == "public",
                Portfolio.is_active == True,
            )
            .order_by(desc(Portfolio.updated_at))
            .limit(limit)
            .offset(offset)
        )
        return [_portfolio_to_dict(p) for p in result.scalars().all()]


# ───────────────────────────────────────────────────────────────────────────────
# Holdings
# ───────────────────────────────────────────────────────────────────────────────


def _holding_to_dict(h: PortfolioHolding) -> dict[str, Any]:
    """Convert PortfolioHolding ORM object to dictionary."""
    return {
        "id": h.id,
        "portfolio_id": h.portfolio_id,
        "symbol": h.symbol,
        "quantity": h.quantity,
        "avg_cost": h.avg_cost,
        "target_weight": h.target_weight,
        "created_at": h.created_at,
        "updated_at": h.updated_at,
    }


async def list_holdings(portfolio_id: int) -> list[dict[str, Any]]:
    """List holdings for a portfolio."""
    async with get_session() as session:
        result = await session.execute(
            select(PortfolioHolding)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .order_by(PortfolioHolding.symbol)
        )
        return [_holding_to_dict(h) for h in result.scalars().all()]


async def upsert_holding(
    portfolio_id: int,
    symbol: str,
    *,
    quantity: Decimal | float | int,
    avg_cost: Decimal | float | int | None = None,
    target_weight: Decimal | float | int | None = None,
) -> dict[str, Any]:
    """Create or update a holding."""
    async with get_session() as session:
        stmt = insert(PortfolioHolding).values(
            portfolio_id=portfolio_id,
            symbol=symbol.upper(),
            quantity=Decimal(str(quantity)),
            avg_cost=Decimal(str(avg_cost)) if avg_cost is not None else None,
            target_weight=Decimal(str(target_weight)) if target_weight is not None else None,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["portfolio_id", "symbol"],
            set_={
                "quantity": stmt.excluded.quantity,
                "avg_cost": stmt.excluded.avg_cost,
                "target_weight": stmt.excluded.target_weight,
                "updated_at": datetime.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()

        # Fetch the updated/inserted record
        result = await session.execute(
            select(PortfolioHolding).where(
                PortfolioHolding.portfolio_id == portfolio_id,
                PortfolioHolding.symbol == symbol.upper(),
            )
        )
        holding = result.scalar_one()
        return _holding_to_dict(holding)


async def delete_holding(portfolio_id: int, symbol: str) -> bool:
    """Delete a holding."""
    async with get_session() as session:
        result = await session.execute(
            delete(PortfolioHolding).where(
                PortfolioHolding.portfolio_id == portfolio_id,
                PortfolioHolding.symbol == symbol.upper(),
            )
        )
        await session.commit()
        return result.rowcount > 0


# ───────────────────────────────────────────────────────────────────────────────
# Transactions
# ───────────────────────────────────────────────────────────────────────────────


def _transaction_to_dict(t: PortfolioTransaction) -> dict[str, Any]:
    """Convert PortfolioTransaction ORM object to dictionary."""
    return {
        "id": t.id,
        "portfolio_id": t.portfolio_id,
        "symbol": t.symbol,
        "side": t.side,
        "quantity": t.quantity,
        "price": t.price,
        "fees": t.fees,
        "trade_date": t.trade_date,
        "notes": t.notes,
        "created_at": t.created_at,
    }


async def add_transaction(
    portfolio_id: int,
    symbol: str,
    *,
    side: str,
    quantity: Decimal | float | int | None,
    price: Decimal | float | int | None,
    fees: Decimal | float | int | None,
    trade_date: date,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add a portfolio transaction."""
    async with get_session() as session:
        txn = PortfolioTransaction(
            portfolio_id=portfolio_id,
            symbol=symbol.upper(),
            side=side,
            quantity=Decimal(str(quantity)) if quantity is not None else None,
            price=Decimal(str(price)) if price is not None else None,
            fees=Decimal(str(fees)) if fees is not None else None,
            trade_date=trade_date,
            notes=notes,
        )
        session.add(txn)
        await session.commit()
        await session.refresh(txn)
        return _transaction_to_dict(txn)


async def list_transactions(portfolio_id: int, limit: int = 200) -> list[dict[str, Any]]:
    """List recent transactions."""
    async with get_session() as session:
        result = await session.execute(
            select(PortfolioTransaction)
            .where(PortfolioTransaction.portfolio_id == portfolio_id)
            .order_by(
                desc(PortfolioTransaction.trade_date),
                desc(PortfolioTransaction.created_at),
            )
            .limit(limit)
        )
        return [_transaction_to_dict(t) for t in result.scalars().all()]


async def delete_transaction(portfolio_id: int, transaction_id: int) -> bool:
    """Delete a transaction."""
    async with get_session() as session:
        result = await session.execute(
            delete(PortfolioTransaction).where(
                PortfolioTransaction.id == transaction_id,
                PortfolioTransaction.portfolio_id == portfolio_id,
            )
        )
        await session.commit()
        return result.rowcount > 0


# ───────────────────────────────────────────────────────────────────────────────
# Analytics
# ───────────────────────────────────────────────────────────────────────────────


def _analytics_to_dict(a: PortfolioAnalytics) -> dict[str, Any]:
    """Convert PortfolioAnalytics ORM object to dictionary."""
    return {
        "id": a.id,
        "portfolio_id": a.portfolio_id,
        "tool": a.tool,
        "as_of_date": a.as_of_date,
        "window": a.window,
        "params": a.params,
        "payload": a.payload,
        "status": a.status,
        "created_at": a.created_at,
    }


async def save_portfolio_analytics(
    portfolio_id: int,
    *,
    tool: str,
    payload: dict[str, Any],
    status: str = "ok",
    as_of_date: date | None = None,
    window: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store analytics output for a portfolio tool."""
    async with get_session() as session:
        analytics = PortfolioAnalytics(
            portfolio_id=portfolio_id,
            tool=tool,
            as_of_date=as_of_date,
            window=window,
            params=params,
            payload=payload,
            status=status,
        )
        session.add(analytics)
        await session.commit()
        await session.refresh(analytics)
        return _analytics_to_dict(analytics)


async def get_latest_analytics(
    portfolio_id: int,
    *,
    tool: str,
    window: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Get latest analytics for a tool."""
    async with get_session() as session:
        query = (
            select(PortfolioAnalytics)
            .where(
                PortfolioAnalytics.portfolio_id == portfolio_id,
                PortfolioAnalytics.tool == tool,
            )
            .order_by(desc(PortfolioAnalytics.created_at))
            .limit(1)
        )
        if window is not None:
            query = query.where(PortfolioAnalytics.window == window)
        if params is not None:
            query = query.where(PortfolioAnalytics.params == params)

        result = await session.execute(query)
        analytics = result.scalar_one_or_none()
        return _analytics_to_dict(analytics) if analytics else None


# ───────────────────────────────────────────────────────────────────────────────
# Transaction + Holdings adjustment (transactional)
# ───────────────────────────────────────────────────────────────────────────────


async def apply_transaction_to_holdings(
    portfolio_id: int,
    *,
    symbol: str,
    side: str,
    quantity: Decimal | float | int,
    price: Decimal | float | int,
) -> None:
    """Apply a transaction to holdings inside a transaction."""
    qty = Decimal(str(quantity))
    px = Decimal(str(price))
    symbol = symbol.upper()

    async with get_session() as session:
        # Get current holding with lock
        result = await session.execute(
            select(PortfolioHolding)
            .where(
                PortfolioHolding.portfolio_id == portfolio_id,
                PortfolioHolding.symbol == symbol,
            )
            .with_for_update()
        )
        holding = result.scalar_one_or_none()

        current_qty = holding.quantity if holding else Decimal("0")
        current_cost = holding.avg_cost if holding and holding.avg_cost is not None else None

        if side == "buy":
            new_qty = current_qty + qty
            if current_cost is None:
                new_cost = px
            else:
                new_cost = (current_cost * current_qty + px * qty) / new_qty if new_qty != 0 else current_cost
        elif side == "sell":
            new_qty = current_qty - qty
            new_cost = current_cost
        else:
            return

        if holding:
            if new_qty <= 0:
                # Delete the holding
                await session.delete(holding)
            else:
                holding.quantity = new_qty
                holding.avg_cost = new_cost
                holding.updated_at = datetime.now()
        else:
            # Insert new holding
            new_holding = PortfolioHolding(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=new_qty,
                avg_cost=new_cost,
            )
            session.add(new_holding)

        await session.commit()


# ───────────────────────────────────────────────────────────────────────────────
# Portfolio AI Analysis
# ───────────────────────────────────────────────────────────────────────────────


async def get_portfolios_needing_ai_analysis() -> list[dict[str, Any]]:
    """
    Get portfolios that need AI analysis.
    
    A portfolio needs analysis if:
    - It has never been analyzed (ai_analysis_hash is None)
    - Its current holdings hash differs from stored hash (portfolio changed)
    """
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio)
            .where(Portfolio.is_active == True)
            .order_by(Portfolio.id)
        )
        portfolios = result.scalars().all()
        
        portfolios_needing_analysis = []
        for p in portfolios:
            # Get current holdings to compute hash
            holdings_result = await session.execute(
                select(PortfolioHolding)
                .where(PortfolioHolding.portfolio_id == p.id)
                .order_by(PortfolioHolding.symbol)
            )
            holdings = holdings_result.scalars().all()
            
            if not holdings:
                continue  # Skip empty portfolios
            
            # Import here to avoid circular dependency
            from app.hedge_fund.agents.portfolio_advisor import compute_portfolio_hash
            
            holdings_data = [
                {
                    "symbol": h.symbol,
                    "quantity": float(h.quantity),
                    "avg_cost": float(h.avg_cost) if h.avg_cost else 0,
                }
                for h in holdings
            ]
            current_hash = compute_portfolio_hash(holdings_data)
            
            # Check if analysis is needed
            if p.ai_analysis_hash != current_hash:
                portfolios_needing_analysis.append({
                    **_portfolio_to_dict(p),
                    "holdings": holdings_data,
                    "current_hash": current_hash,
                })
        
        return portfolios_needing_analysis


async def update_portfolio_ai_analysis(
    portfolio_id: int,
    summary: str,
    holdings_hash: str,
) -> bool:
    """
    Update portfolio with AI analysis results.
    
    Args:
        portfolio_id: Portfolio ID
        summary: AI-generated analysis summary
        holdings_hash: Hash of holdings at time of analysis
        
    Returns:
        True if updated successfully
    """
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return False
        
        portfolio.ai_analysis_summary = summary
        portfolio.ai_analysis_hash = holdings_hash
        portfolio.ai_analysis_at = datetime.now()
        
        await session.commit()
        return True


async def get_portfolio_with_holdings_for_analysis(
    portfolio_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """
    Get portfolio and its holdings for AI analysis.
    
    Returns:
        Tuple of (portfolio_dict, holdings_list) or None if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(
                Portfolio.id == portfolio_id,
                Portfolio.is_active == True,
            )
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            return None
        
        holdings_result = await session.execute(
            select(PortfolioHolding)
            .where(PortfolioHolding.portfolio_id == portfolio_id)
            .order_by(PortfolioHolding.symbol)
        )
        holdings = holdings_result.scalars().all()
        
        holdings_data = [
            {
                "symbol": h.symbol,
                "quantity": float(h.quantity),
                "avg_cost": float(h.avg_cost) if h.avg_cost else 0,
            }
            for h in holdings
        ]
        
        return _portfolio_to_dict(portfolio), holdings_data
