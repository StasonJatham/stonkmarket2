"""Data fetching for notification triggers.

Batch-optimized data retrieval from various sources:
- DipState for dip metrics
- StockFundamentals for fundamentals
- QuantPrecomputed for quant scores
- DipfinderSignal for dipfinder alerts
- StrategySignal for strategy signals
- AiAgentAnalysis for AI ratings
- Portfolio data for portfolio triggers
- Calendar data for earnings/dividends/splits
"""

from __future__ import annotations

from datetime import UTC, datetime, date
from typing import Any

from sqlalchemy import select, and_

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import (
    DipState,
    StockFundamentals,
    QuantPrecomputed,
    DipfinderSignal,
    StrategySignal,
    AiAgentAnalysis,
    Portfolio,
    PortfolioHolding,
    PortfolioAnalytics,
    CalendarEarnings,
    CalendarSplit,
)


logger = get_logger("notifications.data_fetcher")


async def get_trigger_data(
    symbol: str | None = None,
    portfolio_id: int | None = None,
    trigger_type: str | None = None,
) -> dict[str, Any]:
    """Get all data needed for trigger evaluation.
    
    This is a convenience wrapper that fetches data for a single symbol
    or portfolio. For batch processing, use batch_fetch_* functions.
    
    Args:
        symbol: Stock symbol (for symbol-specific triggers)
        portfolio_id: Portfolio ID (for portfolio triggers)
        trigger_type: Optional trigger type to optimize fetching
        
    Returns:
        Dict with all available metrics
    """
    data: dict[str, Any] = {
        "fetched_at": datetime.now(UTC),
    }
    
    if symbol:
        symbol_data = await batch_fetch_symbol_data([symbol.upper()])
        data.update(symbol_data.get(symbol.upper(), {}))
    
    if portfolio_id:
        portfolio_data = await batch_fetch_portfolio_data([portfolio_id])
        data.update(portfolio_data.get(portfolio_id, {}))
    
    return data


async def batch_fetch_symbol_data(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch data for multiple symbols in a single batch.
    
    Optimized for the notification checker to minimize database queries.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dict mapping symbol -> data dict
    """
    if not symbols:
        return {}
    
    symbols = [s.upper() for s in symbols]
    result: dict[str, dict[str, Any]] = {s: {} for s in symbols}
    
    async with get_session() as session:
        # Fetch DipState data
        await _fetch_dip_state_data(session, symbols, result)
        
        # Fetch Fundamentals data
        await _fetch_fundamentals_data(session, symbols, result)
        
        # Fetch Quant scores
        await _fetch_quant_data(session, symbols, result)
        
        # Fetch Dipfinder signals
        await _fetch_dipfinder_signals(session, symbols, result)
        
        # Fetch Strategy signals
        await _fetch_strategy_signals(session, symbols, result)
        
        # Fetch AI analysis
        await _fetch_ai_analysis(session, symbols, result)
        
        # Fetch calendar data
        await _fetch_calendar_data(session, symbols, result)
    
    return result


async def _fetch_dip_state_data(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch DipState data for symbols."""
    stmt = select(DipState).where(DipState.symbol.in_(symbols))
    rows = await session.execute(stmt)
    
    for dip in rows.scalars():
        data = result[dip.symbol]
        data["current_price"] = float(dip.current_price) if dip.current_price else None
        data["dip_percent"] = float(dip.dip_percent) if dip.dip_percent else None
        data["recovery_percent"] = float(dip.recovery_percent) if dip.recovery_percent else None
        data["dip_from_52w_high"] = float(dip.week52_low_distance) if dip.week52_low_distance else None
        data["price_change_percent"] = float(dip.change_percent) if dip.change_percent else None
        data["dip_updated_at"] = dip.updated_at


async def _fetch_fundamentals_data(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch StockFundamentals data for symbols."""
    stmt = select(StockFundamentals).where(StockFundamentals.symbol.in_(symbols))
    rows = await session.execute(stmt)
    
    for fund in rows.scalars():
        data = result[fund.symbol]
        data["pe_ratio"] = float(fund.pe_ratio) if fund.pe_ratio else None
        data["dividend_yield"] = float(fund.dividend_yield) if fund.dividend_yield else None
        data["market_cap"] = float(fund.market_cap) if fund.market_cap else None
        data["fundamentals_updated_at"] = fund.updated_at


async def _fetch_quant_data(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch QuantPrecomputed data for symbols."""
    stmt = select(QuantPrecomputed).where(QuantPrecomputed.symbol.in_(symbols))
    rows = await session.execute(stmt)
    
    for quant in rows.scalars():
        data = result[quant.symbol]
        data["quant_score"] = quant.composite_score
        data["value_score"] = quant.value_score
        data["growth_score"] = quant.growth_score
        data["momentum_score"] = quant.momentum_score
        data["quant_recommendation"] = quant.recommendation
        data["quant_updated_at"] = quant.updated_at


async def _fetch_dipfinder_signals(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch active Dipfinder signals for symbols."""
    stmt = select(DipfinderSignal).where(
        and_(
            DipfinderSignal.symbol.in_(symbols),
            DipfinderSignal.is_active == True,  # noqa: E712
        )
    )
    rows = await session.execute(stmt)
    
    for signal in rows.scalars():
        data = result[signal.symbol]
        data["dipfinder_active"] = True
        data["dipfinder_signal_type"] = signal.signal_type
        data["dipfinder_strength"] = signal.strength
        data["dipfinder_confidence"] = float(signal.confidence) if signal.confidence else None


async def _fetch_strategy_signals(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch latest Strategy signals for symbols."""
    # Get latest signal per symbol
    from sqlalchemy import func
    
    subq = (
        select(
            StrategySignal.symbol,
            func.max(StrategySignal.generated_at).label("max_date"),
        )
        .where(StrategySignal.symbol.in_(symbols))
        .group_by(StrategySignal.symbol)
        .subquery()
    )
    
    stmt = select(StrategySignal).join(
        subq,
        and_(
            StrategySignal.symbol == subq.c.symbol,
            StrategySignal.generated_at == subq.c.max_date,
        )
    )
    rows = await session.execute(stmt)
    
    for signal in rows.scalars():
        data = result[signal.symbol]
        data["strategy_signal"] = signal.signal_type
        data["strategy_name"] = signal.strategy_name
        data["strategy_confidence"] = float(signal.confidence) if signal.confidence else None


async def _fetch_ai_analysis(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch latest AI analysis for symbols."""
    from sqlalchemy import func
    
    subq = (
        select(
            AiAgentAnalysis.symbol,
            func.max(AiAgentAnalysis.analyzed_at).label("max_date"),
        )
        .where(AiAgentAnalysis.symbol.in_(symbols))
        .group_by(AiAgentAnalysis.symbol)
        .subquery()
    )
    
    stmt = select(AiAgentAnalysis).join(
        subq,
        and_(
            AiAgentAnalysis.symbol == subq.c.symbol,
            AiAgentAnalysis.analyzed_at == subq.c.max_date,
        )
    )
    rows = await session.execute(stmt)
    
    for analysis in rows.scalars():
        data = result[analysis.symbol]
        data["ai_rating"] = analysis.rating
        data["ai_opportunity"] = analysis.opportunity_type is not None
        data["ai_opportunity_type"] = analysis.opportunity_type
        data["ai_risk_alert"] = analysis.risk_level in ("high", "critical")
        data["ai_risk_type"] = analysis.risk_level


async def _fetch_calendar_data(
    session: Any,
    symbols: list[str],
    result: dict[str, dict[str, Any]],
) -> None:
    """Fetch upcoming calendar events for symbols."""
    today = date.today()
    
    # Earnings
    stmt = select(CalendarEarnings).where(
        and_(
            CalendarEarnings.symbol.in_(symbols),
            CalendarEarnings.earnings_date >= today,
        )
    ).order_by(CalendarEarnings.earnings_date)
    rows = await session.execute(stmt)
    
    for earnings in rows.scalars():
        data = result.get(earnings.symbol, {})
        if "earnings_days_until" not in data:  # Only take the soonest
            days = (earnings.earnings_date - today).days
            data["earnings_days_until"] = days
    
    # Splits
    stmt = select(CalendarSplit).where(
        and_(
            CalendarSplit.symbol.in_(symbols),
            CalendarSplit.split_date >= today,
        )
    ).order_by(CalendarSplit.split_date)
    rows = await session.execute(stmt)
    
    for split in rows.scalars():
        data = result.get(split.symbol, {})
        if "split_days_until" not in data:
            days = (split.split_date - today).days
            data["split_days_until"] = days


async def batch_fetch_portfolio_data(
    portfolio_ids: list[int],
) -> dict[int, dict[str, Any]]:
    """Fetch data for multiple portfolios.
    
    Args:
        portfolio_ids: List of portfolio IDs
        
    Returns:
        Dict mapping portfolio_id -> data dict
    """
    if not portfolio_ids:
        return {}
    
    result: dict[int, dict[str, Any]] = {pid: {} for pid in portfolio_ids}
    
    async with get_session() as session:
        # Fetch portfolios with holdings
        stmt = select(Portfolio).where(Portfolio.id.in_(portfolio_ids))
        rows = await session.execute(stmt)
        
        for portfolio in rows.scalars():
            data = result[portfolio.id]
            data["portfolio_id"] = portfolio.id
            data["portfolio_name"] = portfolio.name
        
        # Fetch latest analytics
        from sqlalchemy import func
        
        subq = (
            select(
                PortfolioAnalytics.portfolio_id,
                func.max(PortfolioAnalytics.calculated_at).label("max_date"),
            )
            .where(PortfolioAnalytics.portfolio_id.in_(portfolio_ids))
            .group_by(PortfolioAnalytics.portfolio_id)
            .subquery()
        )
        
        stmt = select(PortfolioAnalytics).join(
            subq,
            and_(
                PortfolioAnalytics.portfolio_id == subq.c.portfolio_id,
                PortfolioAnalytics.calculated_at == subq.c.max_date,
            )
        )
        rows = await session.execute(stmt)
        
        for analytics in rows.scalars():
            data = result[analytics.portfolio_id]
            data["portfolio_total_value"] = float(analytics.total_value) if analytics.total_value else None
            data["portfolio_daily_change_percent"] = float(analytics.daily_return_pct) if analytics.daily_return_pct else None
            data["portfolio_drawdown"] = float(analytics.max_drawdown) if analytics.max_drawdown else None
            data["portfolio_updated_at"] = analytics.calculated_at
        
        # Fetch holdings for position-level triggers
        stmt = select(PortfolioHolding).where(PortfolioHolding.portfolio_id.in_(portfolio_ids))
        rows = await session.execute(stmt)
        
        for holding in rows.scalars():
            data = result[holding.portfolio_id]
            if "holdings" not in data:
                data["holdings"] = []
            
            data["holdings"].append({
                "symbol": holding.symbol,
                "quantity": float(holding.quantity),
                "avg_cost": float(holding.avg_cost) if holding.avg_cost else None,
            })
    
    return result
