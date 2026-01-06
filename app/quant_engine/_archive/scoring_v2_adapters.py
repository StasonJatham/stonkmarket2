"""
Data adapters for scoring_v2 integration.

DEPRECATED: This module is deprecated along with scoring_v2.
Use app.quant_engine.scoring.ScoringOrchestrator for all new code.

These adapters transform existing data sources (quant_precomputed, strategy_signals, 
stock_fundamentals) into the input formats required by scoring_v2.

The purpose is to:
1. Read all precomputed data (no recalculation)
2. Transform to scoring_v2 input formats
3. Enable batch scoring with universe statistics

Author: Quant Engine Team
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Emit deprecation warning on import
warnings.warn(
    "scoring_v2_adapters is deprecated. Use app.quant_engine.scoring.ScoringOrchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

from app.database.orm import (
    DipState,
    QuantPrecomputed,
    QuantScore,
    StrategySignal,
    StockFundamentals,
    Symbol,
)
from app.quant_engine.scoring_v2 import (
    BacktestV2Input,
    DipEntryInput,
    FundamentalsInput,
)

logger = logging.getLogger(__name__)


def _float_or_default(val: Any, default: float = 0.0) -> float:
    """Safely convert to float, handling Decimal, None, etc."""
    if val is None:
        return default
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_nested(data: dict | None, *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    if data is None:
        return default
    
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


async def load_backtest_input(
    session: AsyncSession,
    symbol: str,
    precomputed: QuantPrecomputed | None = None,
    strategy: StrategySignal | None = None,
) -> BacktestV2Input:
    """
    Load backtest_v2 metrics from quant_precomputed and strategy_signals.
    
    Data sources:
    - quant_precomputed: Individual backtest columns
    - strategy_signals: Summary metrics from baseline comparison
    
    Args:
        session: Database session
        symbol: Stock ticker
        precomputed: Pre-loaded QuantPrecomputed (optional)
        strategy: Pre-loaded StrategySignal (optional)
        
    Returns:
        BacktestV2Input populated from database
    """
    # Load if not provided
    if precomputed is None:
        result = await session.execute(
            select(QuantPrecomputed).where(QuantPrecomputed.symbol == symbol)
        )
        precomputed = result.scalar_one_or_none()
    
    if strategy is None:
        result = await session.execute(
            select(StrategySignal).where(StrategySignal.symbol == symbol)
        )
        strategy = result.scalar_one_or_none()
    
    input_data = BacktestV2Input()
    
    # Extract from quant_precomputed individual columns
    if precomputed:
        # From trade_* columns
        input_data.total_return_pct = _float_or_default(precomputed.trade_total_return_pct)
        input_data.win_rate = _float_or_default(precomputed.trade_win_rate) * 100  # Convert to %
        input_data.sharpe_ratio = _float_or_default(precomputed.trade_sharpe_ratio)
        input_data.n_trades = int(precomputed.trade_n_trades or 0)
        input_data.avg_trade_return = _float_or_default(precomputed.trade_avg_return_pct)
        
        # From backtest_* columns  
        if not input_data.total_return_pct:
            input_data.total_return_pct = _float_or_default(precomputed.backtest_total_return_pct)
        if not input_data.win_rate:
            input_data.win_rate = _float_or_default(precomputed.backtest_win_rate) * 100
        if not input_data.n_trades:
            input_data.n_trades = int(precomputed.backtest_n_trades or 0)
        
        # Calculate vs buy-and-hold if available
        bh_return = _float_or_default(precomputed.trade_buy_hold_return_pct)
        if bh_return and input_data.total_return_pct:
            input_data.vs_buyhold_pct = input_data.total_return_pct - bh_return
        
        # Calculate vs SPY if available
        spy_return = _float_or_default(precomputed.trade_spy_return_pct)
        if spy_return and input_data.total_return_pct:
            input_data.vs_spy_pct = input_data.total_return_pct - spy_return
    
    # Supplement from strategy_signals (more comprehensive data)
    if strategy:
        if not input_data.total_return_pct:
            input_data.total_return_pct = _float_or_default(strategy.total_return_pct)
        if not input_data.sharpe_ratio:
            input_data.sharpe_ratio = _float_or_default(strategy.sharpe_ratio)
        if not input_data.max_drawdown_pct:
            input_data.max_drawdown_pct = _float_or_default(strategy.max_drawdown_pct)
        if not input_data.n_trades:
            input_data.n_trades = strategy.n_trades or 0
        if not input_data.vs_buyhold_pct:
            input_data.vs_buyhold_pct = _float_or_default(strategy.vs_buy_hold_pct)
        if not input_data.vs_spy_pct:
            input_data.vs_spy_pct = _float_or_default(strategy.vs_spy_pct)
        if not input_data.win_rate:
            input_data.win_rate = _float_or_default(strategy.win_rate)
    
    return input_data


async def load_dip_entry_input(
    session: AsyncSession,
    symbol: str,
    precomputed: QuantPrecomputed | None = None,
    dip_state: DipState | None = None,
    symbol_obj: Symbol | None = None,
) -> DipEntryInput:
    """
    Load dip entry optimizer metrics from quant_precomputed.
    
    Data sources:
    - quant_precomputed.dip_entry_threshold_analysis: Per-threshold stats
    - quant_precomputed.dip_entry_optimal: Best threshold
    - dip_state: Current drawdown state
    
    Args:
        session: Database session
        symbol: Stock ticker
        precomputed: Pre-loaded QuantPrecomputed (optional)
        dip_state: Pre-loaded DipState (optional)
        symbol_obj: Pre-loaded Symbol (optional)
        
    Returns:
        DipEntryInput populated from database
    """
    # Load if not provided
    if precomputed is None:
        result = await session.execute(
            select(QuantPrecomputed).where(QuantPrecomputed.symbol == symbol)
        )
        precomputed = result.scalar_one_or_none()
    
    if dip_state is None:
        result = await session.execute(
            select(DipState).where(DipState.symbol == symbol)
        )
        dip_state = result.scalar_one_or_none()
    
    if symbol_obj is None:
        result = await session.execute(
            select(Symbol).where(Symbol.symbol == symbol)
        )
        symbol_obj = result.scalar_one_or_none()
    
    input_data = DipEntryInput()
    
    # Current state from dip_state
    if dip_state:
        input_data.current_drawdown_pct = _float_or_default(dip_state.dip_percentage)
    
    # Extract from quant_precomputed
    if precomputed:
        # Optimal thresholds (discovered by statistics) - individual columns
        input_data.optimal_threshold_pct = _float_or_default(precomputed.dip_entry_optimal_threshold)
        input_data.max_profit_threshold_pct = _float_or_default(precomputed.dip_entry_max_profit_threshold)
        
        # Find the threshold analysis closest to current drawdown
        threshold_analysis = precomputed.dip_entry_threshold_analysis or []
        matched = _find_matched_threshold(threshold_analysis, input_data.current_drawdown_pct)
        
        if matched:
            input_data.matched_threshold_pct = _float_or_default(matched.get("threshold_pct"))
            input_data.n_occurrences = int(matched.get("n_occurrences", 0))
            input_data.per_year = _float_or_default(matched.get("avg_per_year"))
            
            # Recovery metrics
            input_data.recovery_rate = _float_or_default(matched.get("recovery_threshold_rate"))
            input_data.full_recovery_rate = _float_or_default(matched.get("full_recovery_rate"))
            input_data.avg_recovery_days = _float_or_default(matched.get("avg_recovery_days"))
            input_data.avg_recovery_velocity = _float_or_default(matched.get("avg_recovery_velocity"))
            
            # Return metrics at optimal holding period
            input_data.win_rate_optimal_hold = _float_or_default(matched.get("optimal_win_rate"))
            input_data.avg_return_optimal_hold = _float_or_default(matched.get("optimal_avg_return"))
            input_data.sharpe_optimal_hold = _float_or_default(matched.get("optimal_sharpe"))
            input_data.total_profit_compounded = _float_or_default(matched.get("optimal_total_profit"))
            
            # Risk metrics
            input_data.max_further_drawdown = _float_or_default(matched.get("max_further_drawdown"))
            input_data.avg_further_drawdown = _float_or_default(matched.get("avg_further_drawdown"))
            input_data.prob_further_drop = _float_or_default(matched.get("prob_further_drop"))
            input_data.continuation_risk = matched.get("continuation_risk", "medium")
            
            # Entry quality
            input_data.entry_score = _float_or_default(matched.get("entry_score"))
        
        # Data years calculated from date range
        if precomputed.data_start and precomputed.data_end:
            days = (precomputed.data_end - precomputed.data_start).days
            input_data.data_years = days / 365.25 if days > 0 else 0.0
        
        # Signal strength and buy now from individual columns
        input_data.signal_strength = _float_or_default(precomputed.dip_entry_signal_strength)
        input_data.is_buy_now = bool(precomputed.dip_entry_is_buy_now)
    
    return input_data


def _find_matched_threshold(
    threshold_analysis: list[dict],
    current_drawdown_pct: float,
) -> dict | None:
    """
    Find the threshold analysis entry closest to current drawdown.
    
    We match by finding the threshold that's closest to (but not exceeding)
    the current drawdown. If we're at -12% dip, we use the -10% threshold data.
    """
    if not threshold_analysis:
        return None
    
    # Convert drawdown to magnitude for comparison
    current_mag = abs(current_drawdown_pct)
    
    best_match = None
    best_diff = float("inf")
    
    for entry in threshold_analysis:
        threshold = abs(_float_or_default(entry.get("threshold_pct")))
        
        # Only consider thresholds we've crossed
        if threshold <= current_mag:
            diff = current_mag - threshold
            if diff < best_diff:
                best_diff = diff
                best_match = entry
    
    # If no threshold crossed, use the smallest one
    if best_match is None and threshold_analysis:
        smallest = min(threshold_analysis, key=lambda x: abs(_float_or_default(x.get("threshold_pct"))))
        return smallest
    
    return best_match


async def load_fundamentals_input(
    session: AsyncSession,
    symbol: str,
    fundamentals: StockFundamentals | None = None,
) -> FundamentalsInput:
    """
    Load fundamental metrics from stock_fundamentals.
    
    Also determines domain (bank, reit, insurance, general) based on sector.
    
    Args:
        session: Database session
        symbol: Stock ticker
        fundamentals: Pre-loaded StockFundamentals (optional)
        
    Returns:
        FundamentalsInput populated from database
    """
    # Load if not provided
    if fundamentals is None:
        result = await session.execute(
            select(StockFundamentals).where(StockFundamentals.symbol == symbol)
        )
        fundamentals = result.scalar_one_or_none()
    
    input_data = FundamentalsInput()
    
    if not fundamentals:
        return input_data
    
    # Valuation
    input_data.pe_ratio = _float_or_default(fundamentals.pe_ratio) or None
    input_data.forward_pe = _float_or_default(fundamentals.forward_pe) or None
    input_data.peg_ratio = _float_or_default(fundamentals.peg_ratio) or None
    input_data.pb_ratio = _float_or_default(fundamentals.price_to_book) or None
    input_data.ps_ratio = _float_or_default(fundamentals.price_to_sales) or None
    
    # Profitability
    input_data.profit_margin = _float_or_default(fundamentals.profit_margin) or None
    input_data.operating_margin = _float_or_default(fundamentals.operating_margin) or None
    input_data.gross_margin = _float_or_default(fundamentals.gross_margin) or None
    input_data.roe = _float_or_default(fundamentals.return_on_equity) or None
    input_data.roa = _float_or_default(fundamentals.return_on_assets) or None
    
    # Growth
    input_data.revenue_growth = _float_or_default(fundamentals.revenue_growth) or None
    input_data.earnings_growth = _float_or_default(fundamentals.earnings_growth) or None
    
    # Financial health
    input_data.debt_to_equity = _float_or_default(fundamentals.debt_to_equity) or None
    input_data.current_ratio = _float_or_default(fundamentals.current_ratio) or None
    input_data.quick_ratio = _float_or_default(fundamentals.quick_ratio) or None
    input_data.free_cash_flow = _float_or_default(fundamentals.free_cash_flow) or None
    
    # Risk
    input_data.beta = _float_or_default(fundamentals.beta) or None
    input_data.short_ratio = _float_or_default(fundamentals.short_ratio) or None
    
    # Analyst - use recommendation_mean (1-5 scale, lower is better/buy)
    input_data.analyst_rating = _float_or_default(fundamentals.recommendation_mean) or None
    
    # Target upside calculated from current price vs analyst target
    # We can compute this when we have current_price available from dip_state
    input_data.target_upside_pct = None  # Will be computed if target_mean_price available
    
    # Determine domain based on sector/industry
    sector = getattr(fundamentals, "sector", "") or ""
    industry = getattr(fundamentals, "industry", "") or ""
    
    sector_lower = sector.lower() if sector else ""
    industry_lower = industry.lower() if industry else ""
    
    if "bank" in sector_lower or "bank" in industry_lower:
        input_data.domain = "bank"
    elif "reit" in sector_lower or "reit" in industry_lower or "real estate" in industry_lower:
        input_data.domain = "reit"
    elif "insurance" in sector_lower or "insurance" in industry_lower:
        input_data.domain = "insurance"
    else:
        input_data.domain = "general"
    
    # Domain-specific metrics would be loaded here from additional data sources
    # For now, just flag the domain for potential future enhancement
    
    return input_data


async def load_all_inputs_for_symbol(
    session: AsyncSession,
    symbol: str,
) -> tuple[BacktestV2Input, DipEntryInput, FundamentalsInput]:
    """
    Load all scoring inputs for a single symbol in one batch.
    
    Optimizes database queries by loading related data together.
    
    Args:
        session: Database session
        symbol: Stock ticker
        
    Returns:
        Tuple of (BacktestV2Input, DipEntryInput, FundamentalsInput)
    """
    # Load all data in parallel-ish (single transaction)
    precomputed_result = await session.execute(
        select(QuantPrecomputed).where(QuantPrecomputed.symbol == symbol)
    )
    precomputed = precomputed_result.scalar_one_or_none()
    
    strategy_result = await session.execute(
        select(StrategySignal).where(StrategySignal.symbol == symbol)
    )
    strategy = strategy_result.scalar_one_or_none()
    
    dip_state_result = await session.execute(
        select(DipState).where(DipState.symbol == symbol)
    )
    dip_state = dip_state_result.scalar_one_or_none()
    
    fundamentals_result = await session.execute(
        select(StockFundamentals).where(StockFundamentals.symbol == symbol)
    )
    fundamentals = fundamentals_result.scalar_one_or_none()
    
    # Convert to input formats
    backtest = await load_backtest_input(session, symbol, precomputed, strategy)
    dip_entry = await load_dip_entry_input(session, symbol, precomputed, dip_state)
    fund_input = await load_fundamentals_input(session, symbol, fundamentals)
    
    return backtest, dip_entry, fund_input


async def load_all_inputs_batch(
    session: AsyncSession,
    symbols: list[str],
) -> dict[str, tuple[BacktestV2Input, DipEntryInput, FundamentalsInput]]:
    """
    Load all scoring inputs for multiple symbols efficiently.
    
    Uses bulk queries to minimize database round trips.
    
    Args:
        session: Database session
        symbols: List of stock tickers
        
    Returns:
        Dict mapping symbol -> (BacktestV2Input, DipEntryInput, FundamentalsInput)
    """
    # Bulk load all data
    precomputed_result = await session.execute(
        select(QuantPrecomputed).where(QuantPrecomputed.symbol.in_(symbols))
    )
    precomputed_map = {p.symbol: p for p in precomputed_result.scalars().all()}
    
    strategy_result = await session.execute(
        select(StrategySignal).where(StrategySignal.symbol.in_(symbols))
    )
    strategy_map = {s.symbol: s for s in strategy_result.scalars().all()}
    
    dip_state_result = await session.execute(
        select(DipState).where(DipState.symbol.in_(symbols))
    )
    dip_state_map = {d.symbol: d for d in dip_state_result.scalars().all()}
    
    fundamentals_result = await session.execute(
        select(StockFundamentals).where(StockFundamentals.symbol.in_(symbols))
    )
    fundamentals_map = {f.symbol: f for f in fundamentals_result.scalars().all()}
    
    # Convert to input formats
    result = {}
    for symbol in symbols:
        precomputed = precomputed_map.get(symbol)
        strategy = strategy_map.get(symbol)
        dip_state = dip_state_map.get(symbol)
        fundamentals = fundamentals_map.get(symbol)
        
        backtest = await load_backtest_input(session, symbol, precomputed, strategy)
        dip_entry = await load_dip_entry_input(session, symbol, precomputed, dip_state)
        fund_input = await load_fundamentals_input(session, symbol, fundamentals)
        
        result[symbol] = (backtest, dip_entry, fund_input)
    
    return result
