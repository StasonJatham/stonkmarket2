"""Quant engine job definitions.

This module contains jobs for:
- Monthly quant cache refresh (quant_monthly)
- Nightly strategy optimization (strategy_nightly)
- Daily scoring pipeline - APUS/DOUS dual-mode (quant_scoring_daily)
- Nightly pre-computation of all quant analysis (quant_analysis_nightly)

These jobs form the quantitative analysis backbone of the trading system.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.repositories import price_history_orm as price_history_repo

from ..registry import register_job
from ..utils import get_close_column, log_job_success


if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("jobs.quant")


# =============================================================================
# QUANT MONTHLY - Cache refresh
# =============================================================================


@register_job("quant_monthly")
async def quant_monthly_job() -> str:
    """
    Monthly quant engine optimization.

    Runs on the 1st of each month to:
    - Recalculate expected returns and risk models
    - Update portfolio weights
    - Refresh signal analysis cache

    Schedule: Monthly on 1st at 3 AM UTC (0 3 1 * *)
    """
    from app.cache.cache import Cache
    from app.repositories import symbols_orm as symbols_repo

    logger.info("Starting quant_monthly job")
    job_start = time.monotonic()

    try:
        # Get all tracked symbols
        symbols = await symbols_repo.list_symbols()
        symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]

        if not symbol_list:
            log_job_success("quant_monthly", "No symbols to process",
                            caches_cleared=0, symbols_count=0,
                            duration_ms=int((time.monotonic() - job_start) * 1000))
            return "No symbols to process"

        cache = Cache()

        # Clear old quant caches to force refresh
        await cache.delete("quant:recommendations")
        await cache.delete("quant:signals")
        await cache.delete("regime")

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Cleared quant caches for {len(symbol_list)} symbols, ready for fresh calculations"
        
        # Structured success log
        log_job_success(
            "quant_monthly",
            message,
            caches_cleared=3,
            symbols_count=len(symbol_list),
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"quant_monthly failed: {e}")
        raise


# =============================================================================
# STRATEGY OPTIMIZATION - Nightly after prices (uses backtest)
# =============================================================================


@register_job("strategy_nightly")
async def strategy_nightly_job() -> str:
    """
    Nightly strategy optimization using backtest BaselineEngine.
    
    Uses the comprehensive backtest system to compare:
    1. DCA (Dollar Cost Average) - monthly buying
    2. Buy & Hold - buy once, hold forever  
    3. Buy Dips & Hold - only buy on dips
    4. Technical Trading - optimized entry/exit signals
    5. SPY benchmarks for comparison
    
    The recommendation tells which strategy ACTUALLY beats buy & hold.
    Results include full trade history for chart display.
    
    Schedule: Mon-Fri at 11:30 PM UTC (30 min after prices_daily)
    """
    import pandas as pd
    from decimal import Decimal
    from sqlalchemy.dialects.postgresql import insert
    
    from app.cache.cache import Cache
    from app.database.connection import get_session
    from app.database.orm import StrategySignal, StockFundamentals
    from app.repositories import symbols_orm as symbols_repo
    from app.quant_engine.backtest.baseline_strategies import (
        BaselineEngine,
        BaselineStrategyType,
        RecommendationType,
    )
    from app.quant_engine.dipfinder.entry_optimizer import DipEntryOptimizer
    
    logger.info("Starting strategy_nightly job (using backtest)")
    job_start = time.monotonic()
    
    # Initialize dip entry optimizer for recovery time calculation
    dip_optimizer = DipEntryOptimizer()
    
    try:
        # Get all active symbols
        all_symbols = await symbols_repo.list_symbols()
        symbol_list = [
            s.symbol for s in all_symbols 
            if s.symbol not in ("SPY", "^GSPC", "URTH", "^VIX")
            and s.is_active
        ]
        
        if not symbol_list:
            return "No symbols to process"
        
        logger.info(f"[STRATEGY] Optimizing strategies for {len(symbol_list)} symbols using backtest")
        
        # Get SPY for benchmark comparison
        # Use 3 years by default (matches hero chart), 5 years if stock has enough data
        from datetime import date, timedelta
        end_date = date.today()
        # Fetch 5 years of SPY, will be trimmed to match each stock's period
        spy_start_date = end_date - timedelta(days=1825)  # 5 years
        spy_df = await price_history_repo.get_prices_as_dataframe("SPY", spy_start_date, end_date)
        
        processed = 0
        failed = 0
        signals_saved = 0
        
        async with get_session() as session:
            # Load all fundamentals in bulk
            from sqlalchemy import select
            result = await session.execute(
                select(StockFundamentals).where(
                    StockFundamentals.symbol.in_(symbol_list)
                )
            )
            fundamentals_map = {f.symbol: f for f in result.scalars().all()}
            
            for symbol in symbol_list:
                try:
                    # Get price history - use 3 years (matches hero chart), 5 years for established stocks
                    # This ensures strategy comparison matches what user sees on chart
                    start_3y = end_date - timedelta(days=1095)  # 3 years
                    start_5y = end_date - timedelta(days=1825)  # 5 years
                    
                    # First check how much data the stock has
                    df_all = await price_history_repo.get_prices_as_dataframe(symbol, start_5y, end_date)
                    
                    if df_all is None or len(df_all) < 200:
                        logger.warning(f"[STRATEGY] Insufficient data for {symbol}, skipping")
                        continue
                    
                    # Use 3 years by default (matches hero chart)
                    # Only use 5 years if stock has 5+ years of data
                    if len(df_all) >= 1250:  # ~5 years of trading days
                        df = df_all
                    else:
                        # Use 3 years - trim to match hero chart period
                        df = await price_history_repo.get_prices_as_dataframe(symbol, start_3y, end_date)
                        if df is None or len(df) < 200:
                            # Fallback to all available data if 3y is too short
                            df = df_all
                    
                    # Trim SPY to match the stock's date range for fair comparison
                    stock_start = df.index[0]
                    spy_trimmed = spy_df[spy_df.index >= stock_start].copy() if spy_df is not None else None
                    
                    # Run BaselineEngine comparison (DCA vs B&H vs Dips vs Technical)
                    engine = BaselineEngine(
                        prices=df,
                        spy_prices=spy_trimmed,
                        symbol=symbol,
                        initial_capital=10_000.0,
                        monthly_contribution=1_000.0,
                    )
                    comparison = engine.run_all()
                    
                    # Determine the winning strategy from recommendation
                    rec = comparison.recommendation
                    
                    # Map recommendation type to strategy name for display
                    strategy_name_map = {
                        RecommendationType.DCA: "dollar_cost_average",
                        RecommendationType.BUY_AND_HOLD: "buy_and_hold",
                        RecommendationType.BUY_DIPS: "buy_dips_hold",
                        RecommendationType.OPTIMIZED_STRATEGY: "optimized_technical",
                        RecommendationType.SPY_DCA: "spy_dca",
                        RecommendationType.SWITCH_TO_SPY: "switch_to_spy",
                    }
                    strategy_name = strategy_name_map.get(rec.recommendation, "dca")
                    
                    # Get the best performing strategy result
                    best_result = None
                    best_trades = []
                    
                    # Find the actual best result from comparison
                    results_map = {
                        "DCA Monthly": comparison.dca,
                        "Buy & Hold": comparison.buy_hold,
                        "Lump Sum": comparison.lump_sum,
                        "Buy Dips & Hold": comparison.buy_dips,
                        "Perfect Dip Trading": comparison.dip_trading,
                        "Technical Trading": comparison.technical_trading,
                        "Regime-Aware Technical": comparison.regime_aware_technical,
                        "SPY DCA": comparison.spy_dca,
                        "SPY Buy & Hold": comparison.spy_buy_hold,
                    }
                    
                    # Use the #1 ranked strategy
                    if comparison.ranked_by_return:
                        best_name = comparison.ranked_by_return[0]
                        best_result = results_map.get(best_name)
                    
                    if best_result is None:
                        best_result = comparison.dca  # Fallback to DCA
                    
                    # Extract trade details for chart display
                    # For DCA-type strategies, generate synthetic buy trades from price data
                    if best_result.trade_details:
                        best_trades = [
                            {
                                "entry_date": t.entry_date,
                                "exit_date": t.exit_date,
                                "entry_price": t.entry_price,
                                "exit_price": t.exit_price,
                                "pnl_pct": t.return_pct * 100,  # Convert to %
                                "exit_reason": t.exit_reason,
                                "holding_days": t.holding_days,
                                # Calculate amount invested from shares * entry_price
                                "amount_invested": float(t.shares * t.entry_price) if t.shares and t.entry_price else None,
                            }
                            for t in best_result.trade_details[:20]  # Last 20 trades
                        ]
                    elif strategy_name in ("dollar_cost_average", "buy_dips_hold", "buy_and_hold"):
                        # Generate synthetic buy trades for accumulation strategies
                        # These show green dots on the chart at buy points
                        close_col = get_close_column(df)
                        close_prices = df[close_col]
                        
                        if strategy_name == "buy_and_hold":
                            # Single buy on first day with initial capital
                            initial_capital = 10_000.0  # Match engine default
                            first_date = close_prices.index[0]
                            entry_price = float(close_prices.iloc[0])
                            best_trades = [{
                                "entry_date": first_date.strftime("%Y-%m-%d"),
                                "exit_date": None,
                                "entry_price": entry_price,
                                "exit_price": None,
                                "pnl_pct": best_result.total_return_pct,
                                "exit_reason": "holding",
                                "holding_days": len(close_prices),
                                "amount_invested": initial_capital,
                            }]
                        elif strategy_name == "dollar_cost_average":
                            # Monthly buys - sample ~20 buys evenly spaced
                            monthly_contribution = 1_000.0  # Match engine default
                            total_months = int(len(close_prices) / 21)  # ~21 trading days/month
                            step = max(1, total_months // 20)  # Get ~20 samples
                            
                            for month_idx in range(0, total_months, step):
                                day_idx = min(month_idx * 21, len(close_prices) - 1)
                                entry_date = close_prices.index[day_idx]
                                entry_price = float(close_prices.iloc[day_idx])
                                
                                best_trades.append({
                                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                                    "exit_date": None,
                                    "entry_price": entry_price,
                                    "exit_price": None,
                                    "pnl_pct": 0,  # DCA doesn't have individual trade P&L
                                    "exit_reason": "accumulating",
                                    "holding_days": None,
                                    "amount_invested": monthly_contribution,
                                })
                            
                            # Limit to recent 20 buys
                            best_trades = best_trades[-20:]
                        elif strategy_name == "buy_dips_hold":
                            # Buy on dips - simulates actual cash accumulation logic
                            # $1k/month accumulates, ALL cash deployed on first dip day with cash
                            dip_threshold = engine.dip_threshold_pct / 100 if engine.dip_threshold_pct else -0.10
                            
                            # Find dip days (days where price dropped from recent high)
                            rolling_high = close_prices.rolling(window=63).max()  # 3-month high
                            drawdown = (close_prices - rolling_high) / rolling_high
                            dip_mask = drawdown <= dip_threshold
                            
                            # Simulate the actual strategy cash flow
                            monthly_contribution = 1000.0
                            cash_waiting = 10000.0  # Initial capital
                            last_month = None
                            
                            for i, (date, price) in enumerate(close_prices.items()):
                                is_dip = dip_mask.iloc[i] if i < len(dip_mask) else False
                                
                                # Track month changes for contributions
                                current_month = (date.year, date.month)
                                if last_month is not None and current_month != last_month:
                                    # New month - add contribution
                                    cash_waiting += monthly_contribution
                                last_month = current_month
                                
                                # Buy on dip IF we have cash
                                if is_dip and cash_waiting > 0:
                                    best_trades.append({
                                        "entry_date": date.strftime("%Y-%m-%d"),
                                        "exit_date": None,
                                        "entry_price": float(price),
                                        "exit_price": None,
                                        "pnl_pct": 0,
                                        "exit_reason": "dip_buy",
                                        "holding_days": None,
                                        "amount_invested": cash_waiting,  # Track how much was deployed
                                    })
                                    cash_waiting = 0  # All cash deployed
                            
                            # Keep most recent 20 buys
                            best_trades = best_trades[-20:]
                    
                    # Calculate vs buy & hold - compare DCA/Dips to Buy & Hold for THIS STOCK
                    # This answers: "Does our active strategy beat passive investing in this stock?"
                    # We compare our best active strategy (DCA or Buy Dips) vs Buy & Hold
                    dca_return = comparison.dca.total_return_pct
                    buy_dips_return = comparison.buy_dips.total_return_pct
                    buy_hold_return = comparison.buy_hold.total_return_pct
                    
                    # Our best active strategy for this stock
                    best_active_return = max(dca_return, buy_dips_return)
                    best_active_name = "dollar_cost_average" if dca_return >= buy_dips_return else "buy_dips_hold"
                    
                    # Does our active strategy beat passive Buy & Hold?
                    active_vs_buy_hold = best_active_return - buy_hold_return
                    beats_buy_hold = active_vs_buy_hold > 0
                    
                    # vs_buy_hold_pct stores how much better our best active strategy is
                    # Positive = our strategy beats passive, Negative = passive wins
                    vs_buy_hold = active_vs_buy_hold
                    
                    # Override strategy_name if our active strategy is the winner
                    # This makes the landing page show the correct strategy
                    if beats_buy_hold:
                        strategy_name = best_active_name
                        # Also update best_result to reflect the active strategy we're showcasing
                        if best_active_name == "dollar_cost_average":
                            best_result = comparison.dca
                        else:
                            best_result = comparison.buy_dips
                        
                        # Regenerate trades for the winning active strategy
                        # since original trade generation used the overall best strategy
                        close_col = get_close_column(df)
                        close_prices = df[close_col]
                        
                        if strategy_name == "dollar_cost_average":
                            # Monthly buys - sample ~20 buys evenly spaced
                            monthly_contribution = 1_000.0
                            total_months = int(len(close_prices) / 21)
                            step = max(1, total_months // 20)
                            best_trades = []
                            
                            for month_idx in range(0, total_months, step):
                                day_idx = min(month_idx * 21, len(close_prices) - 1)
                                entry_date = close_prices.index[day_idx]
                                entry_price = float(close_prices.iloc[day_idx])
                                
                                best_trades.append({
                                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                                    "exit_date": None,
                                    "entry_price": entry_price,
                                    "exit_price": None,
                                    "pnl_pct": 0,
                                    "exit_reason": "accumulating",
                                    "holding_days": None,
                                    "amount_invested": monthly_contribution,
                                })
                            best_trades = best_trades[-20:]
                        elif strategy_name == "buy_dips_hold":
                            # Buy on dips
                            dip_threshold = engine.dip_threshold_pct / 100 if engine.dip_threshold_pct else -0.10
                            rolling_high = close_prices.rolling(window=63).max()
                            drawdown = (close_prices - rolling_high) / rolling_high
                            dip_mask = drawdown <= dip_threshold
                            
                            monthly_contribution = 1000.0
                            cash_waiting = 10000.0
                            last_month = None
                            best_trades = []
                            
                            for i, (date, price) in enumerate(close_prices.items()):
                                is_dip = dip_mask.iloc[i] if i < len(dip_mask) else False
                                current_month = (date.year, date.month)
                                if last_month is not None and current_month != last_month:
                                    cash_waiting += monthly_contribution
                                last_month = current_month
                                
                                if is_dip and cash_waiting > 0:
                                    best_trades.append({
                                        "entry_date": date.strftime("%Y-%m-%d"),
                                        "exit_date": None,
                                        "entry_price": float(price),
                                        "exit_price": None,
                                        "pnl_pct": 0,
                                        "exit_reason": "dip_buy",
                                        "holding_days": None,
                                        "amount_invested": cash_waiting,
                                    })
                                    cash_waiting = 0
                            best_trades = best_trades[-20:]
                    
                    # Calculate vs SPY
                    vs_spy = best_result.total_return_pct - comparison.spy_dca.total_return_pct
                    beats_spy = vs_spy > 0
                    
                    # Check fundamentals
                    fund_obj = fundamentals_map.get(symbol)
                    fundamentals_healthy = True
                    fundamental_concerns = []
                    if fund_obj:
                        if fund_obj.pe_ratio and fund_obj.pe_ratio > 50:
                            fundamental_concerns.append("high_pe")
                        if fund_obj.debt_to_equity and fund_obj.debt_to_equity > 3:
                            fundamental_concerns.append("high_debt")
                        if fund_obj.profit_margin and fund_obj.profit_margin < 0:
                            fundamental_concerns.append("unprofitable")
                        fundamentals_healthy = len(fundamental_concerns) == 0
                    
                    # Calculate typical recovery days from dip entry optimizer
                    typical_recovery_days = None
                    try:
                        dip_entry_result = dip_optimizer.analyze(df, symbol)
                        if dip_entry_result and dip_entry_result.typical_recovery_days > 0:
                            typical_recovery_days = int(dip_entry_result.typical_recovery_days)
                    except Exception as dip_err:
                        logger.debug(f"[STRATEGY] Dip entry analysis failed for {symbol}: {dip_err}")
                    
                    # Determine current signal
                    signal_type = "HOLD"
                    signal_reason = rec.headline
                    has_active_signal = False
                    
                    # If DCA or Buy Dips is recommended and fundamentals are healthy
                    if rec.recommendation in (RecommendationType.DCA, RecommendationType.BUY_DIPS) and fundamentals_healthy:
                        signal_type = "BUY"
                        has_active_signal = True
                        signal_reason = f"{rec.headline} - Fundamentals healthy"
                    
                    # Strategy params (for reference)
                    strategy_params = {
                        "recommendation": rec.recommendation.value,
                        "best_strategy": comparison.ranked_by_return[0] if comparison.ranked_by_return else "DCA",
                        "initial_capital": 10_000.0,
                        "monthly_contribution": 1_000.0,
                        "dip_threshold": engine.dip_threshold_pct,
                    }
                    
                    # Build full strategy comparison for UI display
                    # This shows the user exactly what each strategy would have produced
                    strategy_comparison = {
                        "initial_capital": 10_000.0,
                        "monthly_contribution": 1_000.0,
                        "backtest_days": len(df),
                        "strategies": {
                            "dca": {
                                "name": "DCA Monthly",
                                "description": "Invest $1k every month regardless of price",
                                "total_invested": comparison.dca.total_invested,
                                "final_value": comparison.dca.final_value,
                                "total_return_pct": comparison.dca.total_return_pct,
                                "n_buys": comparison.dca.total_buys,
                            },
                            "buy_dips": {
                                "name": "Buy Dips & Hold",
                                "description": "Accumulate $1k/mo cash, deploy only on dips",
                                "total_invested": comparison.buy_dips.total_invested,
                                "final_value": comparison.buy_dips.final_value,
                                "total_return_pct": comparison.buy_dips.total_return_pct,
                                "n_buys": comparison.buy_dips.total_buys,
                            },
                            "buy_hold": {
                                "name": "Buy & Hold",
                                "description": "Invest $10k on day 1, hold forever",
                                "total_invested": comparison.buy_hold.total_invested,
                                "final_value": comparison.buy_hold.final_value,
                                "total_return_pct": comparison.buy_hold.total_return_pct,
                                "n_buys": 1,
                            },
                            "spy_dca": {
                                "name": "SPY DCA",
                                "description": "Same schedule but buying SPY instead",
                                "total_invested": comparison.spy_dca.total_invested,
                                "final_value": comparison.spy_dca.final_value,
                                "total_return_pct": comparison.spy_dca.total_return_pct,
                                "n_buys": comparison.spy_dca.total_buys,
                            },
                            "spy_buy_hold": {
                                "name": "SPY Buy & Hold",
                                "description": "Invest $10k in SPY on day 1, hold forever",
                                "total_invested": comparison.spy_buy_hold.total_invested,
                                "final_value": comparison.spy_buy_hold.final_value,
                                "total_return_pct": comparison.spy_buy_hold.total_return_pct,
                                "n_buys": 1,
                            },
                            "lump_sum": {
                                "name": "Lump Sum",
                                "description": "Same total capital invested on day 1",
                                "total_invested": comparison.lump_sum.total_invested,
                                "final_value": comparison.lump_sum.final_value,
                                "total_return_pct": comparison.lump_sum.total_return_pct,
                                "n_buys": 1,
                            },
                        },
                        "ranked_by_return": comparison.ranked_by_return[:7],
                        "winner": comparison.ranked_by_return[0] if comparison.ranked_by_return else "DCA Monthly",
                    }
                    
                    # Add technical trading if available
                    if comparison.technical_trading:
                        strategy_comparison["strategies"]["technical"] = {
                            "name": "Technical Trading",
                            "description": "Optimized entry/exit signals",
                            "total_invested": comparison.technical_trading.total_invested,
                            "final_value": comparison.technical_trading.final_value,
                            "total_return_pct": comparison.technical_trading.total_return_pct,
                            "n_buys": comparison.technical_trading.total_buys,
                        }
                    
                    # Indicators used
                    indicators_used = ["price", "dip_detection"]
                    if comparison.technical_trading:
                        indicators_used.extend(["rsi", "macd", "sma"])
                    
                    # Upsert to database
                    stmt = insert(StrategySignal).values(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        strategy_params=strategy_params,
                        signal_type=signal_type,
                        signal_reason=signal_reason,
                        has_active_signal=has_active_signal,
                        total_return_pct=Decimal(str(best_result.total_return_pct)),
                        sharpe_ratio=Decimal(str(best_result.sharpe_ratio)),
                        win_rate=Decimal(str(best_result.win_rate_pct)),
                        max_drawdown_pct=Decimal(str(best_result.max_drawdown_pct)),
                        n_trades=best_result.total_buys,
                        recency_weighted_return=Decimal(str(best_result.annualized_return_pct)),
                        current_year_return_pct=Decimal(str(best_result.total_return_pct)),  # TODO: Calculate current year
                        current_year_win_rate=Decimal(str(best_result.win_rate_pct)),
                        current_year_trades=0,  # TODO: Calculate current year trades
                        vs_buy_hold_pct=Decimal(str(vs_buy_hold)),
                        vs_spy_pct=Decimal(str(vs_spy)),
                        beats_buy_hold=beats_buy_hold,
                        beats_spy=beats_spy,
                        fundamentals_healthy=fundamentals_healthy,
                        fundamental_concerns=fundamental_concerns,
                        is_statistically_valid=best_result.total_buys >= 10,
                        recent_trades=best_trades,
                        strategy_comparison=strategy_comparison,
                        indicators_used=indicators_used,
                        typical_recovery_days=typical_recovery_days,
                    ).on_conflict_do_update(
                        index_elements=["symbol"],
                        set_={
                            "strategy_name": strategy_name,
                            "strategy_params": strategy_params,
                            "signal_type": signal_type,
                            "signal_reason": signal_reason,
                            "has_active_signal": has_active_signal,
                            "total_return_pct": Decimal(str(best_result.total_return_pct)),
                            "sharpe_ratio": Decimal(str(best_result.sharpe_ratio)),
                            "win_rate": Decimal(str(best_result.win_rate_pct)),
                            "max_drawdown_pct": Decimal(str(best_result.max_drawdown_pct)),
                            "n_trades": best_result.total_buys,
                            "recency_weighted_return": Decimal(str(best_result.annualized_return_pct)),
                            "current_year_return_pct": Decimal(str(best_result.total_return_pct)),
                            "current_year_win_rate": Decimal(str(best_result.win_rate_pct)),
                            "current_year_trades": 0,
                            "vs_buy_hold_pct": Decimal(str(vs_buy_hold)),
                            "vs_spy_pct": Decimal(str(vs_spy)),
                            "beats_buy_hold": beats_buy_hold,
                            "beats_spy": beats_spy,
                            "fundamentals_healthy": fundamentals_healthy,
                            "fundamental_concerns": fundamental_concerns,
                            "is_statistically_valid": best_result.total_buys >= 10,
                            "recent_trades": best_trades,
                            "strategy_comparison": strategy_comparison,
                            "indicators_used": indicators_used,
                            "typical_recovery_days": typical_recovery_days,
                            "optimized_at": datetime.now(UTC),
                        }
                    )
                    await session.execute(stmt)
                    
                    processed += 1
                    signals_saved += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"[STRATEGY] Processed {processed}/{len(symbol_list)} symbols")
                        await session.commit()
                    
                except Exception as e:
                    logger.exception(f"[STRATEGY] Failed to optimize {symbol}: {e}")
                    failed += 1
                    continue
            
            await session.commit()
        
        # Clear cache
        cache = Cache()
        await cache.delete("strategy_signals:*")
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Optimized strategies for {processed} symbols ({failed} failed), saved {signals_saved} signals"
        
        # Structured success log
        log_job_success(
            "strategy_nightly",
            message,
            symbols_optimized=processed,
            symbols_failed=failed,
            signals_saved=signals_saved,
            symbols_total=len(symbol_list),
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.exception(f"strategy_nightly failed: {e}")
        raise


# =============================================================================
# Helper functions
# =============================================================================


# =============================================================================
# QUANT SCORING V3 - Unified scoring with ScoringOrchestrator
# =============================================================================


@register_job("quant_scoring_daily")
async def quant_scoring_daily_job() -> str:
    """
    Daily unified scoring pipeline using V3 ScoringOrchestrator.
    
    Uses the unified scoring system:
    - TechnicalService for indicators
    - RegimeService for market regime detection  
    - DomainScoring for fundamental quality
    - ScoringOrchestrator combines all scores
    
    Schedule: Mon-Fri 11:45 PM UTC (15 min after strategy_nightly)
    """
    from datetime import date, timedelta
    from decimal import Decimal
    
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert
    
    from app.cache.cache import Cache
    from app.database.connection import get_session
    from app.database.orm import QuantScore, StockFundamentals, Symbol
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import price_history_orm as price_history_repo
    from app.quant_engine.scoring import ScoringOrchestrator, get_scoring_orchestrator
    from app.quant_engine.scoring.orchestrator import SCORING_VERSION
    
    logger.info("Starting quant_scoring_daily job (V3 unified scoring)")
    job_start = time.monotonic()
    
    try:
        # Get all active symbols
        all_symbols = await symbols_repo.list_symbols()
        symbol_list = [
            s.symbol for s in all_symbols 
            if s.symbol not in ("SPY", "^GSPC", "URTH", "^VIX")
            and s.is_active
        ]
        
        if not symbol_list:
            return "No symbols to process"
        
        logger.info(f"[SCORING V3] Computing scores for {len(symbol_list)} symbols")
        
        # Get SPY prices for regime detection
        end_date = date.today()
        start_date = end_date - timedelta(days=750)  # 3 years for good MA data
        
        spy_df = await price_history_repo.get_prices_as_dataframe("SPY", start_date, end_date)
        if spy_df is None or spy_df.empty:
            logger.error("[SCORING V3] Could not fetch SPY prices")
            return "Failed: No SPY data"
        
        # Initialize orchestrator
        orchestrator = get_scoring_orchestrator()
        
        processed = 0
        failed = 0
        rec_counts = {"STRONG_BUY": 0, "BUY": 0, "ACCUMULATE": 0, "HOLD": 0, "AVOID": 0, "SELL": 0}
        
        async with get_session() as session:
            # Load fundamentals in batch
            result = await session.execute(
                select(StockFundamentals).where(StockFundamentals.symbol.in_(symbol_list))
            )
            fundamentals_map = {f.symbol: f for f in result.scalars().all()}
            
            # Load symbol info
            result = await session.execute(
                select(Symbol).where(Symbol.symbol.in_(symbol_list))
            )
            symbol_info = {s.symbol: s for s in result.scalars().all()}
            
            for symbol in symbol_list:
                try:
                    # Get price data
                    stock_df = await price_history_repo.get_prices_as_dataframe(
                        symbol, start_date, end_date
                    )
                    
                    if stock_df is None or len(stock_df) < 100:
                        logger.debug(f"[SCORING V3] Insufficient data for {symbol}")
                        failed += 1
                        continue
                    
                    # Get fundamentals dict
                    fund_row = fundamentals_map.get(symbol)
                    fundamentals = _orm_to_fundamentals_dict(fund_row) if fund_row else None
                    
                    # Get symbol info
                    sym_info = symbol_info.get(symbol)
                    name = sym_info.name if sym_info else None
                    sector = sym_info.sector if sym_info else None
                    
                    # Run orchestrator analysis
                    dashboard = await orchestrator.analyze(
                        symbol=symbol,
                        stock_prices=stock_df,
                        spy_prices=spy_df,
                        fundamentals=fundamentals,
                        name=name,
                        sector=sector,
                    )
                    
                    # Track recommendations
                    rec_counts[dashboard.recommendation] = rec_counts.get(dashboard.recommendation, 0) + 1
                    
                    # Build evidence dict
                    evidence_dict = {
                        "scores": dashboard.scores.to_dict(),
                        "entry": dashboard.entry.to_dict(),
                        "risk": dashboard.risk.to_dict(),
                        "recommendation": dashboard.recommendation,
                        "summary": dashboard.summary,
                        "confidence": dashboard.confidence,
                        "regime": dashboard.regime.regime.value if dashboard.regime else None,
                        "fundamental_notes": dashboard.fundamental_notes,
                    }
                    
                    # Map recommendation to mode
                    mode = _recommendation_to_mode(dashboard.recommendation)
                    
                    # Upsert to database
                    stmt = insert(QuantScore).values(
                        symbol=symbol,
                        best_score=Decimal(str(round(dashboard.scores.composite, 2))),
                        mode=mode,
                        score_a=Decimal(str(round(dashboard.scores.technical, 2))),
                        score_b=Decimal(str(round(dashboard.scores.entry_timing, 2))),
                        gate_pass=dashboard.recommendation in ("STRONG_BUY", "BUY"),
                        # Statistical validation
                        p_outperf=Decimal(str(round(dashboard.confidence / 100, 4))),
                        ci_low=Decimal("0.0"),
                        ci_high=Decimal("0.0"),
                        dsr=Decimal("0.0"),
                        # Edge metrics
                        median_edge=Decimal(str(round(dashboard.scores.composite / 100, 4))),
                        edge_vs_stock=Decimal("0.0"),
                        edge_vs_spy=Decimal("0.0"),
                        worst_regime_edge=Decimal("0.0"),
                        cvar_5=Decimal("0.0"),
                        # Fundamental metrics
                        fund_mom=Decimal(str(round(dashboard.scores.fundamental / 100, 4))),
                        val_z=Decimal("0.0"),
                        event_risk=False,
                        # Dip metrics
                        p_recovery=Decimal(str(round(dashboard.scores.entry_timing / 100, 4))),
                        expected_value=Decimal(str(round(dashboard.scores.composite / 100, 4))),
                        sector_relative=Decimal("0.0"),
                        # Metadata
                        config_hash="v3-" + SCORING_VERSION,
                        scoring_version=SCORING_VERSION,
                        data_start=stock_df.index[0].date() if hasattr(stock_df.index[0], 'date') else start_date,
                        data_end=stock_df.index[-1].date() if hasattr(stock_df.index[-1], 'date') else end_date,
                        evidence=evidence_dict,
                    ).on_conflict_do_update(
                        index_elements=["symbol"],
                        set_={
                            "best_score": Decimal(str(round(dashboard.scores.composite, 2))),
                            "mode": mode,
                            "score_a": Decimal(str(round(dashboard.scores.technical, 2))),
                            "score_b": Decimal(str(round(dashboard.scores.entry_timing, 2))),
                            "gate_pass": dashboard.recommendation in ("STRONG_BUY", "BUY"),
                            "p_outperf": Decimal(str(round(dashboard.confidence / 100, 4))),
                            "fund_mom": Decimal(str(round(dashboard.scores.fundamental / 100, 4))),
                            "p_recovery": Decimal(str(round(dashboard.scores.entry_timing / 100, 4))),
                            "expected_value": Decimal(str(round(dashboard.scores.composite / 100, 4))),
                            "config_hash": "v3-" + SCORING_VERSION,
                            "scoring_version": SCORING_VERSION,
                            "computed_at": datetime.now(UTC),
                            "evidence": evidence_dict,
                        }
                    )
                    await session.execute(stmt)
                    
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"[SCORING V3] Processed {processed}/{len(symbol_list)} symbols")
                        await session.commit()
                    
                except Exception as e:
                    logger.exception(f"[SCORING V3] Failed to score {symbol}: {e}")
                    failed += 1
                    continue
            
            await session.commit()
        
        # Clear cache
        cache = Cache(prefix="quant_scores", default_ttl=86400)
        await cache.invalidate_pattern("*")
        
        recs_cache = Cache(prefix="recommendations", default_ttl=300)
        await recs_cache.invalidate_pattern("*")
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = (
            f"Scored {processed} symbols ({failed} failed) using V3. "
            f"Recs: {rec_counts}"
        )
        
        log_job_success(
            "quant_scoring_daily",
            message,
            symbols_scored=processed,
            symbols_failed=failed,
            recommendations=rec_counts,
            symbols_total=len(symbol_list),
            duration_ms=duration_ms,
            scoring_version=SCORING_VERSION,
        )
        return message
        
    except Exception as e:
        logger.exception(f"quant_scoring_daily failed: {e}")
        raise


def _orm_to_fundamentals_dict(fund: "StockFundamentals") -> dict:
    """Convert ORM StockFundamentals to dict for scoring."""
    if fund is None:
        return {}
    return {
        "pe_ratio": float(fund.pe_ratio) if fund.pe_ratio else None,
        "forward_pe": float(fund.forward_pe) if fund.forward_pe else None,
        "peg_ratio": float(fund.peg_ratio) if fund.peg_ratio else None,
        "price_to_book": float(fund.price_to_book) if fund.price_to_book else None,
        "price_to_sales": float(fund.price_to_sales) if fund.price_to_sales else None,
        "profit_margin": float(fund.profit_margin) if fund.profit_margin else None,
        "operating_margin": float(fund.operating_margin) if fund.operating_margin else None,
        "gross_margin": float(fund.gross_margin) if fund.gross_margin else None,
        "return_on_equity": float(fund.return_on_equity) if fund.return_on_equity else None,
        "return_on_assets": float(fund.return_on_assets) if fund.return_on_assets else None,
        "debt_to_equity": float(fund.debt_to_equity) if fund.debt_to_equity else None,
        "current_ratio": float(fund.current_ratio) if fund.current_ratio else None,
        "quick_ratio": float(fund.quick_ratio) if fund.quick_ratio else None,
        "free_cash_flow": fund.free_cash_flow,
        "revenue_growth": float(fund.revenue_growth) if fund.revenue_growth else None,
        "earnings_growth": float(fund.earnings_growth) if fund.earnings_growth else None,
        "beta": float(fund.beta) if fund.beta else None,
        "recommendation_mean": float(fund.recommendation_mean) if fund.recommendation_mean else None,
    }


def _recommendation_to_mode(rec: str) -> str:
    """Map recommendation to legacy mode."""
    if rec in ("STRONG_BUY", "BUY"):
        return "CERTIFIED_BUY"
    elif rec == "ACCUMULATE":
        return "DIP_ENTRY"
    elif rec == "AVOID":
        return "DOWNTREND"
    else:
        return "HOLD"
        _add("current_drawdown_pct", dip_entry.current_drawdown_pct)
        
        # Fundamental metrics
        _add("pe_ratio", fundamentals.pe_ratio)
        _add("peg_ratio", fundamentals.peg_ratio)
        _add("ev_ebitda", getattr(fundamentals, "ev_ebitda", None))
        _add("profit_margin", fundamentals.profit_margin)
        _add("roe", fundamentals.roe)
        _add("revenue_growth", fundamentals.revenue_growth)
        _add("earnings_growth", fundamentals.earnings_growth)
        _add("debt_to_equity", fundamentals.debt_to_equity)
        _add("current_ratio", fundamentals.current_ratio)
        _add("free_cash_flow", fundamentals.free_cash_flow)
        _add("target_upside_pct", fundamentals.target_upside_pct)
    
    # Convert to numpy arrays
    return {k: np.array(v) for k, v in stats.items()}


# =============================================================================
# QUANT ANALYSIS NIGHTLY - Pre-compute all quant engine results
# =============================================================================


@register_job("quant_analysis_nightly")
async def quant_analysis_nightly_job() -> str:
    """
    Pre-compute all quant engine analysis results for tracked symbols.
    
    This job runs nightly after market close and pre-computes:
    - Technical snapshot (from TechnicalService)
    - Signal triggers (from signals module)
    - Dip analysis (from dip_entry_optimizer)
    - Current indicator states
    
    Results are stored in quant_precomputed table. API endpoints read from
    here instead of computing inline.
    
    Schedule: Nightly after market close (e.g., 11:55 PM after other jobs)
    """
    from datetime import date, timedelta
    
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import quant_precomputed_orm as quant_repo
    from app.quant_engine.signals.scanner import get_historical_triggers
    from app.quant_engine.core import TechnicalService, get_technical_service
    from app.services.prices import get_price_service

    logger.info("Starting quant_analysis_nightly job (V3 - unified services)")
    job_start = time.monotonic()
    
    # Initialize technical service
    tech_service = get_technical_service()

    symbols_list = await symbols_repo.list_symbols()
    tickers = [s.symbol for s in symbols_list if s.is_active]
    
    if not tickers:
        log_job_success("quant_analysis_nightly", "No active symbols to process",
                        symbols_processed=0, symbols_failed=0,
                        duration_ms=int((time.monotonic() - job_start) * 1000))
        return "No active symbols to process"
    
    logger.info(f"quant_analysis_nightly: Processing {len(tickers)} symbols")
    
    processed = 0
    failed = 0
    
    # Fetch SPY prices once for benchmarking
    price_service = get_price_service()
    end_date = date.today()
    start_date = end_date - timedelta(days=1260)
    
    spy_df = await price_history_repo.get_prices_as_dataframe("SPY", start_date, end_date)
    spy_prices = None
    if spy_df is not None and "Close" in spy_df.columns and len(spy_df) >= 60:
        spy_prices = spy_df["Close"].dropna()
    else:
        # Try PriceService fallback
        try:
            spy_df = await price_service.get_prices("SPY", start_date, end_date)
            if spy_df is not None and "Close" in spy_df.columns:
                spy_prices = spy_df["Close"].dropna()
        except Exception:
            pass
    
    for symbol in tickers:
        try:
            # Fetch price data
            df = await price_history_repo.get_prices_as_dataframe(
                symbol, start_date, end_date
            )
            
            if df is None or "Close" not in df.columns or len(df) < 60:
                continue
            
            prices = df["Close"].dropna()
            price_data = {"close": prices}
            
            data_start_date = prices.index[0].date() if hasattr(prices.index[0], 'date') else start_date
            data_end_date = prices.index[-1].date() if hasattr(prices.index[-1], 'date') else end_date
            
            # 1. Signal Backtest using historical triggers
            backtest_data = None
            try:
                triggers = get_historical_triggers(price_data, lookback_days=730)
                if triggers:
                    holding_days = triggers[0].holding_days if triggers else 20
                    signal_name = triggers[0].signal_name if triggers else "Unknown"
                    
                    # Simulate trades
                    entry_triggers = [t for t in triggers if t.signal_type == "entry"]
                    exit_triggers = [t for t in triggers if t.signal_type == "exit"]
                    
                    n_trades = len(entry_triggers)
                    wins = sum(1 for t in exit_triggers if t.avg_return_pct > 0)
                    win_rate = wins / n_trades if n_trades > 0 else 0
                    avg_return = sum(t.avg_return_pct for t in exit_triggers) / len(exit_triggers) if exit_triggers else 0
                    total_return = sum(t.avg_return_pct for t in exit_triggers)
                    
                    # Buy and hold comparison
                    if len(prices) >= 730:
                        bh_prices = prices.iloc[-730:]
                    else:
                        bh_prices = prices
                    buy_hold_return = ((bh_prices.iloc[-1] / bh_prices.iloc[0]) - 1) * 100 if len(bh_prices) >= 2 else 0
                    
                    backtest_data = {
                        "signal_name": signal_name,
                        "n_trades": n_trades,
                        "win_rate": win_rate,
                        "avg_return_pct": avg_return,
                        "total_return_pct": total_return,
                        "buy_hold_return_pct": buy_hold_return,
                        "edge_vs_buy_hold_pct": total_return - buy_hold_return,
                        "holding_days": holding_days,
                    }
            except Exception as e:
                logger.debug(f"Backtest failed for {symbol}: {e}")
            
            # 2. Technical Snapshot using TechnicalService
            trade_data = None
            try:
                # Get technical snapshot from unified service
                snapshot = tech_service.get_snapshot(df)
                
                # Calculate buy-hold and SPY returns for comparison
                if len(prices) >= 252:
                    bh_prices = prices.iloc[-252:]
                else:
                    bh_prices = prices
                buy_hold_return = ((bh_prices.iloc[-1] / bh_prices.iloc[0]) - 1) * 100 if len(bh_prices) >= 2 else 0
                
                spy_return = 0.0
                if spy_prices is not None and len(spy_prices) >= 252:
                    spy_slice = spy_prices.iloc[-252:]
                    spy_return = ((spy_slice.iloc[-1] / spy_slice.iloc[0]) - 1) * 100 if len(spy_slice) >= 2 else 0
                
                trade_data = {
                    "rsi_14": snapshot.rsi_14,
                    "macd_histogram": snapshot.macd_histogram,
                    "momentum_score": snapshot.momentum_score,
                    "trend_direction": snapshot.trend_direction,
                    "volatility_regime": snapshot.volatility_regime,
                    "adx": snapshot.adx,
                    "sma_50": snapshot.sma_50,
                    "sma_200": snapshot.sma_200,
                    "golden_cross": snapshot.golden_cross,
                    "death_cross": snapshot.death_cross,
                    "buy_hold_return_pct": buy_hold_return,
                    "spy_return_pct": spy_return,
                }
            except Exception as e:
                logger.debug(f"Technical analysis failed for {symbol}: {e}")
            
            # 3. Signal Combinations - skip legacy (use signals from backtest)
            combinations_data = None
            
            # 4. Dip Analysis using TechnicalService
            dip_data = None
            try:
                # Get drawdown metrics from technical snapshot
                snapshot = tech_service.get_snapshot(df)
                
                # Calculate current drawdown from price data
                if len(prices) >= 252:
                    high_252 = prices.iloc[-252:].max()
                else:
                    high_252 = prices.max()
                current_price = prices.iloc[-1]
                current_drawdown_pct = ((current_price / high_252) - 1) * 100 if high_252 > 0 else 0
                
                # Historical drawdown analysis
                rolling_max = prices.rolling(window=252, min_periods=1).max()
                drawdowns = ((prices - rolling_max) / rolling_max) * 100
                typical_dip_pct = abs(drawdowns.quantile(0.25))  # 25th percentile
                max_historical_dip_pct = abs(drawdowns.min())
                
                # Z-score of current drawdown
                drawdown_std = drawdowns.std()
                dip_zscore = (current_drawdown_pct - drawdowns.mean()) / drawdown_std if drawdown_std > 0 else 0
                
                # Classify dip type based on severity
                if current_drawdown_pct >= -5:
                    dip_type = "minor"
                    action = "hold"
                    confidence = 0.5
                elif current_drawdown_pct >= -10:
                    dip_type = "moderate"
                    action = "consider_buying"
                    confidence = 0.6
                elif current_drawdown_pct >= -20:
                    dip_type = "significant"
                    action = "buy"
                    confidence = 0.7
                else:
                    dip_type = "severe"
                    action = "buy_aggressive"
                    confidence = 0.8
                
                # Recovery probability based on historical patterns
                historical_recoveries = (drawdowns < current_drawdown_pct).sum()
                recovery_probability = historical_recoveries / len(drawdowns) if len(drawdowns) > 0 else 0.5
                
                dip_data = {
                    "current_drawdown_pct": round(current_drawdown_pct, 2),
                    "typical_pct": round(typical_dip_pct, 2),
                    "max_historical_pct": round(max_historical_dip_pct, 2),
                    "zscore": round(dip_zscore, 2),
                    "type": dip_type,
                    "action": action,
                    "confidence": round(confidence, 2),
                    "reasoning": f"Drawdown of {abs(current_drawdown_pct):.1f}% vs typical {typical_dip_pct:.1f}%",
                    "recovery_probability": round(recovery_probability, 2),
                }
            except Exception as e:
                logger.debug(f"Dip analysis failed for {symbol}: {e}")
            
            # 5. Current Signals from TechnicalService
            signals_data = None
            try:
                snapshot = tech_service.get_snapshot(df)
                
                # Build signals dict from snapshot
                signals_data = {
                    "rsi_14": snapshot.rsi_14,
                    "rsi_signal": "oversold" if snapshot.rsi_14 < 30 else ("overbought" if snapshot.rsi_14 > 70 else "neutral"),
                    "macd_histogram": snapshot.macd_histogram,
                    "macd_signal": "bullish" if snapshot.macd_histogram > 0 else "bearish",
                    "momentum_score": snapshot.momentum_score,
                    "momentum_signal": "strong" if abs(snapshot.momentum_score) > 50 else "weak",
                    "trend_direction": snapshot.trend_direction,
                    "volatility_regime": snapshot.volatility_regime,
                    "golden_cross": snapshot.golden_cross,
                    "death_cross": snapshot.death_cross,
                    "adx": snapshot.adx,
                    "trend_strength": "strong" if snapshot.adx > 25 else "weak",
                }
            except Exception as e:
                logger.debug(f"Current signals failed for {symbol}: {e}")
            
            # 6. Dip Entry Analysis
            dip_entry_data = None
            try:
                from app.quant_engine.dipfinder.entry_optimizer import DipEntryOptimizer, get_dip_summary, get_dip_signal_triggers
                
                optimizer = DipEntryOptimizer()
                result = optimizer.analyze(df, symbol, fundamentals=None)
                summary = get_dip_summary(result)
                
                # Generate dip signal triggers for chart overlay
                dip_triggers = get_dip_signal_triggers(result)
                
                dip_entry_data = {
                    "optimal_threshold": summary["optimal_dip_threshold"],
                    "optimal_price": summary["optimal_entry_price"],
                    "is_buy_now": summary["is_buy_now"],
                    "signal_strength": summary["buy_signal_strength"],
                    "signal_reason": summary["signal_reason"],
                    "recovery_days": int(summary["typical_recovery_days"]) if summary["typical_recovery_days"] else None,
                    "threshold_analysis": summary["threshold_analysis"],
                    "signal_triggers": dip_triggers,  # Include dip signal triggers
                }
            except Exception as e:
                logger.debug(f"Dip entry failed for {symbol}: {e}")
            
            # 7. Signal Triggers (for chart overlays)
            signal_triggers_data = None
            try:
                triggers_list = get_historical_triggers(price_data, lookback_days=365)
                if triggers_list:
                    # Get benchmark comparison
                    buy_hold_return_pct = 0.0
                    signal_return_pct = 0.0
                    
                    if len(prices) >= 365:
                        lookback_slice = prices.iloc[-365:]
                        if len(lookback_slice) >= 2:
                            first_price = float(lookback_slice.iloc[0])
                            last_price = float(lookback_slice.iloc[-1])
                            buy_hold_return_pct = ((last_price / first_price) - 1) * 100
                    
                    exit_triggers = [t for t in triggers_list if t.signal_type == "exit"]
                    if exit_triggers:
                        signal_return_pct = sum(t.avg_return_pct for t in exit_triggers)
                    
                    signal_triggers_data = {
                        "signal_name": triggers_list[0].signal_name if triggers_list else None,
                        "n_trades": len([t for t in triggers_list if t.signal_type == "entry"]),
                        "buy_hold_return_pct": round(buy_hold_return_pct, 2),
                        "signal_return_pct": round(signal_return_pct, 2),
                        "edge_vs_buy_hold_pct": round(signal_return_pct - buy_hold_return_pct, 2),
                        "triggers": [
                            {
                                "date": t.date.isoformat() if hasattr(t.date, 'isoformat') else str(t.date),
                                "signal_name": t.signal_name,
                                "price": t.price,
                                "win_rate": t.win_rate,
                                "avg_return_pct": t.avg_return_pct,
                                "signal_type": t.signal_type,
                            }
                            for t in triggers_list
                        ],
                    }
            except Exception as e:
                logger.debug(f"Signal triggers failed for {symbol}: {e}")
            
            # Store all results
            await quant_repo.upsert_all_quant_data(
                symbol=symbol,
                backtest=backtest_data,
                trade_strategy=trade_data,
                combinations=combinations_data,
                dip_analysis=dip_data,
                current_signals=signals_data,
                dip_entry=dip_entry_data,
                signal_triggers=signal_triggers_data,
                data_start=data_start_date,
                data_end=data_end_date,
            )
            
            processed += 1
            
            if processed % 10 == 0:
                logger.info(f"quant_analysis_nightly: Processed {processed}/{len(tickers)} symbols")
            
        except Exception as e:
            logger.exception(f"quant_analysis_nightly: Failed for {symbol}: {e}")
            failed += 1
    
    duration_ms = int((time.monotonic() - job_start) * 1000)
    message = f"Pre-computed quant analysis for {processed}/{len(tickers)} symbols ({failed} failed)"
    
    # Structured success log
    log_job_success(
        "quant_analysis_nightly",
        message,
        symbols_processed=processed,
        symbols_failed=failed,
        symbols_total=len(tickers),
        duration_ms=duration_ms,
    )
    return message


# Public exports
__all__ = [
    "quant_monthly_job",
    "strategy_nightly_job",
    "quant_scoring_daily_job",
    "quant_analysis_nightly_job",
]
