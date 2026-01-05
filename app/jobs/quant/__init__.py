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
# STRATEGY OPTIMIZATION - Nightly after prices
# =============================================================================


@register_job("strategy_nightly")
async def strategy_nightly_job() -> str:
    """
    Nightly strategy optimization for all tracked symbols.
    
    Runs AFTER prices_daily to:
    1. Run full backtest optimization with recency weighting
    2. Find best strategy for each symbol that works NOW (not just historically)
    3. Check fundamentals for entry/exit signals
    4. Store results in strategy_signals table for API access
    
    Key features:
    - Recency weighting: Recent trades (last 6mo) matter 3x more
    - Current year validation: Strategy must be profitable in 2025
    - Fundamental filters: Only signal entry when financials healthy
    - Statistically sound: Walk-forward validation, min 30 trades
    
    Schedule: Mon-Fri at 11:30 PM UTC (30 min after prices_daily)
    """
    import pandas as pd
    from decimal import Decimal
    from sqlalchemy.dialects.postgresql import insert
    
    from app.cache.cache import Cache
    from app.database.connection import get_session
    from app.database.orm import StrategySignal, StockFundamentals
    from app.repositories import symbols_orm as symbols_repo
    from app.quant_engine.strategy_optimizer import (
        StrategyOptimizer, TradingConfig, RecencyConfig, FundamentalFilter,
        result_to_dict,
    )
    from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer
    
    logger.info("Starting strategy_nightly job")
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
        
        logger.info(f"[STRATEGY] Optimizing strategies for {len(symbol_list)} symbols")
        
        # Get SPY for benchmark comparison
        from datetime import date, timedelta
        end_date = date.today()
        start_date = end_date - timedelta(days=1260)  # 5 years
        spy_df = await price_history_repo.get_prices_as_dataframe("SPY", start_date, end_date)
        spy_prices = None
        if spy_df is not None and len(spy_df) > 0:
            spy_close_col = get_close_column(spy_df)
            spy_prices = spy_df[spy_close_col]
        
        # Initialize optimizer
        optimizer = StrategyOptimizer(
            config=TradingConfig(),
            recency_config=RecencyConfig(),
            fundamental_filter=FundamentalFilter(),
        )
        
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
                    # Get price history
                    df = await price_history_repo.get_prices_as_dataframe(symbol, start_date, end_date)
                    
                    if df is None or len(df) < 200:
                        logger.warning(f"[STRATEGY] Insufficient data for {symbol}, skipping")
                        continue
                    
                    # Convert fundamentals to dict
                    fund_obj = fundamentals_map.get(symbol)
                    fundamentals = None
                    if fund_obj:
                        fundamentals = {
                            "pe_ratio": float(fund_obj.pe_ratio) if fund_obj.pe_ratio else None,
                            "forward_pe": float(fund_obj.forward_pe) if fund_obj.forward_pe else None,
                            "peg_ratio": float(fund_obj.peg_ratio) if fund_obj.peg_ratio else None,
                            "profit_margin": float(fund_obj.profit_margin) if fund_obj.profit_margin else None,
                            "free_cash_flow": fund_obj.free_cash_flow,
                            "market_cap": None,  # Would need from symbol info
                            "debt_to_equity": float(fund_obj.debt_to_equity) if fund_obj.debt_to_equity else None,
                            "current_ratio": float(fund_obj.current_ratio) if fund_obj.current_ratio else None,
                            "revenue_growth": float(fund_obj.revenue_growth) if fund_obj.revenue_growth else None,
                            "recommendation_mean": float(fund_obj.recommendation_mean) if fund_obj.recommendation_mean else None,
                            "target_mean_price": float(fund_obj.target_mean_price) if fund_obj.target_mean_price else None,
                        }
                    
                    # Run optimization (use fewer trials for nightly batch)
                    opt_result = optimizer.optimize_for_symbol(
                        df=df,
                        symbol=symbol,
                        fundamentals=fundamentals,
                        spy_prices=spy_prices,
                        n_trials=50,  # Reduced for batch processing
                    )
                    
                    # Calculate typical recovery days from dip entry optimizer
                    typical_recovery_days = None
                    try:
                        dip_entry_result = dip_optimizer.analyze(df, symbol, fundamentals)
                        if dip_entry_result and dip_entry_result.typical_recovery_days > 0:
                            typical_recovery_days = int(dip_entry_result.typical_recovery_days)
                    except Exception as dip_err:
                        logger.debug(f"[STRATEGY] Dip entry analysis failed for {symbol}: {dip_err}")
                    
                    # Upsert to database
                    stmt = insert(StrategySignal).values(
                        symbol=symbol,
                        strategy_name=opt_result.best_strategy_name,
                        strategy_params=opt_result.best_params,
                        signal_type=opt_result.signal_type,
                        signal_reason=opt_result.signal_reason,
                        has_active_signal=opt_result.has_active_signal,
                        total_return_pct=Decimal(str(opt_result.total_return_pct)),
                        sharpe_ratio=Decimal(str(opt_result.sharpe_ratio)),
                        win_rate=Decimal(str(opt_result.win_rate)),
                        max_drawdown_pct=Decimal(str(opt_result.max_drawdown_pct)),
                        n_trades=opt_result.n_trades,
                        recency_weighted_return=Decimal(str(opt_result.recency_weighted_return)),
                        current_year_return_pct=Decimal(str(opt_result.current_year_return_pct)),
                        current_year_win_rate=Decimal(str(opt_result.current_year_win_rate)),
                        current_year_trades=opt_result.current_year_trades,
                        vs_buy_hold_pct=Decimal(str(opt_result.vs_buy_hold)),
                        vs_spy_pct=Decimal(str(opt_result.vs_spy)) if opt_result.vs_spy else None,
                        beats_buy_hold=opt_result.beats_buy_hold,
                        beats_spy=opt_result.beats_spy,
                        fundamentals_healthy=opt_result.fundamentals_healthy,
                        fundamental_concerns=opt_result.fundamental_concerns,
                        is_statistically_valid=opt_result.is_statistically_valid,
                        recent_trades=opt_result.recent_trades,
                        indicators_used=opt_result.indicators_used,
                        typical_recovery_days=typical_recovery_days,
                    ).on_conflict_do_update(
                        index_elements=["symbol"],
                        set_={
                            "strategy_name": opt_result.best_strategy_name,
                            "strategy_params": opt_result.best_params,
                            "signal_type": opt_result.signal_type,
                            "signal_reason": opt_result.signal_reason,
                            "has_active_signal": opt_result.has_active_signal,
                            "total_return_pct": Decimal(str(opt_result.total_return_pct)),
                            "sharpe_ratio": Decimal(str(opt_result.sharpe_ratio)),
                            "win_rate": Decimal(str(opt_result.win_rate)),
                            "max_drawdown_pct": Decimal(str(opt_result.max_drawdown_pct)),
                            "n_trades": opt_result.n_trades,
                            "recency_weighted_return": Decimal(str(opt_result.recency_weighted_return)),
                            "current_year_return_pct": Decimal(str(opt_result.current_year_return_pct)),
                            "current_year_win_rate": Decimal(str(opt_result.current_year_win_rate)),
                            "current_year_trades": opt_result.current_year_trades,
                            "vs_buy_hold_pct": Decimal(str(opt_result.vs_buy_hold)),
                            "vs_spy_pct": Decimal(str(opt_result.vs_spy)) if opt_result.vs_spy else None,
                            "beats_buy_hold": opt_result.beats_buy_hold,
                            "beats_spy": opt_result.beats_spy,
                            "fundamentals_healthy": opt_result.fundamentals_healthy,
                            "fundamental_concerns": opt_result.fundamental_concerns,
                            "is_statistically_valid": opt_result.is_statistically_valid,
                            "recent_trades": opt_result.recent_trades,
                            "indicators_used": opt_result.indicators_used,
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


def _to_z(value: float | None, mean: float, std: float, invert: bool = False) -> float:
    """Convert value to z-score. If invert=True, lower is better."""
    if value is None:
        return 0.0
    z = (float(value) - mean) / std if std != 0 else 0.0
    return -z if invert else z


# =============================================================================
# QUANT SCORING - Dual-mode scoring pipeline (daily)
# =============================================================================


@register_job("quant_scoring_daily")
async def quant_scoring_daily_job() -> str:
    """
    Daily dual-mode scoring pipeline (APUS + DOUS).
    
    Computes comprehensive scores for all tracked symbols using:
    - Mode A (APUS): Certified Buy - statistically proven edge over benchmarks
    - Mode B (DOUS): Dip Entry - fundamental + technical opportunity scoring
    
    Key features:
    - Stationary bootstrap (Politis & Romano) for P(edge > 0) and CI
    - Deflated Sharpe ratio (Lopez de Prado) for multiple testing correction
    - Walk-forward OOS with embargo
    - Regime robustness (bull/bear/high-vol)
    - Fundamental momentum z-scores
    - BestScore 0-100 per symbol
    
    Schedule: Mon-Fri 11:45 PM UTC (15 min after strategy_nightly)
    """
    from datetime import date, timedelta
    from decimal import Decimal
    
    import pandas as pd
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert
    
    from app.cache.cache import Cache
    from app.database.connection import get_session
    from app.database.orm import (
        DipState,
        QuantScore,
        StrategySignal,
        StockFundamentals,
        Symbol,
    )
    from app.repositories import symbols_orm as symbols_repo
    from app.quant_engine.scoring import (
        ScoringConfig,
        compute_symbol_score,
        generate_historical_weights,
        SCORING_VERSION,
    )
    
    logger.info("Starting quant_scoring_daily job")
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
        
        logger.info(f"[SCORING] Computing scores for {len(symbol_list)} symbols")
        
        # Scoring configuration
        config = ScoringConfig()
        
        # Get SPY prices for benchmark
        end_date = date.today()
        start_date = end_date - timedelta(days=1260)  # 5 years
        spy_df = await price_history_repo.get_prices_as_dataframe("SPY", start_date, end_date)
        spy_prices = None
        if spy_df is not None and len(spy_df) > 0:
            spy_close_col = get_close_column(spy_df)
            spy_prices = spy_df[spy_close_col]
        
        if spy_prices is None:
            logger.error("[SCORING] No SPY data available")
            return "No SPY data available"
        
        # Compute SPY drawdown from 52w high for market normalization
        spy_52w_window = min(252, len(spy_prices))
        spy_52w_high = spy_prices.rolling(spy_52w_window).max().iloc[-1]
        spy_current = spy_prices.iloc[-1]
        spy_dip_pct = ((spy_52w_high - spy_current) / spy_52w_high * 100) if spy_52w_high > 0 else 0
        logger.info(f"[SCORING] Market (SPY) is {spy_dip_pct:.1f}% below 52w high")
        
        processed = 0
        failed = 0
        mode_a_count = 0
        mode_b_count = 0
        mode_hold_count = 0
        
        async with get_session() as session:
            # Load all strategy signals (for weights)
            result = await session.execute(select(StrategySignal))
            strategy_map = {s.symbol: s for s in result.scalars().all()}
            
            # Load all fundamentals
            result = await session.execute(
                select(StockFundamentals).where(
                    StockFundamentals.symbol.in_(symbol_list)
                )
            )
            fundamentals_map = {f.symbol: f for f in result.scalars().all()}
            
            # Load dip state for all symbols
            result = await session.execute(
                select(DipState).where(DipState.symbol.in_(symbol_list))
            )
            dip_state_map = {d.symbol: d for d in result.scalars().all()}
            
            # Load min_dip_pct thresholds from symbols
            result = await session.execute(
                select(Symbol).where(Symbol.symbol.in_(symbol_list))
            )
            symbol_thresholds = {s.symbol: float(s.min_dip_pct or 0.15) for s in result.scalars().all()}
            
            # Count strategies tested for deflated Sharpe
            n_strategies_tested = len(strategy_map)
            
            for symbol in symbol_list:
                try:
                    # Get price history
                    df = await price_history_repo.get_prices_as_dataframe(
                        symbol, start_date, end_date
                    )
                    
                    if df is None or len(df) < 252:
                        logger.warning(f"[SCORING] Insufficient data for {symbol}, skipping")
                        continue
                    
                    # Get strategy signal for weights
                    strategy = strategy_map.get(symbol)
                    
                    # Build strategy weights from HISTORICAL signal triggers
                    # This computes actual backtested positions, not just current signal
                    if df is not None and len(df) > 0:
                        df_close_col = get_close_column(df)
                        prices_series = df[df_close_col]
                        
                        # Generate historical weights from best signal
                        strategy_weights = generate_historical_weights(prices_series, holding_days=20)
                    else:
                        strategy_weights = pd.Series()
                    
                    # Convert fundamentals to z-scores (simplified)
                    fund_obj = fundamentals_map.get(symbol)
                    fundamentals_dict = None
                    if fund_obj:
                        fundamentals_dict = {
                            "revenue_z": _to_z(fund_obj.revenue_growth, 0.10, 0.15),
                            "earnings_z": _to_z(fund_obj.profit_margin, 0.10, 0.10),
                            "margin_z": _to_z(fund_obj.profit_margin, 0.08, 0.10),
                            "pe_z": _to_z(fund_obj.pe_ratio, 20, 15, invert=True) if fund_obj.pe_ratio else 0,
                            "ev_ebitda_z": 0.0,
                            "ps_z": 0.0,
                        }
                    
                    # Get event dates
                    earnings_date = None
                    dividend_date = None
                    if fund_obj:
                        earnings_date = fund_obj.earnings_date if hasattr(fund_obj, 'earnings_date') else None
                        dividend_date = fund_obj.dividend_date if hasattr(fund_obj, 'dividend_date') else None
                    
                    # Compute score
                    result = compute_symbol_score(
                        symbol=symbol,
                        prices=df,
                        spy_prices=spy_prices,
                        strategy_weights=strategy_weights,
                        fundamentals=fundamentals_dict,
                        earnings_date=earnings_date,
                        dividend_date=dividend_date,
                        n_strategies_tested=max(1, n_strategies_tested),
                        config=config,
                    )
                    
                    # Check if stock is actually in a qualifying dip
                    dip_state = dip_state_map.get(symbol)
                    min_dip_pct = symbol_thresholds.get(symbol, 0.15) * 100
                    dip_pct = float(dip_state.dip_percentage) if dip_state and dip_state.dip_percentage else 0
                    
                    # Normalize dip against market (SPY)
                    normalized_dip_pct = max(0, dip_pct - spy_dip_pct)
                    
                    # Check how long the stock has been in this "dip"
                    MAX_DIP_DAYS = 365
                    days_in_dip = 0
                    if dip_state and dip_state.dip_start_date:
                        days_in_dip = (date.today() - dip_state.dip_start_date).days
                    
                    # Use normalized dip for entry decision
                    is_in_dip = normalized_dip_pct >= min_dip_pct and days_in_dip <= MAX_DIP_DAYS
                    is_downtrend = dip_pct >= min_dip_pct and days_in_dip > MAX_DIP_DAYS
                    
                    # Override mode if not in qualifying dip
                    final_mode = result.mode
                    if not result.gate_pass:
                        if is_downtrend:
                            final_mode = "DOWNTREND"
                        elif not is_in_dip:
                            final_mode = "HOLD"
                    
                    # Track modes
                    if result.gate_pass:
                        mode_a_count += 1
                    elif is_in_dip:
                        mode_b_count += 1
                    elif is_downtrend:
                        mode_hold_count += 1
                    else:
                        mode_hold_count += 1
                    
                    # Upsert to database
                    evidence_dict = result.evidence.to_dict() if result.evidence else None
                    
                    stmt = insert(QuantScore).values(
                        symbol=symbol,
                        best_score=Decimal(str(round(result.best_score, 2))),
                        mode=final_mode,
                        score_a=Decimal(str(round(result.score_a, 2))),
                        score_b=Decimal(str(round(result.score_b, 2))),
                        gate_pass=result.gate_pass,
                        p_outperf=Decimal(str(round(result.evidence.p_outperf, 4))),
                        ci_low=Decimal(str(round(result.evidence.ci_low, 4))),
                        ci_high=Decimal(str(round(result.evidence.ci_high, 4))),
                        dsr=Decimal(str(round(result.evidence.dsr, 4))),
                        median_edge=Decimal(str(round(result.evidence.median_edge, 4))),
                        edge_vs_stock=Decimal(str(round(result.evidence.edge_vs_stock, 4))),
                        edge_vs_spy=Decimal(str(round(result.evidence.edge_vs_spy, 4))),
                        worst_regime_edge=Decimal(str(round(result.evidence.worst_regime_edge, 4))),
                        cvar_5=Decimal(str(round(result.evidence.cvar_5, 4))),
                        fund_mom=Decimal(str(round(result.evidence.fund_mom, 4))),
                        val_z=Decimal(str(round(result.evidence.val_z, 4))),
                        event_risk=result.evidence.event_risk,
                        p_recovery=Decimal(str(round(result.evidence.p_recovery, 4))),
                        expected_value=Decimal(str(round(result.evidence.expected_value, 4))),
                        sector_relative=Decimal(str(round(result.evidence.sector_relative, 4))),
                        config_hash=result.config_hash,
                        scoring_version=result.scoring_version,
                        data_start=result.data_start,
                        data_end=result.data_end,
                        evidence=evidence_dict,
                    ).on_conflict_do_update(
                        index_elements=["symbol"],
                        set_={
                            "best_score": Decimal(str(round(result.best_score, 2))),
                            "mode": final_mode,
                            "score_a": Decimal(str(round(result.score_a, 2))),
                            "score_b": Decimal(str(round(result.score_b, 2))),
                            "gate_pass": result.gate_pass,
                            "p_outperf": Decimal(str(round(result.evidence.p_outperf, 4))),
                            "ci_low": Decimal(str(round(result.evidence.ci_low, 4))),
                            "ci_high": Decimal(str(round(result.evidence.ci_high, 4))),
                            "dsr": Decimal(str(round(result.evidence.dsr, 4))),
                            "median_edge": Decimal(str(round(result.evidence.median_edge, 4))),
                            "edge_vs_stock": Decimal(str(round(result.evidence.edge_vs_stock, 4))),
                            "edge_vs_spy": Decimal(str(round(result.evidence.edge_vs_spy, 4))),
                            "worst_regime_edge": Decimal(str(round(result.evidence.worst_regime_edge, 4))),
                            "cvar_5": Decimal(str(round(result.evidence.cvar_5, 4))),
                            "fund_mom": Decimal(str(round(result.evidence.fund_mom, 4))),
                            "val_z": Decimal(str(round(result.evidence.val_z, 4))),
                            "event_risk": result.evidence.event_risk,
                            "p_recovery": Decimal(str(round(result.evidence.p_recovery, 4))),
                            "expected_value": Decimal(str(round(result.evidence.expected_value, 4))),
                            "sector_relative": Decimal(str(round(result.evidence.sector_relative, 4))),
                            "config_hash": result.config_hash,
                            "scoring_version": result.scoring_version,
                            "data_start": result.data_start,
                            "data_end": result.data_end,
                            "computed_at": datetime.now(UTC),
                            "evidence": evidence_dict,
                        }
                    )
                    await session.execute(stmt)
                    
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"[SCORING] Processed {processed}/{len(symbol_list)} symbols")
                        await session.commit()
                    
                except Exception as e:
                    logger.exception(f"[SCORING] Failed to score {symbol}: {e}")
                    failed += 1
                    continue
            
            await session.commit()
        
        # Clear cache
        cache = Cache(prefix="quant_scores", default_ttl=86400)
        await cache.invalidate_pattern("*")
        
        # Also clear recommendations cache
        recs_cache = Cache(prefix="recommendations", default_ttl=300)
        await recs_cache.invalidate_pattern("*")
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = (
            f"Scored {processed} symbols ({failed} failed), "
            f"Mode A: {mode_a_count}, Dip Entry: {mode_b_count}, Hold: {mode_hold_count}"
        )
        
        # Structured success log
        log_job_success(
            "quant_scoring_daily",
            message,
            symbols_scored=processed,
            symbols_failed=failed,
            mode_a_count=mode_a_count,
            mode_b_count=mode_b_count,
            mode_hold_count=mode_hold_count,
            symbols_total=len(symbol_list),
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.exception(f"quant_scoring_daily failed: {e}")
        raise


# =============================================================================
# QUANT ANALYSIS NIGHTLY - Pre-compute all quant engine results
# =============================================================================


@register_job("quant_analysis_nightly")
async def quant_analysis_nightly_job() -> str:
    """
    Pre-compute all quant engine analysis results for tracked symbols.
    
    This job runs nightly after market close and pre-computes:
    - Signal backtest results (signal vs buy-and-hold comparison)
    - Full trade strategies (optimized entry/exit)
    - Signal combinations
    - Dip analysis (overreaction vs falling knife)
    - Current signals
    
    Results are stored in quant_precomputed table. API endpoints read from
    here instead of computing inline.
    
    Schedule: Nightly after market close (e.g., 11:55 PM after other jobs)
    """
    from datetime import date, timedelta
    
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import quant_precomputed_orm as quant_repo
    from app.quant_engine.signals import get_historical_triggers
    from app.quant_engine.trade_engine import (
        get_best_trade_strategy,
        test_signal_combination,
        analyze_dip,
        get_current_signals,
        SIGNAL_COMBINATIONS,
    )
    from app.services.prices import get_price_service

    logger.info("Starting quant_analysis_nightly job")
    job_start = time.monotonic()

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
            
            # 1. Signal Backtest
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
            
            # 2. Trade Strategy
            trade_data = None
            try:
                result = get_best_trade_strategy(price_data, symbol, spy_prices=spy_prices)
                if result:
                    trade_data = {
                        "entry_signal": result.entry_signal,
                        "entry_threshold": result.entry_threshold,
                        "exit_signal": result.exit_signal,
                        "exit_threshold": result.exit_threshold,
                        "n_trades": result.n_trades,
                        "win_rate": result.win_rate,
                        "avg_return_pct": result.avg_return_pct,
                        "sharpe_ratio": result.sharpe_ratio,
                        "buy_hold_return_pct": result.buy_hold_return_pct,
                        "spy_return_pct": result.spy_return_pct,
                        "beats_both": result.beats_both_benchmarks,
                    }
            except Exception as e:
                logger.debug(f"Trade strategy failed for {symbol}: {e}")
            
            # 3. Signal Combinations
            combinations_data = None
            try:
                combos = []
                for combo_cfg in SIGNAL_COMBINATIONS:
                    combo = test_signal_combination(price_data, combo_cfg, holding_days=20, min_signals=3)
                    if combo is not None:
                        combos.append({
                            "name": combo.name,
                            "component_signals": combo.component_signals,
                            "logic": combo.logic,
                            "win_rate": combo.win_rate,
                            "avg_return_pct": combo.avg_return_pct,
                            "n_signals": combo.n_signals,
                            "improvement_vs_best_single": combo.improvement_vs_best_single,
                        })
                combos.sort(key=lambda c: c["win_rate"] * c["avg_return_pct"], reverse=True)
                combinations_data = combos
            except Exception as e:
                logger.debug(f"Combinations failed for {symbol}: {e}")
            
            # 4. Dip Analysis
            dip_data = None
            try:
                analysis = analyze_dip(price_data, symbol)
                dip_type_str = analysis.dip_type.value if hasattr(analysis.dip_type, 'value') else str(analysis.dip_type)
                dip_data = {
                    "current_drawdown_pct": analysis.current_drawdown_pct,
                    "typical_pct": analysis.typical_dip_pct,
                    "max_historical_pct": analysis.max_historical_dip_pct,
                    "zscore": analysis.dip_zscore,
                    "type": dip_type_str,
                    "action": analysis.action,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning,
                    "recovery_probability": analysis.recovery_probability,
                }
            except Exception as e:
                logger.debug(f"Dip analysis failed for {symbol}: {e}")
            
            # 5. Current Signals
            signals_data = None
            try:
                signals = get_current_signals(price_data, symbol)
                signals_data = signals
            except Exception as e:
                logger.debug(f"Current signals failed for {symbol}: {e}")
            
            # 6. Dip Entry Analysis
            dip_entry_data = None
            try:
                from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer, get_dip_summary, get_dip_signal_triggers
                
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
