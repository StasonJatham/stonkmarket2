"""Analysis job definitions.

This module contains jobs for:
- Daily signal scanning for buy opportunities (signals_daily)
- DipFinder signal refresh for all tracked symbols (dipfinder_daily)
- Market regime detection and caching (regime_daily)

These jobs form the analysis backbone of the trading system,
providing buy signals, dip opportunities, and market regime context.
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

logger = get_logger("jobs.analysis")


# =============================================================================
# SIGNALS DAILY - Signal scanner for buy opportunities
# =============================================================================


@register_job("signals_daily")
async def signals_daily_job() -> str:
    """
    Daily signal scanner to cache buy opportunities.

    Scans all tracked symbols for technical buy signals, caching the results
    for fast Dashboard access.

    Schedule: Daily at 10 PM UTC (after US market close)
    """
    from datetime import date, timedelta

    import pandas as pd

    from app.cache.cache import Cache
    from app.quant_engine import scan_all_stocks
    from app.repositories import price_history_orm as price_history_repo
    from app.repositories import symbols_orm as symbols_repo

    logger.info("Starting signals_daily job")
    job_start = time.monotonic()

    cache = Cache(prefix="signals", default_ttl=86400)  # 24 hour TTL

    try:
        # Get all tracked symbols
        symbols_list = await symbols_repo.list_symbols()

        if not symbols_list:
            return "No symbols tracked"

        # Filter out benchmarks
        symbol_list = [
            s.symbol for s in symbols_list 
            if s.symbol not in ("SPY", "^GSPC", "URTH")
        ]
        symbol_names = {s.symbol: s.name or s.symbol for s in symbols_list}
        
        logger.info(f"Scanning {len(symbol_list)} symbols for signals")

        # Fetch price history (5 years for backtesting)
        end_date = date.today()
        start_date = end_date - timedelta(days=1260)

        price_dfs: dict[str, pd.Series] = {}
        for symbol in symbol_list:
            df = await price_history_repo.get_prices_as_dataframe(
                symbol, start_date, end_date
            )
            if df is not None:
                close_col = get_close_column(df)
                price_dfs[symbol] = df[close_col]

        if not price_dfs:
            logger.warning("No price data available")
            return "No price data"

        # Run signal scanner - uses dynamic holding period range internally
        opportunities = scan_all_stocks(price_dfs, symbol_names)

        # Cache results
        cache_data = {
            "scanned_at": str(date.today()),
            "n_symbols": len(opportunities),
            "opportunities": [
                {
                    "symbol": opp.symbol,
                    "name": opp.name,
                    "buy_score": opp.buy_score,
                    "opportunity_type": opp.opportunity_type,
                    "opportunity_reason": opp.opportunity_reason,
                    "current_price": opp.current_price,
                    "zscore_20d": opp.zscore_20d,
                    "rsi_14": opp.rsi_14,
                    "best_signal_name": opp.best_signal_name,
                    "best_holding_days": opp.best_holding_days,
                    "best_expected_return": opp.best_expected_return,
                    "n_active_signals": len(opp.active_signals),
                }
                for opp in opportunities[:50]  # Top 50
            ],
        }

        await cache.set("daily_scan", cache_data)

        # Count active signals
        total_active = sum(len(opp.active_signals) for opp in opportunities)
        strong_buys = sum(1 for opp in opportunities if opp.opportunity_type == "STRONG_BUY")

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Scanned {len(opportunities)} stocks, {strong_buys} strong buys, {total_active} active signals"
        
        # Structured success log
        log_job_success(
            "signals_daily",
            message,
            symbols_scanned=len(opportunities),
            strong_buys=strong_buys,
            active_signals=total_active,
            with_price_data=len(price_dfs),
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"signals_daily failed: {e}")
        raise


# =============================================================================
# DIPFINDER DAILY - Dip signal refresh and entry optimization
# =============================================================================


@register_job("dipfinder_daily")
async def dipfinder_daily_job() -> str:
    """
    Daily DipFinder signal refresh for all tracked symbols.
    
    Computes dip metrics, scores, and enhanced analysis for each symbol.
    Also runs dip entry optimizer to find optimal buy thresholds.
    Must run AFTER quant_scoring_daily since DipFinder requires quant scores.
    
    Schedule: Daily at 11:50 PM UTC (after quant_scoring_daily at 11:45 PM)
    """
    from datetime import date
    
    from app.dipfinder.config import get_dipfinder_config
    from app.dipfinder.service import DipFinderService
    from app.repositories import symbols_orm as symbols_repo
    
    logger.info("Starting dipfinder_daily job")
    job_start = time.monotonic()
    
    try:
        # Get all tracked symbols
        symbols_list = await symbols_repo.list_symbols()
        
        if not symbols_list:
            return "No symbols tracked"
        
        # Filter out benchmarks
        tickers = [
            s.symbol for s in symbols_list
            if s.symbol not in ("SPY", "^GSPC", "URTH")
        ]
        
        config = get_dipfinder_config()
        service = DipFinderService(config)
        
        logger.info(f"[DIPFINDER] Refreshing signals for {len(tickers)} symbols")
        
        processed = 0
        skipped = 0
        failed = 0
        
        for ticker in tickers:
            try:
                signal = await service.get_signal(
                    ticker=ticker,
                    window=config.windows[0] if config.windows else 60,
                    benchmark=config.default_benchmark,
                    force_refresh=True,
                )
                
                if signal:
                    processed += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                logger.debug(f"[DIPFINDER] Failed to process {ticker}: {e}")
                failed += 1
        
        message = f"Processed {processed} signals, {skipped} skipped (no quant score), {failed} failed"
        logger.info(f"[DIPFINDER] {message}")
        
        # Phase 2: Run dip entry optimizer for all symbols
        from datetime import timedelta
        from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer, get_dip_summary
        from app.repositories import price_history_orm as price_history_repo
        from app.repositories import quant_precomputed_orm as quant_repo
        
        logger.info(f"[DIP_ENTRY] Starting dip entry optimization for {len(tickers)} symbols")
        
        dip_entry_processed = 0
        dip_entry_skipped = 0
        dip_entry_failed = 0
        end_date = date.today()
        start_date = end_date - timedelta(days=1825)  # 5 years
        
        for ticker in tickers:
            try:
                # Get symbol's min_dip_pct from DB
                db_symbol = next((s for s in symbols_list if s.symbol == ticker), None)
                min_dip_threshold = None
                if db_symbol and db_symbol.min_dip_pct:
                    min_dip_threshold = -float(db_symbol.min_dip_pct) * 100
                
                # Fetch price history
                df = await price_history_repo.get_prices_as_dataframe(ticker, start_date, end_date)
                
                if df is None or df.empty or len(df) < 252:
                    dip_entry_skipped += 1
                    continue
                
                # Run optimizer
                optimizer = DipEntryOptimizer()
                result = optimizer.analyze(df, ticker, None, min_dip_threshold=min_dip_threshold)
                summary = get_dip_summary(result)
                
                # Update quant_precomputed table
                await quant_repo.update_dip_entry(
                    symbol=ticker,
                    optimal_threshold=summary["optimal_dip_threshold"],
                    optimal_price=summary["optimal_entry_price"],
                    max_profit_threshold=summary["max_profit_threshold"],
                    max_profit_price=summary["max_profit_entry_price"],
                    max_profit_total_return=summary["max_profit_total_return"],
                    is_buy_now=summary["is_buy_now"],
                    signal_strength=summary["buy_signal_strength"],
                    signal_reason=summary["signal_reason"],
                    recovery_days=summary["typical_recovery_days"],
                    threshold_analysis=summary["threshold_analysis"],
                )
                dip_entry_processed += 1
                
            except Exception as e:
                logger.debug(f"[DIP_ENTRY] Failed to process {ticker}: {e}")
                dip_entry_failed += 1
        
        dip_entry_message = f"Dip entry: {dip_entry_processed} optimized, {dip_entry_skipped} skipped, {dip_entry_failed} failed"
        logger.info(f"[DIP_ENTRY] {dip_entry_message}")
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        full_message = f"{message} | {dip_entry_message}"
        
        # Structured success log
        log_job_success(
            "dipfinder_daily",
            full_message,
            signals_processed=processed,
            signals_skipped=skipped,
            signals_failed=failed,
            dip_entries_optimized=dip_entry_processed,
            dip_entries_skipped=dip_entry_skipped,
            dip_entries_failed=dip_entry_failed,
            symbols_total=len(tickers),
            duration_ms=duration_ms,
        )
        return full_message
        
    except Exception as e:
        logger.error(f"dipfinder_daily failed: {e}", exc_info=True)
        raise


# =============================================================================
# REGIME DAILY - Market regime detection
# =============================================================================


@register_job("regime_daily")
async def regime_daily_job() -> str:
    """
    Daily market regime detection and caching.

    Detects current market regime (bull/bear, high/low vol) and caches
    for Dashboard display.

    Schedule: Daily at 10:30 PM UTC (after signal scanner)
    """
    from datetime import date, timedelta

    import pandas as pd

    from app.cache.cache import Cache
    from app.quant_engine.analytics import detect_regime, compute_correlation_analysis
    from app.repositories import price_history_orm as price_history_repo
    from app.repositories import symbols_orm as symbols_repo

    logger.info("Starting regime_daily job")
    job_start = time.monotonic()

    cache = Cache(prefix="market", default_ttl=86400)

    try:
        symbols_list = await symbols_repo.list_symbols()

        if not symbols_list:
            return "No symbols tracked"

        # Get all symbols except market indices for asset returns
        asset_symbols = [
            s.symbol for s in symbols_list 
            if s.symbol not in ("SPY", "^GSPC", "URTH")
        ]

        end_date = date.today()
        start_date = end_date - timedelta(days=400)

        # Get SPY as market returns
        spy_df = await price_history_repo.get_prices_as_dataframe(
            "SPY", start_date, end_date
        )
        if spy_df is None or len(spy_df) < 60:
            return "Insufficient SPY data for regime detection"
        
        spy_close = get_close_column(spy_df)
        market_returns = spy_df[spy_close].pct_change().dropna()

        # Get asset price data
        price_dfs: dict[str, pd.Series] = {}
        for symbol in asset_symbols:
            df = await price_history_repo.get_prices_as_dataframe(
                symbol, start_date, end_date
            )
            if df is not None:
                close_col = get_close_column(df)
                price_dfs[symbol] = df[close_col]

        if not price_dfs:
            return "No asset price data"

        prices = pd.DataFrame(price_dfs).dropna(how="all").ffill()
        
        if len(prices) < 60:
            return "Insufficient asset data"

        asset_returns = prices.pct_change().dropna()

        # Detect regime - requires market_returns (Series) and asset_returns (DataFrame)
        regime = detect_regime(market_returns, asset_returns)

        # Correlation analysis
        corr = compute_correlation_analysis(asset_returns)

        cache_data = {
            "as_of": str(date.today()),
            "regime": regime.regime,
            "trend": "bull" if regime.is_bull else "bear",
            "volatility": "high" if regime.is_high_vol else "low",
            "description": regime.regime_description,
            "recommendation": regime.risk_budget_recommendation,
            "avg_correlation": corr.avg_correlation,
            "n_clusters": corr.n_clusters,
            "stress_correlation": corr.stress_avg_correlation,
        }

        await cache.set("regime", cache_data)

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Regime: {regime.regime}, Avg Corr: {corr.avg_correlation:.1%}"
        
        # Structured success log
        log_job_success(
            "regime_daily",
            message,
            regime=regime.regime,
            trend="bull" if regime.is_bull else "bear",
            volatility="high" if regime.is_high_vol else "low",
            avg_correlation=round(corr.avg_correlation, 4),
            n_clusters=corr.n_clusters,
            symbols_analyzed=len(price_dfs),
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"regime_daily failed: {e}")
        raise
