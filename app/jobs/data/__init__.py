"""Data ingestion and sync job definitions.

This module contains jobs for:
- Universe sync from FinanceDatabase (universe_sync)
- Symbol ingestion queue processing (symbol_ingest)
- Weekly data backfill for all tracked symbols (data_backfill)
- Daily price updates and dip state refresh (prices_daily)

These jobs form the data ingestion backbone of the system.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.repositories import jobs_orm as jobs_repo
from app.repositories import price_history_orm as price_history_repo

from ..registry import register_job
from ..utils import get_close_column, log_job_success


if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("jobs.data")


# =============================================================================
# UNIVERSE SYNC - Sync FinanceDatabase to local (weekly, before data_backfill)
# =============================================================================


@register_job("universe_sync")
async def universe_sync_job() -> str:
    """
    Sync FinanceDatabase to local financial_universe table.
    
    This provides a comprehensive local database of ~130K financial instruments:
    - Equities (~24K) - with sector, industry, country metadata
    - ETFs (~3K) - with category and family info
    - Funds (~31K) - mutual funds with category info
    - Indices (~62K) - market indices
    - Cryptos (~3K) - cryptocurrencies
    - Currencies (~3K) - forex pairs
    - Moneymarkets (~1K) - money market instruments
    
    Benefits:
    - Fast local autocomplete/search (no API calls)
    - Symbol validation without external API
    - Sector/industry/country faceting
    - ISIN/CUSIP/FIGI identifier resolution
    
    Schedule: Sunday 1:30 AM UTC (before weekly_ai_pipeline)
    Runtime: ~2-5 minutes (depends on network)
    """
    job_start = time.monotonic()
    
    from app.services.financedatabase_service import ingest_universe
    
    logger.info("Starting universe sync from FinanceDatabase...")
    
    try:
        # Ingest all asset classes
        stats = await ingest_universe()
        
        total = sum(stats.values())
        duration_ms = int((time.monotonic() - job_start) * 1000)
        
        message = f"Synced {total} symbols from FinanceDatabase"
        
        # Structured success log
        log_job_success(
            "universe_sync",
            message,
            total_symbols=total,
            equities=stats.get("equity", 0),
            etfs=stats.get("etf", 0),
            funds=stats.get("fund", 0),
            indices=stats.get("index", 0),
            cryptos=stats.get("crypto", 0),
            currencies=stats.get("currency", 0),
            moneymarkets=stats.get("moneymarket", 0),
            duration_ms=duration_ms,
        )
        
        return message
        
    except Exception as e:
        duration_ms = int((time.monotonic() - job_start) * 1000)
        logger.exception(f"universe_sync failed after {duration_ms}ms: {e}")
        raise


# =============================================================================
# SYMBOL INGEST - Process new symbols (every 15 min)
# =============================================================================


@register_job("symbol_ingest")
async def symbol_ingest_job() -> str:
    """
    Process new symbols - unified, idempotent symbol data ingestion.
    
    This is an IDEMPOTENT job that:
    1. Fetches symbols from the ingest queue
    2. Checks what data is MISSING for each symbol
    3. Only fetches/generates what's needed
    4. Submits AI batch for symbols missing AI content
    
    Data checked (skipped if already exists):
    - Price history (5 years)
    - Fundamentals
    - Dip state
    - AI bio and rating (submitted as batch)

    Schedule: Every 15 minutes (processes queue as symbols are added)
    """
    from datetime import date, timedelta

    from app.cache.cache import Cache
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import dip_state_orm as dip_state_repo
    from app.repositories import dip_votes_orm as dip_votes_repo
    from app.services.prices import get_price_service
    from app.services.openai import TaskType, submit_batch
    from app.services.fundamentals import get_fundamentals_for_analysis, refresh_fundamentals
    from app.services.stock_info import get_stock_info_batch_async

    BATCH_SIZE = 20
    logger.info("Starting symbol_ingest job")
    job_start = time.monotonic()

    try:
        # Get pending symbols from the queue
        rows = await jobs_repo.get_pending_ingest_symbols(BATCH_SIZE)

        if not rows:
            log_job_success(
                "symbol_ingest",
                "No symbols in queue",
                symbols_processed=0,
                symbols_failed=0,
                queue_remaining=0,
                duration_ms=int((time.monotonic() - job_start) * 1000),
            )
            return "No symbols in queue"

        symbols = [row.symbol for row in rows]
        queue_map = {row.symbol: row for row in rows}  # For updating status
        
        logger.info(f"[INGEST] Processing {len(symbols)} queued symbols: {symbols}")

        price_service = get_price_service()
        processed = 0
        failed = 0
        ai_items = []  # Collect items needing AI generation
        
        # Mark all as processing
        for row in rows:
            await jobs_repo.mark_ingest_processing(row.id, (row.attempts or 0) + 1)

        # Step 1: Check what each symbol already has
        symbols_needing_data = []
        for symbol in symbols:
            has_prices = await price_history_repo.has_price_history(symbol)
            if not has_prices:
                symbols_needing_data.append(symbol)
            else:
                logger.debug(f"[INGEST] {symbol} already has price data, checking other fields")

        # Step 2: Batch fetch stock info (cheap, always get fresh)
        logger.info(f"[INGEST] Fetching Yahoo Finance info for {len(symbols)} symbols")
        stock_infos = await get_stock_info_batch_async(symbols)

        # Step 3: Batch fetch price history for symbols missing it
        # PriceService handles validation and saving automatically
        price_results = {}
        if symbols_needing_data:
            logger.info(f"[INGEST] Fetching 5y price history for {len(symbols_needing_data)} symbols")
            today = date.today()
            start_date = today - timedelta(days=1825)  # 5 years
            price_results = await price_service.get_prices_batch(
                symbols=symbols_needing_data,
                start_date=start_date,
                end_date=today,
            )

        # Step 4: Process each symbol
        for symbol in symbols:
            queue_row = queue_map[symbol]
            
            try:
                info = stock_infos.get(symbol, {})
                if not info:
                    logger.warning(f"[INGEST] No Yahoo data for {symbol}")
                    await jobs_repo.mark_ingest_failed(queue_row.id, "No Yahoo Finance data")
                    failed += 1
                    continue

                name = info.get("name") or info.get("short_name")
                sector = info.get("sector")
                full_summary = info.get("summary")
                current_price = info.get("current_price", 0)
                ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
                dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0

                # Update symbol info (idempotent - just updates)
                if name or sector:
                    await symbols_repo.update_symbol_info(symbol, name=name, sector=sector)

                # Upsert dip state (idempotent)
                await dip_state_repo.upsert_dip_state(
                    symbol=symbol,
                    current_price=current_price,
                    ath_price=ath_price,
                    dip_percentage=dip_pct,
                )

                # Check if we got price data (already saved by PriceService)
                if symbol in price_results:
                    prices = price_results[symbol]
                    if prices is not None and not prices.empty:
                        logger.info(f"[INGEST] Got {len(prices)} days of price history for {symbol}")

                # Refresh fundamentals (idempotent - checks freshness internally)
                await refresh_fundamentals(symbol)

                # Check if symbol needs AI content
                existing_ai = await dip_votes_repo.get_ai_analysis(symbol)
                needs_ai = not existing_ai or not existing_ai.swipe_bio
                
                if needs_ai and full_summary and len(full_summary) > 100:
                    fundamentals = await get_fundamentals_for_analysis(symbol)
                    ai_items.append({
                        "symbol": symbol,
                        "name": name,
                        "sector": sector,
                        "summary": full_summary,
                        "current_price": current_price,
                        "ref_high": ath_price,
                        "dip_pct": dip_pct,
                        "days_below": 0,
                        **fundamentals,
                    })

                # Mark queue item as completed
                await jobs_repo.mark_ingest_completed(queue_row.id)
                
                # Update symbol fetch_status to 'fetched' now that data is complete
                await symbols_repo.update_fetch_status(symbol, fetch_status="fetched")
                
                processed += 1
                logger.info(f"[INGEST] Completed {symbol}")

            except Exception as e:
                logger.error(f"[INGEST] Failed to process {symbol}: {e}")
                max_attempts = 3
                if (queue_row.attempts or 0) + 1 >= max_attempts:
                    await jobs_repo.mark_ingest_failed(queue_row.id, str(e)[:500])
                else:
                    await jobs_repo.mark_ingest_pending_retry(queue_row.id, str(e)[:500])
                failed += 1

        # Step 5: Submit AI batch for symbols needing AI content
        batch_id = None
        if ai_items:
            logger.info(f"[INGEST] Submitting {len(ai_items)} items for AI batch processing")
            try:
                batch_id = await submit_batch(task=TaskType.RATING, items=ai_items)
                if batch_id:
                    logger.info(f"[INGEST] Submitted AI rating batch: {batch_id}")
                
                bio_batch_id = await submit_batch(task=TaskType.BIO, items=ai_items)
                if bio_batch_id:
                    logger.info(f"[INGEST] Submitted AI bio batch: {bio_batch_id}")
            except Exception as e:
                logger.warning(f"[INGEST] Failed to submit AI batch: {e}")

        # Step 6: Invalidate caches if we processed anything
        if processed > 0:
            ranking_cache = Cache(prefix="ranking", default_ttl=3600)
            await ranking_cache.invalidate_pattern("*")
            symbols_cache = Cache(prefix="symbols", default_ttl=3600)
            await symbols_cache.invalidate_pattern("*")

        remaining = await jobs_repo.get_ingest_queue_count()
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Processed {processed}/{len(symbols)} ({failed} failed), {remaining} remaining"
        if batch_id:
            message += f", AI batch: {batch_id}"
        
        # Structured success log
        log_job_success(
            "symbol_ingest",
            message,
            symbols_processed=processed,
            symbols_failed=failed,
            symbols_total=len(symbols),
            ai_items_submitted=len(ai_items),
            ai_batch_id=batch_id,
            queue_remaining=remaining,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"symbol_ingest failed: {e}", exc_info=True)
        raise


async def _store_price_history(symbol: str, prices: "pd.DataFrame") -> None:
    """Store price history in database."""
    if prices.empty:
        return

    await price_history_repo.save_prices(symbol, prices)


async def add_to_ingest_queue(symbol: str, priority: int = 0) -> bool:
    """
    Add a symbol to the ingest queue.

    Called when:
    - New symbol is added to watchlist
    - Symbol is suggested and approved
    - Manual trigger via admin

    Returns True if added, False if already in queue.
    """
    return await jobs_repo.add_to_ingest_queue(symbol, priority)


@register_job("data_backfill")
async def data_backfill_job() -> str:
    """
    Comprehensive data backfill for all tracked symbols.
    
    This job runs weekly to ensure data integrity by filling gaps caused by:
    - yfinance returning incomplete data at import time
    - Failed API calls during symbol_ingest
    - New data fields added after symbols were imported
    - Expired or missing computed data
    
    Checks and fills (in order):
    1. Missing sector/summary_ai (from yfinance cache)
    2. Missing price history (fetch from yfinance)
    3. Missing fundamentals (refresh from yfinance)
    4. Missing quant scores (compute fresh)
    5. Missing dipfinder signals (compute fresh)
    
    Schedule: Weekly (Sunday 2 AM UTC, before AI jobs)
    """
    from datetime import date, timedelta
    
    from app.dipfinder.config import get_dipfinder_config
    from app.dipfinder.service import DipFinderService
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import quant_scores_orm as quant_repo
    from app.services.prices import get_price_service
    from app.services.fundamentals import refresh_fundamentals
    from app.services.openai import summarize_company
    from app.services.stock_info import get_stock_info_async
    
    logger.info("Starting data_backfill job")
    job_start = time.monotonic()
    
    stats = {
        "sectors": 0,
        "summaries": 0,
        "prices": 0,
        "fundamentals": 0,
        "quant_scores": 0,
        "dipfinder": 0,
    }
    
    try:
        # Get all symbols
        all_symbols = await symbols_repo.list_symbols()
        tickers = [s.symbol for s in all_symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]
        
        if not tickers:
            log_job_success("data_backfill", "No symbols to backfill",
                            total_filled=0, symbols_checked=0, duration_ms=int((time.monotonic() - job_start) * 1000))
            return "No symbols to backfill"
        
        logger.info(f"[BACKFILL] Checking {len(tickers)} symbols for data gaps")
        
        # =====================================================================
        # PHASE 1: Symbol metadata (sector, summary_ai)
        # =====================================================================
        symbols_needing_sector = [s.symbol for s in all_symbols if not s.sector and s.symbol in tickers]
        symbols_needing_summary = [s.symbol for s in all_symbols if not s.summary_ai and s.symbol in tickers]
        metadata_symbols = list(set(symbols_needing_sector + symbols_needing_summary))
        
        if metadata_symbols:
            logger.info(f"[BACKFILL] Phase 1: Filling metadata for {len(metadata_symbols)} symbols")
            
            for symbol in metadata_symbols:
                try:
                    info = await get_stock_info_async(symbol)
                    if not info:
                        continue
                    
                    name = info.get("name")
                    sector = info.get("sector")
                    summary = info.get("summary")
                    
                    if symbol in symbols_needing_sector and sector:
                        await symbols_repo.update_symbol_info(symbol, name=name, sector=sector)
                        stats["sectors"] += 1
                    
                    if symbol in symbols_needing_summary and summary and len(summary) > 100:
                        ai_summary = await summarize_company(symbol=symbol, name=name, description=summary)
                        if ai_summary:
                            await symbols_repo.update_symbol_info(symbol, summary_ai=ai_summary)
                            stats["summaries"] += 1
                            
                except Exception as e:
                    logger.debug(f"[BACKFILL] Metadata failed for {symbol}: {e}")
        
        # =====================================================================
        # PHASE 2: Price history - Use unified PriceService
        # =====================================================================
        price_service = get_price_service()
        
        logger.info(f"[BACKFILL] Phase 2: Refreshing prices for {len(tickers)} symbols")
        
        # Refresh recent prices (last 30 days covers most gaps)
        price_results = await price_service.refresh_prices(tickers, days=30)
        
        stats["prices"] = sum(1 for count in price_results.values() if count > 0)
        
        if stats["prices"] > 0:
            total_records = sum(price_results.values())
            logger.info(
                f"[BACKFILL] Phase 2: Updated prices for {stats['prices']} symbols "
                f"({total_records} records)"
            )
        
        # =====================================================================
        # PHASE 3: Fundamentals (symbols without recent fundamentals)
        # =====================================================================
        symbols_needing_fundamentals = []
        for symbol in tickers[:50]:  # Limit to avoid rate limiting
            from app.services.fundamentals import get_fundamentals_from_db
            try:
                fundamentals = await get_fundamentals_from_db(symbol)
                if not fundamentals:
                    symbols_needing_fundamentals.append(symbol)
            except Exception:
                symbols_needing_fundamentals.append(symbol)
        
        if symbols_needing_fundamentals:
            logger.info(f"[BACKFILL] Phase 3: Refreshing fundamentals for {len(symbols_needing_fundamentals)} symbols")
            
            for symbol in symbols_needing_fundamentals[:20]:  # Limit batch
                try:
                    await refresh_fundamentals(symbol)
                    stats["fundamentals"] += 1
                except Exception as e:
                    logger.debug(f"[BACKFILL] Fundamentals failed for {symbol}: {e}")
        
        # =====================================================================
        # PHASE 4: Quant scores (symbols without scores)
        # =====================================================================
        symbols_needing_quant = []
        for symbol in tickers:
            score = await quant_repo.get_quant_score(symbol)
            if not score:
                symbols_needing_quant.append(symbol)
        
        if symbols_needing_quant:
            logger.info(f"[BACKFILL] Phase 4: Computing quant scores for {len(symbols_needing_quant)} symbols")
            
            from app.quant_engine.dual_scoring import compute_and_store_scores
            
            for symbol in symbols_needing_quant[:30]:  # Limit batch
                try:
                    await compute_and_store_scores(symbol)
                    stats["quant_scores"] += 1
                except Exception as e:
                    logger.debug(f"[BACKFILL] Quant score failed for {symbol}: {e}")
        
        # =====================================================================
        # PHASE 5: Dipfinder signals (symbols without recent signals)
        # =====================================================================
        from app.repositories import dipfinder_orm
        
        # Get all existing signals at once
        existing_signals = await dipfinder_orm.get_latest_signals_for_tickers(tickers)
        signals_by_symbol = {s.ticker: s for s in existing_signals}
        
        symbols_needing_dipfinder = []
        today = date.today()
        for symbol in tickers:
            signal = signals_by_symbol.get(symbol)
            if not signal or (today - signal.as_of_date).days > 7:
                symbols_needing_dipfinder.append(symbol)
        
        if symbols_needing_dipfinder:
            logger.info(f"[BACKFILL] Phase 5: Computing dipfinder signals for {len(symbols_needing_dipfinder)} symbols")
            
            config = get_dipfinder_config()
            service = DipFinderService(config)
            
            for symbol in symbols_needing_dipfinder[:30]:  # Limit batch
                try:
                    signal = await service.get_signal(
                        ticker=symbol,
                        window=config.windows[0] if config.windows else 60,
                        benchmark=config.default_benchmark,
                        force_refresh=True,
                    )
                    if signal:
                        stats["dipfinder"] += 1
                except Exception as e:
                    logger.debug(f"[BACKFILL] Dipfinder failed for {symbol}: {e}")
        
        # Build summary message
        filled = sum(stats.values())
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = (
            f"Backfilled {filled} data gaps: "
            f"{stats['sectors']} sectors, {stats['summaries']} summaries, "
            f"{stats['prices']} prices, {stats['fundamentals']} fundamentals, "
            f"{stats['quant_scores']} quant scores, {stats['dipfinder']} dipfinder"
        )
        
        # Structured success log
        log_job_success(
            "data_backfill",
            message,
            total_filled=filled,
            symbols_checked=len(tickers),
            sectors_filled=stats["sectors"],
            summaries_filled=stats["summaries"],
            prices_filled=stats["prices"],
            fundamentals_filled=stats["fundamentals"],
            quant_scores_filled=stats["quant_scores"],
            dipfinder_filled=stats["dipfinder"],
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"data_backfill failed: {e}", exc_info=True)
        raise


@register_job("prices_daily")
async def prices_daily_job() -> str:
    """
    Fetch stock data and update dip_state with latest prices.
    Uses unified PriceService with automatic validation.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from datetime import date

    from app.cache.cache import Cache
    from app.repositories import symbols_orm as symbols_repo
    from app.services.prices import get_price_service
    from app.services.runtime_settings import get_runtime_setting

    logger.info("Starting prices_daily job")
    job_start = time.monotonic()

    try:
        # Ensure sector ETFs exist in symbols table before processing
        # This prevents FK violations when updating dip_state for sector ETFs
        created_etfs = await jobs_repo.ensure_sector_etfs_exist()
        if created_etfs > 0:
            logger.info(f"Created {created_etfs} missing sector ETF symbols")

        tickers = await jobs_repo.get_active_symbol_tickers()

        if not tickers:
            log_job_success("prices_daily", "No active symbols",
                            symbols_updated=0, symbols_changed=0, benchmarks_updated=0,
                            duration_ms=int((time.monotonic() - job_start) * 1000))
            return "No active symbols"

        price_service = get_price_service()

        # Fetch latest prices and update dip_state
        updated_count = 0
        changed_symbols = []  # Track which symbols had data changes

        # Get min_dip_pct for each symbol to calculate dip start date
        dip_thresholds = await jobs_repo.get_symbol_dip_thresholds()
        latest_dates = await price_history_repo.get_latest_price_dates(tickers)

        from datetime import timedelta

        from app.services.fundamentals import update_price_based_metrics

        today = date.today()
        stale_cutoff_days = 7
        batch_size = 10

        full_refresh = []
        incremental = []
        for ticker in tickers:
            last_date = latest_dates.get(ticker)
            if not last_date or (today - last_date).days > stale_cutoff_days:
                full_refresh.append(ticker)
            else:
                incremental.append(ticker)

        def _chunked(items: list[str], size: int) -> list[list[str]]:
            return [items[i:i + size] for i in range(0, len(items), size)]

        async def _process_batch(symbols: list[str], start_date: date) -> None:
            nonlocal updated_count
            if not symbols:
                return
            if start_date >= today:
                return

            # Import dipfinder signal computation for opportunity_type
            from app.dipfinder.signal import compute_signal, OpportunityType
            from app.dipfinder.config import DipFinderConfig
            dipfinder_config = DipFinderConfig()

            # Use PriceService - it handles fetching, validation, and saving
            results = await price_service.get_prices_batch(symbols, start_date, today)

            for symbol in symbols:
                try:
                    prices = results.get(symbol)
                    if prices is None or prices.empty:
                        continue

                    # Get last valid (non-NaN) close price (prefer adjusted close)
                    close_col = get_close_column(prices)
                    valid_closes = prices[close_col].dropna()
                    if valid_closes.empty:
                        continue
                    current_price = float(valid_closes.iloc[-1])

                    # Mark as changed if we got new data
                    changed_symbols.append(symbol)

                    # ATH from our price_history table (single source of truth)
                    ath_price = await jobs_repo.get_ath_price(
                        symbol, fallback=float(prices[close_col].max())
                    )

                    # Calculate dip percentage
                    dip_percentage = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0

                    # Calculate dip start date
                    dip_threshold = dip_thresholds.get(symbol, 0.15)
                    dip_start_date = await jobs_repo.calculate_dip_start_date(
                        symbol, ath_price, dip_threshold
                    )

                    # Compute opportunity_type from dipfinder signal
                    opportunity_type = "NONE"
                    is_tail_event = False
                    return_period_years = None
                    regime_dip_percentile = None
                    try:
                        from app.dipfinder.fundamentals import QualityMetrics, compute_quality_score
                        from app.dipfinder.stability import compute_stability_score
                        from app.dipfinder.dip import compute_dip_series_windowed
                        from app.dipfinder.extreme_value import analyze_tail_events
                        from app.services.fundamentals import get_fundamentals_from_db
                        from datetime import datetime
                        
                        stock_prices = prices[close_col].dropna().values
                        volumes = prices.get("Volume", prices.get("volume"))
                        volumes = volumes.fillna(0).values if volumes is not None else None
                        
                        if len(stock_prices) >= 50:
                            # Try to get stored fundamentals (no yfinance call)
                            stored_fundamentals = await get_fundamentals_from_db(symbol)
                            info = stored_fundamentals if stored_fundamentals else None
                            
                            # Compute stability metrics with stored fundamentals
                            stability_metrics = compute_stability_score(
                                ticker=symbol,
                                close_prices=stock_prices,
                                info=info,
                                config=dipfinder_config,
                            )
                            
                            # Compute quality metrics with stored fundamentals
                            quality_metrics = await compute_quality_score(
                                ticker=symbol,
                                info=info,
                                config=dipfinder_config,
                            )
                            
                            signal = compute_signal(
                                ticker=symbol,
                                stock_prices=stock_prices,
                                benchmark_prices=stock_prices,  # Self as benchmark
                                benchmark_ticker=symbol,
                                window=dipfinder_config.window,
                                quality_metrics=quality_metrics,
                                stability_metrics=stability_metrics,
                                as_of_date=datetime.now().strftime("%Y-%m-%d"),
                                config=dipfinder_config,
                                volumes=volumes,
                            )
                            opportunity_type = signal.opportunity_type.value
                            
                            # Compute Extreme Value Analysis (EVA) metrics
                            try:
                                dip_series = compute_dip_series_windowed(stock_prices, dipfinder_config.window)
                                current_dip = float(dip_series[-1]) if len(dip_series) > 0 else 0.0
                                
                                tail_analysis = analyze_tail_events(
                                    dip_series=dip_series,
                                    current_dip=current_dip,
                                    threshold_percentile=95.0,
                                    min_threshold=0.40,
                                    max_threshold=0.70,
                                )
                                is_tail_event = tail_analysis.is_tail_event
                                return_period_years = tail_analysis.return_period_years
                                regime_dip_percentile = tail_analysis.regime_dip_percentile
                            except Exception as eva_err:
                                logger.debug(f"EVA computation failed for {symbol}: {eva_err}")
                    except Exception as e:
                        logger.debug(f"Could not compute opportunity_type for {symbol}: {e}")

                    # Check for active strategy signal (buy signal that beats buy & hold)
                    try:
                        from app.database.orm import StrategySignal
                        from app.database.connection import get_session
                        async with get_session() as ss:
                            strategy = await ss.get(StrategySignal, symbol)
                            # Removed extra get_session call
                            from sqlalchemy import select
                            stmt = select(StrategySignal).where(StrategySignal.symbol == symbol)
                            result = await ss.execute(stmt)
                            strategy = result.scalar_one_or_none()
                            if (
                                strategy is not None
                                and strategy.has_active_signal
                                and strategy.signal_type == "BUY"
                                and strategy.beats_buy_hold
                            ):
                                # STRATEGY takes priority if dipfinder says NONE,
                                # otherwise keep the dipfinder signal (they're complementary)
                                if opportunity_type == "NONE":
                                    opportunity_type = "STRATEGY"
                    except Exception as strat_err:
                        logger.debug(f"Could not check strategy signal for {symbol}: {strat_err}")

                    # Update dip_state with opportunity_type and EVA fields
                    await jobs_repo.upsert_dip_state_with_dates(
                        symbol, current_price, ath_price, dip_percentage, dip_start_date,
                        opportunity_type=opportunity_type,
                        is_tail_event=is_tail_event,
                        return_period_years=return_period_years,
                        regime_dip_percentile=regime_dip_percentile,
                    )

                    # PriceService already saved prices to DB

                    # Update price-based ratios daily
                    await update_price_based_metrics(symbol, current_price)
                    
                    # Ensure fetch_status is 'fetched' now that we have price data
                    await symbols_repo.update_fetch_status(symbol, fetch_status="fetched")

                    updated_count += 1
                    days_in_dip = (date.today() - dip_start_date).days if dip_start_date else 0
                    logger.debug(
                        f"Updated dip_state for {symbol}: ${current_price:.2f}, "
                        f"ATH: ${ath_price:.2f}, dip: {dip_percentage:.1f}%, days: {days_in_dip}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update {symbol}: {e}")

        # Full refresh batches (5y window for quant analysis)
        if full_refresh:
            full_start = today - timedelta(days=1825)
            for batch in _chunked(full_refresh, batch_size):
                await _process_batch(batch, full_start)

        # Incremental batches (only new data since last date)
        # Use minimum 5-day window to avoid holiday no-data errors
        if incremental:
            min_fetch_days = 5
            for batch in _chunked(incremental, batch_size):
                batch_start = min(
                    latest_dates[symbol] for symbol in batch if latest_dates.get(symbol)
                ) + timedelta(days=1)
                # Ensure minimum fetch window to handle holidays
                if (today - batch_start).days < min_fetch_days:
                    batch_start = today - timedelta(days=min_fetch_days)
                await _process_batch(batch, batch_start)

        # Update stock info cache for all symbols
        # This caches 52w high/low, previous_close, avg_volume, pe_ratio, market_cap
        # so the ranking endpoint doesn't need live yfinance calls
        logger.info(f"Updating stock info cache for {len(tickers)} symbols")
        
        from app.services.stock_info import get_stock_info_batch_async
        from app.repositories import symbols_orm as symbols_repo
        
        stock_info_map = await get_stock_info_batch_async(tickers)
        
        stock_info_updates = []
        for symbol, info in stock_info_map.items():
            if info:
                stock_info_updates.append({
                    "symbol": symbol,
                    "fifty_two_week_low": info.get("fifty_two_week_low"),
                    "fifty_two_week_high": info.get("fifty_two_week_high"),
                    "previous_close": info.get("previous_close"),
                    "avg_volume": info.get("avg_volume"),
                    "pe_ratio": info.get("pe_ratio"),
                    "market_cap": info.get("market_cap"),
                })
        
        if stock_info_updates:
            info_updated = await symbols_repo.batch_update_stock_info_cache(stock_info_updates)
            logger.info(f"Updated stock info cache for {info_updated} symbols")

        # Also fetch latest benchmark data
        benchmarks = get_runtime_setting("benchmarks", [])
        benchmark_symbols = [b.get("symbol") for b in benchmarks if b.get("symbol")]
        benchmark_count = 0

        if benchmark_symbols:
            logger.info(f"Fetching data for {len(benchmark_symbols)} benchmarks")
            for symbol in benchmark_symbols:
                try:
                    await price_service.refresh_prices([symbol], days=5)
                    benchmark_count += 1
                except Exception as e:
                    logger.warning(f"Failed to fetch benchmark {symbol}: {e}")

        # Targeted cache invalidation - only invalidate caches for symbols that actually changed
        ranking_cache = Cache(prefix="ranking", default_ttl=1800)
        chart_cache = Cache(prefix="chart", default_ttl=3600)

        if changed_symbols:
            logger.info(f"Data changed for {len(changed_symbols)} symbols, invalidating their caches")
            for symbol in changed_symbols:
                await chart_cache.invalidate_pattern(f"*{symbol}*")
            # Ranking cache needs full invalidation since order may have changed
            await ranking_cache.invalidate_pattern("*")
        else:
            logger.info("No data changes detected, skipping cache invalidation")

        # Bump data version so frontend caches invalidate
        from app.cache.data_version import bump_data_version
        await bump_data_version()

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Updated {updated_count}/{len(tickers)} symbols ({len(changed_symbols)} changed), {benchmark_count} benchmarks"
        
        # Structured success log
        log_job_success(
            "prices_daily",
            message,
            symbols_total=len(tickers),
            symbols_updated=updated_count,
            symbols_changed=len(changed_symbols),
            benchmarks_updated=benchmark_count,
            sector_etfs_created=created_etfs,
            full_refresh_count=len(full_refresh),
            incremental_count=len(incremental),
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"prices_daily failed: {e}")
        raise
