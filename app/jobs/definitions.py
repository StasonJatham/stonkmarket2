"""Built-in job definitions for scheduled tasks.

Job Categories:

1. REAL-TIME PROCESSING (every 5-15 min)
   - symbol_ingest: Process NEW symbols from queue
   - ai_batch_poll: Check OpenAI batch results
   - portfolio_worker: Process analytics queue
   - cache_warmup: Pre-cache chart data

2. DAILY MARKET CLOSE PIPELINE (single orchestrator, Mon-Fri 10 PM UTC)
   - market_close_pipeline: Runs all steps sequentially, each waits for previous
     Steps: prices → signals → regime → strategy → quant_scoring → dipfinder → quant_analysis
   - Individual jobs exist for manual retries but are NOT scheduled

3. WEEKLY AI ANALYSIS (Sunday morning)
   - ai_personas_weekly: Warren Buffett, Peter Lynch etc.
   - ai_bios_weekly: Swipe-style stock bios

4. WEEKLY MAINTENANCE (Sunday)
   - data_backfill: Fill ALL data gaps (comprehensive)

5. MONTHLY MAINTENANCE
   - fundamentals_monthly: Refresh company fundamentals
   - quant_monthly: Portfolio optimization

6. DAILY CLEANUP
   - cleanup_daily: Remove expired data
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.repositories import jobs_orm as jobs_repo
from app.repositories import price_history_orm as price_history_repo

from .registry import register_job


if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("jobs.definitions")


def log_job_success(job_name: str, message: str, **metrics: Any) -> None:
    """Log a structured job success message with metrics.
    
    Args:
        job_name: Name of the job (e.g., "cache_warmup")
        message: Human-readable summary message
        **metrics: Key-value pairs of metrics to include in structured log
    
    Example:
        log_job_success("cache_warmup", "Warmed 60 chart caches",
            items_warmed=60, symbols_cached=20, duration_ms=1234)
    """
    # Build structured log data
    log_data = {
        "job": job_name,
        "status": "success",
        **metrics,
    }
    
    # Log with structured data for JSON parsing, and human message for text logs
    # The message format is: "job_name completed: message | metrics_json"
    metrics_str = " ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"{job_name} completed: {message} | {metrics_str}", extra={"extra_fields": log_data})


def get_close_column(df: "pd.DataFrame") -> str:
    """Get the best close column name, preferring adjusted close.
    
    Adjusted close accounts for stock splits and dividends, making
    historical price comparisons accurate.
    
    Args:
        df: Price DataFrame with Close and/or Adj Close columns
        
    Returns:
        Column name to use ('Adj Close' if available, else 'Close')
    """
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        return "Adj Close"
    return "Close"


# =============================================================================
# MARKET CLOSE PIPELINE - Orchestrates all daily analysis jobs
# =============================================================================
# This is the ONLY scheduled daily job. It runs each step sequentially,
# waiting for completion before starting the next. No timing issues!
# =============================================================================

# Pipeline steps in execution order
MARKET_CLOSE_PIPELINE_STEPS = [
    "prices_daily",        # 1. Fetch closing prices - MUST be first
    "fundamentals_daily",  # 2. Refresh fundamentals (earnings-driven, smart)
    "signals_daily",       # 3. Technical signals (needs prices)
    "regime_daily",        # 4. Market regime detection (needs prices)
    "strategy_nightly",    # 5. Strategy optimization (needs prices + signals)
    "quant_scoring_daily", # 6. Quant scoring (needs prices + signals)
    "dipfinder_daily",     # 7. DipFinder + entry optimizer (needs quant scores)
    "quant_analysis_nightly",  # 8. Pre-compute quant results - MUST be last
]


@register_job("market_close_pipeline")
async def market_close_pipeline_job() -> str:
    """
    Daily market close pipeline - orchestrates all analysis jobs sequentially.
    
    This is the SINGLE SCHEDULED JOB for daily market analysis.
    Each step waits for the previous to complete - no timing issues!
    
    Pipeline:
    1. prices_daily - Fetch closing prices (required for all others)
    2. fundamentals_daily - Refresh fundamentals (earnings-driven)
    3. signals_daily - Technical signal scanner
    4. regime_daily - Market regime detection
    5. strategy_nightly - Strategy optimization & backtesting
    6. quant_scoring_daily - Quant metrics computation
    7. dipfinder_daily - DipFinder signals + entry optimizer
    8. quant_analysis_nightly - Pre-compute all quant results for API
    
    Each step is tracked with timing. If a step fails, the pipeline
    continues with remaining steps but reports the failure.
    
    Schedule: Mon-Fri at 10 PM UTC (after US market close)
    """
    import time
    from app.jobs.executor import execute_job
    from app.repositories import cronjobs_orm as cron_repo
    
    logger.info("=" * 60)
    logger.info("MARKET CLOSE PIPELINE - Starting")
    logger.info("=" * 60)
    
    results = []
    total_start = time.monotonic()
    
    for step_num, job_name in enumerate(MARKET_CLOSE_PIPELINE_STEPS, 1):
        step_start = time.monotonic()
        logger.info(f"[PIPELINE] Step {step_num}/{len(MARKET_CLOSE_PIPELINE_STEPS)}: {job_name}")
        
        try:
            # Execute the job directly (not via Celery task to ensure sequential)
            message = await execute_job(job_name)
            duration_s = time.monotonic() - step_start
            
            # Update job stats
            try:
                await cron_repo.update_job_stats(job_name, "ok", int(duration_s * 1000))
            except Exception:
                pass
            
            results.append({
                "step": step_num,
                "job": job_name,
                "status": "ok",
                "duration_s": round(duration_s, 1),
                "message": message[:100] if message else "Done",
            })
            logger.info(f"[PIPELINE] ✓ {job_name} completed in {duration_s:.1f}s")
            
        except Exception as e:
            duration_s = time.monotonic() - step_start
            error_msg = str(e)[:200]
            
            # Update job stats with error
            try:
                await cron_repo.update_job_stats(job_name, "error", int(duration_s * 1000), error_msg)
            except Exception:
                pass
            
            results.append({
                "step": step_num,
                "job": job_name,
                "status": "error",
                "duration_s": round(duration_s, 1),
                "message": error_msg,
            })
            logger.error(f"[PIPELINE] ✗ {job_name} FAILED after {duration_s:.1f}s: {error_msg}")
            # Continue with next step - don't abort entire pipeline
    
    total_duration = time.monotonic() - total_start
    total_duration_ms = int(total_duration * 1000)
    
    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info("=" * 60)
    logger.info(f"MARKET CLOSE PIPELINE - Completed in {total_duration:.1f}s")
    logger.info(f"Results: {ok_count} succeeded, {error_count} failed")
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else "✗"
        logger.info(f"  {status_icon} {r['job']}: {r['duration_s']}s - {r['message'][:50]}")
    logger.info("=" * 60)
    
    summary = f"Pipeline: {ok_count}/{len(MARKET_CLOSE_PIPELINE_STEPS)} steps OK in {total_duration:.0f}s"
    if error_count > 0:
        failed_jobs = [r["job"] for r in results if r["status"] == "error"]
        summary += f" (FAILED: {', '.join(failed_jobs)})"
    
    # Structured success log
    log_job_success(
        "market_close_pipeline",
        summary,
        steps_total=len(MARKET_CLOSE_PIPELINE_STEPS),
        steps_ok=ok_count,
        steps_failed=error_count,
        duration_ms=total_duration_ms,
        failed_steps=[r["job"] for r in results if r["status"] == "error"],
        step_durations={r["job"]: r["duration_s"] for r in results},
    )
    
    return summary


# =============================================================================
# WEEKLY AI PIPELINE - Orchestrated AI analysis (Sunday morning)
# =============================================================================

WEEKLY_AI_PIPELINE_STEPS = [
    "data_backfill",       # 1. Fill any data gaps first
    "ai_personas_weekly",  # 2. Generate AI investor personas
    "ai_bios_weekly",      # 3. Generate swipe bios
]


@register_job("weekly_ai_pipeline")
async def weekly_ai_pipeline_job() -> str:
    """
    Weekly AI pipeline - orchestrates weekly maintenance and AI jobs.
    
    This ensures proper ordering:
    1. data_backfill - Fill ALL data gaps (sectors, summaries, prices, etc.)
    2. ai_personas_weekly - Warren Buffett, Peter Lynch etc. analysis
    3. ai_bios_weekly - Fun "dating profile" style descriptions
    
    Each step waits for previous to complete. If data_backfill fails,
    AI jobs still run but may have incomplete data.
    
    Schedule: Sunday 2 AM UTC
    """
    import time
    from app.jobs.executor import execute_job
    from app.repositories import cronjobs_orm as cron_repo
    
    logger.info("=" * 60)
    logger.info("WEEKLY AI PIPELINE - Starting")
    logger.info("=" * 60)
    
    results = []
    total_start = time.monotonic()
    
    for step_num, job_name in enumerate(WEEKLY_AI_PIPELINE_STEPS, 1):
        step_start = time.monotonic()
        logger.info(f"[WEEKLY AI] Step {step_num}/{len(WEEKLY_AI_PIPELINE_STEPS)}: {job_name}")
        
        try:
            message = await execute_job(job_name)
            duration_s = time.monotonic() - step_start
            
            try:
                await cron_repo.update_job_stats(job_name, "ok", int(duration_s * 1000))
            except Exception:
                pass
            
            results.append({
                "step": step_num,
                "job": job_name,
                "status": "ok",
                "duration_s": round(duration_s, 1),
                "message": message[:100] if message else "Done",
            })
            logger.info(f"[WEEKLY AI] ✓ {job_name} completed in {duration_s:.1f}s")
            
        except Exception as e:
            duration_s = time.monotonic() - step_start
            error_msg = str(e)[:200]
            
            try:
                await cron_repo.update_job_stats(job_name, "error", int(duration_s * 1000), error_msg)
            except Exception:
                pass
            
            results.append({
                "step": step_num,
                "job": job_name,
                "status": "error",
                "duration_s": round(duration_s, 1),
                "message": error_msg,
            })
            logger.error(f"[WEEKLY AI] ✗ {job_name} FAILED after {duration_s:.1f}s: {error_msg}")
    
    total_duration = time.monotonic() - total_start
    total_duration_ms = int(total_duration * 1000)
    
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info("=" * 60)
    logger.info(f"WEEKLY AI PIPELINE - Completed in {total_duration:.1f}s")
    logger.info(f"Results: {ok_count} succeeded, {error_count} failed")
    logger.info("=" * 60)
    
    summary = f"Weekly AI: {ok_count}/{len(WEEKLY_AI_PIPELINE_STEPS)} steps OK in {total_duration:.0f}s"
    if error_count > 0:
        failed_jobs = [r["job"] for r in results if r["status"] == "error"]
        summary += f" (FAILED: {', '.join(failed_jobs)})"
    
    # Structured success log
    log_job_success(
        "weekly_ai_pipeline",
        summary,
        steps_total=len(WEEKLY_AI_PIPELINE_STEPS),
        steps_ok=ok_count,
        steps_failed=error_count,
        duration_ms=total_duration_ms,
        failed_steps=[r["job"] for r in results if r["status"] == "error"],
    )
    
    return summary


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
    from app.services.data_providers import get_yfinance_service
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

        yf_service = get_yfinance_service()
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
        price_results = {}
        if symbols_needing_data:
            logger.info(f"[INGEST] Fetching 5y price history for {len(symbols_needing_data)} symbols")
            today = date.today()
            start_date = today - timedelta(days=1825)  # 5 years
            price_results = await yf_service.get_price_history_batch(
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

                # Store price history if we fetched it
                if symbol in price_results:
                    price_data = price_results[symbol]
                    if price_data:
                        prices, _version = price_data
                        if prices is not None and not prices.empty:
                            await price_history_repo.save_prices(symbol, prices)
                            logger.info(f"[INGEST] Saved {len(prices)} days of price history for {symbol}")

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


async def _store_price_history(symbol: str, prices: pd.DataFrame) -> None:
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
    from app.services.data_providers import get_yfinance_service
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
        # PHASE 2: Price history (symbols with < 200 days of data)
        # =====================================================================
        symbols_needing_prices = []
        for symbol in tickers:
            count = await price_history_repo.get_price_count(symbol)
            if count < 200:  # Need at least ~1 year of data
                symbols_needing_prices.append(symbol)
        
        if symbols_needing_prices:
            logger.info(f"[BACKFILL] Phase 2: Fetching prices for {len(symbols_needing_prices)} symbols")
            
            yf_service = get_yfinance_service()
            today = date.today()
            start_date = today - timedelta(days=1825)  # 5 years
            
            # Batch fetch prices
            price_results = await yf_service.get_price_history_batch(
                symbols=symbols_needing_prices[:20],  # Limit batch size
                start_date=start_date,
                end_date=today,
            )
            
            for symbol, result in price_results.items():
                if result:
                    prices, _version = result
                    if prices is not None and not prices.empty:
                        await price_history_repo.save_prices(symbol, prices)
                        stats["prices"] += 1
        
        # =====================================================================
        # PHASE 3: Fundamentals (symbols without recent fundamentals)
        # =====================================================================
        symbols_needing_fundamentals = []
        for symbol in tickers[:50]:  # Limit to avoid rate limiting
            from app.repositories import financials_orm
            try:
                fundamentals = await financials_orm.get_fundamentals(symbol)
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
            score = await quant_repo.get_score(symbol)
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
        
        symbols_needing_dipfinder = []
        today = date.today()
        for symbol in tickers:
            signal = await dipfinder_orm.get_latest_signal(symbol)
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
    Fetch stock data from yfinance and update dip_state with latest prices.
    Uses unified YFinanceService with change detection for targeted cache invalidation.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from datetime import date

    from app.cache.cache import Cache
    from app.repositories import symbols_orm as symbols_repo
    from app.services.data_providers import get_yfinance_service
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

        yf_service = get_yfinance_service()

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

            results = await yf_service.get_price_history_batch(symbols, start_date, today)

            for symbol in symbols:
                try:
                    data = results.get(symbol)
                    if not data:
                        continue
                    prices, price_version = data
                    if prices is None or prices.empty:
                        continue

                    # Get last valid (non-NaN) close price (prefer adjusted close)
                    close_col = get_close_column(prices)
                    valid_closes = prices[close_col].dropna()
                    if valid_closes.empty:
                        continue
                    current_price = float(valid_closes.iloc[-1])

                    # Check if data actually changed
                    data_changed = False
                    if price_version:
                        data_changed = await yf_service.has_data_changed(
                            symbol, "prices", price_version.hash
                        )
                        if data_changed:
                            await yf_service.save_data_version(symbol, price_version)
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

                    # Store new price history (only new rows)
                    last_date = latest_dates.get(symbol)
                    if last_date:
                        prices_to_store = prices.loc[prices.index.date > last_date]
                    else:
                        prices_to_store = prices
                    if prices_to_store is not None and not prices_to_store.empty:
                        await _store_price_history(symbol, prices_to_store)

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
                    await yf_service.get_price_history(
                        symbol,
                        period="5d",
                    )
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


@register_job("cache_warmup")
async def cache_warmup_job() -> str:
    """
    Pre-cache chart data for top dips and benchmarks.
    Forces refresh of existing cache entries.

    Schedule: After data_grab (Mon-Fri at 11:30pm) or on demand
    """
    from datetime import date, timedelta

    from app.cache.cache import Cache
    from app.dipfinder.service import get_dipfinder_service
    from app.services.runtime_settings import get_runtime_setting

    logger.info("Starting cache_warmup job")
    job_start = time.monotonic()

    try:
        # Get top 20 active symbols ordered by dip percentage
        symbols = await jobs_repo.get_top_dip_symbols(limit=20)

        # Add benchmark symbols from runtime settings
        benchmarks = get_runtime_setting("benchmarks", [])
        benchmark_symbols = [b.get("symbol") for b in benchmarks if b.get("symbol")]

        # Chart periods to pre-cache
        periods = [90, 180, 365]

        service = get_dipfinder_service()
        cached_count = 0
        chart_cache = Cache(prefix="chart", default_ttl=3600)

        all_symbols = list(set(symbols + benchmark_symbols))

        for symbol in all_symbols:
            for days in periods:
                cache_key = f"{symbol}:{days}"

                try:
                    # Fetch fresh price data (always refresh)
                    prices = await service.price_provider.get_prices(
                        symbol,
                        start_date=date.today() - timedelta(days=days),
                        end_date=date.today(),
                    )

                    if prices is not None and not prices.empty:
                        # Get min_dip_pct for the symbol
                        min_dip_pct = await jobs_repo.get_symbol_min_dip_pct(symbol)

                        # Build chart data (use adjusted close for accuracy)
                        close_col = get_close_column(prices)
                        ref_high = float(prices[close_col].max())
                        threshold = ref_high * (1.0 - min_dip_pct)

                        ref_high_date = None
                        dip_start_date = None
                        if close_col in prices.columns and not prices.empty:
                            ref_high_idx = prices[close_col].idxmax()
                            ref_high_date = str(ref_high_idx.date()) if hasattr(ref_high_idx, "date") else str(ref_high_idx)
                            prices_after_peak = prices.loc[ref_high_idx:]
                            if len(prices_after_peak) > 1:
                                dip_low_idx = prices_after_peak[close_col].idxmin()
                                dip_start_date = str(dip_low_idx.date()) if hasattr(dip_low_idx, "date") else str(dip_low_idx)

                        chart_points = []
                        for idx, row_data in prices.iterrows():
                            close = float(row_data[close_col])
                            drawdown = (close - ref_high) / ref_high if ref_high > 0 else 0.0
                            chart_points.append({
                                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                                "close": close,
                                "ref_high": ref_high,
                                "threshold": threshold,
                                "drawdown": drawdown,
                                "since_dip": None,
                                "dip_start_date": dip_start_date,
                                "ref_high_date": ref_high_date,
                            })

                        await chart_cache.set(cache_key, chart_points)
                        cached_count += 1
                        logger.debug(f"Cached chart for {symbol}:{days}")

                except Exception as e:
                    logger.warning(f"Failed to cache {symbol}:{days}: {e}")
                    continue

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Warmed up {cached_count} chart caches for {len(all_symbols)} symbols"
        
        # Structured success log
        log_job_success(
            "cache_warmup",
            message,
            items_warmed=cached_count,
            symbols_cached=len(all_symbols),
            periods_per_symbol=len(periods),
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"cache_warmup failed: {e}")
        raise


@register_job("ai_bios_weekly")
async def ai_bios_weekly_job() -> str:
    """
    Generate swipe-style bios for dips using OpenAI Batch API.

    Schedule: Weekly Sunday 4am
    """
    from app.services.batch_scheduler import (
        process_completed_batch_jobs,
        schedule_batch_swipe_bios,
    )

    logger.info("Starting ai_bios_weekly job")
    job_start = time.monotonic()

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_swipe_bios()

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        
        # Structured success log
        log_job_success(
            "ai_bios_weekly",
            message,
            batch_id=batch_id,
            items_processed=processed,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"ai_bios_weekly failed: {e}")
        raise


@register_job("ai_batch_poll")
async def ai_batch_poll_job() -> str:
    """
    Poll for completed OpenAI batch jobs.

    Schedule: Every 5 minutes (idempotent - only calls API if pending jobs exist)

    OpenAI Batch API can take up to 24 hours to complete.
    This job polls for completed batches and processes their results.
    """
    from app.services.batch_scheduler import process_completed_batch_jobs

    logger.info("Starting ai_batch_poll job")
    job_start = time.monotonic()

    try:
        processed = await process_completed_batch_jobs()

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Polled batches, processed: {processed}"
        
        # Structured success log
        log_job_success(
            "ai_batch_poll",
            message,
            items_processed=processed,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"ai_batch_poll failed: {e}")
        raise


@register_job("batch_watchdog")
async def batch_watchdog_job() -> str:
    """
    Batch job watchdog - expires stale batch jobs and reports health.
    
    Actions:
    1. Mark jobs stuck in pending/in_progress for >24h as 'expired'
    2. Log warnings for stale jobs (useful for alerting)
    3. Return health summary
    
    Schedule: Every hour
    """
    from app.repositories import api_usage_orm
    
    logger.info("Starting batch_watchdog job")
    job_start = time.monotonic()
    
    try:
        # Expire stale jobs (>24h old)
        expired_jobs = await api_usage_orm.expire_stale_batch_jobs(max_age_hours=24)
        
        # Get current health metrics
        health = await api_usage_orm.get_batch_job_health()
        
        if expired_jobs:
            logger.warning(
                f"batch_watchdog: Expired {len(expired_jobs)} stale jobs: "
                f"{[j['batch_id'] for j in expired_jobs]}"
            )
        
        if health.get("has_stale_jobs"):
            logger.warning(
                f"batch_watchdog: Oldest pending job is {health['oldest_pending_hours']}h old"
            )
        
        pending = health.get("status_counts", {}).get("pending", 0)
        in_progress = health.get("status_counts", {}).get("in_progress", 0)
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Batch watchdog: expired={len(expired_jobs)}, pending={pending}, in_progress={in_progress}"
        
        # Structured success log
        log_job_success(
            "batch_watchdog",
            message,
            expired_count=len(expired_jobs),
            pending_count=pending,
            in_progress_count=in_progress,
            has_stale_jobs=health.get("has_stale_jobs", False),
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"batch_watchdog failed: {e}")
        raise


@register_job("fundamentals_daily")
async def fundamentals_daily_job() -> str:
    """
    Refresh stock fundamentals and financial statements from Yahoo Finance.

    Smart, earnings-driven refresh criteria:
    1. Basic fundamentals never fetched
    2. Basic fundamentals expired (>30 days old)
    3. Earnings date has passed since last fetch (new quarterly data available)
    4. Financial statements never fetched (financials_fetched_at is NULL)
    5. Financial statements stale (>90 days old) - statements change less frequently
    6. Next earnings date is within 7 days (pre-fetch for upcoming data)

    Runs daily as part of market_close_pipeline but only refreshes what's needed.
    """
    from datetime import datetime, timedelta

    from app.services.fundamentals import refresh_all_fundamentals

    logger.info("Starting fundamentals_daily job")
    job_start = time.monotonic()

    try:
        # Find symbols that need refresh based on multiple criteria
        rows = await jobs_repo.get_stocks_needing_fundamentals_refresh()

        symbols_to_refresh = []
        now = datetime.now(UTC)
        seven_days_from_now = now + timedelta(days=7)

        for row in rows:
            symbol = row["symbol"]
            fetched_at = row["fetched_at"]
            earnings_date = row["earnings_date"]
            next_earnings_date = row["next_earnings_date"]
            financials_fetched_at = row["financials_fetched_at"]

            # Criterion 1: Basic fundamentals never fetched
            if not fetched_at:
                symbols_to_refresh.append((symbol, "never_fetched"))
                continue

            # Normalize fetched_at to UTC
            fetched_at_utc = fetched_at.replace(tzinfo=UTC) if fetched_at.tzinfo is None else fetched_at

            # Criterion 2: Basic fundamentals expired (>30 days old)
            age_days = (now - fetched_at_utc).days
            if age_days > 30:
                symbols_to_refresh.append((symbol, f"expired_{age_days}d"))
                continue

            # Criterion 3: Earnings date passed since last fetch
            if earnings_date:
                earnings_dt = _parse_datetime(earnings_date)
                if earnings_dt and earnings_dt < now and earnings_dt > fetched_at_utc:
                    symbols_to_refresh.append((symbol, "earnings_passed"))
                    continue

            # Criterion 4: Financial statements never fetched
            if not financials_fetched_at:
                symbols_to_refresh.append((symbol, "financials_never_fetched"))
                continue

            # Criterion 5: Financial statements stale (>90 days - quarterly data)
            financials_at_utc = financials_fetched_at.replace(tzinfo=UTC) if financials_fetched_at.tzinfo is None else financials_fetched_at
            financials_age_days = (now - financials_at_utc).days
            if financials_age_days > 90:
                symbols_to_refresh.append((symbol, f"financials_stale_{financials_age_days}d"))
                continue

            # Criterion 6: Next earnings date within 7 days (pre-fetch)
            if next_earnings_date:
                next_earnings_dt = _parse_datetime(next_earnings_date)
                if next_earnings_dt and now <= next_earnings_dt <= seven_days_from_now:
                    # Only add if we haven't fetched since 1 week before earnings
                    if fetched_at_utc < (next_earnings_dt - timedelta(days=7)):
                        symbols_to_refresh.append((symbol, "upcoming_earnings"))
                        continue

            # Otherwise skip - data is fresh enough
            logger.debug(f"Skipping {symbol}: fresh data (age={age_days}d, financials_age={financials_age_days}d)")

        if not symbols_to_refresh:
            duration_ms = int((time.monotonic() - job_start) * 1000)
            message = "No symbols need fundamentals refresh"
            log_job_success(
                "fundamentals_daily",
                message,
                symbols_refreshed=0,
                symbols_failed=0,
                symbols_skipped=len(rows),
                duration_ms=duration_ms,
            )
            return message

        logger.info(f"Refreshing fundamentals for {len(symbols_to_refresh)} symbols")

        # Log reasons for refresh
        reasons = {}
        for _, reason in symbols_to_refresh:
            base_reason = reason.split("_")[0] if reason.startswith("expired") or reason.startswith("financials_stale") else reason
            reasons[base_reason] = reasons.get(base_reason, 0) + 1
        logger.info(f"Refresh reasons: {reasons}")

        # Use the existing refresh function with specific symbols
        symbols_only = [s for s, _ in symbols_to_refresh]
        rows_by_symbol = {row["symbol"]: row for row in rows}

        include_financials_for: set[str] = set()
        include_calendar_for: set[str] = set()

        for symbol, reason in symbols_to_refresh:
            row = rows_by_symbol.get(symbol)
            if not row:
                continue

            fetched_at = row["fetched_at"]
            financials_fetched_at = row["financials_fetched_at"]
            earnings_date = row["earnings_date"]
            next_earnings_date = row["next_earnings_date"]

            # Calendar updates for refreshed symbols (keep upcoming earnings accurate)
            if not next_earnings_date:
                include_calendar_for.add(symbol)
            else:
                next_dt = _parse_datetime(next_earnings_date)
                if not next_dt or next_dt <= (now + timedelta(days=30)) or next_dt < now:
                    include_calendar_for.add(symbol)

            # Decide whether to include financial statements
            include_financials = False
            if not financials_fetched_at:
                include_financials = True
            else:
                financials_at_utc = financials_fetched_at.replace(tzinfo=UTC) if financials_fetched_at.tzinfo is None else financials_fetched_at
                if (now - financials_at_utc).days > 90:
                    include_financials = True

            if earnings_date and fetched_at:
                earnings_dt = _parse_datetime(earnings_date)
                fetched_at_utc = fetched_at.replace(tzinfo=UTC) if fetched_at.tzinfo is None else fetched_at
                if earnings_dt and earnings_dt < now and earnings_dt > fetched_at_utc:
                    include_financials = True

            if next_earnings_date:
                next_earnings_dt = _parse_datetime(next_earnings_date)
                fetched_at_utc = fetched_at.replace(tzinfo=UTC) if fetched_at and fetched_at.tzinfo is None else fetched_at
                if next_earnings_dt and fetched_at_utc and fetched_at_utc < (next_earnings_dt - timedelta(days=7)):
                    include_financials = True

            if include_financials:
                include_financials_for.add(symbol)

        result = await refresh_all_fundamentals(
            symbols=symbols_only,
            batch_size=5,
            include_financials_for=include_financials_for,
            include_calendar_for=include_calendar_for,
        )

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Fundamentals refresh: {result['refreshed']} updated, {result['failed']} failed, {result['skipped']} skipped"
        
        # Structured success log
        log_job_success(
            "fundamentals_daily",
            message,
            symbols_refreshed=result["refreshed"],
            symbols_failed=result["failed"],
            symbols_skipped=result["skipped"],
            refresh_reasons=reasons,
            with_financials=len(include_financials_for),
            with_calendar=len(include_calendar_for),
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"fundamentals_daily failed: {e}")
        raise


def _parse_datetime(dt_value: Any) -> datetime | None:
    """Parse datetime from various formats."""
    from datetime import datetime

    if not dt_value:
        return None

    if isinstance(dt_value, datetime):
        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=UTC)
        return dt_value

    if isinstance(dt_value, str):
        try:
            parsed = datetime.fromisoformat(dt_value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed
        except ValueError:
            return None

    return None


@register_job("ai_personas_weekly")
async def ai_personas_weekly_job() -> str:
    """
    Run AI persona analysis (Warren Buffett, Peter Lynch, etc.) on stocks.

    Schedule: Weekly Sunday 3am (first AI job of the day)

    Each persona analyzes stocks using their investment philosophy.
    Results are stored for frontend display.

    Uses Batch API for 50% cost savings. Results collected by ai_batch_poll job.
    Uses input version checking to skip symbols whose data hasn't changed.
    """
    from app.services.ai_agents import run_all_agent_analyses_batch
    from app.services.batch_scheduler import process_completed_batch_jobs

    logger.info("Starting ai_personas_weekly job")
    job_start = time.monotonic()

    try:
        # Process any completed agent batches first
        processed = await process_completed_batch_jobs()

        # Submit new batch
        result = await run_all_agent_analyses_batch()

        batch_id = result.get("batch_id")
        submitted = result.get("submitted", 0)
        skipped = result.get("skipped", 0)

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Batch: {batch_id or 'none needed'}, submitted: {submitted}, skipped: {skipped}, processed: {processed}"
        
        # Structured success log
        log_job_success(
            "ai_personas_weekly",
            message,
            batch_id=batch_id,
            submitted_count=submitted,
            skipped_count=skipped,
            processed_count=processed,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"ai_personas_weekly failed: {e}")
        raise


@register_job("cleanup_daily")
async def cleanup_daily_job() -> str:
    """
    Clean up expired suggestions and old API keys.

    Schedule: Daily midnight
    """
    logger.info("Starting cleanup_daily job")
    job_start = time.monotonic()

    try:
        # Rejected suggestions > 7 days
        suggestions_cleaned = await jobs_repo.cleanup_expired_suggestions()

        # Pending suggestions > 30 days
        stale_suggestions = await jobs_repo.cleanup_stale_pending_suggestions()

        # Expired AI analyses
        ai_analyses = await jobs_repo.cleanup_expired_ai_analyses()

        # Expired AI agent analyses
        agent_analyses = await jobs_repo.cleanup_expired_agent_analyses()

        # Expired cached symbol search results
        search_results = await jobs_repo.cleanup_expired_symbol_search_results()

        # Expired user API keys
        api_keys = await jobs_repo.cleanup_expired_api_keys()

        total_cleaned = (suggestions_cleaned or 0) + (stale_suggestions or 0) + (ai_analyses or 0) + (agent_analyses or 0) + (search_results or 0) + (api_keys or 0)
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Cleanup completed: {total_cleaned} items removed"
        
        # Structured success log
        log_job_success(
            "cleanup_daily",
            message,
            total_cleaned=total_cleaned,
            suggestions_cleaned=suggestions_cleaned or 0,
            stale_suggestions=stale_suggestions or 0,
            ai_analyses=ai_analyses or 0,
            agent_analyses=agent_analyses or 0,
            search_results=search_results or 0,
            api_keys_expired=api_keys or 0,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"cleanup_daily failed: {e}")
        raise


@register_job("portfolio_worker")
async def portfolio_worker_job() -> str:
    """
    Process queued portfolio analytics jobs.

    Schedule: Every 5 minutes
    """
    from app.portfolio.jobs import process_pending_jobs

    logger.info("Starting portfolio_worker job")
    job_start = time.monotonic()

    try:
        processed = await process_pending_jobs(limit=3)
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Processed {processed} portfolio analytics jobs"
        
        # Structured success log
        log_job_success(
            "portfolio_worker",
            message,
            jobs_processed=processed,
            duration_ms=duration_ms,
        )
        return message
    except Exception as e:
        logger.error(f"portfolio_worker failed: {e}")
        raise


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

        # Run signal scanner
        holding_days_options = [5, 10, 20, 40, 60]
        opportunities = scan_all_stocks(price_dfs, symbol_names, holding_days_options)

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


# =============================================================================
# QUANT MONTHLY
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
                        from app.quant_engine.scoring import generate_historical_weights
                        strategy_weights = generate_historical_weights(prices_series, holding_days=20)
                    else:
                        strategy_weights = pd.Series()
                    
                    # Convert fundamentals to z-scores (simplified)
                    fund_obj = fundamentals_map.get(symbol)
                    fundamentals_dict = None
                    if fund_obj:
                        # Very simplified z-score computation
                        # In production, compute vs sector/market medians
                        fundamentals_dict = {
                            "revenue_z": _to_z(fund_obj.revenue_growth, 0.10, 0.15),
                            "earnings_z": _to_z(fund_obj.profit_margin, 0.10, 0.10),
                            "margin_z": _to_z(fund_obj.profit_margin, 0.08, 0.10),
                            "pe_z": _to_z(fund_obj.pe_ratio, 20, 15, invert=True) if fund_obj.pe_ratio else 0,
                            "ev_ebitda_z": 0.0,  # Would need EV/EBITDA data
                            "ps_z": 0.0,  # Would need P/S data
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
                    min_dip_pct = symbol_thresholds.get(symbol, 0.15) * 100  # Convert to percentage
                    dip_pct = float(dip_state.dip_percentage) if dip_state and dip_state.dip_percentage else 0
                    
                    # Normalize dip against market (SPY)
                    # If market is down 10% and stock is down 20%, the stock-specific dip is 10%
                    # This helps filter out market-wide corrections vs stock-specific opportunities
                    normalized_dip_pct = max(0, dip_pct - spy_dip_pct)
                    
                    # Check how long the stock has been in this "dip"
                    # If it's been more than 365 days, it's a downtrend, not a dip
                    MAX_DIP_DAYS = 365
                    days_in_dip = 0
                    if dip_state and dip_state.dip_start_date:
                        days_in_dip = (date.today() - dip_state.dip_start_date).days
                    
                    # Use normalized dip for entry decision
                    # Stock must have a stock-specific dip (above market decline) to qualify
                    is_in_dip = normalized_dip_pct >= min_dip_pct and days_in_dip <= MAX_DIP_DAYS
                    is_downtrend = dip_pct >= min_dip_pct and days_in_dip > MAX_DIP_DAYS
                    
                    # Override mode if not in qualifying dip
                    final_mode = result.mode
                    if not result.gate_pass:
                        if is_downtrend:
                            final_mode = "DOWNTREND"  # Long-term decline, not a recoverable dip
                        elif not is_in_dip:
                            final_mode = "HOLD"  # Not a certified buy AND not in a qualifying dip
                    
                    # Track modes
                    if result.gate_pass:
                        mode_a_count += 1
                    elif is_in_dip:
                        mode_b_count += 1
                    elif is_downtrend:
                        mode_hold_count += 1  # Count downtrends with hold
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


def _to_z(value: float | None, mean: float, std: float, invert: bool = False) -> float:
    """Convert value to z-score. If invert=True, lower is better."""
    if value is None:
        return 0.0
    z = (float(value) - mean) / std if std != 0 else 0.0
    return -z if invert else z


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
    from app.repositories import price_history_orm as price_history_repo
    from app.quant_engine.signals import get_historical_triggers
    from app.quant_engine.trade_engine import (
        get_best_trade_strategy,
        test_signal_combination,
        analyze_dip,
        get_current_signals,
        SIGNAL_COMBINATIONS,
    )
    from app.services.data_providers.yfinance_service import get_yfinance_service

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
    yf_service = get_yfinance_service()
    end_date = date.today()
    start_date = end_date - timedelta(days=1260)
    
    spy_df = await price_history_repo.get_prices_as_dataframe("SPY", start_date, end_date)
    spy_prices = None
    if spy_df is not None and "Close" in spy_df.columns and len(spy_df) >= 60:
        spy_prices = spy_df["Close"].dropna()
    else:
        # Try yfinance
        try:
            yf_results = await yf_service.get_price_history_batch(
                ["SPY"], start_date, end_date
            )
            if "SPY" in yf_results:
                df, _ = yf_results["SPY"]
                if df is not None and "Close" in df.columns:
                    spy_prices = df["Close"].dropna()
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
                    total_return = 0.0
                    wins = 0
                    for trigger in triggers:
                        try:
                            entry_idx = prices.index.get_loc(trigger.date)
                        except KeyError:
                            try:
                                entry_idx = prices.index.get_indexer([trigger.date], method='nearest')[0]
                            except Exception:
                                continue
                        
                        exit_idx = min(entry_idx + holding_days, len(prices) - 1)
                        exit_price = float(prices.iloc[exit_idx])
                        ret = ((exit_price / trigger.price) - 1) * 100
                        total_return += ret
                        if ret > 0:
                            wins += 1
                    
                    n_trades = len(triggers)
                    win_rate = wins / n_trades if n_trades > 0 else 0.0
                    avg_return = total_return / n_trades if n_trades > 0 else 0.0
                    
                    # Buy-and-hold
                    lookback_days = 730
                    start_idx = max(0, len(prices) - lookback_days)
                    bh_start = float(prices.iloc[start_idx])
                    bh_end = float(prices.iloc[-1])
                    buy_hold_return = ((bh_end / bh_start) - 1) * 100
                    edge = total_return - buy_hold_return
                    
                    backtest_data = {
                        "signal_name": signal_name,
                        "n_trades": n_trades,
                        "win_rate": win_rate,
                        "total_return_pct": total_return,
                        "avg_return_per_trade": avg_return,
                        "holding_days": holding_days,
                        "buy_hold_return_pct": buy_hold_return,
                        "edge_pct": edge,
                        "outperformed": edge > 0,
                    }
            except Exception as e:
                logger.debug(f"Backtest failed for {symbol}: {e}")
            
            # 2. Full Trade Strategy
            trade_data = None
            try:
                result, _ = get_best_trade_strategy(
                    price_data, symbol, spy_prices=spy_prices, test_combinations=False
                )
                if result:
                    trade_data = {
                        "entry_signal": result.entry_signal_name,
                        "entry_threshold": result.entry_threshold,
                        "exit_signal": result.exit_strategy_name,
                        "exit_threshold": result.exit_threshold,
                        "n_trades": result.n_complete_trades,
                        "win_rate": result.win_rate,
                        "total_return_pct": result.total_return_pct,
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
                from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer, get_dip_summary
                
                optimizer = DipEntryOptimizer()
                result = optimizer.analyze(df, symbol, fundamentals=None)
                summary = get_dip_summary(result)
                
                dip_entry_data = {
                    "optimal_threshold": summary["optimal_dip_threshold"],
                    "optimal_price": summary["optimal_entry_price"],
                    "is_buy_now": summary["is_buy_now"],
                    "signal_strength": summary["buy_signal_strength"],
                    "signal_reason": summary["signal_reason"],
                    "recovery_days": int(summary["typical_recovery_days"]) if summary["typical_recovery_days"] else None,
                    "threshold_analysis": summary["threshold_analysis"],
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


# =============================================================================
# PORTFOLIO AI ANALYSIS (DAILY)
# =============================================================================


@register_job("portfolio_ai_analysis")
async def portfolio_ai_analysis_job() -> str:
    """
    Run AI analysis on portfolios that have changed.
    
    Schedule: Daily at 6 AM UTC (after market data is updated)
    
    This job:
    1. Finds portfolios where holdings have changed since last analysis
    2. Schedules batch AI analysis using the central batch scheduler
    3. Results are collected later by cron_sync_batch_jobs
    
    Only runs for portfolios with at least one holding.
    Uses batch processing for cost efficiency.
    """
    from app.services.batch_scheduler import cron_portfolio_ai_analysis

    logger.info("Starting portfolio_ai_analysis job")
    job_start = time.monotonic()
    
    try:
        result = await cron_portfolio_ai_analysis()
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        batch_id = result.get("batch_id")
        
        if batch_id:
            message = f"Scheduled portfolio AI batch: {batch_id}"
        else:
            message = "No portfolios need AI analysis"
        
        log_job_success(
            "portfolio_ai_analysis",
            message,
            batch_id=batch_id,
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"portfolio_ai_analysis failed: {e}")
        raise