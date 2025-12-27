"""Built-in job definitions for scheduled tasks.

Jobs:
- initial_data_ingest: Process queued symbols (15min window, batch 20)
- data_grab: Daily price updates with change detection
- cache_warmup: Pre-cache chart data for top dips
- batch_ai_*: OpenAI Batch API for AI analysis
- fundamentals_refresh: Monthly fundamentals update (change-driven)
- ai_agents_analysis: Weekly AI agent analysis
- portfolio_analytics_worker: Process queued portfolio analytics jobs
- cleanup: Daily cleanup of expired data
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.repositories import jobs_orm as jobs_repo
from app.repositories import price_history_orm as price_history_repo

from .registry import register_job


if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("jobs.definitions")


# =============================================================================
# INITIAL DATA INGEST (Symbol Queue Processing)
# =============================================================================


@register_job("initial_data_ingest")
async def initial_data_ingest_job() -> str:
    """
    Process queued symbols that need initial data fetch.

    Uses a 15-minute aggregation window and processes up to 20 symbols per batch.
    - If <20 symbols in queue: process all
    - If >=20 symbols: process 20, rest will be handled in next run

    For each symbol:
    1. Fetch full price history (1 year)
    2. Fetch ticker info (fundamentals)
    3. Compute data versions for change tracking
    4. Mark symbol as completed in queue

    Schedule: Every 15 minutes
    """
    from app.services.data_providers import get_yfinance_service

    logger.info("Starting initial_data_ingest job")

    BATCH_SIZE = 20

    try:
        # Get pending symbols from the queue (oldest first, limit to batch size)
        rows = await jobs_repo.get_pending_ingest_symbols(BATCH_SIZE)

        if not rows:
            return "No symbols in queue"

        yf_service = get_yfinance_service()
        processed = 0
        failed = 0

        for row in rows:
            queue_id = row.id
            symbol = row.symbol
            attempts = row.attempts or 0

            # Mark as processing
            await jobs_repo.mark_ingest_processing(queue_id, attempts + 1)

            try:
                # Fetch price history (1 year) with version tracking
                prices, price_version = await yf_service.get_price_history(
                    symbol,
                    period="1y",
                )

                if prices is None or prices.empty:
                    raise ValueError(f"No price data for {symbol}")

                # Save price version for change tracking
                if price_version:
                    await yf_service.save_data_version(symbol, price_version)

                # Fetch and STORE fundamentals (not just version tracking)
                from app.services.fundamentals import refresh_fundamentals
                fundamentals = await refresh_fundamentals(symbol)
                if fundamentals:
                    logger.debug(f"Stored fundamentals for {symbol}")
                else:
                    logger.warning(f"No fundamentals available for {symbol}")

                # Get calendar events if available
                _calendar, calendar_version = await yf_service.get_calendar(symbol)
                if calendar_version:
                    await yf_service.save_data_version(symbol, calendar_version)

                # Insert price history into database
                await _store_price_history(symbol, prices)

                # Mark as completed
                await jobs_repo.mark_ingest_completed(queue_id)
                processed += 1
                logger.info(f"Ingested initial data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")
                failed += 1

                # Check max attempts
                max_attempts = 3
                if attempts + 1 >= max_attempts:
                    await jobs_repo.mark_ingest_failed(queue_id, str(e)[:500])
                else:
                    # Reset to pending for retry
                    await jobs_repo.mark_ingest_pending_retry(queue_id, str(e)[:500])

        remaining = await jobs_repo.get_ingest_queue_count()
        message = f"Processed {processed}/{len(rows)} symbols ({failed} failed), {remaining} remaining in queue"
        logger.info(f"initial_data_ingest: {message}")
        return message

    except Exception as e:
        logger.error(f"initial_data_ingest failed: {e}")
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


# =============================================================================
# BATCH PROCESS NEW SYMBOLS (Efficient batched yfinance + OpenAI)
# =============================================================================


@register_job("process_new_symbols_batch")
async def process_new_symbols_batch_job() -> str:
    """
    Batch process all symbols with fetch_status='pending'.

    Runs every 10 minutes to aggregate new symbol requests and process them
    efficiently using:
    - yfinance batch download (single API call for all symbols)
    - OpenAI Batch API for AI content generation (50% cost savings)

    Schedule: Every 10 minutes (*/10 * * * *)
    """
    from datetime import date, timedelta

    from app.cache.cache import Cache
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import price_history_orm as price_history_repo
    from app.repositories import dip_state_orm as dip_state_repo
    from app.services.data_providers import get_yfinance_service
    from app.services.openai_client import TaskType, submit_batch
    from app.services.fundamentals import get_fundamentals_for_analysis, refresh_fundamentals
    from app.services.stock_info import get_stock_info_batch_async

    logger.info("Starting process_new_symbols_batch job")

    try:
        # Get all symbols with pending status
        pending_symbols = await symbols_repo.get_symbols_by_status("pending")

        if not pending_symbols:
            return "No pending symbols to process"

        symbols = [s.symbol for s in pending_symbols]
        logger.info(f"[BATCH] Processing {len(symbols)} pending symbols: {symbols}")

        # Mark all as fetching
        for symbol in symbols:
            await symbols_repo.update_fetch_status(symbol, fetch_status="fetching")

        yf_service = get_yfinance_service()
        processed = 0
        failed = 0
        ai_items = []  # Collect items for OpenAI batch

        # Step 1: Batch fetch stock info from yfinance
        logger.info(f"[BATCH] Step 1: Fetching Yahoo Finance info for {len(symbols)} symbols")
        stock_infos = await get_stock_info_batch_async(symbols)

        # Step 2: Batch fetch 5 years of price history
        logger.info(f"[BATCH] Step 2: Batch fetching 5y price history for {len(symbols)} symbols")
        today = date.today()
        start_date = today - timedelta(days=1825)  # 5 years

        price_results = await yf_service.get_price_history_batch(
            symbols=symbols,
            start_date=start_date,
            end_date=today,
        )

        # Step 3: Process each symbol with the fetched data
        for symbol in symbols:
            try:
                info = stock_infos.get(symbol, {})
                if not info:
                    logger.warning(f"[BATCH] No Yahoo data for {symbol}")
                    await symbols_repo.update_fetch_status(
                        symbol, fetch_status="error", fetch_error="No Yahoo Finance data"
                    )
                    failed += 1
                    continue

                name = info.get("name") or info.get("short_name")
                sector = info.get("sector")
                full_summary = info.get("summary")
                current_price = info.get("current_price", 0)
                ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
                dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0

                # Update symbol info
                if name or sector:
                    await symbols_repo.update_symbol_info(symbol, name=name, sector=sector)

                # Add to dip_state
                await dip_state_repo.upsert_dip_state(
                    symbol=symbol,
                    current_price=current_price,
                    ath_price=ath_price,
                    dip_percentage=dip_pct,
                )

                # Store price history
                price_data = price_results.get(symbol)
                if price_data:
                    prices, _version = price_data
                    if prices is not None and not prices.empty:
                        await price_history_repo.save_prices(symbol, prices)
                        logger.info(f"[BATCH] Saved {len(prices)} days of price history for {symbol}")

                # Fetch and store fundamentals
                await refresh_fundamentals(symbol)

                # Collect for AI batch processing
                if full_summary and len(full_summary) > 100:
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

                processed += 1

            except Exception as e:
                logger.error(f"[BATCH] Failed to process {symbol}: {e}")
                await symbols_repo.update_fetch_status(
                    symbol, fetch_status="error", fetch_error=str(e)[:500]
                )
                failed += 1

        # Step 4: Submit AI content generation as batch (if items)
        batch_id = None
        if ai_items:
            logger.info(f"[BATCH] Step 4: Submitting {len(ai_items)} items for AI batch processing")
            try:
                batch_id = await submit_batch(
                    task=TaskType.RATING,
                    items=ai_items,
                )
                if batch_id:
                    logger.info(f"[BATCH] Submitted OpenAI batch job: {batch_id}")
                    # Also submit for bio generation
                    bio_batch_id = await submit_batch(
                        task=TaskType.BIO,
                        items=ai_items,
                    )
                    if bio_batch_id:
                        logger.info(f"[BATCH] Submitted bio batch job: {bio_batch_id}")
            except Exception as e:
                logger.warning(f"[BATCH] Failed to submit AI batch: {e}")
                # Fall back to individual AI generation
                await _fallback_individual_ai_generation(ai_items)

        # Step 5: Mark successful symbols as fetched
        for symbol in symbols:
            status = await symbols_repo.get_symbol(symbol)
            if status and status.fetch_status == "fetching":
                # AI content will be populated when batch completes
                await symbols_repo.update_fetch_status(symbol, fetch_status="fetched")

        # Step 6: Invalidate caches
        ranking_cache = Cache(prefix="ranking", default_ttl=3600)
        await ranking_cache.invalidate_pattern("*")
        symbols_cache = Cache(prefix="symbols", default_ttl=3600)
        await symbols_cache.invalidate_pattern("*")

        message = f"Processed {processed}/{len(symbols)} symbols ({failed} failed)"
        if batch_id:
            message += f", AI batch submitted: {batch_id}"
        logger.info(f"[BATCH] {message}")
        return message

    except Exception as e:
        logger.error(f"process_new_symbols_batch failed: {e}", exc_info=True)
        raise


async def _fallback_individual_ai_generation(items: list[dict]) -> None:
    """Fallback to individual AI generation if batch fails."""
    from app.services.openai_client import generate_bio, rate_dip, summarize_company
    from app.repositories import dip_votes_orm as dip_votes_repo
    from app.repositories import symbols_orm as symbols_repo

    for item in items:
        symbol = item["symbol"]
        try:
            # Generate AI summary
            ai_summary = await summarize_company(
                symbol=symbol,
                name=item.get("name"),
                description=item.get("summary"),
            )
            if ai_summary:
                await symbols_repo.update_symbol_info(symbol, summary_ai=ai_summary)

            # Generate bio
            bio = await generate_bio(
                symbol=symbol,
                name=item.get("name"),
                sector=item.get("sector"),
                summary=item.get("summary"),
                dip_pct=item.get("dip_pct", 0),
            )

            # Generate rating
            rating_data = await rate_dip(
                symbol=symbol,
                current_price=item.get("current_price", 0),
                ref_high=item.get("ref_high", 0),
                dip_pct=item.get("dip_pct", 0),
                days_below=item.get("days_below", 0),
                name=item.get("name"),
                sector=item.get("sector"),
                summary=item.get("summary"),
            )

            # Store AI analysis
            if bio or rating_data:
                await dip_votes_repo.upsert_ai_analysis(
                    symbol=symbol,
                    swipe_bio=bio,
                    ai_rating=rating_data.get("rating") if rating_data else None,
                    ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                    is_batch=False,
                )
        except Exception as e:
            logger.warning(f"[BATCH] Individual AI generation failed for {symbol}: {e}")


@register_job("data_grab")
async def data_grab_job() -> str:
    """
    Fetch stock data from yfinance and update dip_state with latest prices.
    Uses unified YFinanceService with change detection for targeted cache invalidation.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from datetime import date

    from app.cache.cache import Cache
    from app.services.data_providers import get_yfinance_service
    from app.services.runtime_settings import get_runtime_setting

    logger.info("Starting data_grab job")

    try:
        tickers = await jobs_repo.get_active_symbol_tickers()

        if not tickers:
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

            results = await yf_service.get_price_history_batch(symbols, start_date, today)

            for symbol in symbols:
                try:
                    data = results.get(symbol)
                    if not data:
                        continue
                    prices, price_version = data
                    if prices is None or prices.empty:
                        continue

                    current_price = float(prices["Close"].iloc[-1])

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
                        symbol, fallback=float(prices["Close"].max())
                    )

                    # Calculate dip percentage
                    dip_percentage = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0

                    # Calculate dip start date
                    dip_threshold = dip_thresholds.get(symbol, 0.15)
                    dip_start_date = await jobs_repo.calculate_dip_start_date(
                        symbol, ath_price, dip_threshold
                    )

                    # Update dip_state
                    await jobs_repo.upsert_dip_state_with_dates(
                        symbol, current_price, ath_price, dip_percentage, dip_start_date
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

        message = f"Updated {updated_count}/{len(tickers)} symbols ({len(changed_symbols)} changed), {benchmark_count} benchmarks"
        logger.info(f"data_grab: {message}")
        return message

    except Exception as e:
        logger.error(f"data_grab failed: {e}")
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

                        # Build chart data
                        ref_high = float(prices["Close"].max())
                        threshold = ref_high * (1.0 - min_dip_pct)

                        ref_high_date = None
                        dip_start_date = None
                        if "Close" in prices.columns and not prices.empty:
                            ref_high_idx = prices["Close"].idxmax()
                            ref_high_date = str(ref_high_idx.date()) if hasattr(ref_high_idx, "date") else str(ref_high_idx)
                            prices_after_peak = prices.loc[ref_high_idx:]
                            if len(prices_after_peak) > 1:
                                dip_low_idx = prices_after_peak["Close"].idxmin()
                                dip_start_date = str(dip_low_idx.date()) if hasattr(dip_low_idx, "date") else str(dip_low_idx)

                        chart_points = []
                        for idx, row_data in prices.iterrows():
                            close = float(row_data["Close"])
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

        message = f"Warmed up {cached_count} chart caches for {len(all_symbols)} symbols"
        logger.info(f"cache_warmup: {message}")
        return message

    except Exception as e:
        logger.error(f"cache_warmup failed: {e}")
        raise


@register_job("batch_ai_swipe")
async def batch_ai_swipe_job() -> str:
    """
    Generate swipe-style bios for dips using OpenAI Batch API.

    Schedule: Weekly Sunday 3am
    """
    from app.services.batch_scheduler import (
        process_completed_batch_jobs,
        schedule_batch_swipe_bios,
    )

    logger.info("Starting batch_ai_swipe job")

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_swipe_bios()

        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        logger.info(f"batch_ai_swipe: {message}")
        return message

    except Exception as e:
        logger.error(f"batch_ai_swipe failed: {e}")
        raise


@register_job("batch_ai_analysis")
async def batch_ai_analysis_job() -> str:
    """
    Generate serious dip analysis using OpenAI Batch API.

    Schedule: Weekly Sunday 4am
    """
    from app.services.batch_scheduler import (
        process_completed_batch_jobs,
        schedule_batch_dip_analysis,
    )

    logger.info("Starting batch_ai_analysis job")

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_dip_analysis()

        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        logger.info(f"batch_ai_analysis: {message}")
        return message

    except Exception as e:
        logger.error(f"batch_ai_analysis failed: {e}")
        raise


@register_job("batch_poll")
async def batch_poll_job() -> str:
    """
    Poll for completed OpenAI batch jobs.

    Schedule: Every 5 minutes (idempotent - only calls API if pending jobs exist)

    OpenAI Batch API can take up to 24 hours to complete.
    This job polls for completed batches and processes their results.
    """
    from app.services.batch_scheduler import process_completed_batch_jobs

    logger.info("Starting batch_poll job")

    try:
        processed = await process_completed_batch_jobs()

        message = f"Polled batches, processed: {processed}"
        logger.info(f"batch_poll: {message}")
        return message

    except Exception as e:
        logger.error(f"batch_poll failed: {e}")
        raise


@register_job("fundamentals_refresh")
async def fundamentals_refresh_job() -> str:
    """
    Refresh stock fundamentals and financial statements from Yahoo Finance.

    Change-driven refresh criteria:
    1. Basic fundamentals never fetched
    2. Basic fundamentals expired (>30 days old)
    3. Earnings date has passed since last fetch (new quarterly data available)
    4. Financial statements never fetched (financials_fetched_at is NULL)
    5. Financial statements stale (>90 days old) - statements change less frequently
    6. Next earnings date is within 7 days (pre-fetch for upcoming data)

    Schedule: Weekly on Sunday at 2am (checks all conditions)
    """
    from datetime import datetime, timedelta

    from app.services.fundamentals import refresh_all_fundamentals

    logger.info("Starting fundamentals_refresh job")

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
            message = "No symbols need fundamentals refresh"
            logger.info(f"fundamentals_refresh: {message}")
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

        message = f"Fundamentals refresh: {result['refreshed']} updated, {result['failed']} failed, {result['skipped']} skipped"
        logger.info(f"fundamentals_refresh: {message}")
        return message

    except Exception as e:
        logger.error(f"fundamentals_refresh failed: {e}")
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


@register_job("ai_agents_analysis")
async def ai_agents_analysis_job() -> str:
    """
    Run AI agent analysis (Warren Buffett, Peter Lynch, etc.) on stocks.

    Schedule: Weekly Sunday 5am (after fundamentals_refresh)

    Each agent analyzes stocks using their investment philosophy.
    Results are stored for frontend display.

    Uses input version checking to skip symbols whose data hasn't changed.
    """
    from app.services.ai_agents import run_all_agent_analyses

    logger.info("Starting ai_agents_analysis job")

    try:
        result = await run_all_agent_analyses()

        message = f"Agent analysis: {result['analyzed']} analyzed, {result.get('skipped', 0)} skipped, {result['failed']} failed"
        logger.info(f"ai_agents_analysis: {message}")
        return message

    except Exception as e:
        logger.error(f"ai_agents_analysis failed: {e}")
        raise


@register_job("ai_agents_batch_submit")
async def ai_agents_batch_submit_job() -> str:
    """
    Submit AI agent analysis as a batch job for cost savings.

    Schedule: Weekly Sunday 3am (before regular ai_agents_analysis)

    Uses OpenAI Batch API for ~50% cost reduction.
    Results are collected separately after batch completes.

    Workflow:
    1. Check which symbols need analysis (expired or new)
    2. Filter symbols where input data hasn't changed
    3. Submit batch job with all agent prompts
    4. Store batch_id for later collection
    """
    from app.services.ai_agents import run_all_agent_analyses_batch

    logger.info("Starting ai_agents_batch_submit job")

    try:
        result = await run_all_agent_analyses_batch()

        if result.get("batch_id"):
            message = f"Batch {result['batch_id']}: {result['submitted']} symbols submitted, {result['skipped']} skipped"
        else:
            message = f"No batch submitted: {result.get('message', 'Unknown reason')}"

        logger.info(f"ai_agents_batch_submit: {message}")
        return message

    except Exception as e:
        logger.error(f"ai_agents_batch_submit failed: {e}")
        raise


@register_job("ai_agents_batch_collect")
async def ai_agents_batch_collect_job() -> str:
    """
    Collect results from pending AI agent batch jobs.

    Schedule: Every 4 hours (batch jobs complete within 24h)

    Checks for any completed batch jobs and processes their results.
    """
    from app.services.ai_agents import collect_agent_batch
    from app.services.openai_client import check_batch

    logger.info("Starting ai_agents_batch_collect job")

    try:
        # Find pending batch jobs
        batch_ids = await jobs_repo.get_pending_batch_jobs()

        if not batch_ids:
            return "No pending batch jobs"

        collected = 0
        pending = 0
        failed = 0

        for batch_id in batch_ids:
            # Check batch status
            status = await check_batch(batch_id)
            if not status:
                logger.warning(f"Could not check batch {batch_id}")
                failed += 1
                continue

            if status["status"] == "completed":
                # Collect and process results
                results = await collect_agent_batch(batch_id)
                collected += len(results)
                logger.info(f"Collected {len(results)} results from batch {batch_id}")

                # Clear batch_job_id for processed symbols
                await jobs_repo.clear_batch_job_references(batch_id)
            elif status["status"] in ("failed", "cancelled", "expired"):
                logger.error(f"Batch {batch_id} failed with status: {status['status']}")
                failed += 1
                # Clear failed batch references
                await jobs_repo.clear_batch_job_references(batch_id)
            else:
                # Still in progress
                pending += 1
                logger.info(f"Batch {batch_id} still {status['status']}: {status.get('completed_count', 0)}/{status.get('total_count', 0)}")

        message = f"Batch collect: {collected} results collected, {pending} pending, {failed} failed"
        logger.info(f"ai_agents_batch_collect: {message}")
        return message

    except Exception as e:
        logger.error(f"ai_agents_batch_collect failed: {e}")
        raise


@register_job("cleanup")
async def cleanup_job() -> str:
    """
    Clean up expired suggestions and old API keys.

    Schedule: Daily midnight
    """
    logger.info("Starting cleanup job")

    try:
        # Rejected suggestions > 7 days
        await jobs_repo.cleanup_expired_suggestions()

        # Pending suggestions > 30 days
        await jobs_repo.cleanup_stale_pending_suggestions()

        # Expired AI analyses
        await jobs_repo.cleanup_expired_ai_analyses()

        # Expired AI agent analyses
        await jobs_repo.cleanup_expired_agent_analyses()

        # Expired cached symbol search results
        await jobs_repo.cleanup_expired_symbol_search_results()

        # Expired user API keys
        await jobs_repo.cleanup_expired_api_keys()

        message = "Cleanup completed"
        logger.info(f"cleanup: {message}")
        return message

    except Exception as e:
        logger.error(f"cleanup failed: {e}")
        raise


@register_job("portfolio_analytics_worker")
async def portfolio_analytics_worker_job() -> str:
    """
    Process queued portfolio analytics jobs.

    Schedule: Every 5 minutes
    """
    from app.portfolio.jobs import process_pending_jobs

    logger.info("Starting portfolio_analytics_worker job")

    try:
        processed = await process_pending_jobs(limit=3)
        message = f"Processed {processed} portfolio analytics jobs"
        logger.info(f"portfolio_analytics_worker: {message}")
        return message
    except Exception as e:
        logger.error(f"portfolio_analytics_worker failed: {e}")
        raise


@register_job("signal_scanner_daily")
async def signal_scanner_daily_job() -> str:
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

    logger.info("Starting signal_scanner_daily job")

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
            if df is not None and "Close" in df.columns:
                price_dfs[symbol] = df["Close"]

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

        message = f"Scanned {len(opportunities)} stocks, {strong_buys} strong buys, {total_active} active signals"
        logger.info(f"signal_scanner_daily: {message}")
        return message

    except Exception as e:
        logger.error(f"signal_scanner_daily failed: {e}")
        raise


@register_job("market_regime_daily")
async def market_regime_daily_job() -> str:
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

    logger.info("Starting market_regime_daily job")

    cache = Cache(prefix="market", default_ttl=86400)

    try:
        symbols_list = await symbols_repo.list_symbols()

        if not symbols_list:
            return "No symbols tracked"

        symbol_list = [
            s.symbol for s in symbols_list 
            if s.symbol not in ("SPY", "^GSPC", "URTH")
        ]

        end_date = date.today()
        start_date = end_date - timedelta(days=400)

        price_dfs: dict[str, pd.Series] = {}
        for symbol in symbol_list:
            df = await price_history_repo.get_prices_as_dataframe(
                symbol, start_date, end_date
            )
            if df is not None and "Close" in df.columns:
                price_dfs[symbol] = df["Close"]

        if not price_dfs:
            return "No price data"

        prices = pd.DataFrame(price_dfs).dropna(how="all").ffill()
        
        if len(prices) < 60:
            return "Insufficient data"

        returns = prices.pct_change().dropna()

        # Detect regime
        regime = detect_regime(returns)

        # Correlation analysis
        corr = compute_correlation_analysis(returns)

        cache_data = {
            "as_of": str(date.today()),
            "regime": regime.regime,
            "trend": regime.trend,
            "volatility": regime.volatility,
            "description": regime.description,
            "recommendation": regime.risk_budget_recommendation,
            "avg_correlation": corr.average_correlation,
            "n_clusters": corr.n_clusters,
            "stress_correlation": corr.stress_correlation,
        }

        await cache.set("regime", cache_data)

        message = f"Regime: {regime.regime}, Avg Corr: {corr.average_correlation:.1%}"
        logger.info(f"market_regime_daily: {message}")
        return message

    except Exception as e:
        logger.error(f"market_regime_daily failed: {e}")
        raise
