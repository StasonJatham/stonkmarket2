"""Built-in job definitions for scheduled tasks.

Jobs (New Names):
- symbol_ingest: Process new symbols (every 15 min, idempotent)
- prices_daily: Daily price updates (Mon-Fri 11 PM UTC)
- signals_daily: Technical signal scanner (Mon-Fri 10 PM UTC)
- regime_daily: Market regime detection (Mon-Fri 10:30 PM UTC)
- ai_personas_weekly: Warren Buffett, Peter Lynch etc. (Sunday 3 AM UTC)
- ai_bios_weekly: Swipe-style stock bios (Sunday 4 AM UTC)
- ai_batch_poll: OpenAI batch result collector (every 5 min)
- fundamentals_monthly: Company fundamentals (1st of month)
- quant_monthly: Portfolio optimization (1st of month)
- portfolio_worker: Portfolio analytics queue (every 5 min)
- cache_warmup: Pre-cache chart data (every 30 min)
- cleanup_daily: Remove expired data (midnight UTC)
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
    from app.services.openai_client import TaskType, submit_batch
    from app.services.fundamentals import get_fundamentals_for_analysis, refresh_fundamentals
    from app.services.stock_info import get_stock_info_batch_async

    BATCH_SIZE = 20
    logger.info("Starting initial_data_ingest job")

    try:
        # Get pending symbols from the queue
        rows = await jobs_repo.get_pending_ingest_symbols(BATCH_SIZE)

        if not rows:
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
        message = f"Processed {processed}/{len(symbols)} ({failed} failed), {remaining} remaining"
        if batch_id:
            message += f", AI batch: {batch_id}"
        logger.info(f"[INGEST] {message}")
        return message

    except Exception as e:
        logger.error(f"initial_data_ingest failed: {e}", exc_info=True)
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


@register_job("prices_daily")
async def prices_daily_job() -> str:
    """
    Fetch stock data from yfinance and update dip_state with latest prices.
    Uses unified YFinanceService with change detection for targeted cache invalidation.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from datetime import date

    from app.cache.cache import Cache
    from app.services.data_providers import get_yfinance_service
    from app.services.runtime_settings import get_runtime_setting

    logger.info("Starting prices_daily job")

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
        logger.info(f"prices_daily: {message}")
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

        message = f"Warmed up {cached_count} chart caches for {len(all_symbols)} symbols"
        logger.info(f"cache_warmup: {message}")
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

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_swipe_bios()

        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        logger.info(f"ai_bios_weekly: {message}")
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

    try:
        processed = await process_completed_batch_jobs()

        message = f"Polled batches, processed: {processed}"
        logger.info(f"ai_batch_poll: {message}")
        return message

    except Exception as e:
        logger.error(f"ai_batch_poll failed: {e}")
        raise


@register_job("fundamentals_monthly")
async def fundamentals_monthly_job() -> str:
    """
    Refresh stock fundamentals and financial statements from Yahoo Finance.

    Change-driven refresh criteria:
    1. Basic fundamentals never fetched
    2. Basic fundamentals expired (>30 days old)
    3. Earnings date has passed since last fetch (new quarterly data available)
    4. Financial statements never fetched (financials_fetched_at is NULL)
    5. Financial statements stale (>90 days old) - statements change less frequently
    6. Next earnings date is within 7 days (pre-fetch for upcoming data)

    Schedule: Monthly 1st at 2am UTC
    """
    from datetime import datetime, timedelta

    from app.services.fundamentals import refresh_all_fundamentals

    logger.info("Starting fundamentals_monthly job")

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
            logger.info(f"fundamentals_monthly: {message}")
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
        logger.info(f"fundamentals_monthly: {message}")
        return message

    except Exception as e:
        logger.error(f"fundamentals_monthly failed: {e}")
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

    try:
        # Process any completed agent batches first
        processed = await process_completed_batch_jobs()

        # Submit new batch
        result = await run_all_agent_analyses_batch()

        batch_id = result.get("batch_id")
        submitted = result.get("submitted", 0)
        skipped = result.get("skipped", 0)

        message = f"Batch: {batch_id or 'none needed'}, submitted: {submitted}, skipped: {skipped}, processed: {processed}"
        logger.info(f"ai_personas_weekly: {message}")
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
        logger.info(f"cleanup_daily: {message}")
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

    try:
        processed = await process_pending_jobs(limit=3)
        message = f"Processed {processed} portfolio analytics jobs"
        logger.info(f"portfolio_worker: {message}")
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

        message = f"Scanned {len(opportunities)} stocks, {strong_buys} strong buys, {total_active} active signals"
        logger.info(f"signals_daily: {message}")
        return message

    except Exception as e:
        logger.error(f"signals_daily failed: {e}")
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
            if df is not None:
                close_col = get_close_column(df)
                price_dfs[symbol] = df[close_col]

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
        logger.info(f"regime_daily: {message}")
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

    try:
        # Get all tracked symbols
        symbols = await symbols_repo.list_symbols()
        symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]

        if not symbol_list:
            return "No symbols to process"

        cache = Cache()

        # Clear old quant caches to force refresh
        await cache.delete("quant:recommendations")
        await cache.delete("quant:signals")
        await cache.delete("regime")

        # Log completion
        message = f"Cleared quant caches for {len(symbol_list)} symbols, ready for fresh calculations"
        logger.info(f"quant_monthly: {message}")
        return message

    except Exception as e:
        logger.error(f"quant_monthly failed: {e}")
        raise


# =============================================================================
# STRATEGY OPTIMIZATION - Nightly after prices
# =============================================================================


@register_job("strategy_optimize_nightly")
async def strategy_optimize_nightly_job() -> str:
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
    
    logger.info("Starting strategy_optimize_nightly job")
    
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
        
        message = f"Optimized strategies for {processed} symbols ({failed} failed), saved {signals_saved} signals"
        logger.info(f"strategy_optimize_nightly: {message}")
        return message
        
    except Exception as e:
        logger.exception(f"strategy_optimize_nightly failed: {e}")
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
    
    Schedule: Mon-Fri 11:45 PM UTC (15 min after strategy_optimize_nightly)
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
        
        message = (
            f"Scored {processed} symbols ({failed} failed), "
            f"Mode A: {mode_a_count}, Dip Entry: {mode_b_count}, Hold: {mode_hold_count}"
        )
        logger.info(f"quant_scoring_daily: {message}")
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