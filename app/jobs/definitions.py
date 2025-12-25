"""Built-in job definitions for scheduled tasks.

Jobs:
- initial_data_ingest: Process queued symbols (15min window, batch 20)
- data_grab: Daily price updates with change detection
- cache_warmup: Pre-cache chart data for top dips
- batch_ai_*: OpenAI Batch API for AI analysis
- fundamentals_refresh: Monthly fundamentals update (change-driven)
- ai_agents_analysis: Weekly AI agent analysis
- cleanup: Daily cleanup of expired data
"""

from __future__ import annotations

from app.core.logging import get_logger

from .registry import register_job

logger = get_logger("jobs.definitions")


async def _calculate_dip_start_date(symbol: str, ath_price: float, dip_threshold: float):
    """
    Calculate when the stock first entered the current dip period.
    
    Uses price_history table to find the first date where the stock dropped
    below the dip threshold (from ATH) and stayed there.
    
    Args:
        symbol: Stock symbol
        ath_price: All-time high price from price_history
        dip_threshold: Minimum dip percentage to consider (e.g., 0.15 for 15%)
    
    Returns:
        Date when dip started, or None if not in dip
    """
    from datetime import date
    from app.database.connection import fetch_all as db_fetch_all
    
    if ath_price <= 0:
        return None
    
    # Calculate the threshold price (e.g., if ATH=100 and threshold=0.15, then dip_price=85)
    dip_threshold_price = ath_price * (1 - dip_threshold)
    
    # Get all price history sorted chronologically
    rows = await db_fetch_all(
        "SELECT date, close FROM price_history WHERE symbol = $1 ORDER BY date ASC",
        symbol
    )
    
    if not rows:
        return None
    
    # Find when the current dip started
    # Walk through prices chronologically, tracking when we enter/exit dip territory
    dip_start = None
    currently_in_dip = False
    
    for row in rows:
        price = float(row["close"])
        is_dip = price <= dip_threshold_price
        
        if is_dip and not currently_in_dip:
            # Entering a dip
            dip_start = row["date"]
            currently_in_dip = True
        elif not is_dip and currently_in_dip:
            # Exited the dip, reset
            dip_start = None
            currently_in_dip = False
    
    # If currently in dip, return when it started
    return dip_start if currently_in_dip else None


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
    from datetime import datetime, timezone, timedelta
    from app.database.connection import fetch_all, execute
    from app.services.data_providers import get_yfinance_service
    from app.services.data_providers.yfinance_service import DataVersion, _compute_hash
    
    logger.info("Starting initial_data_ingest job")
    
    BATCH_SIZE = 20
    
    try:
        # Get pending symbols from the queue (oldest first, limit to batch size)
        rows = await fetch_all(
            """
            SELECT id, symbol, attempts 
            FROM symbol_ingest_queue 
            WHERE status = 'pending' 
            ORDER BY priority DESC, queued_at ASC 
            LIMIT $1
            """,
            BATCH_SIZE
        )
        
        if not rows:
            return "No symbols in queue"
        
        yf_service = get_yfinance_service()
        processed = 0
        failed = 0
        
        for row in rows:
            queue_id = row["id"]
            symbol = row["symbol"]
            attempts = row["attempts"] or 0
            
            # Mark as processing
            await execute(
                """
                UPDATE symbol_ingest_queue 
                SET status = 'processing', 
                    processing_started_at = NOW(),
                    attempts = $2
                WHERE id = $1
                """,
                queue_id, attempts + 1
            )
            
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
                
                # Fetch ticker info (fundamentals)
                info = await yf_service.get_ticker_info(symbol)
                
                if info:
                    # Compute fundamentals version
                    fundamentals_version = DataVersion(
                        hash=_compute_hash(info),
                        timestamp=datetime.now(timezone.utc),
                        source="fundamentals",
                        metadata={
                            "pe_ratio": info.get("pe_ratio"),
                            "market_cap": info.get("market_cap"),
                        },
                    )
                    await yf_service.save_data_version(symbol, fundamentals_version)
                
                # Get calendar events if available
                calendar, calendar_version = await yf_service.get_calendar(symbol)
                if calendar_version:
                    await yf_service.save_data_version(symbol, calendar_version)
                
                # Insert price history into database
                await _store_price_history(symbol, prices)
                
                # Mark as completed
                await execute(
                    """
                    UPDATE symbol_ingest_queue 
                    SET status = 'completed', completed_at = NOW()
                    WHERE id = $1
                    """,
                    queue_id
                )
                processed += 1
                logger.info(f"Ingested initial data for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {symbol}: {e}")
                failed += 1
                
                # Check max attempts
                max_attempts = 3
                if attempts + 1 >= max_attempts:
                    await execute(
                        """
                        UPDATE symbol_ingest_queue 
                        SET status = 'failed', 
                            last_error = $2
                        WHERE id = $1
                        """,
                        queue_id, str(e)[:500]
                    )
                else:
                    # Reset to pending for retry
                    await execute(
                        """
                        UPDATE symbol_ingest_queue 
                        SET status = 'pending', 
                            last_error = $2
                        WHERE id = $1
                        """,
                        queue_id, str(e)[:500]
                    )
        
        remaining = await _get_queue_count()
        message = f"Processed {processed}/{len(rows)} symbols ({failed} failed), {remaining} remaining in queue"
        logger.info(f"initial_data_ingest: {message}")
        return message
        
    except Exception as e:
        logger.error(f"initial_data_ingest failed: {e}")
        raise


async def _store_price_history(symbol: str, prices: "pd.DataFrame") -> None:
    """Store price history in database."""
    from app.database.connection import execute
    
    if prices.empty:
        return
    
    # Convert DataFrame to records
    for idx, row in prices.iterrows():
        price_date = idx.date() if hasattr(idx, 'date') else idx
        await execute(
            """
            INSERT INTO price_history (symbol, date, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            symbol.upper(),
            price_date,
            float(row.get("Open", 0)),
            float(row.get("High", 0)),
            float(row.get("Low", 0)),
            float(row.get("Close", 0)),
            int(row.get("Volume", 0)),
        )


async def _get_queue_count() -> int:
    """Get count of pending symbols in queue."""
    from app.database.connection import fetch_val
    return await fetch_val(
        "SELECT COUNT(*) FROM symbol_ingest_queue WHERE status = 'pending'"
    ) or 0


async def add_to_ingest_queue(symbol: str, priority: int = 0) -> bool:
    """
    Add a symbol to the ingest queue.
    
    Called when:
    - New symbol is added to watchlist
    - Symbol is suggested and approved
    - Manual trigger via admin
    
    Returns True if added, False if already in queue.
    """
    from app.database.connection import execute, fetch_val
    
    # Check if already in queue
    existing = await fetch_val(
        "SELECT 1 FROM symbol_ingest_queue WHERE symbol = $1",
        symbol.upper()
    )
    if existing:
        logger.debug(f"Symbol {symbol} already in ingest queue")
        return False
    
    await execute(
        """
        INSERT INTO symbol_ingest_queue (symbol, status, priority, attempts, max_attempts, queued_at)
        VALUES ($1, 'pending', $2, 0, 3, NOW())
        ON CONFLICT (symbol) DO NOTHING
        """,
        symbol.upper(),
        priority,
    )
    logger.info(f"Added {symbol} to ingest queue (priority={priority})")
    return True


@register_job("data_grab")
async def data_grab_job() -> str:
    """
    Fetch stock data from yfinance and update dip_state with latest prices.
    Uses unified YFinanceService with change detection for targeted cache invalidation.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from app.database.connection import fetch_all, execute, fetch_val
    from app.cache.cache import Cache
    from app.services.runtime_settings import get_runtime_setting
    from app.services.data_providers import get_yfinance_service
    from app.services.data_providers.yfinance_service import _compute_hash
    from datetime import date, datetime, timedelta, timezone

    logger.info("Starting data_grab job")

    try:
        rows = await fetch_all("SELECT symbol FROM symbols WHERE is_active = TRUE")
        tickers = [row["symbol"] for row in rows]

        if not tickers:
            return "No active symbols"

        yf_service = get_yfinance_service()
        
        # Fetch latest prices and update dip_state
        updated_count = 0
        changed_symbols = []  # Track which symbols had data changes
        
        # Get min_dip_pct for each symbol to calculate dip start date
        dip_thresholds = {}
        threshold_rows = await fetch_all("SELECT symbol, min_dip_pct FROM symbols WHERE is_active = TRUE")
        for row in threshold_rows:
            dip_thresholds[row["symbol"]] = float(row["min_dip_pct"]) if row["min_dip_pct"] else 0.15
        
        for ticker in tickers:
            try:
                # Get price history with version tracking
                prices, price_version = await yf_service.get_price_history(
                    ticker,
                    period="1y",
                )
                
                if prices is not None and not prices.empty:
                    current_price = float(prices["Close"].iloc[-1])
                    
                    # Check if data actually changed
                    data_changed = False
                    if price_version:
                        data_changed = await yf_service.has_data_changed(
                            ticker, "prices", price_version.hash
                        )
                        if data_changed:
                            await yf_service.save_data_version(ticker, price_version)
                            changed_symbols.append(ticker)
                    
                    # ATH from our price_history table (single source of truth)
                    ath_price = await fetch_val(
                        "SELECT COALESCE(MAX(close), $2) FROM price_history WHERE symbol = $1",
                        ticker, float(prices["Close"].max())
                    )
                    ath_price = float(ath_price)
                    
                    # Calculate dip percentage
                    dip_percentage = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0
                    
                    # Calculate dip start date
                    dip_threshold = dip_thresholds.get(ticker, 0.15)
                    dip_start_date = await _calculate_dip_start_date(ticker, ath_price, dip_threshold)
                    
                    # Update dip_state
                    await execute(
                        """
                        INSERT INTO dip_state (symbol, current_price, ath_price, dip_percentage, dip_start_date, first_seen, last_updated)
                        VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                        ON CONFLICT (symbol) DO UPDATE SET
                            current_price = EXCLUDED.current_price,
                            ath_price = EXCLUDED.ath_price,
                            dip_percentage = EXCLUDED.dip_percentage,
                            dip_start_date = COALESCE(EXCLUDED.dip_start_date, dip_state.dip_start_date),
                            last_updated = NOW()
                        """,
                        ticker, current_price, ath_price, dip_percentage, dip_start_date
                    )
                    
                    # Store new price history
                    await _store_price_history(ticker, prices)
                    
                    updated_count += 1
                    days_in_dip = (date.today() - dip_start_date).days if dip_start_date else 0
                    logger.debug(f"Updated dip_state for {ticker}: ${current_price:.2f}, ATH: ${ath_price:.2f}, dip: {dip_percentage:.1f}%, days: {days_in_dip}")
                    
            except Exception as e:
                logger.warning(f"Failed to update {ticker}: {e}")
                continue

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
    from app.database.connection import fetch_all, fetch_one
    from app.dipfinder.service import get_dipfinder_service
    from app.cache.cache import Cache
    from app.services.runtime_settings import get_runtime_setting
    from datetime import date, timedelta

    logger.info("Starting cache_warmup job")

    try:
        # Get top 20 active symbols ordered by dip percentage
        rows = await fetch_all("""
            SELECT s.symbol 
            FROM symbols s
            LEFT JOIN dip_state ds ON s.symbol = ds.symbol
            WHERE s.is_active = TRUE 
            ORDER BY COALESCE(ds.dip_percentage, 0) DESC
            LIMIT 20
        """)
        symbols = [row["symbol"] for row in rows]

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
                        symbol_row = await fetch_one(
                            "SELECT min_dip_pct FROM symbols WHERE symbol = $1", symbol
                        )
                        min_dip_pct = float(symbol_row["min_dip_pct"]) if symbol_row and symbol_row.get("min_dip_pct") else 0.10

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
        schedule_batch_swipe_bios,
        process_completed_batch_jobs,
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
        schedule_batch_dip_analysis,
        process_completed_batch_jobs,
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
    Refresh stock fundamentals from Yahoo Finance.
    
    Change-driven: Only refreshes when:
    1. Data is expired (>30 days old)
    2. Earnings date has passed since last refresh
    3. Data was never fetched

    Schedule: Weekly on Sunday at 2am (checks all conditions)
    """
    from app.services.fundamentals import refresh_all_fundamentals
    from app.services.data_providers import get_yfinance_service
    from app.database.connection import fetch_all, execute
    from datetime import datetime, timezone

    logger.info("Starting fundamentals_refresh job")

    try:
        yf_service = get_yfinance_service()
        
        # Find symbols that need refresh based on multiple criteria
        rows = await fetch_all(
            """
            SELECT s.symbol, f.fetched_at, f.earnings_date
            FROM symbols s
            LEFT JOIN stock_fundamentals f ON s.symbol = f.symbol
            WHERE s.symbol_type = 'stock'
              AND s.is_active = TRUE
            """
        )
        
        symbols_to_refresh = []
        now = datetime.now(timezone.utc)
        
        for row in rows:
            symbol = row["symbol"]
            fetched_at = row["fetched_at"]
            earnings_date = row["earnings_date"]
            
            # Criterion 1: Never fetched
            if not fetched_at:
                symbols_to_refresh.append((symbol, "never_fetched"))
                continue
            
            # Criterion 2: Expired (>30 days old)
            age_days = (now - fetched_at.replace(tzinfo=timezone.utc)).days if fetched_at else 999
            if age_days > 30:
                symbols_to_refresh.append((symbol, f"expired_{age_days}d"))
                continue
            
            # Criterion 3: Earnings date passed since last fetch
            if earnings_date and fetched_at:
                # If earnings_date is in the past and after last fetch, refresh
                if isinstance(earnings_date, str):
                    try:
                        earnings_dt = datetime.fromisoformat(earnings_date)
                    except ValueError:
                        earnings_dt = None
                else:
                    earnings_dt = earnings_date
                
                if earnings_dt:
                    if hasattr(earnings_dt, 'tzinfo') and earnings_dt.tzinfo is None:
                        earnings_dt = earnings_dt.replace(tzinfo=timezone.utc)
                    
                    if earnings_dt < now and earnings_dt > fetched_at.replace(tzinfo=timezone.utc):
                        symbols_to_refresh.append((symbol, "earnings_passed"))
                        continue
            
            # Otherwise skip - data is fresh enough
            logger.debug(f"Skipping {symbol}: fresh data (age={age_days}d)")
        
        if not symbols_to_refresh:
            message = "No symbols need fundamentals refresh"
            logger.info(f"fundamentals_refresh: {message}")
            return message
        
        logger.info(f"Refreshing fundamentals for {len(symbols_to_refresh)} symbols")
        
        # Log reasons for refresh
        reasons = {}
        for _, reason in symbols_to_refresh:
            reasons[reason] = reasons.get(reason, 0) + 1
        logger.info(f"Refresh reasons: {reasons}")
        
        # Use the existing refresh function
        symbols_only = [s for s, _ in symbols_to_refresh]
        result = await refresh_all_fundamentals(batch_size=5)

        message = f"Fundamentals refresh: {result['refreshed']} updated, {result['failed']} failed, {result['skipped']} skipped"
        logger.info(f"fundamentals_refresh: {message}")
        return message

    except Exception as e:
        logger.error(f"fundamentals_refresh failed: {e}")
        raise


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
    from app.services.openai_client import check_batch
    from app.services.ai_agents import collect_agent_batch
    from app.database.connection import fetch_all, execute

    logger.info("Starting ai_agents_batch_collect job")

    try:
        # Find pending batch jobs
        rows = await fetch_all(
            """
            SELECT DISTINCT batch_job_id 
            FROM analysis_versions 
            WHERE batch_job_id IS NOT NULL 
              AND analysis_type = 'agent_analysis'
              AND generated_at > NOW() - INTERVAL '24 hours'
            """
        )
        
        if not rows:
            return "No pending batch jobs"
        
        collected = 0
        pending = 0
        failed = 0
        
        for row in rows:
            batch_id = row["batch_job_id"]
            
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
                await execute(
                    """
                    UPDATE analysis_versions 
                    SET batch_job_id = NULL 
                    WHERE batch_job_id = $1
                    """,
                    batch_id
                )
            elif status["status"] in ("failed", "cancelled", "expired"):
                logger.error(f"Batch {batch_id} failed with status: {status['status']}")
                failed += 1
                # Clear failed batch references
                await execute(
                    """
                    UPDATE analysis_versions 
                    SET batch_job_id = NULL 
                    WHERE batch_job_id = $1
                    """,
                    batch_id
                )
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
    from app.database.connection import execute

    logger.info("Starting cleanup job")

    try:
        # Rejected suggestions > 7 days
        await execute(
            "DELETE FROM stock_suggestions WHERE status = 'rejected' AND reviewed_at < NOW() - INTERVAL '7 days'"
        )

        # Pending suggestions > 30 days
        await execute(
            "DELETE FROM stock_suggestions WHERE status = 'pending' AND created_at < NOW() - INTERVAL '30 days'"
        )

        # Expired AI analyses
        await execute(
            "DELETE FROM dip_ai_analysis WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        )
        
        # Expired AI agent analyses
        await execute(
            "DELETE FROM ai_agent_analysis WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        )

        # Expired user API keys
        await execute(
            "DELETE FROM user_api_keys WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        )

        message = "Cleanup completed"
        logger.info(f"cleanup: {message}")
        return message

    except Exception as e:
        logger.error(f"cleanup failed: {e}")
        raise

