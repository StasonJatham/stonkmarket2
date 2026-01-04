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

NOTE: This file is the main job definitions file. Shared utilities are in
app/jobs/utils.py. Pipeline orchestrators are in app/jobs/pipelines/.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger
from app.repositories import jobs_orm as jobs_repo
from app.repositories import price_history_orm as price_history_repo

from .registry import register_job
from .utils import get_close_column, log_job_success  # Shared utilities


if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("jobs.definitions")


# Re-export pipeline jobs so they're registered
from .pipelines import (  # noqa: E402, F401
    market_close_pipeline_job,
    weekly_ai_pipeline_job,
    MARKET_CLOSE_PIPELINE_STEPS,
    WEEKLY_AI_PIPELINE_STEPS,
)

# Re-export AI jobs so they're registered
from .ai import (  # noqa: E402, F401
    ai_bios_weekly_job,
    ai_batch_poll_job,
    batch_watchdog_job,
    ai_personas_weekly_job,
    portfolio_ai_analysis_job,
)

# Re-export quant jobs so they're registered
from .quant import (  # noqa: E402, F401
    quant_monthly_job,
    strategy_nightly_job,
    quant_scoring_daily_job,
    quant_analysis_nightly_job,
)

# Re-export data ingestion jobs so they're registered
from .data import (  # noqa: E402, F401
    universe_sync_job,
    symbol_ingest_job,
    data_backfill_job,
    prices_daily_job,
    add_to_ingest_queue,  # Helper function also exported
)

# Re-export analysis jobs so they're registered
from .analysis import (  # noqa: E402, F401
    signals_daily_job,
    dipfinder_daily_job,
    regime_daily_job,
)


# Data ingestion jobs moved to app/jobs/data/__init__.py:
# - universe_sync_job
# - symbol_ingest_job
# - data_backfill_job
# - prices_daily_job
# - add_to_ingest_queue


# =============================================================================
# CACHE WARMUP - Pre-cache chart data
# =============================================================================


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


# AI jobs moved to app/jobs/ai/__init__.py:
# - ai_bios_weekly_job
# - ai_batch_poll_job  
# - batch_watchdog_job


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


# ai_personas_weekly_job moved to app/jobs/ai/__init__.py


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


# Analysis jobs moved to app/jobs/analysis/__init__.py:
# - signals_daily_job
# - dipfinder_daily_job
# - regime_daily_job


# Quant jobs moved to app/jobs/quant/__init__.py:
# - quant_monthly_job
# - strategy_nightly_job
# - quant_scoring_daily_job
# - quant_analysis_nightly_job


# portfolio_ai_analysis_job moved to app/jobs/ai/__init__.py


# =============================================================================
# MARKET DATA SYNC - Weekly yfinance sector/industry data
# =============================================================================


@register_job("market_data_sync")
async def market_data_sync_job() -> str:
    """
    Sync market sector and industry data from yfinance.
    
    Schedule: Sunday 3 AM UTC (weekly)
    
    This job:
    1. Fetches all 11 sector summaries with top companies, ETFs, mutual funds
    2. Fetches all industries for each sector with top/performing/growth companies
    3. Stores data in market_sectors and market_industries tables
    4. Invalidates cache for sector/industry endpoints
    
    Data is used for:
    - Competitor suggestions (stocks in same industry)
    - Sector analysis and trends
    - Similar stock recommendations
    """
    from app.services.market_data import sync_all_market_data

    logger.info("Starting market_data_sync job")
    job_start = time.monotonic()
    
    try:
        result = await sync_all_market_data()
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        
        message = (
            f"Synced {result['sectors_synced']} sectors, "
            f"{result['industries_synced']} industries"
        )
        
        if result['errors']:
            message += f" ({len(result['errors'])} errors)"
        
        log_job_success(
            "market_data_sync",
            message,
            sectors_synced=result['sectors_synced'],
            industries_synced=result['industries_synced'],
            errors=len(result['errors']),
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"market_data_sync failed: {e}")
        raise


# =============================================================================
# CALENDAR SYNC - Weekly yfinance calendar data (earnings, splits, IPOs)
# =============================================================================


@register_job("calendar_sync")
async def calendar_sync_job() -> str:
    """
    Sync calendar data (earnings, IPOs, splits, economic events) from yfinance.
    
    Schedule: Saturday 5 AM UTC (weekly)
    
    This job:
    1. Fetches earnings calendar for next 5 weeks
    2. Fetches IPO calendar for next 5 weeks
    3. Fetches splits calendar for next 5 weeks
    4. Fetches economic events calendar for next 5 weeks
    5. Stores all events in database
    6. Invalidates cache for calendar endpoints
    
    Data is used for:
    - Calendar widget in UI
    - Earnings alerts and analysis
    - Split tracking for price adjustments
    - Economic event awareness
    """
    from app.services.calendar_data import sync_all_calendar_data

    logger.info("Starting calendar_sync job")
    job_start = time.monotonic()
    
    try:
        result = await sync_all_calendar_data(weeks_ahead=5)
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        
        total_synced = (
            result['earnings_synced'] + 
            result['ipos_synced'] + 
            result['splits_synced'] + 
            result['economic_events_synced']
        )
        
        message = (
            f"Synced {total_synced} calendar events: "
            f"{result['earnings_synced']} earnings, "
            f"{result['ipos_synced']} IPOs, "
            f"{result['splits_synced']} splits, "
            f"{result['economic_events_synced']} economic"
        )
        
        if result['errors']:
            message += f" ({len(result['errors'])} errors)"
        
        log_job_success(
            "calendar_sync",
            message,
            earnings_synced=result['earnings_synced'],
            ipos_synced=result['ipos_synced'],
            splits_synced=result['splits_synced'],
            economic_events_synced=result['economic_events_synced'],
            errors=len(result['errors']),
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"calendar_sync failed: {e}")
        raise