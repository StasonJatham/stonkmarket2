"""Built-in job definitions - exactly 4 jobs."""

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


@register_job("data_grab")
async def data_grab_job() -> str:
    """
    Fetch stock data from yfinance and update dip_state with latest prices.
    Also fetches latest prices for benchmark indices.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from app.dipfinder.service import get_dipfinder_service
    from app.database.connection import fetch_all, execute
    from app.cache.cache import Cache
    from app.services.runtime_settings import get_runtime_setting
    from datetime import date, timedelta
    import yfinance as yf

    logger.info("Starting data_grab job")

    try:
        rows = await fetch_all("SELECT symbol FROM symbols WHERE is_active = TRUE")
        tickers = [row["symbol"] for row in rows]

        if not tickers:
            return "No active symbols"

        service = get_dipfinder_service()
        
        # Fetch latest prices and update dip_state
        updated_count = 0
        
        # Get min_dip_pct for each symbol to calculate dip start date
        dip_thresholds = {}
        threshold_rows = await fetch_all("SELECT symbol, min_dip_pct FROM symbols WHERE is_active = TRUE")
        for row in threshold_rows:
            dip_thresholds[row["symbol"]] = float(row["min_dip_pct"]) if row["min_dip_pct"] else 0.15
        
        for ticker in tickers:
            try:
                # Get full price history (1 year) for dip start calculation
                full_prices = await service.price_provider.get_prices(
                    ticker,
                    start_date=date.today() - timedelta(days=365),
                    end_date=date.today(),
                )
                
                if full_prices is not None and not full_prices.empty:
                    current_price = float(full_prices["Close"].iloc[-1])
                    
                    # ATH is calculated from our price_history table (single source of truth)
                    # This uses our stored historical data (typically ~1 year) rather than yfinance's 52w high
                    from app.database.connection import fetch_val
                    ath_price = await fetch_val(
                        "SELECT COALESCE(MAX(close), $2) FROM price_history WHERE symbol = $1",
                        ticker, float(full_prices["Close"].max())
                    )
                    ath_price = float(ath_price)
                    
                    # Calculate dip percentage using the true ATH
                    dip_percentage = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0
                    
                    # Calculate when the dip started using price_history table
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
                    await service.price_provider.get_prices(
                        symbol,
                        start_date=date.today() - timedelta(days=5),
                        end_date=date.today(),
                    )
                    benchmark_count += 1
                except Exception as e:
                    logger.warning(f"Failed to fetch benchmark {symbol}: {e}")

        # Invalidate caches since new data is available
        ranking_cache = Cache(prefix="ranking", default_ttl=1800)
        chart_cache = Cache(prefix="chart", default_ttl=3600)
        await ranking_cache.invalidate_pattern("*")
        await chart_cache.invalidate_pattern("*")
        logger.info("Invalidated ranking and chart caches")

        message = f"Updated {updated_count}/{len(tickers)} symbols, {benchmark_count} benchmarks"
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
