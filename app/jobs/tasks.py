"""Celery tasks for background jobs."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Iterable

import app.jobs.definitions  # noqa: F401 - register jobs

from app.celery_app import celery_app
from app.core.logging import get_logger
from app.jobs.executor import execute_job
from app.repositories import cronjobs_orm as cron_repo

logger = get_logger("jobs.celery_tasks")

# Per-worker event loop for Celery prefork pool
_worker_loop: asyncio.AbstractEventLoop | None = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the worker process."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def _run_async(coro: Any) -> Any:
    """Run async coroutine in the worker's event loop.
    
    Uses a persistent event loop to avoid 'Event loop is closed' errors
    with async Redis connections.
    """
    loop = _get_worker_loop()
    return loop.run_until_complete(coro)


async def _execute_job_locked(job_name: str) -> str:
    from app.cache.distributed_lock import DistributedLock

    lock = DistributedLock(f"job:{job_name}", timeout=60 * 30, blocking=False)
    acquired = await lock.acquire()
    if not acquired:
        try:
            await cron_repo.update_job_stats(
                job_name, "skipped", 0, "Already running"
            )
        except Exception as stats_exc:
            logger.warning(f"Failed to update job stats for {job_name}: {stats_exc}")
        return f"Skipped {job_name}: already running"

    start = time.monotonic()
    try:
        message = await execute_job(job_name)
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            await cron_repo.update_job_stats(job_name, "ok", duration_ms)
        except Exception as stats_exc:
            logger.warning(f"Failed to update job stats for {job_name}: {stats_exc}")
        return message
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            await cron_repo.update_job_stats(job_name, "error", duration_ms, str(exc))
        except Exception as stats_exc:
            logger.warning(f"Failed to update job stats for {job_name}: {stats_exc}")
        logger.exception("Job failed", extra={"job": job_name})
        raise
    finally:
        await lock.release()


def _run_job(job_name: str) -> str:
    return _run_async(_execute_job_locked(job_name))


async def _execute_symbol_task(symbol: str, coro: Awaitable[None]) -> str:
    from app.cache.distributed_lock import DistributedLock

    normalized = symbol.upper()
    lock = DistributedLock(f"symbol:process:{normalized}", timeout=60 * 30, blocking=False)
    acquired = await lock.acquire()
    if not acquired:
        return f"Skipped {normalized}: already processing"

    try:
        await coro
        return f"Processed {normalized}"
    finally:
        await lock.release()


async def _execute_symbol_ai_task(
    symbol: str,
    coro: Awaitable[str],
    lock_suffix: str,
) -> str:
    from app.cache.distributed_lock import DistributedLock

    normalized = symbol.upper()
    lock = DistributedLock(
        f"symbol:ai:{normalized}:{lock_suffix}", timeout=60 * 20, blocking=False
    )
    acquired = await lock.acquire()
    if not acquired:
        return f"Skipped {normalized}: AI task already running"

    try:
        return await coro
    finally:
        await lock.release()


async def _execute_dipfinder_task(
    tickers: Iterable[str], benchmark: str, windows: Iterable[int]
) -> str:
    from app.cache.distributed_lock import DistributedLock
    from app.dipfinder.service import get_dipfinder_service

    lock = DistributedLock("dipfinder:bulk", timeout=60 * 30, blocking=False)
    acquired = await lock.acquire()
    if not acquired:
        return "Skipped dipfinder run: already running"

    try:
        service = get_dipfinder_service()
        symbols = [ticker.upper() for ticker in tickers]
        if not symbols:
            return "No tickers provided"
        window_list = list(windows)
        for window in window_list:
            await service.get_signals(symbols, benchmark, window, force_refresh=True)
        return f"Processed {len(symbols)} tickers across {len(window_list)} windows"
    finally:
        await lock.release()


@celery_app.task(name="jobs.initial_data_ingest")
def initial_data_ingest_task() -> str:
    return _run_job("initial_data_ingest")


@celery_app.task(name="jobs.data_grab")
def data_grab_task() -> str:
    return _run_job("data_grab")


@celery_app.task(name="jobs.cache_warmup")
def cache_warmup_task() -> str:
    return _run_job("cache_warmup")


@celery_app.task(name="jobs.batch_ai_swipe")
def batch_ai_swipe_task() -> str:
    return _run_job("batch_ai_swipe")


@celery_app.task(name="jobs.batch_ai_analysis")
def batch_ai_analysis_task() -> str:
    return _run_job("batch_ai_analysis")


@celery_app.task(name="jobs.batch_poll")
def batch_poll_task() -> str:
    return _run_job("batch_poll")


@celery_app.task(name="jobs.fundamentals_refresh")
def fundamentals_refresh_task() -> str:
    return _run_job("fundamentals_refresh")


@celery_app.task(name="jobs.refresh_fundamentals_symbol")
def refresh_fundamentals_symbol_task(symbol: str) -> str:
    """Refresh fundamentals for a single symbol with minimal scope."""
    from datetime import datetime, timezone, timedelta
    from app.services.fundamentals import (
        refresh_fundamentals,
        get_fundamentals_with_status,
    )

    async def _run() -> str:
        data, _ = await get_fundamentals_with_status(symbol, allow_stale=True)
        include_financials = True

        if data:
            financials_fetched_at = data.get("financials_fetched_at")
            if financials_fetched_at:
                fin_at_utc = (
                    financials_fetched_at.replace(tzinfo=timezone.utc)
                    if financials_fetched_at.tzinfo is None
                    else financials_fetched_at
                )
                include_financials = (datetime.now(timezone.utc) - fin_at_utc).days > 90

                earnings_date = data.get("earnings_date")
                if earnings_date:
                    earnings_dt = (
                        earnings_date
                        if isinstance(earnings_date, datetime)
                        else datetime.combine(earnings_date, datetime.min.time(), tzinfo=timezone.utc)
                    )
                    if earnings_dt > fin_at_utc:
                        include_financials = True

                next_earnings = data.get("next_earnings_date")
                if next_earnings:
                    next_dt = (
                        next_earnings
                        if isinstance(next_earnings, datetime)
                        else datetime.combine(next_earnings, datetime.min.time(), tzinfo=timezone.utc)
                    )
                    if datetime.now(timezone.utc) <= next_dt <= datetime.now(timezone.utc) + timedelta(days=7):
                        if fin_at_utc < (next_dt - timedelta(days=7)):
                            include_financials = True

        await refresh_fundamentals(
            symbol,
            include_financials=include_financials,
            include_calendar=True,
        )
        return f"Refreshed fundamentals for {symbol} (financials={include_financials})"

    return _run_async(_run())


@celery_app.task(name="jobs.ai_agents_analysis")
def ai_agents_analysis_task() -> str:
    return _run_job("ai_agents_analysis")


@celery_app.task(name="jobs.ai_agents_batch_submit")
def ai_agents_batch_submit_task() -> str:
    return _run_job("ai_agents_batch_submit")


@celery_app.task(name="jobs.ai_agents_batch_collect")
def ai_agents_batch_collect_task() -> str:
    return _run_job("ai_agents_batch_collect")


@celery_app.task(name="jobs.cleanup")
def cleanup_task() -> str:
    return _run_job("cleanup")


@celery_app.task(name="jobs.portfolio_analytics_worker")
def portfolio_analytics_worker_task() -> str:
    return _run_job("portfolio_analytics_worker")


@celery_app.task(name="jobs.process_new_symbol")
def process_new_symbol_task(symbol: str) -> str:
    """Process a newly created symbol in the background."""
    from app.services.symbol_processing import process_new_symbol

    normalized = symbol.upper()
    return _run_async(_execute_symbol_task(normalized, process_new_symbol(normalized)))


@celery_app.task(name="jobs.process_approved_symbol")
def process_approved_symbol_task(symbol: str) -> str:
    """Process an approved suggestion in the background."""
    from app.services.symbol_processing import process_approved_symbol

    normalized = symbol.upper()
    return _run_async(
        _execute_symbol_task(normalized, process_approved_symbol(normalized))
    )


@celery_app.task(name="jobs.regenerate_symbol_summary")
def regenerate_symbol_summary_task(symbol: str) -> str:
    """Regenerate AI summary for a symbol in the background."""

    async def _run() -> str:
        from app.repositories import symbols_orm as symbols_repo
        from app.services.stock_info import get_stock_info_async
        from app.services.openai_client import summarize_company
        from app.services.runtime_settings import get_runtime_setting
        from app.cache.cache import Cache

        normalized = symbol.upper()

        if not get_runtime_setting("ai_enrichment_enabled", True):
            return f"AI enrichment disabled for {normalized}"

        info = await get_stock_info_async(normalized)
        if not info:
            return f"No Yahoo Finance data for {normalized}"

        description = info.get("summary")
        if not description or len(description) < 100:
            return f"No usable description for {normalized}"

        summary = await summarize_company(
            symbol=normalized,
            name=info.get("name"),
            description=description,
        )

        if not summary:
            return f"No AI summary generated for {normalized}"

        await symbols_repo.update_symbol_info(normalized, summary_ai=summary)

        stockinfo_cache = Cache(prefix="stockinfo", default_ttl=3600)
        await stockinfo_cache.delete(normalized)

        symbols_cache = Cache(prefix="symbols", default_ttl=3600)
        await symbols_cache.invalidate_pattern("*")

        return f"Regenerated AI summary for {normalized}"

    return _run_async(
        _execute_symbol_ai_task(symbol, _run(), lock_suffix="summary")
    )


@celery_app.task(name="jobs.refresh_dip_ai")
def refresh_dip_ai_task(symbol: str, field: str | None = None) -> str:
    """Refresh swipe AI analysis for a symbol in the background."""

    async def _run() -> str:
        from app.services import swipe as swipe_service

        normalized = symbol.upper()
        if field and field not in ("rating", "bio"):
            return f"Invalid AI field '{field}' for {normalized}"

        if field:
            card = await swipe_service.regenerate_ai_field(normalized, field)
            if not card:
                return f"No dip card for {normalized}"
            return f"Regenerated AI {field} for {normalized}"

        card = await swipe_service.get_dip_card_with_fresh_ai(
            normalized, force_refresh=True
        )
        if not card:
            return f"No dip card for {normalized}"
        return f"Refreshed swipe AI for {normalized}"

    lock_suffix = f"swipe:{field or 'all'}"
    return _run_async(_execute_symbol_ai_task(symbol, _run(), lock_suffix=lock_suffix))


@celery_app.task(name="jobs.dipfinder_run")
def dipfinder_run_task(tickers: list[str], benchmark: str, windows: list[int]) -> str:
    """Run DipFinder signals for a ticker list."""
    return _run_async(_execute_dipfinder_task(tickers, benchmark, windows))


@celery_app.task(name="jobs.dipfinder_refresh_all")
def dipfinder_refresh_all_task(benchmark: str | None = None) -> str:
    """Refresh DipFinder signals for all tracked symbols."""
    from app.dipfinder.config import get_dipfinder_config
    from app.repositories import symbols_orm as symbols_repo

    async def _run() -> str:
        symbols = await symbols_repo.list_symbols()
        tickers = [symbol.symbol for symbol in symbols]
        if not tickers:
            return "No symbols to refresh"
        config = get_dipfinder_config()
        return await _execute_dipfinder_task(
            tickers, benchmark or config.default_benchmark, config.windows
        )

    return _run_async(_run())
