"""Pipeline orchestrator jobs.

These jobs run multiple child jobs in sequence, ensuring proper data flow.
"""

from __future__ import annotations

import time

from app.core.logging import get_logger
from app.jobs.registry import register_job
from app.jobs.utils import log_job_success


logger = get_logger("jobs.pipelines")


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


WEEKLY_AI_PIPELINE_STEPS = [
    "data_backfill",       # 1. Fill any data gaps first
    "ai_personas_weekly",  # 2. Generate AI investor personas
    "ai_bios_weekly",      # 3. Generate swipe bios
]


async def _run_pipeline(
    name: str,
    steps: list[str],
    log_prefix: str,
) -> str:
    """Generic pipeline runner that executes jobs sequentially.
    
    Args:
        name: Pipeline name for logging
        steps: List of job names to execute in order
        log_prefix: Log prefix like "[PIPELINE]" or "[WEEKLY AI]"
        
    Returns:
        Summary string with results
    """
    from app.jobs.executor import execute_job
    from app.repositories import cronjobs_orm as cron_repo
    
    logger.info("=" * 60)
    logger.info(f"{name} - Starting")
    logger.info("=" * 60)
    
    results = []
    total_start = time.monotonic()
    
    for step_num, job_name in enumerate(steps, 1):
        step_start = time.monotonic()
        logger.info(f"{log_prefix} Step {step_num}/{len(steps)}: {job_name}")
        
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
            logger.info(f"{log_prefix} {job_name} completed in {duration_s:.1f}s")
            
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
            logger.error(f"{log_prefix} {job_name} FAILED after {duration_s:.1f}s: {error_msg}")
            # Continue with next step - don't abort entire pipeline
    
    total_duration = time.monotonic() - total_start
    total_duration_ms = int(total_duration * 1000)
    
    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info("=" * 60)
    logger.info(f"{name} - Completed in {total_duration:.1f}s")
    logger.info(f"Results: {ok_count} succeeded, {error_count} failed")
    for r in results:
        status_icon = "OK" if r["status"] == "ok" else "FAIL"
        logger.info(f"  {status_icon} {r['job']}: {r['duration_s']}s - {r['message'][:50]}")
    logger.info("=" * 60)
    
    summary = f"{name}: {ok_count}/{len(steps)} steps OK in {total_duration:.0f}s"
    if error_count > 0:
        failed_jobs = [r["job"] for r in results if r["status"] == "error"]
        summary += f" (FAILED: {', '.join(failed_jobs)})"
    
    return summary, results, total_duration_ms


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
    summary, results, duration_ms = await _run_pipeline(
        name="MARKET CLOSE PIPELINE",
        steps=MARKET_CLOSE_PIPELINE_STEPS,
        log_prefix="[PIPELINE]",
    )
    
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    log_job_success(
        "market_close_pipeline",
        summary,
        steps_total=len(MARKET_CLOSE_PIPELINE_STEPS),
        steps_ok=ok_count,
        steps_failed=error_count,
        duration_ms=duration_ms,
        failed_steps=[r["job"] for r in results if r["status"] == "error"],
        step_durations={r["job"]: r["duration_s"] for r in results},
    )
    
    return summary


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
    summary, results, duration_ms = await _run_pipeline(
        name="WEEKLY AI PIPELINE",
        steps=WEEKLY_AI_PIPELINE_STEPS,
        log_prefix="[WEEKLY AI]",
    )
    
    ok_count = sum(1 for r in results if r["status"] == "ok")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    log_job_success(
        "weekly_ai_pipeline",
        summary,
        steps_total=len(WEEKLY_AI_PIPELINE_STEPS),
        steps_ok=ok_count,
        steps_failed=error_count,
        duration_ms=duration_ms,
        failed_steps=[r["job"] for r in results if r["status"] == "error"],
    )
    
    return summary
