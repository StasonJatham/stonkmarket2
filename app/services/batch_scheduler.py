"""Batch job scheduler for AI analysis updates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from app.database.connection import fetch_all, execute
from app.services.openai_client import (
    submit_batch,
    check_batch,
    collect_batch,
    generate_bio,
    rate_dip,
    TaskType,
)
from app.repositories import api_usage
from app.core.logging import get_logger

logger = get_logger("batch_scheduler")


async def get_all_dip_symbols() -> list[str]:
    """Get all current dip symbols."""
    rows = await fetch_all("SELECT symbol FROM dip_state")
    return [r["symbol"] for r in rows]


async def get_pending_suggestions() -> list[dict]:
    """Get pending stock suggestions that need AI bios."""
    rows = await fetch_all(
        """
        SELECT id, symbol, company_name, reason
        FROM stock_suggestions
        WHERE status = 'pending'
        """
    )
    return [dict(r) for r in rows]


async def get_dips_needing_analysis() -> list[dict]:
    """Get dips that need AI analysis (new or expired), with fundamentals."""
    rows = await fetch_all(
        """
        SELECT ds.symbol, ds.current_price, ds.ath_price, ds.dip_percentage, ds.first_seen,
               sig.dip_class as classification, sig.excess_dip, sig.quality_score, sig.stability_score,
               s.name, s.sector, s.summary_ai,
               f.pe_ratio, f.forward_pe, f.peg_ratio, f.price_to_book, f.ev_to_ebitda,
               f.profit_margin, f.gross_margin, f.return_on_equity,
               f.revenue_growth, f.earnings_growth,
               f.debt_to_equity, f.current_ratio, f.free_cash_flow,
               f.recommendation, f.target_mean_price, f.num_analyst_opinions,
               f.beta, f.short_percent_of_float, f.held_percent_institutions
        FROM dip_state ds
        LEFT JOIN symbols s ON s.symbol = ds.symbol
        LEFT JOIN stock_fundamentals f ON ds.symbol = f.symbol
        LEFT JOIN LATERAL (
            SELECT dip_class, excess_dip, quality_score, stability_score
            FROM dipfinder_signals
            WHERE ticker = ds.symbol
            ORDER BY as_of_date DESC
            LIMIT 1
        ) sig ON true
        LEFT JOIN dip_ai_analysis ai ON ds.symbol = ai.symbol
        WHERE ai.symbol IS NULL 
           OR ai.expires_at < NOW()
           OR ai.generated_at < NOW() - INTERVAL '7 days'
        """
    )
    return [dict(r) for r in rows]


async def schedule_batch_dip_analysis() -> Optional[str]:
    """
    Schedule a batch job to analyze all current dips.

    Returns:
        Batch job ID if created, None if no dips to analyze
    """
    dips = await get_dips_needing_analysis()

    if not dips:
        logger.info("No dips need AI analysis")
        return None

    logger.info(f"Scheduling batch AI analysis for {len(dips)} dips")

    # Prepare items for batch - use context keys expected by _build_prompt
    items = []
    for dip in dips:
        # Calculate days in dip
        days_below = 0
        if dip.get("first_seen"):
            from datetime import timezone
            first_seen = dip["first_seen"]
            if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=timezone.utc)
            days_below = (datetime.now(timezone.utc) - first_seen).days
        
        # Helper to format percentages
        def fmt_pct(val):
            return f"{val * 100:.1f}%" if val else None
        
        def fmt_ratio(val):
            return f"{val:.2f}" if val else None
        
        def fmt_large_num(val):
            if val is None:
                return None
            if val >= 1e12:
                return f"${val / 1e12:.1f}T"
            if val >= 1e9:
                return f"${val / 1e9:.1f}B"
            if val >= 1e6:
                return f"${val / 1e6:.1f}M"
            return f"${val:,.0f}"
        
        items.append(
            {
                "symbol": dip["symbol"],
                "name": dip.get("name"),
                "sector": dip.get("sector"),
                "summary": dip.get("summary_ai"),
                "current_price": float(dip["current_price"])
                if dip["current_price"]
                else None,
                "ref_high": float(dip["ath_price"]) if dip["ath_price"] else None,
                "dip_pct": float(dip["dip_percentage"])
                if dip["dip_percentage"]
                else None,
                "days_below": days_below,
                # Dip classification and scores
                "dip_classification": dip.get("classification"),
                "excess_dip": float(dip["excess_dip"]) if dip.get("excess_dip") else None,
                "quality_score": float(dip["quality_score"]) if dip.get("quality_score") else None,
                "stability_score": float(dip["stability_score"]) if dip.get("stability_score") else None,
                # Fundamentals from database
                "pe_ratio": float(dip["pe_ratio"]) if dip.get("pe_ratio") else None,
                "forward_pe": float(dip["forward_pe"]) if dip.get("forward_pe") else None,
                "peg_ratio": fmt_ratio(dip.get("peg_ratio")),
                "price_to_book": fmt_ratio(dip.get("price_to_book")),
                "ev_to_ebitda": fmt_ratio(dip.get("ev_to_ebitda")),
                "profit_margin": fmt_pct(dip.get("profit_margin")),
                "gross_margin": fmt_pct(dip.get("gross_margin")),
                "return_on_equity": fmt_pct(dip.get("return_on_equity")),
                "revenue_growth": fmt_pct(dip.get("revenue_growth")),
                "earnings_growth": fmt_pct(dip.get("earnings_growth")),
                "debt_to_equity": fmt_ratio(dip.get("debt_to_equity")),
                "current_ratio": fmt_ratio(dip.get("current_ratio")),
                "free_cash_flow": fmt_large_num(dip.get("free_cash_flow")),
                "recommendation": dip.get("recommendation"),
                "target_mean_price": float(dip["target_mean_price"]) if dip.get("target_mean_price") else None,
                "num_analyst_opinions": dip.get("num_analyst_opinions"),
                "beta": fmt_ratio(dip.get("beta")),
                "short_percent_of_float": fmt_pct(dip.get("short_percent_of_float")),
                "institutional_ownership": fmt_pct(dip.get("held_percent_institutions")),
            }
        )

    # Submit batch job for RATING (includes reasoning)
    batch_id = await submit_batch(
        task=TaskType.RATING,
        items=items,
    )

    if batch_id:
        # Record batch job
        await api_usage.record_batch_job(
            batch_id=batch_id,
            job_type=TaskType.RATING.value,
            total_requests=len(items),
        )

        logger.info(f"Created batch job {batch_id} for {len(dips)} dips")

    return batch_id


async def schedule_batch_suggestion_bios() -> Optional[str]:
    """
    Schedule a batch job to generate bios for pending suggestions.

    Returns:
        Batch job ID if created, None if no suggestions
    """
    suggestions = await get_pending_suggestions()

    if not suggestions:
        logger.info("No pending suggestions need bios")
        return None

    logger.info(f"Scheduling batch AI bios for {len(suggestions)} suggestions")

    # Prepare items for batch
    items = []
    for suggestion in suggestions:
        items.append(
            {
                "symbol": suggestion["symbol"],
                "name": suggestion.get("company_name"),
            }
        )

    # Submit batch job
    batch_id = await submit_batch(
        task=TaskType.BIO,
        items=items,
    )

    if batch_id:
        # Record batch job
        await api_usage.record_batch_job(
            batch_id=batch_id,
            job_type=TaskType.BIO.value,
            total_requests=len(items),
        )

        logger.info(f"Created batch job {batch_id} for {len(suggestions)} suggestions")

    return batch_id


# Alias for swipe bios
schedule_batch_swipe_bios = schedule_batch_suggestion_bios


async def process_completed_batch_jobs() -> int:
    """
    Check for and process completed batch jobs.

    Returns:
        Number of jobs processed
    """
    # Get pending batch jobs
    rows = await fetch_all(
        """
        SELECT batch_id, job_type, status
        FROM batch_jobs
        WHERE status IN ('pending', 'validating', 'in_progress', 'finalizing')
        """
    )

    processed = 0

    for row in rows:
        batch_id = row["batch_id"]
        job_type = row["job_type"]

        # Check status
        status_info = await check_batch(batch_id)

        if not status_info:
            continue

        new_status = status_info.get("status", "unknown")

        # Update status in database
        await execute(
            """
            UPDATE batch_jobs
            SET status = $1,
                completed_requests = $2,
                failed_requests = $3,
                output_file_id = $4,
                error_file_id = $5
            WHERE batch_id = $6
            """,
            new_status,
            status_info.get("completed_count", 0),
            status_info.get("failed_count", 0),
            status_info.get("output_file_id"),
            status_info.get("error_file_id"),
            batch_id,
        )

        # If completed, retrieve and process results
        if new_status == "completed":
            results = await collect_batch(batch_id)

            if results:
                await _process_batch_results(batch_id, job_type, results)
                processed += 1

                # Update completion time and cost
                await execute(
                    """
                    UPDATE batch_jobs
                    SET completed_at = NOW(),
                        actual_cost_usd = $1
                    WHERE batch_id = $2
                    """,
                    status_info.get("total_cost", 0),
                    batch_id,
                )

    return processed


async def _process_batch_results(
    batch_id: str,
    job_type: str,
    results: list[dict],
) -> None:
    """Process results from a completed batch job."""
    logger.info(f"Processing {len(results)} results from batch {batch_id}")

    for result in results:
        custom_id = result.get("custom_id", "")
        # Results from new collect_batch have "result" directly
        content = result.get("result", "")
        symbol = result.get("symbol", "")

        try:
            if job_type == TaskType.RATING.value:
                # RATING batch returns parsed JSON with rating/reasoning/confidence
                if isinstance(content, dict) and not result.get("failed"):
                    await _store_dip_analysis(symbol, content, batch_id)
                else:
                    logger.warning(f"Skipping failed RATING result for {symbol}: {result.get('error')}")

            elif job_type == TaskType.BIO.value:
                # BIO batch returns text
                if content and not result.get("failed"):
                    await _store_suggestion_bio(symbol, content)

        except Exception as e:
            logger.error(f"Error processing batch result for {custom_id}: {e}")


async def _store_dip_analysis(symbol: str, content: dict | str, batch_id: str) -> None:
    """Store AI rating analysis for a dip."""
    # Handle both dict (from RATING batch) and legacy string formats
    if isinstance(content, dict):
        rating = content.get("rating")  # String like "buy", "hold", "sell"
        reasoning = content.get("reasoning", "")
        # Note: Bio is not included in RATING - generate separately if needed
    else:
        # Legacy: plain text (shouldn't happen with RATING task)
        rating = None
        reasoning = str(content)

    expires_at = datetime.utcnow() + timedelta(days=7)

    await execute(
        """
        INSERT INTO dip_ai_analysis (
            symbol, ai_rating, rating_reasoning,
            model_used, is_batch_generated, batch_job_id, generated_at, expires_at
        )
        VALUES ($1, $2, $3, 'gpt-5-mini', TRUE, $4, NOW(), $5)
        ON CONFLICT (symbol) DO UPDATE SET
            ai_rating = EXCLUDED.ai_rating,
            rating_reasoning = EXCLUDED.rating_reasoning,
            is_batch_generated = TRUE,
            batch_job_id = EXCLUDED.batch_job_id,
            generated_at = NOW(),
            expires_at = EXCLUDED.expires_at
        """,
        symbol.upper(),
        rating,
        reasoning,
        batch_id,
        expires_at,
    )

    logger.debug(f"Stored AI rating for {symbol}: {rating}")


async def _store_suggestion_bio(symbol: str, bio: str) -> None:
    """Store AI bio for a suggestion."""
    await execute(
        """
        UPDATE stock_suggestions
        SET ai_bio = $1, updated_at = NOW()
        WHERE symbol = $2
        """,
        bio,
        symbol.upper(),
    )

    logger.debug(f"Stored AI bio for suggestion {symbol}")


async def run_realtime_analysis_for_new_stock(
    symbol: str,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    dip_percentage: Optional[float] = None,
) -> dict:
    """
    Run real-time AI analysis for a newly added stock.

    Called when a stock is approved from suggestions and added to dips.
    """
    from app.services import stock_info
    
    logger.info(f"Running real-time AI analysis for new stock: {symbol}")

    try:
        # Fetch stock info for context
        info = await stock_info.get_stock_info_async(symbol)
        name = info.get("name") if info else None
        sector = info.get("sector") if info else None
        summary = info.get("summary") if info else None
        pe_ratio = info.get("pe_ratio") if info else None
        
        # Generate bio
        bio = await generate_bio(
            symbol=symbol,
            name=name,
            sector=sector,
            summary=summary,
            dip_pct=dip_percentage,
        )

        # Get rating
        rating_data = await rate_dip(
            symbol=symbol,
            current_price=current_price,
            ref_high=ath_price,
            dip_pct=dip_percentage,
            days_below=0,  # New stock
            name=name,
            sector=sector,
            summary=summary,
            pe_ratio=pe_ratio,
        )

        # Store the analysis
        expires_at = datetime.utcnow() + timedelta(days=7)

        await execute(
            """
            INSERT INTO dip_ai_analysis (
                symbol, swipe_bio, ai_rating, rating_reasoning,
                model_used, is_batch_generated, generated_at, expires_at
            )
            VALUES ($1, $2, $3, $4, 'gpt-5-mini', FALSE, NOW(), $5)
            ON CONFLICT (symbol) DO UPDATE SET
                swipe_bio = EXCLUDED.swipe_bio,
                ai_rating = EXCLUDED.ai_rating,
                rating_reasoning = EXCLUDED.rating_reasoning,
                is_batch_generated = FALSE,
                generated_at = NOW(),
                expires_at = EXCLUDED.expires_at
            """,
            symbol.upper(),
            bio,
            rating_data.get("rating") if rating_data else None,  # String rating like 'buy', 'hold'
            rating_data.get("reasoning", "") if rating_data else "",
            expires_at,
        )

        logger.info(f"Completed real-time analysis for {symbol}")

        return {
            "symbol": symbol,
            "bio": bio,
            "rating": rating_data.get("rating") if rating_data else None,
            "reasoning": rating_data.get("reasoning", "") if rating_data else "",
        }

    except Exception as e:
        logger.error(f"Failed to run real-time analysis for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


# ============================================================================
# Cron Job Functions (called by scheduler)
# ============================================================================


async def cron_batch_ai_dips() -> dict:
    """
    Cron job: Schedule weekly batch AI analysis for all dips.

    Runs Sunday 3 AM UTC by default.
    """
    logger.info("Running weekly batch AI analysis for dips")

    batch_id = await schedule_batch_dip_analysis()

    return {
        "job": "batch_ai_dips",
        "batch_id": batch_id,
        "status": "scheduled" if batch_id else "no_dips",
    }


async def cron_batch_ai_suggestions() -> dict:
    """
    Cron job: Schedule weekly batch AI bios for suggestions.

    Runs Sunday 4 AM UTC by default.
    """
    logger.info("Running weekly batch AI bios for suggestions")

    batch_id = await schedule_batch_suggestion_bios()

    return {
        "job": "batch_ai_suggestions",
        "batch_id": batch_id,
        "status": "scheduled" if batch_id else "no_suggestions",
    }


async def cron_sync_batch_jobs() -> dict:
    """
    Cron job: Check and process completed batch jobs.

    Runs every 15 minutes.
    """
    logger.info("Syncing batch job statuses")

    processed = await process_completed_batch_jobs()

    return {
        "job": "sync_batch_jobs",
        "processed": processed,
    }


async def cron_cleanup_expired() -> dict:
    """
    Cron job: Clean up expired data.

    Runs daily at midnight.
    """
    from app.repositories.dip_history import cleanup_old_history

    logger.info("Running daily cleanup")

    # Clean up old dip history (keep 90 days)
    history_deleted = await cleanup_old_history(days=90)

    # Clean up old rate limit entries
    await execute("SELECT cleanup_rate_limits()")

    return {
        "job": "cleanup_expired",
        "history_deleted": history_deleted,
    }
