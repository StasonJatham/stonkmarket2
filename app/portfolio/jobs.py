"""Background processing for heavy portfolio analytics."""

from __future__ import annotations

from app.core.logging import get_logger
from app.portfolio.service import run_portfolio_tools
from app.repositories import portfolio_analytics_jobs_orm as jobs_repo

logger = get_logger("portfolio.jobs")


async def process_pending_jobs(limit: int = 3) -> int:
    """Process a limited number of queued analytics jobs."""
    processed = 0

    for _ in range(limit):
        job = await jobs_repo.claim_next_job()
        if not job:
            break

        try:
            results = await run_portfolio_tools(
                job["portfolio_id"],
                user_id=job["user_id"],
                tools=job["tools"],
                window=job.get("window"),
                start_date=job.get("start_date"),
                end_date=job.get("end_date"),
                benchmark=job.get("benchmark"),
                params=job.get("params") or {},
                force_refresh=job.get("force_refresh", False),
            )
            await jobs_repo.mark_job_completed(job["job_id"], len(results))
            processed += 1
        except Exception as exc:
            logger.exception("Portfolio analytics job failed", extra={"job_id": job.get("job_id")})
            await jobs_repo.mark_job_failed(job["job_id"], str(exc))

    return processed
