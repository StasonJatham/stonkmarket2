"""Repository for portfolio analytics job queue."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Optional

from app.database.connection import execute, fetch_all, fetch_one, transaction


def _normalize_tools(tools: list[str]) -> list[str]:
    return sorted({t.strip().lower() for t in tools if t and t.strip()})


async def create_job(
    portfolio_id: int,
    user_id: int,
    *,
    tools: list[str],
    window: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    benchmark: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Create or reuse a pending analytics job."""
    normalized_tools = _normalize_tools(tools)

    existing = await fetch_one(
        """
        SELECT id, job_id, portfolio_id, user_id, status, tools, params, window,
               start_date, end_date, benchmark, force_refresh, results_count,
               error_message, created_at, started_at, completed_at
        FROM portfolio_analytics_jobs
        WHERE portfolio_id = $1
          AND status IN ('pending', 'running')
          AND tools = $2
          AND window IS NOT DISTINCT FROM $3
          AND start_date IS NOT DISTINCT FROM $4
          AND end_date IS NOT DISTINCT FROM $5
          AND benchmark IS NOT DISTINCT FROM $6
          AND params IS NOT DISTINCT FROM $7
          AND force_refresh = $8
        ORDER BY created_at DESC
        LIMIT 1
        """,
        portfolio_id,
        normalized_tools,
        window,
        start_date,
        end_date,
        benchmark,
        params,
        force_refresh,
    )
    if existing:
        return dict(existing)

    job_id = uuid.uuid4().hex[:16]
    row = await fetch_one(
        """
        INSERT INTO portfolio_analytics_jobs (
            job_id, portfolio_id, user_id, status, tools, params, window,
            start_date, end_date, benchmark, force_refresh
        )
        VALUES ($1, $2, $3, 'pending', $4, $5, $6, $7, $8, $9, $10)
        RETURNING id, job_id, portfolio_id, user_id, status, tools, params, window,
                  start_date, end_date, benchmark, force_refresh, results_count,
                  error_message, created_at, started_at, completed_at
        """,
        job_id,
        portfolio_id,
        user_id,
        normalized_tools,
        params,
        window,
        start_date,
        end_date,
        benchmark,
        force_refresh,
    )
    return dict(row) if row else {}


async def list_jobs(
    user_id: int,
    *,
    portfolio_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List portfolio analytics jobs for a user."""
    rows = await fetch_all(
        """
        SELECT id, job_id, portfolio_id, user_id, status, tools, params, window,
               start_date, end_date, benchmark, force_refresh, results_count,
               error_message, created_at, started_at, completed_at
        FROM portfolio_analytics_jobs
        WHERE user_id = $1
          AND ($2::int IS NULL OR portfolio_id = $2)
          AND ($3::text IS NULL OR status = $3)
        ORDER BY created_at DESC
        LIMIT $4
        """,
        user_id,
        portfolio_id,
        status,
        limit,
    )
    return [dict(r) for r in rows]


async def get_job(job_id: str, user_id: int) -> Optional[dict[str, Any]]:
    """Fetch a single analytics job."""
    row = await fetch_one(
        """
        SELECT id, job_id, portfolio_id, user_id, status, tools, params, window,
               start_date, end_date, benchmark, force_refresh, results_count,
               error_message, created_at, started_at, completed_at
        FROM portfolio_analytics_jobs
        WHERE job_id = $1 AND user_id = $2
        """,
        job_id,
        user_id,
    )
    return dict(row) if row else None


async def claim_next_job() -> Optional[dict[str, Any]]:
    """Claim the next pending analytics job for processing."""
    async with transaction() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, job_id, portfolio_id, user_id, tools, params, window,
                   start_date, end_date, benchmark, force_refresh
            FROM portfolio_analytics_jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        )
        if not row:
            return None
        await conn.execute(
            """
            UPDATE portfolio_analytics_jobs
            SET status = 'running', started_at = NOW(), error_message = NULL
            WHERE id = $1
            """,
            row["id"],
        )
        updated = await conn.fetchrow(
            """
            SELECT id, job_id, portfolio_id, user_id, status, tools, params, window,
                   start_date, end_date, benchmark, force_refresh, results_count,
                   error_message, created_at, started_at, completed_at
            FROM portfolio_analytics_jobs
            WHERE id = $1
            """,
            row["id"],
        )
        return dict(updated) if updated else None


async def mark_job_completed(job_id: str, results_count: int) -> None:
    """Mark analytics job as completed."""
    await execute(
        """
        UPDATE portfolio_analytics_jobs
        SET status = 'completed', completed_at = NOW(), results_count = $2
        WHERE job_id = $1
        """,
        job_id,
        results_count,
    )


async def mark_job_failed(job_id: str, error_message: str) -> None:
    """Mark analytics job as failed."""
    await execute(
        """
        UPDATE portfolio_analytics_jobs
        SET status = 'failed', completed_at = NOW(), error_message = $2
        WHERE job_id = $1
        """,
        job_id,
        error_message[:1000],
    )
