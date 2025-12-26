"""Repository for portfolio analytics job queue - SQLAlchemy ORM version."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import and_, select, update

from app.database.connection import get_session
from app.database.orm import PortfolioAnalyticsJob


def _normalize_tools(tools: list[str]) -> list[str]:
    return sorted({t.strip().lower() for t in tools if t and t.strip()})


def _job_to_dict(job: PortfolioAnalyticsJob) -> dict[str, Any]:
    """Convert ORM model to dictionary."""
    return {
        "id": job.id,
        "job_id": job.job_id,
        "portfolio_id": job.portfolio_id,
        "user_id": job.user_id,
        "status": job.status,
        "tools": job.tools,
        "params": job.params,
        "window": job.window,
        "start_date": job.start_date,
        "end_date": job.end_date,
        "benchmark": job.benchmark,
        "force_refresh": job.force_refresh,
        "results_count": job.results_count,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


async def create_job(
    portfolio_id: int,
    user_id: int,
    *,
    tools: list[str],
    window: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    benchmark: str | None = None,
    params: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Create or reuse a pending analytics job."""
    normalized_tools = _normalize_tools(tools)

    async with get_session() as session:
        # Check for existing pending/running job with same parameters
        stmt = select(PortfolioAnalyticsJob).where(
            and_(
                PortfolioAnalyticsJob.portfolio_id == portfolio_id,
                PortfolioAnalyticsJob.status.in_(["pending", "running"]),
                PortfolioAnalyticsJob.tools == normalized_tools,
                PortfolioAnalyticsJob.window == window if window else PortfolioAnalyticsJob.window.is_(None),
                PortfolioAnalyticsJob.start_date == start_date if start_date else PortfolioAnalyticsJob.start_date.is_(None),
                PortfolioAnalyticsJob.end_date == end_date if end_date else PortfolioAnalyticsJob.end_date.is_(None),
                PortfolioAnalyticsJob.benchmark == benchmark if benchmark else PortfolioAnalyticsJob.benchmark.is_(None),
                PortfolioAnalyticsJob.params == params if params else PortfolioAnalyticsJob.params.is_(None),
                PortfolioAnalyticsJob.force_refresh == force_refresh,
            )
        ).order_by(PortfolioAnalyticsJob.created_at.desc()).limit(1)

        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            return _job_to_dict(existing)

        # Create new job
        job = PortfolioAnalyticsJob(
            job_id=uuid.uuid4().hex[:16],
            portfolio_id=portfolio_id,
            user_id=user_id,
            status="pending",
            tools=normalized_tools,
            params=params,
            window=window,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            force_refresh=force_refresh,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return _job_to_dict(job)


async def list_jobs(
    user_id: int,
    *,
    portfolio_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List portfolio analytics jobs for a user."""
    async with get_session() as session:
        stmt = select(PortfolioAnalyticsJob).where(
            PortfolioAnalyticsJob.user_id == user_id
        )

        if portfolio_id is not None:
            stmt = stmt.where(PortfolioAnalyticsJob.portfolio_id == portfolio_id)
        if status is not None:
            stmt = stmt.where(PortfolioAnalyticsJob.status == status)

        stmt = stmt.order_by(PortfolioAnalyticsJob.created_at.desc()).limit(limit)

        result = await session.execute(stmt)
        jobs = result.scalars().all()
        return [_job_to_dict(job) for job in jobs]


async def get_job(job_id: str, user_id: int) -> dict[str, Any] | None:
    """Fetch a single analytics job."""
    async with get_session() as session:
        stmt = select(PortfolioAnalyticsJob).where(
            and_(
                PortfolioAnalyticsJob.job_id == job_id,
                PortfolioAnalyticsJob.user_id == user_id,
            )
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        return _job_to_dict(job) if job else None


async def claim_next_job() -> dict[str, Any] | None:
    """Claim the next pending analytics job for processing.
    
    Uses FOR UPDATE SKIP LOCKED for safe concurrent access.
    """
    async with get_session() as session:
        # Select next pending job with row-level lock
        stmt = (
            select(PortfolioAnalyticsJob)
            .where(PortfolioAnalyticsJob.status == "pending")
            .order_by(PortfolioAnalyticsJob.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return None

        # Update status to running
        job.status = "running"
        job.started_at = datetime.now(UTC)
        job.error_message = None

        await session.commit()
        await session.refresh(job)
        return _job_to_dict(job)


async def mark_job_completed(job_id: str, results_count: int) -> None:
    """Mark analytics job as completed."""
    async with get_session() as session:
        stmt = (
            update(PortfolioAnalyticsJob)
            .where(PortfolioAnalyticsJob.job_id == job_id)
            .values(
                status="completed",
                completed_at=datetime.now(UTC),
                results_count=results_count,
            )
        )
        await session.execute(stmt)
        await session.commit()


async def mark_job_failed(job_id: str, error_message: str) -> None:
    """Mark analytics job as failed."""
    async with get_session() as session:
        stmt = (
            update(PortfolioAnalyticsJob)
            .where(PortfolioAnalyticsJob.job_id == job_id)
            .values(
                status="failed",
                completed_at=datetime.now(UTC),
                error_message=error_message[:1000],
            )
        )
        await session.execute(stmt)
        await session.commit()
