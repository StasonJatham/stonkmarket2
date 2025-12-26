"""API usage tracking repository using SQLAlchemy ORM.

Tracks OpenAI and other API costs and batch job status.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import case, func, select

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import ApiUsage, BatchJob


logger = get_logger("repositories.api_usage")


async def record_usage(
    service: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    is_batch: bool = False,
    metadata: dict | None = None,
) -> int:
    """Record an API usage entry."""
    async with get_session() as session:
        usage = ApiUsage(
            service=service,
            endpoint=operation,
            model=metadata.get("model") if metadata else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=Decimal(str(cost_usd)) if cost_usd else None,
            is_batch=is_batch,
            request_metadata=metadata or {},
        )
        session.add(usage)
        await session.commit()
        await session.refresh(usage)
        return usage.id


async def get_usage_summary(days: int = 30) -> dict[str, Any]:
    """Get usage summary for the last N days."""
    cutoff = datetime.now(UTC) - timedelta(days=days)

    async with get_session() as session:
        result = await session.execute(
            select(
                ApiUsage.service,
                func.count().label("request_count"),
                func.coalesce(func.sum(ApiUsage.input_tokens), 0).label("total_input_tokens"),
                func.coalesce(func.sum(ApiUsage.output_tokens), 0).label("total_output_tokens"),
                func.coalesce(func.sum(ApiUsage.cost_usd), 0).label("total_cost_usd"),
                func.sum(case((ApiUsage.is_batch == True, 1), else_=0)).label("batch_requests"),
                func.sum(case((ApiUsage.is_batch == False, 1), else_=0)).label("realtime_requests"),
            )
            .where(ApiUsage.recorded_at > cutoff)
            .group_by(ApiUsage.service)
        )
        rows = result.all()

    by_service = {}
    total_cost = 0.0
    total_requests = 0

    for row in rows:
        service = row.service
        by_service[service] = {
            "request_count": row.request_count,
            "input_tokens": int(row.total_input_tokens or 0),
            "output_tokens": int(row.total_output_tokens or 0),
            "cost_usd": round(float(row.total_cost_usd or 0), 4),
            "batch_requests": int(row.batch_requests or 0),
            "realtime_requests": int(row.realtime_requests or 0),
        }
        total_cost += float(row.total_cost_usd or 0)
        total_requests += row.request_count

    return {
        "period_days": days,
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 4),
        "by_service": by_service,
    }


async def get_daily_costs(days: int = 30) -> list[dict[str, Any]]:
    """Get daily cost breakdown."""
    cutoff = datetime.now(UTC) - timedelta(days=days)

    async with get_session() as session:
        result = await session.execute(
            select(
                func.date(ApiUsage.recorded_at).label("date"),
                func.coalesce(func.sum(ApiUsage.cost_usd), 0).label("cost_usd"),
                func.count().label("request_count"),
                func.coalesce(
                    func.sum(ApiUsage.input_tokens + ApiUsage.output_tokens), 0
                ).label("total_tokens"),
            )
            .where(ApiUsage.recorded_at > cutoff)
            .group_by(func.date(ApiUsage.recorded_at))
            .order_by(func.date(ApiUsage.recorded_at).desc())
        )
        rows = result.all()

    return [
        {
            "date": str(row.date),
            "cost_usd": round(float(row.cost_usd or 0), 4),
            "request_count": row.request_count,
            "total_tokens": int(row.total_tokens or 0),
        }
        for row in rows
    ]


async def get_batch_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Get recent batch jobs."""
    async with get_session() as session:
        result = await session.execute(
            select(BatchJob)
            .order_by(BatchJob.created_at.desc())
            .limit(limit)
        )
        jobs = result.scalars().all()

    return [
        {
            "id": job.id,
            "batch_id": job.batch_id,
            "job_type": job.job_type,
            "status": job.status,
            "item_count": job.total_requests,
            "cost_usd": float(job.actual_cost_usd) if job.actual_cost_usd else None,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
            "error_message": None,  # Stored in errors relationship
        }
        for job in jobs
    ]


async def record_batch_job(
    batch_id: str,
    job_type: str,
    total_requests: int,
) -> int:
    """Record a new batch job."""
    async with get_session() as session:
        job = BatchJob(
            batch_id=batch_id,
            job_type=job_type,
            status="pending",
            total_requests=total_requests,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job.id


async def update_batch_job(
    batch_id: str,
    status: str,
    cost_usd: float | None = None,
    error_message: str | None = None,
) -> bool:
    """Update batch job status."""
    now = datetime.now(UTC)

    async with get_session() as session:
        result = await session.execute(
            select(BatchJob).where(BatchJob.batch_id == batch_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.status = status

        if status in ("completed", "failed"):
            job.completed_at = now

        if status == "completed" and cost_usd is not None:
            job.actual_cost_usd = Decimal(str(cost_usd))

        # Note: error_message handling would go to BatchTaskError if needed

        await session.commit()
        return True
