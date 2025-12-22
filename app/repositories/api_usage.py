"""API usage tracking repository for OpenAI costs - PostgreSQL async."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional, Any

from app.database.connection import fetch_one, fetch_all, execute


async def record_usage(
    service: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    is_batch: bool = False,
    metadata: Optional[dict] = None,
) -> int:
    """Record an API usage entry."""
    result = await execute(
        """
        INSERT INTO api_usage (service, operation, input_tokens, output_tokens, cost_usd, is_batch, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        RETURNING id
        """,
        service,
        operation,
        input_tokens,
        output_tokens,
        cost_usd,
        is_batch,
        json.dumps(metadata or {}),
    )
    # Parse "INSERT 0 1" or similar
    return 0


async def get_usage_summary(
    days: int = 30,
) -> dict[str, Any]:
    """Get usage summary for the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    rows = await fetch_all(
        """
        SELECT 
            service,
            COUNT(*) as request_count,
            COALESCE(SUM(input_tokens), 0) as total_input_tokens,
            COALESCE(SUM(output_tokens), 0) as total_output_tokens,
            COALESCE(SUM(cost_usd), 0) as total_cost_usd,
            SUM(CASE WHEN is_batch THEN 1 ELSE 0 END) as batch_requests,
            SUM(CASE WHEN NOT is_batch THEN 1 ELSE 0 END) as realtime_requests
        FROM api_usage
        WHERE created_at > $1
        GROUP BY service
        """,
        cutoff,
    )

    by_service = {}
    total_cost = 0.0
    total_requests = 0

    for row in rows:
        service = row["service"]
        by_service[service] = {
            "request_count": row["request_count"],
            "input_tokens": row["total_input_tokens"] or 0,
            "output_tokens": row["total_output_tokens"] or 0,
            "cost_usd": round(float(row["total_cost_usd"] or 0), 4),
            "batch_requests": row["batch_requests"],
            "realtime_requests": row["realtime_requests"],
        }
        total_cost += float(row["total_cost_usd"] or 0)
        total_requests += row["request_count"]

    return {
        "period_days": days,
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 4),
        "by_service": by_service,
    }


async def get_daily_costs(
    days: int = 30,
) -> list[dict[str, Any]]:
    """Get daily cost breakdown."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    rows = await fetch_all(
        """
        SELECT 
            DATE(created_at) as date,
            COALESCE(SUM(cost_usd), 0) as cost_usd,
            COUNT(*) as request_count,
            COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens
        FROM api_usage
        WHERE created_at > $1
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        """,
        cutoff,
    )

    return [
        {
            "date": str(row["date"]),
            "cost_usd": round(float(row["cost_usd"] or 0), 4),
            "request_count": row["request_count"],
            "total_tokens": row["total_tokens"] or 0,
        }
        for row in rows
    ]


async def get_batch_jobs(
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get recent batch jobs."""
    rows = await fetch_all(
        """
        SELECT id, batch_id, job_type, status, item_count, cost_usd, created_at, completed_at, error_message
        FROM batch_jobs
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )

    return [dict(row) for row in rows]


async def record_batch_job(
    batch_id: str,
    job_type: str,
    total_requests: int,
) -> int:
    """Record a new batch job."""
    await execute(
        """
        INSERT INTO batch_jobs (batch_id, job_type, status, item_count, created_at)
        VALUES ($1, $2, 'pending', $3, NOW())
        """,
        batch_id, job_type, total_requests,
    )
    return 0


async def update_batch_job(
    batch_id: str,
    status: str,
    cost_usd: Optional[float] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Update batch job status."""
    if status == "completed":
        await execute(
            """
            UPDATE batch_jobs 
            SET status = $1, cost_usd = $2, completed_at = NOW()
            WHERE batch_id = $3
            """,
            status, cost_usd, batch_id,
        )
    elif status == "failed":
        await execute(
            """
            UPDATE batch_jobs 
            SET status = $1, error_message = $2, completed_at = NOW()
            WHERE batch_id = $3
            """,
            status, error_message, batch_id,
        )
    else:
        await execute(
            """
            UPDATE batch_jobs SET status = $1 WHERE batch_id = $2
            """,
            status, batch_id,
        )
    return True
