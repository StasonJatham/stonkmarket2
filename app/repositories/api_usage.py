"""API usage tracking repository for OpenAI costs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Any


def record_usage(
    conn: sqlite3.Connection,
    service: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    is_batch: bool = False,
    metadata: Optional[dict] = None,
) -> int:
    """Record an API usage entry."""
    import json
    
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO api_usage (service, operation, input_tokens, output_tokens, cost_usd, is_batch, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (service, operation, input_tokens, output_tokens, cost_usd, int(is_batch), json.dumps(metadata or {}), now),
    )
    conn.commit()
    return cur.lastrowid or 0


def get_usage_summary(
    conn: sqlite3.Connection,
    days: int = 30,
) -> dict[str, Any]:
    """Get usage summary for the last N days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    cur = conn.execute(
        """
        SELECT 
            service,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(cost_usd) as total_cost_usd,
            SUM(CASE WHEN is_batch = 1 THEN 1 ELSE 0 END) as batch_requests,
            SUM(CASE WHEN is_batch = 0 THEN 1 ELSE 0 END) as realtime_requests
        FROM api_usage
        WHERE created_at > ?
        GROUP BY service
        """,
        (cutoff,),
    )
    
    by_service = {}
    total_cost = 0.0
    total_requests = 0
    
    for row in cur.fetchall():
        service = row["service"]
        by_service[service] = {
            "request_count": row["request_count"],
            "input_tokens": row["total_input_tokens"] or 0,
            "output_tokens": row["total_output_tokens"] or 0,
            "cost_usd": round(row["total_cost_usd"] or 0, 4),
            "batch_requests": row["batch_requests"],
            "realtime_requests": row["realtime_requests"],
        }
        total_cost += row["total_cost_usd"] or 0
        total_requests += row["request_count"]
    
    return {
        "period_days": days,
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 4),
        "by_service": by_service,
    }


def get_daily_costs(
    conn: sqlite3.Connection,
    days: int = 30,
) -> list[dict[str, Any]]:
    """Get daily cost breakdown."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    cur = conn.execute(
        """
        SELECT 
            DATE(created_at) as date,
            SUM(cost_usd) as cost_usd,
            COUNT(*) as request_count,
            SUM(input_tokens + output_tokens) as total_tokens
        FROM api_usage
        WHERE created_at > ?
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        """,
        (cutoff,),
    )
    
    return [
        {
            "date": row["date"],
            "cost_usd": round(row["cost_usd"] or 0, 4),
            "request_count": row["request_count"],
            "total_tokens": row["total_tokens"] or 0,
        }
        for row in cur.fetchall()
    ]


def get_batch_jobs(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get recent batch jobs."""
    cur = conn.execute(
        """
        SELECT id, batch_id, job_type, status, item_count, cost_usd, created_at, completed_at, error_message
        FROM batch_jobs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    
    return [dict(row) for row in cur.fetchall()]


def record_batch_job(
    conn: sqlite3.Connection,
    batch_id: str,
    job_type: str,
    item_count: int,
) -> int:
    """Record a new batch job."""
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        INSERT INTO batch_jobs (batch_id, job_type, status, item_count, created_at)
        VALUES (?, ?, 'pending', ?, ?)
        """,
        (batch_id, job_type, item_count, now),
    )
    conn.commit()
    return cur.lastrowid or 0


def update_batch_job(
    conn: sqlite3.Connection,
    batch_id: str,
    status: str,
    cost_usd: Optional[float] = None,
    error_message: Optional[str] = None,
) -> bool:
    """Update batch job status."""
    now = datetime.utcnow().isoformat()
    
    if status == "completed":
        cur = conn.execute(
            """
            UPDATE batch_jobs 
            SET status = ?, cost_usd = ?, completed_at = ?
            WHERE batch_id = ?
            """,
            (status, cost_usd, now, batch_id),
        )
    elif error_message:
        cur = conn.execute(
            """
            UPDATE batch_jobs 
            SET status = ?, error_message = ?, completed_at = ?
            WHERE batch_id = ?
            """,
            (status, error_message, now, batch_id),
        )
    else:
        cur = conn.execute(
            "UPDATE batch_jobs SET status = ? WHERE batch_id = ?",
            (status, batch_id),
        )
    
    conn.commit()
    return cur.rowcount > 0
