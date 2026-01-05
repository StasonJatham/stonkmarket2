"""Notification system jobs.

Periodic tasks for checking notification rules and sending alerts.
"""

from __future__ import annotations

import time

from app.core.logging import get_logger
from app.jobs.registry import register_job


logger = get_logger("jobs.notifications")


@register_job("notification_check")
async def notification_check_job() -> str:
    """
    Check all active notification rules and send triggered notifications.
    
    Evaluates all user-defined notification rules against current data
    and sends alerts via configured channels (Discord, Telegram, etc.).
    
    Features:
    - Batch-optimized data fetching (group by symbol/portfolio)
    - Cooldown enforcement (prevent spam)
    - Rate limiting (50 notifications/hour/user)
    - Data staleness protection (skip if data >24h old)
    - Content deduplication (avoid identical messages)
    
    Schedule: Every 5 minutes (*/5 * * * *)
    Time limits: 60s soft / 120s hard
    
    Returns:
        Summary of the check run
    """
    from app.services.notifications import check_all_rules
    
    job_start = time.monotonic()
    
    try:
        stats = await check_all_rules()
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        
        summary = (
            f"Checked {stats['rules_checked']} rules, "
            f"triggered {stats['rules_triggered']}, "
            f"sent {stats['notifications_sent']}, "
            f"skipped {stats['notifications_skipped']}, "
            f"failed {stats['notifications_failed']} "
            f"in {duration_ms}ms"
        )
        
        if stats.get("errors"):
            logger.warning(
                "Notification check had errors",
                extra={
                    "errors": stats["errors"][:5],  # Only log first 5
                    "total_errors": len(stats["errors"]),
                }
            )
        
        logger.info(
            "Notification check completed",
            extra={
                **stats,
                "duration_ms": duration_ms,
            }
        )
        
        return summary
        
    except Exception as e:
        duration_ms = int((time.monotonic() - job_start) * 1000)
        logger.exception("Notification check failed")
        return f"FAILED after {duration_ms}ms: {e}"


@register_job("notification_cleanup")
async def notification_cleanup_job() -> str:
    """
    Clean up old notification logs.
    
    Removes notification logs older than 30 days to prevent database bloat.
    Keeps recent logs for debugging and user visibility.
    
    Schedule: Daily at 3 AM UTC (0 3 * * *)
    Time limits: 120s soft / 300s hard
    
    Returns:
        Number of logs deleted
    """
    from datetime import timedelta
    from sqlalchemy import delete
    from app.database.connection import get_session
    from app.database.orm import NotificationLog
    
    job_start = time.monotonic()
    
    try:
        from datetime import datetime, UTC
        
        cutoff = datetime.now(UTC) - timedelta(days=30)
        
        async with get_session() as session:
            result = await session.execute(
                delete(NotificationLog).where(
                    NotificationLog.triggered_at < cutoff
                )
            )
            await session.commit()
            deleted = result.rowcount
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        
        logger.info(
            "Notification cleanup completed",
            extra={
                "deleted": deleted,
                "cutoff": cutoff.isoformat(),
                "duration_ms": duration_ms,
            }
        )
        
        return f"Deleted {deleted} old notification logs in {duration_ms}ms"
        
    except Exception as e:
        duration_ms = int((time.monotonic() - job_start) * 1000)
        logger.exception("Notification cleanup failed")
        return f"FAILED after {duration_ms}ms: {e}"
