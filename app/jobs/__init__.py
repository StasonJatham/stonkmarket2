"""Background job scheduler with distributed locking."""

from .scheduler import (
    JobScheduler,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)
from .registry import (
    JobRegistry,
    register_job,
    get_job,
)
from .executor import (
    execute_job,
    execute_job_with_retry,
)
from .dispatch import (
    enqueue_job,
    get_task_status,
)


async def reschedule_job(name: str, cron_expression: str) -> bool:
    """Reschedule a running job with a new cron expression."""
    scheduler = get_scheduler()
    if scheduler is None:
        return False
    return await scheduler.reschedule_job(name, cron_expression)


__all__ = [
    "JobScheduler",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
    "JobRegistry",
    "register_job",
    "get_job",
    "execute_job",
    "execute_job_with_retry",
    "enqueue_job",
    "get_task_status",
    "reschedule_job",
]
