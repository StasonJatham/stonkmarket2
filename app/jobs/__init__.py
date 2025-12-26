"""Background job utilities."""
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
    """No-op reschedule hook (Celery beat polls DB for changes)."""
    return True


__all__ = [
    "JobRegistry",
    "register_job",
    "get_job",
    "execute_job",
    "execute_job_with_retry",
    "enqueue_job",
    "get_task_status",
    "reschedule_job",
]
