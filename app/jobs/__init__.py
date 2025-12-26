"""Background job utilities."""
from .dispatch import (
    enqueue_job,
    get_task_status,
)
from .executor import (
    execute_job,
    execute_job_with_retry,
)
from .registry import (
    JobRegistry,
    get_job,
    register_job,
)


async def reschedule_job(name: str, _cron_expression: str) -> bool:
    """No-op reschedule hook (Celery beat polls DB for changes)."""
    return True


__all__ = [
    "JobRegistry",
    "enqueue_job",
    "execute_job",
    "execute_job_with_retry",
    "get_job",
    "get_task_status",
    "register_job",
    "reschedule_job",
]
