"""
Celery Task Synchronization - Wait for async tasks to complete.

This module solves the problem of:
1. Test triggers API call
2. API queues Celery task
3. Test ends before task completes
4. Task fails silently
5. Test appears to pass

Solution: Wait for Celery queue to drain before declaring success.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docker.models.containers import Container


@dataclass
class TaskStatus:
    """Status of a Celery task."""

    task_id: str
    task_name: str
    status: str  # received, started, succeeded, failed, retried
    timestamp: datetime | None = None


class CeleryTaskTracker:
    """
    Track Celery task execution to ensure completion before test ends.

    Strategies:
    1. Monitor Celery worker logs for task completion
    2. Track received vs completed tasks
    3. Wait until queue is empty or timeout

    Usage:
        tracker = CeleryTaskTracker(worker_container, timeout=30)

        # Trigger action that queues Celery task
        response = client.post("/api/dipfinder/run", ...)

        # Wait for all tasks to complete
        tracker.wait_for_drain(timeout=30)

        # Verify no tasks failed
        tracker.assert_no_failed_tasks()
    """

    # Patterns to match Celery task log messages
    TASK_RECEIVED = re.compile(
        r"Task (\S+)\[([a-f0-9-]+)\] received"
    )
    TASK_STARTED = re.compile(
        r"Task (\S+)\[([a-f0-9-]+)\] started"
    )
    TASK_SUCCEEDED = re.compile(
        r"Task (\S+)\[([a-f0-9-]+)\] succeeded"
    )
    TASK_FAILED = re.compile(
        r"Task (\S+)\[([a-f0-9-]+)\] (failed|raised)"
    )
    TASK_RETRIED = re.compile(
        r"Task (\S+)\[([a-f0-9-]+)\] retry"
    )

    def __init__(
        self,
        worker_container: "Container",
        timeout: float = 30.0,
    ):
        self.worker = worker_container
        self.timeout = timeout
        self._mark_time: datetime | None = None

    def mark_start(self) -> None:
        """Mark the start time for task tracking."""
        self._mark_time = datetime.now(UTC)

    def wait_for_drain(self, timeout: float | None = None) -> None:
        """
        Wait for all pending Celery tasks to complete.

        Monitors worker logs for task completion messages.
        Raises TimeoutError if tasks don't complete in time.
        """
        timeout = timeout or self.timeout
        start = time.time()

        # Wait a moment for tasks to be received
        time.sleep(0.5)

        while (time.time() - start) < timeout:
            pending = self._get_pending_task_ids()

            if not pending:
                return  # All tasks completed

            time.sleep(0.5)

        # Timeout - get details about pending tasks
        pending = self._get_pending_task_ids()
        if pending:
            raise TimeoutError(
                f"Celery tasks did not complete within {timeout}s. "
                f"Pending task IDs: {list(pending)[:5]}"
            )

    def _get_logs(self, tail: int = 500) -> str:
        """Get recent worker logs."""
        try:
            return self.worker.logs(tail=tail).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _get_pending_task_ids(self) -> set[str]:
        """Get task IDs that were received but not completed."""
        logs = self._get_logs()

        received: set[str] = set()
        completed: set[str] = set()

        for match in self.TASK_RECEIVED.finditer(logs):
            received.add(match.group(2))

        for match in self.TASK_SUCCEEDED.finditer(logs):
            completed.add(match.group(2))

        for match in self.TASK_FAILED.finditer(logs):
            completed.add(match.group(2))

        return received - completed

    def get_completed_tasks(self) -> list[TaskStatus]:
        """Get all completed tasks from recent logs."""
        logs = self._get_logs()
        tasks = []

        for match in self.TASK_SUCCEEDED.finditer(logs):
            tasks.append(TaskStatus(
                task_id=match.group(2),
                task_name=match.group(1),
                status="succeeded",
            ))

        for match in self.TASK_FAILED.finditer(logs):
            tasks.append(TaskStatus(
                task_id=match.group(2),
                task_name=match.group(1),
                status="failed",
            ))

        return tasks

    def get_failed_tasks(self) -> list[TaskStatus]:
        """Get all failed tasks from recent logs."""
        logs = self._get_logs()
        tasks = []

        for match in self.TASK_FAILED.finditer(logs):
            tasks.append(TaskStatus(
                task_id=match.group(2),
                task_name=match.group(1),
                status="failed",
            ))

        return tasks

    def assert_no_failed_tasks(self) -> None:
        """Assert that no Celery tasks failed during the test window."""
        failed = self.get_failed_tasks()

        if failed:
            task_list = ", ".join(f"{t.task_name}[{t.task_id}]" for t in failed[:5])
            raise AssertionError(
                f"Celery tasks failed during test: {task_list}"
            )

    def get_task_count(self) -> dict[str, int]:
        """Get count of tasks by status."""
        logs = self._get_logs()

        return {
            "received": len(self.TASK_RECEIVED.findall(logs)),
            "started": len(self.TASK_STARTED.findall(logs)),
            "succeeded": len(self.TASK_SUCCEEDED.findall(logs)),
            "failed": len(self.TASK_FAILED.findall(logs)),
        }
