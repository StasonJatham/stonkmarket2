"""
Docker Log Watcher - Captures and analyzes container logs.

This module provides real-time log monitoring with pattern matching
to detect errors, warnings, and tracebacks across all containers.

Architecture:
1. mark_positions() - Called BEFORE test, records current log position
2. Test executes (API calls, UI actions, etc.)
3. capture_since_mark() - Called AFTER test, captures new logs
4. Analyze logs for errors/warnings/tracebacks
5. Fail test if any issues found
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docker.models.containers import Container

from system_tests.config import SystemTestConfig


@dataclass
class LogPosition:
    """Marker for a position in a container's log stream."""

    container_id: str
    timestamp: datetime
    # Last N log lines for deduplication
    tail_lines: list[str] = field(default_factory=list)


@dataclass
class LogCapture:
    """Logs captured between two positions with analysis."""

    container: str
    logs: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tracebacks: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """Check if any issues were found."""
        return bool(self.errors or self.tracebacks)

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return bool(self.warnings)


class DockerLogWatcher:
    """
    Watches Docker container logs for errors during test execution.

    Usage:
        watcher = DockerLogWatcher(containers, config)
        watcher.mark_positions()  # Before test
        # ... test runs ...
        captures = watcher.capture_since_mark()  # After test
        issues = watcher.get_issues(captures)
        if issues:
            pytest.fail(issues)
    """

    def __init__(
        self,
        containers: dict[str, "Container"],
        config: SystemTestConfig,
    ):
        self.containers = containers
        self.config = config
        self._positions: dict[str, LogPosition] = {}

        # Compile regex patterns for efficiency
        self._error_patterns = [
            re.compile(p, re.IGNORECASE) for p in config.error_patterns
        ]
        self._warning_patterns = [
            re.compile(p, re.IGNORECASE) for p in config.warning_patterns
        ]
        self._ignore_patterns = [
            re.compile(p, re.IGNORECASE) for p in config.ignore_patterns
        ]
        self._traceback_start = re.compile(r"Traceback \(most recent call last\)")
        self._exception_line = re.compile(r"^\w+Error:|^\w+Exception:")

    def mark_positions(self) -> None:
        """
        Mark current position in all container logs.

        Called BEFORE each test to establish baseline.
        """
        self._positions.clear()
        now = datetime.now(UTC)

        for name, container in self.containers.items():
            try:
                # Get last N lines for deduplication
                tail = container.logs(
                    tail=self.config.log_buffer_lines,
                    timestamps=True,
                ).decode("utf-8", errors="replace")
                tail_lines = tail.strip().split("\n") if tail.strip() else []
            except Exception:
                tail_lines = []

            self._positions[name] = LogPosition(
                container_id=container.id,
                timestamp=now,
                tail_lines=tail_lines[-50:],  # Keep last 50 for dedup
            )

    def capture_since_mark(self) -> list[LogCapture]:
        """
        Capture all logs since the last mark.

        Called AFTER each test to analyze what happened.
        """
        captures = []

        for name, container in self.containers.items():
            position = self._positions.get(name)
            if not position:
                continue

            try:
                # Get logs since the mark timestamp
                raw_logs = container.logs(
                    since=position.timestamp,
                    timestamps=True,
                ).decode("utf-8", errors="replace")
            except Exception as e:
                raw_logs = f"[ERROR FETCHING LOGS: {e}]"

            log_lines = raw_logs.strip().split("\n") if raw_logs.strip() else []

            # Filter out lines that were in the tail before the test
            # (Docker 'since' can be slightly imprecise)
            new_lines = []
            for line in log_lines:
                if line and line not in position.tail_lines:
                    new_lines.append(line)

            # Analyze logs
            capture = self._analyze_logs(name, new_lines)
            captures.append(capture)

        return captures

    def _analyze_logs(self, container: str, lines: list[str]) -> LogCapture:
        """Analyze log lines for errors, warnings, and tracebacks."""
        capture = LogCapture(container=container, logs=lines)

        # Track traceback state
        in_traceback = False
        current_traceback: list[str] = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Skip ignored patterns first
            if any(p.search(line) for p in self._ignore_patterns):
                continue

            # Check for traceback start
            if self._traceback_start.search(line):
                in_traceback = True
                current_traceback = [line]
                continue

            # Continue collecting traceback
            if in_traceback:
                current_traceback.append(line)
                # Tracebacks end with the exception line
                if self._exception_line.search(line):
                    capture.tracebacks.append("\n".join(current_traceback))
                    in_traceback = False
                    current_traceback = []
                # Limit traceback length
                elif len(current_traceback) > 50:
                    capture.tracebacks.append(
                        "\n".join(current_traceback) + "\n[TRUNCATED]"
                    )
                    in_traceback = False
                    current_traceback = []
                continue

            # Check for error patterns
            if any(p.search(line) for p in self._error_patterns):
                capture.errors.append(line)
                continue

            # Check for warning patterns
            if any(p.search(line) for p in self._warning_patterns):
                capture.warnings.append(line)

        return capture

    def get_issues_summary(
        self, captures: list[LogCapture], strict_mode: bool = True
    ) -> str | None:
        """
        Generate a summary of all issues found.

        Returns None if no issues, otherwise a formatted error message.
        """
        issues = []

        for capture in captures:
            if capture.errors:
                issues.append(
                    f"🔴 Container '{capture.container}' emitted {len(capture.errors)} error(s):\n"
                    + "\n".join(f"  → {e[:300]}" for e in capture.errors[:5])
                )

            if capture.tracebacks:
                issues.append(
                    f"💥 Container '{capture.container}' has {len(capture.tracebacks)} traceback(s):\n"
                    + "\n---\n".join(capture.tracebacks[:2])
                )

            if strict_mode and capture.warnings:
                issues.append(
                    f"⚠️  Container '{capture.container}' emitted {len(capture.warnings)} warning(s) (strict mode):\n"
                    + "\n".join(f"  → {w[:300]}" for w in capture.warnings[:5])
                )

        if not issues:
            return None

        return "\n\n".join(issues)

    def get_all_logs_since_mark(self) -> dict[str, list[str]]:
        """Get all logs since mark, organized by container."""
        captures = self.capture_since_mark()
        return {c.container: c.logs for c in captures}
