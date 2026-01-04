"""
AI Debug Report - Structured failure reports for automated debugging.

Format designed to be:
1. Machine-parseable (JSON)
2. Contains all context needed to diagnose and fix
3. Includes correlation between test actions and log errors

These reports enable an AI assistant to read the failure and
immediately diagnose the root cause without additional context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AIDebugReport:
    """
    Structured report for test failures.

    Designed to provide an AI assistant with everything needed
    to diagnose and fix the issue without additional context.
    """

    # Test identification
    test_id: str  # Full pytest node ID (e.g., system_tests/workflows/test_voting.py::test_upvote)
    test_name: str  # Short name (e.g., test_upvote)
    test_start: datetime
    test_end: datetime

    # Log evidence
    container_logs: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    tracebacks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Optional: Request/Response data
    request_method: str | None = None
    request_url: str | None = None
    request_headers: dict[str, str] | None = None
    request_body: dict[str, Any] | None = None
    response_status: int | None = None
    response_body: dict[str, Any] | None = None

    # Optional: Database state
    db_before: dict[str, Any] | None = None
    db_after: dict[str, Any] | None = None

    # Optional: Test assertion that failed
    assertion_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "duration_seconds": (self.test_end - self.test_start).total_seconds(),
            "summary": self._generate_summary(),
            "errors": self.errors,
            "tracebacks": self.tracebacks,
            "warnings": self.warnings,
            "container_logs": self.container_logs,
            "request": {
                "method": self.request_method,
                "url": self.request_url,
                "headers": self.request_headers,
                "body": self.request_body,
            } if self.request_url else None,
            "response": {
                "status": self.response_status,
                "body": self.response_body,
            } if self.response_status else None,
            "database": {
                "before": self.db_before,
                "after": self.db_after,
            } if self.db_before or self.db_after else None,
            "assertion_error": self.assertion_error,
        }

    def to_json(self) -> str:
        """Serialize to JSON for file output."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, directory: Path | str = "test-results/debug-reports") -> Path:
        """Save report to a JSON file and return the path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Create filename from test name and timestamp
        safe_name = self.test_name.replace("/", "_").replace("::", "_")
        timestamp = self.test_start.strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"

        filepath = directory / filename
        filepath.write_text(self.to_json())

        return filepath

    def _generate_summary(self) -> str:
        """Generate a human-readable summary for quick triage."""
        parts = []

        # Extract exception type from first traceback
        if self.tracebacks:
            tb = self.tracebacks[0]
            lines = tb.strip().split("\n")
            if lines:
                # Last line usually has the exception
                parts.append(f"Exception: {lines[-1][:100]}")

        if self.errors:
            parts.append(f"{len(self.errors)} error(s) in logs")

        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")

        if self.response_status and self.response_status >= 400:
            parts.append(f"HTTP {self.response_status}")

        if self.assertion_error:
            parts.append(f"Assertion: {self.assertion_error[:80]}")

        return " | ".join(parts) if parts else "Unknown failure"

    def to_markdown(self) -> str:
        """Generate markdown report for human review."""
        duration = (self.test_end - self.test_start).total_seconds()

        md = f"""# Test Failure Report

## Test: `{self.test_name}`

**ID:** `{self.test_id}`
**Duration:** {duration:.2f}s
**Summary:** {self._generate_summary()}

---
"""

        # Assertion error
        if self.assertion_error:
            md += f"\n## Assertion Failed\n\n```\n{self.assertion_error}\n```\n"

        # Errors
        if self.errors:
            md += "\n## Errors Found\n\n"
            for error in self.errors[:10]:
                md += f"- `{error[:300]}`\n"

        # Tracebacks
        if self.tracebacks:
            md += "\n## Tracebacks\n\n"
            for tb in self.tracebacks[:3]:
                md += f"```\n{tb}\n```\n\n"

        # Request/Response
        if self.request_url:
            md += f"\n## Request\n\n"
            md += f"**{self.request_method}** `{self.request_url}`\n\n"
            if self.request_body:
                md += f"```json\n{json.dumps(self.request_body, indent=2)}\n```\n"

        if self.response_status:
            md += f"\n## Response (HTTP {self.response_status})\n\n"
            if self.response_body:
                md += f"```json\n{json.dumps(self.response_body, indent=2)}\n```\n"

        # Container logs
        if self.container_logs:
            md += "\n## Container Logs\n\n"
            for container, logs in self.container_logs.items():
                if logs:
                    md += f"### {container}\n\n```\n"
                    md += "\n".join(logs[-20:])  # Last 20 lines
                    md += "\n```\n\n"

        return md


def create_report_from_test_failure(
    test_id: str,
    test_name: str,
    test_start: datetime,
    container_logs: dict[str, list[str]],
    errors: list[str],
    tracebacks: list[str],
    warnings: list[str],
    assertion_error: str | None = None,
) -> AIDebugReport:
    """Factory function to create a report from test failure data."""
    return AIDebugReport(
        test_id=test_id,
        test_name=test_name,
        test_start=test_start,
        test_end=datetime.now(),
        container_logs=container_logs,
        errors=errors,
        tracebacks=tracebacks,
        warnings=warnings,
        assertion_error=assertion_error,
    )
