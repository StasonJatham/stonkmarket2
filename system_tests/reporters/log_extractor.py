"""
Log Extractor - Extract relevant log snippets from container logs.

This module helps correlate errors with surrounding context,
making it easier to understand what happened before and after
an error occurred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LogSnippet:
    """A snippet of logs with context around an error."""

    container: str
    error_line: str
    context_before: list[str]
    context_after: list[str]
    line_number: int

    def to_string(self, include_line_numbers: bool = True) -> str:
        """Format snippet as a string."""
        lines = []

        start_num = max(1, self.line_number - len(self.context_before))

        for i, line in enumerate(self.context_before):
            num = start_num + i
            if include_line_numbers:
                lines.append(f"  {num:4d} │ {line}")
            else:
                lines.append(f"       │ {line}")

        # Error line with marker
        if include_line_numbers:
            lines.append(f"→ {self.line_number:4d} │ {self.error_line}")
        else:
            lines.append(f"    →  │ {self.error_line}")

        for i, line in enumerate(self.context_after):
            num = self.line_number + 1 + i
            if include_line_numbers:
                lines.append(f"  {num:4d} │ {line}")
            else:
                lines.append(f"       │ {line}")

        return "\n".join(lines)


class LogExtractor:
    """
    Extract relevant snippets from container logs.

    Provides context around errors to help understand the
    sequence of events leading to the failure.
    """

    def __init__(self, context_lines: int = 5):
        self.context_lines = context_lines

    def extract_error_snippets(
        self,
        logs: list[str],
        container: str,
        error_patterns: list[re.Pattern] | None = None,
    ) -> list[LogSnippet]:
        """
        Extract snippets around error lines.

        Args:
            logs: List of log lines
            container: Container name for the snippet
            error_patterns: Compiled regex patterns to match errors

        Returns:
            List of LogSnippet objects with context
        """
        if error_patterns is None:
            error_patterns = [
                re.compile(r"ERROR", re.IGNORECASE),
                re.compile(r"Exception:", re.IGNORECASE),
                re.compile(r"Traceback", re.IGNORECASE),
            ]

        snippets = []

        for i, line in enumerate(logs):
            # Check if this line matches any error pattern
            is_error = any(p.search(line) for p in error_patterns)

            if is_error:
                # Get context before
                start = max(0, i - self.context_lines)
                context_before = logs[start:i]

                # Get context after
                end = min(len(logs), i + self.context_lines + 1)
                context_after = logs[i + 1:end]

                snippets.append(LogSnippet(
                    container=container,
                    error_line=line,
                    context_before=context_before,
                    context_after=context_after,
                    line_number=i + 1,  # 1-based
                ))

        return snippets

    def extract_traceback(
        self,
        logs: list[str],
        container: str,
    ) -> list[str]:
        """
        Extract complete tracebacks from logs.

        Returns list of complete traceback strings.
        """
        tracebacks = []
        current_tb: list[str] = []
        in_traceback = False

        traceback_start = re.compile(r"Traceback \(most recent call last\)")
        exception_end = re.compile(r"^\w+Error:|^\w+Exception:")

        for line in logs:
            if traceback_start.search(line):
                in_traceback = True
                current_tb = [line]
                continue

            if in_traceback:
                current_tb.append(line)

                if exception_end.search(line):
                    tracebacks.append("\n".join(current_tb))
                    in_traceback = False
                    current_tb = []
                elif len(current_tb) > 50:
                    # Truncate very long tracebacks
                    current_tb.append("[TRUNCATED]")
                    tracebacks.append("\n".join(current_tb))
                    in_traceback = False
                    current_tb = []

        return tracebacks

    def find_request_response_pair(
        self,
        logs: list[str],
        request_id: str | None = None,
        path_pattern: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Find request/response log entries.

        Looks for common patterns like:
        - "POST /api/endpoint"
        - "response_status=200"
        - Request correlation IDs

        Returns dict with 'request' and 'response' log lines.
        """
        result = {"request": [], "response": []}

        request_patterns = [
            re.compile(r"(GET|POST|PUT|DELETE|PATCH)\s+/\S+"),
            re.compile(r"Request:\s"),
            re.compile(r"request_id="),
        ]

        response_patterns = [
            re.compile(r"response_status=\d+"),
            re.compile(r"Response:\s"),
            re.compile(r"HTTP/\d\.\d\"\s+\d{3}"),
        ]

        for line in logs:
            # Filter by request_id if provided
            if request_id and request_id not in line:
                continue

            # Filter by path pattern if provided
            if path_pattern and path_pattern not in line:
                continue

            # Check request patterns
            if any(p.search(line) for p in request_patterns):
                result["request"].append(line)

            # Check response patterns
            if any(p.search(line) for p in response_patterns):
                result["response"].append(line)

        return result

    def summarize_logs(
        self,
        logs: list[str],
        max_lines: int = 50,
    ) -> str:
        """
        Create a summary of logs, prioritizing errors and important events.

        Returns a condensed view of the most relevant log entries.
        """
        # Priority categories
        errors = []
        warnings = []
        important = []
        other = []

        error_pattern = re.compile(r"ERROR|Exception|Traceback|FATAL", re.IGNORECASE)
        warning_pattern = re.compile(r"WARNING|WARN", re.IGNORECASE)
        important_pattern = re.compile(
            r"started|completed|succeeded|failed|timeout|connected",
            re.IGNORECASE
        )

        for line in logs:
            if error_pattern.search(line):
                errors.append(line)
            elif warning_pattern.search(line):
                warnings.append(line)
            elif important_pattern.search(line):
                important.append(line)
            else:
                other.append(line)

        # Build summary prioritizing errors
        summary_lines = []

        # Always include all errors
        summary_lines.extend(errors)

        # Add warnings up to limit
        remaining = max_lines - len(summary_lines)
        if remaining > 0:
            summary_lines.extend(warnings[:remaining])

        # Add important events
        remaining = max_lines - len(summary_lines)
        if remaining > 0:
            summary_lines.extend(important[:remaining])

        # Fill with other if space
        remaining = max_lines - len(summary_lines)
        if remaining > 0:
            # Take from end (most recent)
            summary_lines.extend(other[-remaining:])

        return "\n".join(summary_lines)
