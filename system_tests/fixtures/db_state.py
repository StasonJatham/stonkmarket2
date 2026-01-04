"""
Database State Verification - Capture and compare database state.

This module enables tests to verify that database changes occurred correctly
by capturing snapshots before and after test actions.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from docker.models.containers import Container


@dataclass
class QueryResult:
    """Result of a database query."""

    query: str
    rows: list[tuple[Any, ...]]
    columns: list[str] | None = None

    @property
    def scalar(self) -> Any:
        """Get single scalar value from result."""
        if not self.rows:
            return None
        return self.rows[0][0]

    @property
    def count(self) -> int:
        """Get row count."""
        return len(self.rows)


class DatabaseSnapshot:
    """
    Capture database state for before/after comparison.

    Usage:
        snapshot = DatabaseSnapshot(postgres_container)

        # Capture before state
        votes_before = snapshot.query_scalar("SELECT COUNT(*) FROM community_votes")

        # Perform action
        response = client.post("/api/swipe/vote", ...)

        # Verify after state
        votes_after = snapshot.query_scalar("SELECT COUNT(*) FROM community_votes")
        assert votes_after > votes_before
    """

    def __init__(
        self,
        postgres_container: "Container",
        database: str = "stonkmarket",
        user: str = "stonkmarket",
    ):
        self.container = postgres_container
        self.database = database
        self.user = user

    def query(self, sql: str) -> QueryResult:
        """
        Execute a SQL query and return results.

        Args:
            sql: SQL query to execute

        Returns:
            QueryResult with rows and metadata
        """
        # Execute via docker exec
        cmd = [
            "docker", "exec", self.container.name,
            "psql", "-U", self.user, "-d", self.database,
            "-t", "-A", "-F", "|",  # Tuples only, unaligned, pipe separator
            "-c", sql,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Query failed: {result.stderr}")

            # Parse output
            output = result.stdout.strip()
            if not output:
                return QueryResult(query=sql, rows=[])

            rows = []
            for line in output.split("\n"):
                if line.strip():
                    # Split by pipe and convert types
                    values = tuple(
                        self._parse_value(v) for v in line.split("|")
                    )
                    rows.append(values)

            return QueryResult(query=sql, rows=rows)

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Query timed out: {sql}")

    def query_scalar(self, sql: str) -> Any:
        """Execute query and return single scalar value."""
        result = self.query(sql)
        return result.scalar

    def query_one(self, sql: str) -> tuple[Any, ...] | None:
        """Execute query and return single row."""
        result = self.query(sql)
        return result.rows[0] if result.rows else None

    def query_all(self, sql: str) -> list[tuple[Any, ...]]:
        """Execute query and return all rows."""
        return self.query(sql).rows

    def table_count(self, table: str) -> int:
        """Get row count for a table."""
        result = self.query_scalar(f"SELECT COUNT(*) FROM {table}")
        return int(result) if result else 0

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        result = self.query_scalar(
            f"SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = '{table}')"
        )
        return result == "t" or result is True

    def row_exists(self, table: str, where: str) -> bool:
        """Check if a row exists matching the condition."""
        result = self.query_scalar(
            f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {where})"
        )
        return result == "t" or result is True

    def _parse_value(self, value: str) -> Any:
        """Parse a string value to appropriate Python type."""
        if value == "" or value.lower() == "null":
            return None

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Boolean
        if value.lower() in ("t", "true"):
            return True
        if value.lower() in ("f", "false"):
            return False

        # String
        return value


class DatabaseStateManager:
    """
    Manage database state for test isolation.

    Provides methods to capture full state, reset tables,
    and restore to a known state.
    """

    def __init__(self, snapshot: DatabaseSnapshot):
        self.snapshot = snapshot
        self._initial_counts: dict[str, int] = {}

    def capture_initial_state(self, tables: list[str]) -> dict[str, int]:
        """Capture initial row counts for specified tables."""
        self._initial_counts = {
            table: self.snapshot.table_count(table)
            for table in tables
        }
        return self._initial_counts

    def get_changes(self, tables: list[str]) -> dict[str, int]:
        """Get row count changes since initial capture."""
        changes = {}
        for table in tables:
            current = self.snapshot.table_count(table)
            initial = self._initial_counts.get(table, current)
            changes[table] = current - initial
        return changes

    def assert_no_data_loss(self, tables: list[str]) -> None:
        """Assert that no rows were deleted from specified tables."""
        for table in tables:
            current = self.snapshot.table_count(table)
            initial = self._initial_counts.get(table, 0)
            if current < initial:
                raise AssertionError(
                    f"Data loss detected in {table}: "
                    f"was {initial}, now {current}"
                )
