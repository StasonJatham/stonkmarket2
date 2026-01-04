"""
Stack Health Smoke Tests - Verify Docker stack is operational.

These tests run first to ensure the entire infrastructure is healthy
before running more complex workflow tests.

Run with: pytest system_tests/smoke/ -v
"""

from __future__ import annotations

import pytest

from system_tests.fixtures.stack_health import StackHealthChecker
from system_tests.fixtures.docker_logs import DockerLogWatcher


class TestStackHealth:
    """Verify the Docker stack is healthy and ready for testing."""

    def test_all_containers_running(
        self,
        stack_health: StackHealthChecker,
    ):
        """All required containers must be running."""
        results = stack_health.check_all()

        for result in results:
            if "container:" in result.name:
                assert result.healthy, f"{result.name}: {result.message}"

    def test_api_health_endpoint(
        self,
        api_client,
    ):
        """API health endpoint returns 200."""
        response = api_client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ("healthy", "ok", True)

    def test_database_connected(
        self,
        db_snapshot,
    ):
        """Database is accessible and has expected tables."""
        # Query pg_tables to verify connection works
        result = db_snapshot.query_scalar(
            "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public'"
        )

        assert result is not None
        assert int(result) > 0, "No tables found in public schema"

    def test_celery_worker_responsive(
        self,
        celery_tracker,
    ):
        """Celery worker is running and processing tasks."""
        # Check that we can get task counts (worker is responsive)
        counts = celery_tracker.get_task_count()

        # Counts should be a dict (even if all zeros)
        assert isinstance(counts, dict)
        assert "received" in counts
        assert "succeeded" in counts

    def test_no_startup_errors_in_logs(
        self,
        docker_log_watcher: DockerLogWatcher,
    ):
        """No error patterns in container startup logs."""
        # This test uses the autouse fixture which checks logs
        # If there are errors, the fixture will fail the test
        # Here we just verify the watcher is working
        captures = docker_log_watcher.capture_since_mark()

        # Log how many lines captured for visibility
        total_lines = sum(len(c.logs) for c in captures)
        print(f"\n  Captured {total_lines} log lines across {len(captures)} containers")


class TestCriticalEndpoints:
    """Verify critical API endpoints are functional."""

    def test_symbols_list(self, api_client):
        """GET /api/symbols returns list of symbols."""
        response = api_client.get("/api/symbols")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have some symbols if database is seeded

    def test_dips_list(self, api_client, db_snapshot):
        """GET /api/dipfinder/signals returns dip signals."""
        # Get a symbol to query
        symbol = db_snapshot.query_scalar(
            "SELECT symbol FROM symbols WHERE is_active = true LIMIT 1"
        )

        if not symbol:
            pytest.skip("No active symbols in database")

        response = api_client.get(f"/api/dipfinder/signals?tickers={symbol}")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "signals" in data

    def test_cronjobs_list(self, auth_client):
        """GET /api/cronjobs requires auth and returns job list."""
        response = auth_client.get("/api/cronjobs/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_api_docs(self, api_client):
        """Swagger UI is accessible."""
        # Try docs endpoint
        response = api_client.get("/api/docs")

        # May redirect, so accept 2xx or 3xx
        assert response.status_code < 400, f"Docs not accessible: {response.status_code}"


class TestDatabaseIntegrity:
    """Verify database schema and critical data."""

    def test_symbols_table_exists(self, db_snapshot):
        """Symbols table exists with expected structure."""
        exists = db_snapshot.table_exists("symbols")
        assert exists, "symbols table does not exist"

    def test_price_history_table_exists(self, db_snapshot):
        """Price history table exists."""
        exists = db_snapshot.table_exists("price_history")
        assert exists, "price_history table does not exist"

    def test_dip_votes_table_exists(self, db_snapshot):
        """Dip votes table exists."""
        exists = db_snapshot.table_exists("dip_votes")
        assert exists, "dip_votes table does not exist"

    def test_dipfinder_signals_table_exists(self, db_snapshot):
        """Dipfinder signals table exists."""
        exists = db_snapshot.table_exists("dipfinder_signals")
        assert exists, "dipfinder_signals table does not exist"

    def test_no_null_symbols_in_symbols(self, db_snapshot):
        """No NULL symbols in symbols table."""
        count = db_snapshot.query_scalar(
            "SELECT COUNT(*) FROM symbols WHERE symbol IS NULL"
        )
        assert int(count) == 0, f"Found {count} symbols with NULL symbol"
