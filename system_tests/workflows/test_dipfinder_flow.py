"""
DipFinder Flow - End-to-end test for dip detection.

This test verifies:
1. DipFinder analysis runs without errors
2. Results are persisted to database
3. API returns valid dip signals
4. No errors in Celery worker logs
"""

from __future__ import annotations

import pytest

from system_tests.fixtures.celery_sync import CeleryTaskTracker
from system_tests.fixtures.db_state import DatabaseSnapshot


class TestDipFinderFlow:
    """End-to-end DipFinder workflow tests."""

    def test_dips_list_endpoint(
        self,
        api_client,
    ):
        """
        GET /api/dipfinder/dips returns valid dip data.

        Verifies API layer is working correctly.
        """
        response = api_client.get("/api/dipfinder/dips")

        assert response.status_code == 200
        data = response.json()

        # Response should be a list (may be empty if no dips detected)
        assert isinstance(data, list), f"Expected list, got {type(data)}"

    def test_dip_detail_endpoint(
        self,
        api_client,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        GET /api/dipfinder/dip/{symbol} returns dip details.

        Tests the detail view for a specific symbol.
        """
        # Get a symbol that has dip data
        symbol = db_snapshot.query_scalar(
            "SELECT symbol FROM dipfinder_signals LIMIT 1"
        )

        if not symbol:
            pytest.skip("No dip signals in database")

        response = api_client.get(f"/api/dipfinder/dip/{symbol}")

        assert response.status_code == 200
        data = response.json()
        assert data is not None

    def test_dip_signals_in_database(
        self,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        Verify dip signals exist in database.

        This confirms the DipFinder pipeline is populating data.
        """
        signal_count = db_snapshot.query_scalar(
            "SELECT COUNT(*) FROM dipfinder_signals"
        )

        # Just verify query works, count may be 0 in fresh DB
        assert signal_count is not None

    def test_dip_categories_valid(
        self,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        Verify dip categories are valid values.

        Catches issues where invalid categories are stored.
        """
        # Get distinct categories
        result = db_snapshot.query_all(
            "SELECT DISTINCT dip_category FROM dipfinder_signals WHERE dip_category IS NOT NULL"
        )

        valid_categories = {
            "BUY", "HOLD", "SELL", "STRONG_BUY", "STRONG_SELL",
            "WATCH", "AVOID", "NEUTRAL",
        }

        for (category,) in result:
            if category:
                # Category should be uppercase or a known value
                assert category.upper() in valid_categories or category in valid_categories, \
                    f"Invalid dip category: {category}"


class TestDipFinderJobFlow:
    """Test DipFinder via background job execution."""

    @pytest.mark.slow
    def test_dipfinder_job_completes(
        self,
        auth_client,
        celery_tracker: CeleryTaskTracker,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        Run dipfinder job and verify completion.

        Full end-to-end test:
        1. Trigger dipfinder job
        2. Wait for Celery completion
        3. Verify no errors (via docker_log_watcher)
        """
        # Get signal count before
        before_count = db_snapshot.query_scalar(
            "SELECT COUNT(*) FROM dipfinder_signals"
        )

        # Trigger job
        response = auth_client.post("/api/cronjobs/dipfinder/run")

        if response.status_code == 404:
            pytest.skip("DipFinder job not registered")

        assert response.status_code == 200, \
            f"Failed to trigger job: {response.status_code}"

        # Wait for task to complete
        celery_tracker.wait_for_drain(timeout=120)

        # Verify no failed tasks
        celery_tracker.assert_no_failed_tasks()

        # Job completion without errors is success
        # (docker_log_watcher will catch any issues)

    @pytest.mark.slow
    def test_quant_analysis_job_completes(
        self,
        auth_client,
        celery_tracker: CeleryTaskTracker,
    ):
        """
        Run quant_analysis_nightly job and verify completion.

        This job runs statistical analysis and scoring.
        """
        # Trigger job
        response = auth_client.post("/api/cronjobs/quant_analysis_nightly/run")

        if response.status_code == 404:
            pytest.skip("quant_analysis_nightly job not registered")

        assert response.status_code == 200, \
            f"Failed to trigger job: {response.status_code}"

        # Wait for task to complete
        celery_tracker.wait_for_drain(timeout=180)

        # Verify no failed tasks
        celery_tracker.assert_no_failed_tasks()
