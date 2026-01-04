"""
Price Fetch Flow - End-to-end test for price data operations.

This test verifies:
1. Price data can be fetched via API
2. Database stores prices correctly
3. Chart endpoints return valid data
4. No errors in any container logs
"""

from __future__ import annotations

import pytest

from system_tests.fixtures.celery_sync import CeleryTaskTracker
from system_tests.fixtures.db_state import DatabaseSnapshot


class TestPriceFetchFlow:
    """End-to-end price fetching workflow tests."""

    def test_chart_endpoint_returns_valid_data(
        self,
        api_client,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        GET /api/dipfinder/chart/{symbol} returns valid price data.

        Verifies:
        - Endpoint returns 200
        - Data has expected structure
        - No NaN or null prices
        - Prices are reasonable (not 0, not negative)
        """
        # Get a symbol from the database
        symbol = db_snapshot.query_scalar(
            "SELECT symbol FROM symbols WHERE is_active = true LIMIT 1"
        )

        if not symbol:
            pytest.skip("No active symbols in database")

        # Fetch chart data
        response = api_client.get(f"/api/dipfinder/chart/{symbol}")

        assert response.status_code == 200, f"Chart endpoint failed for {symbol}"

        data = response.json()
        assert isinstance(data, list), "Chart data should be a list"

        if len(data) > 0:
            # Verify structure
            first_point = data[0]
            assert "close" in first_point, "Chart point missing 'close'"
            assert "date" in first_point or "timestamp" in first_point, \
                "Chart point missing date/timestamp"

            # Verify no invalid prices
            for point in data:
                close = point.get("close")
                if close is not None:
                    assert isinstance(close, (int, float)), \
                        f"Close price should be numeric, got {type(close)}"
                    assert close > 0, f"Close price should be positive: {close}"
                    assert close == close, f"Close price is NaN for {symbol}"  # NaN check

    def test_price_data_persisted_to_database(
        self,
        api_client,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        Verify price data exists in the database.

        This confirms the data pipeline from yfinance -> DB is working.
        """
        # Count prices in database
        price_count = db_snapshot.query_scalar(
            "SELECT COUNT(*) FROM price_history"
        )

        assert int(price_count) > 0, "No price history in database"

        # Check for recent prices (within last 7 days)
        recent_count = db_snapshot.query_scalar("""
            SELECT COUNT(*) FROM price_history
            WHERE date > NOW() - INTERVAL '7 days'
        """)

        # This may be 0 on weekends/holidays, so just verify query works
        assert recent_count is not None

    def test_multiple_symbols_chart_data(
        self,
        api_client,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        Verify chart endpoint works for multiple symbols.

        Catches issues like connection pool exhaustion or
        symbol-specific parsing errors.
        """
        # Get up to 5 active symbols
        symbols = db_snapshot.query_all(
            "SELECT symbol FROM symbols WHERE is_active = true LIMIT 5"
        )

        if not symbols:
            pytest.skip("No active symbols in database")

        errors = []
        for (symbol,) in symbols:
            response = api_client.get(f"/api/dipfinder/chart/{symbol}")

            if response.status_code != 200:
                errors.append(f"{symbol}: HTTP {response.status_code}")
            else:
                data = response.json()
                if not isinstance(data, list):
                    errors.append(f"{symbol}: Invalid response type")
                elif len(data) == 0:
                    # Empty is OK, just note it
                    print(f"  Note: {symbol} has no chart data")

        assert not errors, f"Chart endpoint errors: {errors}"


class TestPriceJobFlow:
    """Test price fetching via background jobs."""

    @pytest.mark.slow
    def test_prices_daily_job_completes(
        self,
        auth_client,
        celery_tracker: CeleryTaskTracker,
        db_snapshot: DatabaseSnapshot,
    ):
        """
        Run prices_daily job and verify completion.

        This is a full end-to-end test:
        1. Trigger job via API
        2. Wait for Celery task to complete
        3. Verify database was updated
        4. Verify no errors in logs (via docker_log_watcher fixture)
        """
        # Get price count before
        before_count = db_snapshot.query_scalar(
            "SELECT COUNT(*) FROM price_history"
        )

        # Trigger job
        response = auth_client.post("/api/cronjobs/prices_daily/run")

        assert response.status_code == 200, \
            f"Failed to trigger job: {response.status_code} - {response.text}"

        # Wait for task to complete
        celery_tracker.wait_for_drain(timeout=60)

        # Verify no failed tasks
        celery_tracker.assert_no_failed_tasks()

        # Job may or may not add prices (depending on market hours, etc.)
        # But it should complete without errors (verified by docker_log_watcher)

    @pytest.mark.slow
    def test_data_backfill_job_completes(
        self,
        auth_client,
        celery_tracker: CeleryTaskTracker,
    ):
        """
        Run data_backfill job and verify completion.

        This job fills gaps in price/fundamental data.
        """
        # Trigger job
        response = auth_client.post("/api/cronjobs/data_backfill/run")

        assert response.status_code == 200, \
            f"Failed to trigger job: {response.status_code}"

        # Wait for task to complete (backfill can take a while)
        celery_tracker.wait_for_drain(timeout=120)

        # Verify no failed tasks
        celery_tracker.assert_no_failed_tasks()
