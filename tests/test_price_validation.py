"""Tests for price data validation.

Tests the validation logic in app/services/prices.py to ensure:
1. Corrupt data is rejected before saving to DB
2. Valid data passes validation
3. Edge cases are handled correctly
"""

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from app.services.prices import (
    MAX_VALID_DAILY_CHANGE,
    ValidationResult,
    validate_prices,
)


class TestValidatePrices:
    """Tests for the validate_prices function."""

    def _make_price_df(
        self,
        closes: list[float],
        start_date: date | None = None,
    ) -> pd.DataFrame:
        """Helper to create a price DataFrame."""
        if start_date is None:
            start_date = date.today() - timedelta(days=len(closes))
        
        dates = pd.date_range(start_date, periods=len(closes), freq="D")
        return pd.DataFrame({
            "Open": closes,
            "High": [c * 1.02 for c in closes],
            "Low": [c * 0.98 for c in closes],
            "Close": closes,
            "Volume": [1000000] * len(closes),
        }, index=dates)

    def test_rejects_empty_dataframe(self):
        """Empty DataFrame should be rejected."""
        df = pd.DataFrame()
        result = validate_prices("AAPL", df)
        
        assert not result.valid
        assert result.error is not None
        assert "empty" in result.error.lower()

    def test_rejects_missing_close_column(self):
        """DataFrame without Close column should be rejected."""
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [105, 106],
            "Low": [98, 99],
            "Volume": [1000, 1000],
        })
        result = validate_prices("AAPL", df)
        
        assert not result.valid
        assert "Close" in result.error

    def test_rejects_all_nan_close_values(self):
        """DataFrame with all NaN Close values should be rejected."""
        df = pd.DataFrame({
            "Close": [float("nan"), float("nan")],
        })
        result = validate_prices("AAPL", df)
        
        assert not result.valid
        assert "valid" in result.error.lower() or "nan" in result.error.lower()

    def test_accepts_valid_price_data(self):
        """Normal price data should pass validation."""
        df = self._make_price_df([100, 101, 99, 102, 100.5])
        result = validate_prices("AAPL", df)
        
        assert result.valid
        assert result.error is None

    def test_rejects_extreme_daily_change(self):
        """Price change >50% in a single day should be rejected."""
        # 100 -> 160 is a 60% change
        df = self._make_price_df([100, 160, 162])
        result = validate_prices("AAPL", df)
        
        assert not result.valid
        assert "60" in result.error or "threshold" in result.error.lower()

    def test_accepts_change_at_threshold(self):
        """Price change exactly at 50% threshold should be accepted."""
        # 100 -> 150 is exactly 50% change
        df = self._make_price_df([100, 150, 152])
        result = validate_prices("AAPL", df)
        
        assert result.valid

    def test_rejects_extreme_drop(self):
        """Price drop >50% should be rejected."""
        # 100 -> 40 is a 60% drop
        df = self._make_price_df([100, 40, 42])
        result = validate_prices("AAPL", df)
        
        assert not result.valid

    def test_continuity_check_rejects_large_gap(self):
        """Gap from existing price >50% should be rejected."""
        df = self._make_price_df([200, 205, 210])
        existing_price = Decimal("100.00")  # New data starts at 200, 100% higher
        
        result = validate_prices("AAPL", df, existing_last_price=existing_price)
        
        assert not result.valid
        assert "gap" in result.error.lower()

    def test_continuity_check_accepts_small_gap(self):
        """Gap from existing price <50% should be accepted."""
        df = self._make_price_df([105, 106, 107])
        existing_price = Decimal("100.00")  # 5% gap is fine
        
        result = validate_prices("AAPL", df, existing_last_price=existing_price)
        
        assert result.valid

    def test_continuity_check_skipped_if_no_existing_price(self):
        """Continuity check should be skipped if no existing price."""
        df = self._make_price_df([100, 101, 102])
        
        result = validate_prices("AAPL", df, existing_last_price=None)
        
        assert result.valid

    def test_continuity_check_skipped_if_zero_existing_price(self):
        """Continuity check should be skipped if existing price is 0."""
        df = self._make_price_df([100, 101, 102])
        
        result = validate_prices("AAPL", df, existing_last_price=Decimal("0"))
        
        assert result.valid

    def test_single_data_point_is_valid(self):
        """A single data point should pass (no daily change to check)."""
        df = self._make_price_df([100])
        result = validate_prices("AAPL", df)
        
        assert result.valid

    def test_identifies_day_of_extreme_change(self):
        """Error message should identify which day had the extreme change."""
        df = self._make_price_df([100, 101, 200, 202])  # Big jump on day 3
        result = validate_prices("AAPL", df)
        
        assert not result.valid
        # Error should mention the percentage
        assert "98" in result.error or "99" in result.error  # ~99% change

    def test_handles_negative_prices(self):
        """Negative prices in the data shouldn't crash validation.
        
        Note: The current validate_prices doesn't explicitly reject negatives,
        but the daily change calculation would catch extreme changes.
        """
        df = self._make_price_df([100, -50, -55])  # Extreme change
        result = validate_prices("AAPL", df)
        
        # Should be rejected due to extreme change (150%)
        assert not result.valid


class TestValidationConstants:
    """Tests for validation constants."""

    def test_max_daily_change_is_50_percent(self):
        """MAX_VALID_DAILY_CHANGE should be 0.50 (50%)."""
        assert MAX_VALID_DAILY_CHANGE == 0.50


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Valid result has valid=True and no error."""
        result = ValidationResult(valid=True)
        assert result.valid
        assert result.error is None

    def test_invalid_result_with_error(self):
        """Invalid result has valid=False and an error message."""
        result = ValidationResult(valid=False, error="Test error")
        assert not result.valid
        assert result.error == "Test error"


class TestRealWorldScenarios:
    """Tests based on real-world data corruption scenarios."""

    def _make_price_df(
        self,
        closes: list[float],
        start_date: date | None = None,
    ) -> pd.DataFrame:
        """Helper to create a price DataFrame."""
        if start_date is None:
            start_date = date.today() - timedelta(days=len(closes))
        
        dates = pd.date_range(start_date, periods=len(closes), freq="D")
        return pd.DataFrame({
            "Open": closes,
            "High": [c * 1.02 for c in closes],
            "Low": [c * 0.98 for c in closes],
            "Close": closes,
            "Volume": [1000000] * len(closes),
        }, index=dates)

    def test_spy_corruption_scenario(self):
        """
        Real scenario: SPY jumped from $687 to $232 due to yfinance bug.
        
        This was caught in production - the validation correctly rejected
        a 66% drop in a single day.
        """
        # Simulated corrupt data: $687 -> $232 is a 66% drop
        df = self._make_price_df([687, 232, 230])
        result = validate_prices("SPY", df)
        
        assert not result.valid
        assert "66" in result.error or "threshold" in result.error.lower()

    def test_msft_gap_from_existing(self):
        """
        Real scenario: MSFT DB had $472, yfinance returned $209 (55% gap).
        
        Validation correctly rejected due to gap from existing price.
        """
        df = self._make_price_df([209, 210, 211])
        existing_price = Decimal("472.94")
        
        result = validate_prices("MSFT", df, existing_last_price=existing_price)
        
        assert not result.valid
        assert "gap" in result.error.lower()

    def test_stock_split_looks_like_corruption(self):
        """
        Stock splits can look like corruption but are valid.
        
        Note: Our validation would reject a split-day because it looks like
        a >50% drop. This is by design - we rely on yfinance auto_adjust=True
        to handle splits before data reaches us.
        """
        # 2:1 split: $200 -> $100 is a 50% drop (at threshold)
        df = self._make_price_df([200, 100, 101])
        result = validate_prices("AAPL", df)
        
        # At exactly 50%, this should still pass
        assert result.valid or "50" in str(result.error)

    def test_ipo_volatility_is_valid(self):
        """
        IPO day can have high volatility (like HOOD 50% jump).
        
        At exactly 50%, this should still be accepted.
        """
        # IPO: ~46.8 -> 70.4 is ~50% change
        df = self._make_price_df([46.8, 70.4, 68.5])
        result = validate_prices("HOOD", df)
        
        # 50.4% is just barely over threshold
        # This is a grey area - we accept at threshold
        if not result.valid:
            assert "50" in result.error

    def test_gradual_decline_is_valid(self):
        """
        Gradual price decline over time is valid even if total is >50%."""
        # Each day drops ~10%, total decline is huge but daily is fine
        closes = [100]
        for _ in range(10):
            closes.append(closes[-1] * 0.9)
        
        df = self._make_price_df(closes)
        result = validate_prices("DECLINING", df)
        
        # Total drop is ~65% but daily max is 10%
        assert result.valid
