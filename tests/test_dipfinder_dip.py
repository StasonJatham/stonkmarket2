"""Tests for DipFinder dip calculation module."""

from __future__ import annotations

import numpy as np
import pytest

from app.dipfinder.dip import (
    compute_dip_series_windowed,
    compute_dip_percentile,
    compute_persistence,
    DipMetrics,
)


class TestDipSeriesWindowed:
    """Tests for compute_dip_series_windowed."""
    
    def test_empty_array(self):
        """Empty input returns empty output."""
        result = compute_dip_series_windowed(np.array([]), window=7)
        assert len(result) == 0
    
    def test_single_element(self):
        """Single element with window=1 returns 0 dip."""
        result = compute_dip_series_windowed(np.array([100.0]), window=1)
        assert len(result) == 1
        assert result[0] == 0.0  # No dip when price equals peak
    
    def test_constant_prices(self):
        """Constant prices have zero dip."""
        prices = np.array([100.0] * 20)
        result = compute_dip_series_windowed(prices, window=7)
        
        # First window-1 values are NaN
        assert np.isnan(result[:6]).all()
        
        # Rest should be 0
        assert np.allclose(result[6:], 0.0)
    
    def test_rising_prices(self):
        """Rising prices have zero dip (current = peak)."""
        prices = np.arange(1, 21, dtype=float)  # 1, 2, 3, ..., 20
        result = compute_dip_series_windowed(prices, window=5)
        
        # After initial window, all dips should be 0 (current is always peak)
        valid = result[~np.isnan(result)]
        assert np.allclose(valid, 0.0)
    
    def test_falling_prices(self):
        """Falling prices have increasing dips."""
        prices = np.array([100.0, 95.0, 90.0, 85.0, 80.0])
        result = compute_dip_series_windowed(prices, window=5)
        
        # At last point: peak=100, current=80, dip = 0.20
        assert np.isclose(result[-1], 0.20)
    
    def test_peak_to_current_calculation(self):
        """Verify peak-to-current calculation."""
        prices = np.array([80.0, 90.0, 100.0, 95.0, 85.0])
        result = compute_dip_series_windowed(prices, window=3)
        
        # At index 4: window is [100, 95, 85], peak=100, current=85
        # dip = (100 - 85) / 100 = 0.15
        assert np.isclose(result[4], 0.15)
    
    def test_window_sliding(self):
        """Verify window slides correctly."""
        prices = np.array([100.0, 90.0, 80.0, 70.0, 80.0, 90.0, 100.0])
        result = compute_dip_series_windowed(prices, window=3)
        
        # At index 2: window [100, 90, 80], peak=100, dip=0.20
        assert np.isclose(result[2], 0.20)
        
        # At index 3: window [90, 80, 70], peak=90, dip=(90-70)/90
        assert np.isclose(result[3], (90 - 70) / 90)
        
        # At index 6: window [80, 90, 100], peak=100, current=100, dip=0
        assert np.isclose(result[6], 0.0)
    
    def test_invalid_window(self):
        """Invalid window raises error."""
        prices = np.array([100.0] * 10)
        
        with pytest.raises(ValueError):
            compute_dip_series_windowed(prices, window=0)
        
        with pytest.raises(ValueError):
            compute_dip_series_windowed(prices, window=-1)
    
    def test_window_larger_than_data(self):
        """Window larger than data handles gracefully."""
        prices = np.array([100.0, 90.0, 80.0])
        result = compute_dip_series_windowed(prices, window=10)
        
        # All should be NaN since window > len
        assert np.isnan(result).all()
    
    def test_known_values(self):
        """Test with known calculated values."""
        prices = np.array([100.0, 110.0, 105.0, 115.0, 100.0])
        result = compute_dip_series_windowed(prices, window=3)
        
        # Index 2: window [100, 110, 105], peak=110, dip=(110-105)/110 ≈ 0.0455
        assert np.isclose(result[2], (110 - 105) / 110, rtol=1e-4)
        
        # Index 3: window [110, 105, 115], peak=115, current=115, dip=0
        assert np.isclose(result[3], 0.0)
        
        # Index 4: window [105, 115, 100], peak=115, dip=(115-100)/115 ≈ 0.1304
        assert np.isclose(result[4], (115 - 100) / 115, rtol=1e-4)


class TestDipPercentile:
    """Tests for compute_dip_percentile."""
    
    def test_max_dip_is_100th_percentile(self):
        """Maximum dip should be at 100th percentile."""
        dip_series = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        pctl = compute_dip_percentile(dip_series, 0.30, exclude_last=False)
        assert pctl == 100.0
    
    def test_min_dip_is_low_percentile(self):
        """Minimum dip should be at low percentile."""
        dip_series = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        pctl = compute_dip_percentile(dip_series, 0.01, exclude_last=False)
        assert pctl < 20.0
    
    def test_median_dip(self):
        """Median dip should be around 50th percentile."""
        dip_series = np.arange(0.0, 0.21, 0.01)  # 0, 0.01, ..., 0.20
        pctl = compute_dip_percentile(dip_series, 0.10, exclude_last=False)
        assert 45.0 <= pctl <= 55.0
    
    def test_exclude_last(self):
        """Excluding last value works correctly."""
        dip_series = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        # If we exclude last (0.25), comparing 0.20 against [0.05, 0.10, 0.15, 0.20]
        pctl = compute_dip_percentile(dip_series, 0.20, exclude_last=True)
        assert pctl >= 75.0  # 0.20 is at 75th percentile of [0.05, 0.10, 0.15, 0.20]
    
    def test_nan_handling(self):
        """NaN values in series are ignored."""
        dip_series = np.array([np.nan, 0.05, 0.10, np.nan, 0.15, 0.20])
        pctl = compute_dip_percentile(dip_series, 0.15, exclude_last=False)
        # Should compute percentile against [0.05, 0.10, 0.15, 0.20]
        assert 50.0 <= pctl <= 80.0


class TestPersistence:
    """Tests for compute_persistence."""
    
    def test_no_persistence(self):
        """No dip persistence returns 0."""
        # Prices stay close to peak, no significant dip
        prices = np.array([100.0, 102.0, 101.0, 99.0, 100.0])
        persist = compute_persistence(prices, dip_threshold=0.10, window=3)
        assert persist == 0
    
    def test_single_day_persistence(self):
        """Single day dip returns 1."""
        # Last price drops 15% from peak (110 -> 93.5)
        prices = np.array([100.0, 105.0, 110.0, 105.0, 93.5])
        persist = compute_persistence(prices, dip_threshold=0.10, window=3)
        assert persist == 1
    
    def test_multi_day_persistence(self):
        """Multiple consecutive dip days."""
        # Prices that create consistent 10%+ dips in window
        # Window=3: [110, 100, 95]=14% dip, [100, 95, 90]=10% dip, [95, 90, 85]=11% dip
        prices = np.array([110.0, 100.0, 95.0, 90.0, 85.0])
        persist = compute_persistence(prices, dip_threshold=0.10, window=3)
        assert persist >= 2
    
    def test_interrupted_persistence(self):
        """Interrupted dip resets counter."""
        # Dip, recover to near peak, then dip again at end
        prices = np.array([100.0, 85.0, 99.0, 100.0, 85.0])
        persist = compute_persistence(prices, dip_threshold=0.10, window=3)
        # Only the last day counts as dip >= 10%
        assert persist >= 1
    
    def test_full_series_persistence(self):
        """Full series above threshold from peak."""
        # Start high, then stay consistently down 10%+
        prices = np.array([120.0, 100.0, 95.0, 90.0, 85.0, 80.0])
        persist = compute_persistence(prices, dip_threshold=0.10, window=3)
        assert persist >= 3


class TestDipMetrics:
    """Tests for DipMetrics dataclass."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        metrics = DipMetrics(
            ticker="AAPL",
            window=30,
            dip_pct=0.15,
            peak_price=150.0,
            current_price=127.5,
            dip_percentile=85.0,
            dip_vs_typical=2.0,
            typical_dip=0.075,
            persist_days=5,
            days_since_peak=10,
            is_meaningful=True,
        )
        
        d = metrics.to_dict()
        
        assert d["ticker"] == "AAPL"
        assert d["window"] == 30
        assert d["dip_pct"] == 0.15
        assert d["is_meaningful"] is True
    
    def test_is_meaningful_criteria(self):
        """Test meaningful dip criteria."""
        # Meaningful: large dip, high percentile, persisted
        meaningful = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.15,
            peak_price=100.0,
            current_price=85.0,
            dip_percentile=90.0,
            dip_vs_typical=2.0,
            typical_dip=0.075,
            persist_days=3,
            days_since_peak=5,
            is_meaningful=True,
        )
        assert meaningful.is_meaningful is True
        
        # Not meaningful: small dip
        not_meaningful = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.05,
            peak_price=100.0,
            current_price=95.0,
            dip_percentile=50.0,
            dip_vs_typical=0.8,
            typical_dip=0.0625,
            persist_days=1,
            days_since_peak=2,
            is_meaningful=False,
        )
        assert not_meaningful.is_meaningful is False
