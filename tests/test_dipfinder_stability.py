"""Tests for DipFinder stability module."""

from __future__ import annotations

import numpy as np
import pytest

from app.dipfinder.stability import (
    compute_volatility,
    compute_max_drawdown,
    score_beta,
    score_volatility,
    score_max_drawdown,
    StabilityMetrics,
)


class TestVolatility:
    """Tests for compute_volatility."""
    
    def test_constant_prices(self):
        """Constant prices have zero volatility."""
        prices = np.array([100.0] * 260)
        vol = compute_volatility(prices)
        assert vol == 0.0
    
    def test_small_variation(self):
        """Small price variations give low volatility."""
        np.random.seed(42)
        # Small daily changes ±0.1%
        returns = np.random.normal(0, 0.001, 252)
        prices = 100.0 * np.cumprod(1 + returns)
        
        vol = compute_volatility(prices)
        # Annualized vol should be around 0.001 * sqrt(252) ≈ 0.016
        assert vol < 0.05
    
    def test_high_variation(self):
        """Large price variations give high volatility."""
        np.random.seed(42)
        # Large daily changes ±3%
        returns = np.random.normal(0, 0.03, 252)
        prices = 100.0 * np.cumprod(1 + returns)
        
        vol = compute_volatility(prices)
        # Annualized vol should be around 0.03 * sqrt(252) ≈ 0.48
        assert vol > 0.30
    
    def test_insufficient_data(self):
        """Insufficient data returns 0."""
        prices = np.array([100.0, 101.0, 99.0])
        vol = compute_volatility(prices, window=252)
        assert vol == 0.0
    
    def test_custom_window(self):
        """Custom window works correctly."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        prices = 100.0 * np.cumprod(1 + returns)
        
        vol = compute_volatility(prices, window=50)
        assert vol > 0.0


class TestMaxDrawdown:
    """Tests for compute_max_drawdown."""
    
    def test_rising_prices(self):
        """Rising prices have zero drawdown."""
        prices = np.arange(100, 200, dtype=float)
        mdd = compute_max_drawdown(prices)
        assert mdd == 0.0
    
    def test_falling_prices(self):
        """Falling prices: drawdown equals total decline."""
        prices = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        mdd = compute_max_drawdown(prices)
        # Peak=100, trough=60, drawdown = 0.40
        assert np.isclose(mdd, 0.40)
    
    def test_recovery_after_drawdown(self):
        """Max drawdown is captured even with recovery."""
        prices = np.array([100.0, 110.0, 80.0, 90.0, 120.0])
        mdd = compute_max_drawdown(prices)
        # Peak before trough: 110, trough: 80
        # Drawdown = (110 - 80) / 110 ≈ 0.2727
        assert np.isclose(mdd, (110 - 80) / 110, rtol=0.01)
    
    def test_multiple_drawdowns(self):
        """Returns largest drawdown among multiple."""
        prices = np.array([100.0, 95.0, 100.0, 80.0, 100.0, 85.0])
        mdd = compute_max_drawdown(prices)
        # Largest drawdown: 100 -> 80 = 20%
        assert np.isclose(mdd, 0.20)
    
    def test_v_shaped_recovery(self):
        """V-shaped pattern captures the full drop."""
        prices = np.array([100.0, 90.0, 80.0, 70.0, 80.0, 90.0, 100.0])
        mdd = compute_max_drawdown(prices)
        assert np.isclose(mdd, 0.30)
    
    def test_empty_prices(self):
        """Empty prices returns 0."""
        prices = np.array([])
        mdd = compute_max_drawdown(prices)
        assert mdd == 0.0
    
    def test_single_price(self):
        """Single price returns 0."""
        prices = np.array([100.0])
        mdd = compute_max_drawdown(prices)
        assert mdd == 0.0


class TestScoreBeta:
    """Tests for score_beta."""
    
    def test_low_beta(self):
        """Low beta gets high score."""
        score = score_beta(0.5)
        assert score > 70
    
    def test_neutral_beta(self):
        """Beta near 1.0 gets medium score."""
        score = score_beta(1.0)
        assert 40 <= score <= 60
    
    def test_high_beta(self):
        """High beta gets low score."""
        score = score_beta(2.0)
        assert score < 30
    
    def test_negative_beta(self):
        """Negative beta (unusual) treated as low."""
        score = score_beta(-0.5)
        assert score > 50
    
    def test_none_beta(self):
        """None beta returns neutral 50."""
        score = score_beta(None)
        assert score == 50.0


class TestScoreVolatility:
    """Tests for score_volatility."""
    
    def test_low_volatility(self):
        """Low volatility gets high score."""
        score = score_volatility(0.10)  # 10% annual vol
        assert score > 70
    
    def test_medium_volatility(self):
        """Medium volatility gets medium score."""
        score = score_volatility(0.25)  # 25% annual vol
        assert 40 <= score <= 60
    
    def test_high_volatility(self):
        """High volatility gets low score."""
        score = score_volatility(0.60)  # 60% annual vol
        assert score < 30
    
    def test_zero_volatility(self):
        """Zero volatility gets perfect score."""
        score = score_volatility(0.0)
        assert score == 100.0


class TestScoreMaxDrawdown:
    """Tests for score_max_drawdown."""
    
    def test_small_drawdown(self):
        """Small drawdown gets high score."""
        score = score_max_drawdown(0.10)  # 10% max drawdown
        assert score > 70
    
    def test_medium_drawdown(self):
        """Medium drawdown gets medium score."""
        score = score_max_drawdown(0.30)  # 30% max drawdown
        assert 40 <= score <= 60
    
    def test_large_drawdown(self):
        """Large drawdown gets low score."""
        score = score_max_drawdown(0.60)  # 60% max drawdown
        assert score < 30
    
    def test_zero_drawdown(self):
        """Zero drawdown gets perfect score."""
        score = score_max_drawdown(0.0)
        assert score == 100.0


class TestStabilityMetrics:
    """Tests for StabilityMetrics dataclass."""
    
    def test_to_dict(self):
        """Test to_dict method."""
        metrics = StabilityMetrics(
            ticker="AAPL",
            score=75.0,
            beta=0.9,
            volatility_252d=0.25,
            max_drawdown_5y=0.35,
            typical_dip_365=0.08,
            beta_score=60.0,
            volatility_score=70.0,
            drawdown_score=55.0,
            typical_dip_score=80.0,
            fundamental_stability_score=65.0,
            has_price_data=True,
            price_data_days=1260,
        )
        
        d = metrics.to_dict()
        
        assert d["ticker"] == "AAPL"
        assert d["score"] == 75.0
        assert d["beta"] == 0.9
        assert d["has_price_data"] is True
    
    def test_default_values(self):
        """Test default values."""
        metrics = StabilityMetrics(
            ticker="TEST",
            score=50.0,
        )
        
        assert metrics.beta is None
        assert metrics.volatility_252d is None
        assert metrics.has_price_data is False
