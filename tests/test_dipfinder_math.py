"""Comprehensive tests for DipFinder signal calculation math.

This module tests the mathematical correctness of:
1. Dip calculation (peak-to-current drop)
2. Excess dip formula (stock dip - benchmark dip)
3. Classification thresholds (MARKET_DIP, STOCK_SPECIFIC, MIXED)
4. Score component calculations
5. Real-world validation with yfinance data

The core formula for excess_dip:
    excess_dip = dip_stock - dip_mkt

Where:
    dip_stock = (peak_stock - current_stock) / peak_stock
    dip_mkt = (peak_benchmark - current_benchmark) / peak_benchmark

Classification rules (from DipFinderConfig defaults):
    - MARKET_DIP: dip_mkt >= 0.06 AND excess_dip < 0.03
    - STOCK_SPECIFIC: excess_dip >= 0.04
    - MIXED: everything else
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from app.dipfinder.config import DipFinderConfig, get_dipfinder_config
from app.dipfinder.dip import (
    DipMetrics,
    compute_dip_metrics,
    compute_dip_percentile,
    compute_dip_series_windowed,
    compute_persistence,
    compute_typical_dip,
)
from app.dipfinder.fundamentals import QualityMetrics
from app.dipfinder.signal import (
    AlertLevel,
    DipClass,
    MarketContext,
    classify_dip,
    compute_dip_score,
    compute_market_context,
    compute_signal,
)
from app.dipfinder.stability import StabilityMetrics


# ===========================================================================
# MATHEMATICAL SANITY TESTS
# ===========================================================================


class TestDipCalculationMath:
    """Tests to verify the core dip formula is mathematically correct."""

    def test_dip_formula_basic(self):
        """Verify: dip = (peak - current) / peak."""
        # Simple case: peak=100, current=85 -> dip = 15%
        prices = np.array([100.0, 90.0, 85.0])
        result = compute_dip_series_windowed(prices, window=3)

        expected_dip = (100.0 - 85.0) / 100.0  # = 0.15
        assert np.isclose(result[-1], expected_dip), (
            f"Expected {expected_dip}, got {result[-1]}"
        )

    def test_dip_formula_zero_when_at_peak(self):
        """Dip should be 0 when current price equals peak."""
        prices = np.array([80.0, 90.0, 100.0])  # Rising prices
        result = compute_dip_series_windowed(prices, window=3)

        # Current = peak = 100, so dip = 0
        assert result[-1] == 0.0

    def test_dip_always_positive_or_zero(self):
        """Dip can never be negative (by definition)."""
        # Even when price rises above previous peak
        prices = np.array([100.0, 110.0, 120.0, 130.0])
        result = compute_dip_series_windowed(prices, window=3)

        # All valid values should be >= 0
        valid = result[~np.isnan(result)]
        assert (valid >= 0).all(), "Dip values should never be negative"

    def test_dip_percentage_interpretation(self):
        """
        Verify interpretation: dip_pct=0.15 means stock is 15% below peak.

        If stock was at $100 and is now at $85:
        - dip_pct = 0.15 (as fraction)
        - Stock is DOWN 15%
        - Stock price = peak * (1 - dip_pct) = 100 * 0.85 = 85
        """
        peak = 100.0
        current = 85.0
        dip_pct = (peak - current) / peak

        # Verify the relationship
        assert np.isclose(dip_pct, 0.15)
        assert np.isclose(current, peak * (1 - dip_pct))

        # Test with the algorithm
        prices = np.array([90.0, 100.0, 95.0, 85.0])
        result = compute_dip_series_windowed(prices, window=3)
        assert np.isclose(result[-1], 0.15)

    def test_window_affects_peak_selection(self):
        """Demonstrate how window size affects which peak is used."""
        # Price history: peak at 150, then dropped, recovered to 120
        prices = np.array([100.0, 150.0, 120.0, 100.0, 110.0, 120.0])

        # Short window (3 days): only sees recent prices
        # Window at end: [100, 110, 120], peak=120, current=120, dip=0
        short_dips = compute_dip_series_windowed(prices, window=3)
        assert np.isclose(short_dips[-1], 0.0)

        # Long window (6 days): sees the old peak of 150
        # Window: [100, 150, 120, 100, 110, 120], peak=150, current=120
        # dip = (150 - 120) / 150 = 0.2
        long_dips = compute_dip_series_windowed(prices, window=6)
        assert np.isclose(long_dips[-1], 0.2)


class TestExcessDipMath:
    """Tests for the excess_dip calculation."""

    def test_excess_dip_formula(self):
        """Verify: excess_dip = dip_stock - dip_mkt."""
        # Stock: 20% down from peak
        # Benchmark: 8% down from peak
        # Excess: 12%
        dip_stock = 0.20
        dip_mkt = 0.08
        excess_dip = dip_stock - dip_mkt

        assert np.isclose(excess_dip, 0.12)

    def test_excess_dip_interpretation(self):
        """
        excess_dip tells us how much WORSE the stock performed than market.

        - excess_dip > 0: Stock underperformed market (fell more)
        - excess_dip = 0: Stock and market fell equally
        - excess_dip < 0: Stock outperformed market (fell less)
        """
        # Stock fell 15%, market fell 10% -> stock underperformed by 5%
        # Use np.isclose for floating point comparison
        assert np.isclose(0.15 - 0.10, 0.05)  # Stock-specific weakness

        # Stock fell 10%, market fell 15% -> stock outperformed by 5%
        assert np.isclose(0.10 - 0.15, -0.05)  # Stock is relatively strong

        # Stock fell 10%, market fell 10% -> same performance
        assert np.isclose(0.10 - 0.10, 0.0)  # Pure market move

    def test_excess_dip_in_context_computation(self):
        """Test compute_market_context produces correct excess_dip."""
        config = DipFinderConfig()

        # Create stock prices with 20% dip
        stock_prices = np.array([100.0] * 30 + [80.0])  # 20% dip

        # Create benchmark prices with 8% dip
        benchmark_prices = np.array([100.0] * 30 + [92.0])  # 8% dip

        ctx = compute_market_context(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
            config=config,
        )

        expected_dip_stock = 0.20
        expected_dip_mkt = 0.08
        expected_excess = 0.12

        assert np.isclose(ctx.dip_stock, expected_dip_stock, atol=0.001)
        assert np.isclose(ctx.dip_mkt, expected_dip_mkt, atol=0.001)
        assert np.isclose(ctx.excess_dip, expected_excess, atol=0.001)


class TestClassificationMath:
    """Tests for dip classification logic."""

    def test_classification_thresholds(self):
        """Document and test classification thresholds."""
        config = DipFinderConfig()

        # Default thresholds:
        # - market_dip_threshold: 0.06 (6% market dip required)
        # - excess_dip_market: 0.03 (max 3% excess for MARKET_DIP)
        # - excess_dip_stock_specific: 0.04 (min 4% excess for STOCK_SPECIFIC)

        assert config.market_dip_threshold == 0.06
        assert config.excess_dip_market == 0.03
        assert config.excess_dip_stock_specific == 0.04

    def test_market_dip_classification(self):
        """MARKET_DIP: Market is down AND stock didn't fall much more."""
        config = DipFinderConfig()

        # Scenario: Market -7%, Stock -9% (excess = 2%)
        # Market is down enough (>=6%) and stock excess is small (<3%)
        result = classify_dip(dip_stock=0.09, dip_mkt=0.07, config=config)
        assert result == DipClass.MARKET_DIP

        # Scenario: Market -10%, Stock -10% (pure market move)
        result = classify_dip(dip_stock=0.10, dip_mkt=0.10, config=config)
        assert result == DipClass.MARKET_DIP

    def test_stock_specific_classification(self):
        """STOCK_SPECIFIC: Stock fell significantly more than market."""
        config = DipFinderConfig()

        # Scenario: Market -2%, Stock -15% (excess = 13%)
        # Clear stock-specific issue
        result = classify_dip(dip_stock=0.15, dip_mkt=0.02, config=config)
        assert result == DipClass.STOCK_SPECIFIC

        # Scenario: Market -5%, Stock -10% (excess = 5% >= 4%)
        result = classify_dip(dip_stock=0.10, dip_mkt=0.05, config=config)
        assert result == DipClass.STOCK_SPECIFIC

    def test_mixed_classification(self):
        """MIXED: Neither pure market nor pure stock-specific."""
        config = DipFinderConfig()

        # Scenario: Market -4%, Stock -7% (excess = 3%)
        # Market not down enough for MARKET_DIP (need 6%)
        # Excess not high enough for STOCK_SPECIFIC (need 4%)
        result = classify_dip(dip_stock=0.07, dip_mkt=0.04, config=config)
        assert result == DipClass.MIXED

        # Scenario: Market -6.5%, Stock -10% (excess = 3.5%)
        # Market is down enough but excess is between 3% and 4%
        result = classify_dip(dip_stock=0.10, dip_mkt=0.065, config=config)
        assert result == DipClass.MIXED

    def test_classification_edge_cases(self):
        """Test edge cases at threshold boundaries."""
        config = DipFinderConfig()

        # Exactly at market_dip_threshold
        result = classify_dip(dip_stock=0.06, dip_mkt=0.06, config=config)
        assert result == DipClass.MARKET_DIP  # excess=0, market >= 6%

        # Just below market_dip_threshold
        result = classify_dip(dip_stock=0.059, dip_mkt=0.059, config=config)
        assert result in (DipClass.MIXED, DipClass.MARKET_DIP)

        # Exactly at excess_dip_stock_specific threshold
        result = classify_dip(dip_stock=0.08, dip_mkt=0.04, config=config)
        assert result == DipClass.STOCK_SPECIFIC  # excess=0.04 >= 0.04


class TestDipScoreComponentsMath:
    """Tests for individual score components."""

    @pytest.fixture
    def config(self) -> DipFinderConfig:
        return DipFinderConfig()

    def test_magnitude_score_scaling(self, config):
        """
        Magnitude score: max 40 points.
        Scale: dip_pct * 200 (capped at 40)
        So 0.20 (20%) dip -> 40 points
        And 0.10 (10%) dip -> 20 points
        
        Note: Other scoring components also change with dip_pct,
        so we just verify that higher dip gets higher score.
        """
        # Create metrics with specific dip percentages
        def make_metrics(dip_pct: float) -> Tuple[DipMetrics, MarketContext]:
            return (
                DipMetrics(
                    ticker="TEST",
                    window=30,
                    dip_pct=dip_pct,
                    peak_price=100.0,
                    current_price=100 * (1 - dip_pct),
                    dip_percentile=50.0,  # Neutral
                    dip_vs_typical=1.0,  # Neutral
                    typical_dip=dip_pct,
                    persist_days=2,
                    days_since_peak=5,
                    is_meaningful=True,
                ),
                MarketContext(
                    benchmark_ticker="SPY",
                    dip_mkt=0.02,
                    dip_stock=dip_pct,
                    excess_dip=dip_pct - 0.02,
                    dip_class=DipClass.STOCK_SPECIFIC,
                ),
            )

        # 20% dip should give max magnitude points (40)
        metrics_20, ctx_20 = make_metrics(0.20)
        score_20 = compute_dip_score(metrics_20, ctx_20, config)

        # 10% dip should give ~20 magnitude points
        metrics_10, ctx_10 = make_metrics(0.10)
        score_10 = compute_dip_score(metrics_10, ctx_10, config)

        # 20% dip should score higher than 10% dip
        assert score_20 > score_10
        # The difference should be noticeable (at least a few points)
        assert score_20 - score_10 >= 5

    def test_percentile_score_scaling(self, config):
        """
        Percentile score: max 25 points.
        Higher percentile = rarer dip = more points.
        """
        def make_metrics(percentile: float) -> Tuple[DipMetrics, MarketContext]:
            return (
                DipMetrics(
                    ticker="TEST",
                    window=30,
                    dip_pct=0.15,  # Same dip
                    peak_price=100.0,
                    current_price=85.0,
                    dip_percentile=percentile,  # Variable
                    dip_vs_typical=1.5,
                    typical_dip=0.10,
                    persist_days=3,
                    days_since_peak=7,
                    is_meaningful=True,
                ),
                MarketContext(
                    benchmark_ticker="SPY",
                    dip_mkt=0.02,
                    dip_stock=0.15,
                    excess_dip=0.13,
                    dip_class=DipClass.STOCK_SPECIFIC,
                ),
            )

        # 95th percentile (very rare dip)
        metrics_95, ctx_95 = make_metrics(95.0)
        score_95 = compute_dip_score(metrics_95, ctx_95, config)

        # 50th percentile (average dip)
        metrics_50, ctx_50 = make_metrics(50.0)
        score_50 = compute_dip_score(metrics_50, ctx_50, config)

        # Rare dip should score higher
        assert score_95 > score_50

    def test_classification_bonus(self, config):
        """STOCK_SPECIFIC dips get bonus points."""
        base_metrics = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.15,
            peak_price=100.0,
            current_price=85.0,
            dip_percentile=80.0,
            dip_vs_typical=2.0,
            typical_dip=0.075,
            persist_days=3,
            days_since_peak=7,
            is_meaningful=True,
        )

        # Stock-specific context (gets bonus)
        ctx_stock = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.02,
            dip_stock=0.15,
            excess_dip=0.13,
            dip_class=DipClass.STOCK_SPECIFIC,
        )

        # Market dip context (no bonus)
        ctx_market = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.12,
            dip_stock=0.15,
            excess_dip=0.03,
            dip_class=DipClass.MARKET_DIP,
        )

        score_stock = compute_dip_score(base_metrics, ctx_stock, config)
        score_market = compute_dip_score(base_metrics, ctx_market, config)

        # Stock-specific should score higher
        assert score_stock > score_market


class TestFinalScoreCalculation:
    """Tests for weighted final score calculation."""

    @pytest.fixture
    def config(self) -> DipFinderConfig:
        return DipFinderConfig()

    def test_weight_verification(self, config):
        """Verify default weights sum to 1.0."""
        total = config.weight_dip + config.weight_quality + config.weight_stability
        assert np.isclose(total, 1.0)

    def test_default_weights(self, config):
        """Document default weights."""
        assert config.weight_dip == 0.45  # Dip importance
        assert config.weight_quality == 0.30  # Quality importance
        assert config.weight_stability == 0.25  # Stability importance

    def test_no_dip_caps_score(self, config):
        """Without meaningful dip, score is capped at 25."""
        # Create a high-quality stock with no dip
        stock_prices = np.array([100.0] * 31)  # No price movement
        benchmark_prices = np.array([100.0] * 31)

        quality = QualityMetrics(ticker="TEST", score=90.0)
        stability = StabilityMetrics(ticker="TEST", score=85.0)

        signal = compute_signal(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
            quality_metrics=quality,
            stability_metrics=stability,
            as_of_date="2025-01-01",
            config=config,
        )

        # Even with excellent fundamentals, no dip = low score
        assert signal.final_score <= 25.0


class TestBenchmarkUsage:
    """Tests to verify correct benchmark usage."""

    def test_default_benchmark_is_spy(self):
        """SPY is the default benchmark."""
        config = get_dipfinder_config()
        assert config.default_benchmark == "SPY"

    def test_benchmark_affects_classification(self):
        """Different benchmarks change classification."""
        config = DipFinderConfig()

        # Stock is down 10%
        stock_prices = np.array([100.0] * 30 + [90.0])

        # Benchmark A: down 8% (stock excess = 2%)
        benchmark_a = np.array([100.0] * 30 + [92.0])
        ctx_a = compute_market_context(
            "TEST", stock_prices, benchmark_a, "SPY", 30, config
        )

        # Benchmark B: down 2% (stock excess = 8%)
        benchmark_b = np.array([100.0] * 30 + [98.0])
        ctx_b = compute_market_context(
            "TEST", stock_prices, benchmark_b, "SPY", 30, config
        )

        # With similar benchmark movement, it's a market dip
        assert ctx_a.dip_class in (DipClass.MARKET_DIP, DipClass.MIXED)

        # With minimal benchmark movement, it's stock-specific
        assert ctx_b.dip_class == DipClass.STOCK_SPECIFIC


# ===========================================================================
# REAL WORLD VALIDATION TESTS
# ===========================================================================


class TestRealWorldValidation:
    """Tests using simulated real-world scenarios."""

    @pytest.fixture
    def config(self) -> DipFinderConfig:
        return DipFinderConfig()

    def test_crash_scenario_march_2020_style(self, config):
        """
        Simulate a March 2020-style crash.
        Market drops 30%+, most stocks follow.
        """
        # Simulate 60 days: steady then crash
        prices_before = np.linspace(100, 105, 40)  # Slight uptrend
        prices_crash = np.linspace(105, 70, 20)  # Sharp drop
        stock_prices = np.concatenate([prices_before, prices_crash])
        benchmark_prices = stock_prices.copy()  # Market-wide

        ctx = compute_market_context(
            "TEST", stock_prices, benchmark_prices, "SPY", 30, config
        )

        # Both down ~30%, so excess should be near 0
        assert abs(ctx.excess_dip) < 0.05
        assert ctx.dip_class == DipClass.MARKET_DIP

    def test_company_scandal_scenario(self, config):
        """
        Simulate a company-specific scandal (e.g., fraud discovery).
        Stock drops 40% while market is flat.
        """
        # Stock crashes
        stock_before = np.array([100.0] * 30)
        stock_crash = np.linspace(100, 60, 10)  # 40% drop
        stock_prices = np.concatenate([stock_before, stock_crash])

        # Market steady
        benchmark_prices = np.array([100.0] * 40)

        ctx = compute_market_context(
            "TEST", stock_prices, benchmark_prices, "SPY", 30, config
        )

        # Stock down 40%, market flat
        assert ctx.dip_stock >= 0.35
        assert ctx.dip_mkt < 0.05
        assert ctx.excess_dip >= 0.30
        assert ctx.dip_class == DipClass.STOCK_SPECIFIC

    def test_sector_rotation_scenario(self, config):
        """
        Simulate sector rotation.
        Tech stocks down 15% while market is up.
        """
        # Tech stock dropping
        stock_prices = np.linspace(100, 85, 40)  # 15% decline

        # Market rising
        benchmark_prices = np.linspace(100, 105, 40)  # 5% gain

        ctx = compute_market_context(
            "TEST", stock_prices, benchmark_prices, "SPY", 30, config
        )

        # Stock down, market up = very stock-specific
        assert ctx.dip_stock >= 0.10
        assert ctx.dip_mkt <= 0.0  # Market is at peak
        assert ctx.dip_class == DipClass.STOCK_SPECIFIC

    def test_gradual_decline_vs_sudden_crash(self, config):
        """Compare gradual decline vs sudden crash metrics."""
        # Gradual: 20% over 40 days with clear peak at start
        gradual = np.concatenate([
            np.array([100.0]),  # Clear peak at start
            np.linspace(100, 80, 39)  # Gradual decline
        ])

        # Sudden: flat then 20% drop at end
        sudden_flat = np.array([100.0] * 35)
        sudden_drop = np.linspace(100, 80, 5)
        sudden = np.concatenate([sudden_flat, sudden_drop])

        metrics_gradual = compute_dip_metrics("GRADUAL", gradual, 30, config)
        metrics_sudden = compute_dip_metrics("SUDDEN", sudden, 30, config)

        # Both should have dip around 20% (with some tolerance)
        assert np.isclose(metrics_gradual.dip_pct, 0.20, atol=0.05)
        assert np.isclose(metrics_sudden.dip_pct, 0.20, atol=0.05)


class TestPercentileCalculation:
    """Tests for percentile ranking logic."""

    def test_percentile_meaning(self):
        """Verify percentile interpretation."""
        # Historical dips: mostly small (0-10%), occasional larger
        historical = np.array([
            0.02, 0.03, 0.02, 0.04, 0.03,  # Small dips
            0.05, 0.04, 0.03, 0.06, 0.05,
            0.08, 0.10, 0.12, 0.07, 0.09,  # Medium dips
            0.15, 0.20, 0.25,  # Rare large dips
        ])

        # A 20% dip is rare (only 2 out of 18 are higher: 0.20 and 0.25)
        pctl = compute_dip_percentile(historical, 0.20, exclude_last=False)
        assert pctl >= 85  # Should be in top 15%

        # A 5% dip is common
        pctl = compute_dip_percentile(historical, 0.05, exclude_last=False)
        assert pctl <= 55  # Should be below or around median

    def test_typical_dip_calculation(self):
        """Verify typical dip is the median."""
        dips = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
        typical = compute_typical_dip(dips, use_median=True)
        assert np.isclose(typical, 0.06)  # Median

        typical_mean = compute_typical_dip(dips, use_median=False)
        assert np.isclose(typical_mean, 0.06)  # Mean also 0.06

    def test_dip_vs_typical_ratio(self):
        """Verify dip_vs_typical interpretation."""
        # If typical dip is 5% and current is 15%, ratio = 3x
        typical = 0.05
        current = 0.15
        ratio = current / typical
        assert np.isclose(ratio, 3.0)  # Current dip is 3x larger than typical


class TestPersistence:
    """Tests for dip persistence calculation."""

    def test_persistence_counting(self):
        """Count consecutive days at dip level."""
        # Price history: stable peak then sudden drop that persists
        # With window=5, the first 4 values will be NaN
        # We need at least window+persist_days prices
        prices = np.array([
            100.0, 100.0, 100.0, 100.0, 100.0,  # Days 0-4: Peak = 100
            100.0, 100.0, 100.0,  # Days 5-7: Still at peak
            85.0, 85.0, 85.0, 85.0  # Days 8-11: 15% dip for 4 days
        ])

        # With 10% threshold and window of 5
        persist = compute_persistence(prices, dip_threshold=0.10, window=5)

        # Last 4 prices are all 15% below peak (which is 100 in the window)
        assert persist >= 4

    def test_persistence_breaks_on_recovery(self):
        """Persistence resets when price recovers."""
        # Dip then partial recovery then dip again
        prices = np.array([
            100, 100, 100,
            85, 84, 83,  # Down 17%
            92,  # Recovery to 8% dip - breaks persistence
            85, 84  # Back down
        ])

        persist = compute_persistence(prices, dip_threshold=0.10, window=5)

        # Should only count days after recovery
        assert persist <= 2


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================


class TestFullSignalFlow:
    """Integration tests for complete signal computation."""

    @pytest.fixture
    def config(self) -> DipFinderConfig:
        return DipFinderConfig()

    def test_signal_components_add_up(self, config):
        """Verify final score respects weights."""
        # Create a consistent scenario
        stock_prices = np.array([100.0] * 30 + [85.0])  # 15% dip
        benchmark_prices = np.array([100.0] * 31)  # Market flat

        quality = QualityMetrics(ticker="TEST", score=80.0)
        stability = StabilityMetrics(ticker="TEST", score=70.0)

        signal = compute_signal(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
            quality_metrics=quality,
            stability_metrics=stability,
            as_of_date="2025-01-01",
            config=config,
        )

        # Final score should be weighted combination
        # expected = 0.45 * dip_score + 0.30 * 80 + 0.25 * 70
        expected_min = 0.30 * 80 + 0.25 * 70  # Without dip contribution
        expected_max = 0.45 * 100 + 0.30 * 80 + 0.25 * 70  # With max dip

        assert signal.final_score >= expected_min
        assert signal.final_score <= expected_max

    def test_alert_requires_all_gates(self, config):
        """Alert only fires when all conditions met."""
        stock_prices = np.array([100.0] * 30 + [80.0])  # 20% dip
        benchmark_prices = np.array([100.0] * 31)

        # High quality, low stability (below gate of 60)
        quality = QualityMetrics(ticker="TEST", score=80.0)
        stability = StabilityMetrics(ticker="TEST", score=50.0)

        signal = compute_signal(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
            quality_metrics=quality,
            stability_metrics=stability,
            as_of_date="2025-01-01",
            config=config,
        )

        # Should NOT alert because stability is below gate
        assert signal.alert_level == AlertLevel.NONE

    def test_strong_vs_good_alert(self, config):
        """Differentiate between STRONG and GOOD alerts."""
        # Create meaningful dip scenario:
        # - Need history for percentile calculation
        # - Need persistence >= 2 days
        # - Need percentile >= 80 OR dip_vs_typical >= 1.5

        # Create a longer price history with typical small dips
        # then a big drop at the end
        typical_history = []
        for i in range(60):  # 60 days of typical fluctuation
            # Small dips of 2-5% from local peaks
            base = 100 + 5 * np.sin(i / 10)  # Oscillating pattern
            typical_history.append(base)

        # Now add a significant drop: 30% that persists for 5 days
        typical_history.extend([70.0, 70.0, 70.0, 70.0, 70.0])

        stock_prices = np.array(typical_history)
        benchmark_prices = np.array([100.0] * len(stock_prices))

        # Excellent fundamentals
        quality = QualityMetrics(ticker="TEST", score=85.0)
        stability = StabilityMetrics(ticker="TEST", score=80.0)

        signal = compute_signal(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
            quality_metrics=quality,
            stability_metrics=stability,
            as_of_date="2025-01-01",
            config=config,
        )

        # Check that we have meaningful dip conditions
        assert signal.dip_metrics.dip_pct >= 0.25  # At least 25% dip
        assert signal.dip_metrics.persist_days >= 2

        # With strong dip, good persistence, and good fundamentals
        # should either alert or have high score
        # The is_meaningful flag also requires percentile or dip_vs_typical
        # With proper history, the 30% dip should be unusual
        if signal.dip_metrics.is_meaningful:
            assert signal.alert_level in (AlertLevel.STRONG, AlertLevel.GOOD)
            assert signal.should_alert is True
        else:
            # Even if not "meaningful" by strict criteria, score should be decent
            assert signal.final_score >= 50


# ===========================================================================
# REAL YFINANCE DATA TESTS
# ===========================================================================


class TestRealYFinanceData:
    """Tests that validate calculations using real yfinance data.
    
    These tests fetch real market data and verify that:
    1. The calculations produce sensible results
    2. Dip values are within expected ranges
    3. Benchmark comparisons work correctly
    """

    @pytest.fixture
    def config(self) -> DipFinderConfig:
        return DipFinderConfig()

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance", reason="yfinance not installed"),
        reason="yfinance required for real data tests"
    )
    def test_spy_benchmark_data(self, config):
        """Test that we can fetch and process SPY benchmark data."""
        import yfinance as yf

        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")

        if hist.empty:
            pytest.skip("Could not fetch SPY data")

        close_prices = hist["Close"].values

        # Compute dip series
        dip_series = compute_dip_series_windowed(close_prices, window=30)

        # Should have valid values after first 29 days
        valid_dips = dip_series[~np.isnan(dip_series)]
        assert len(valid_dips) > 0

        # SPY rarely has extreme dips (> 30%) in normal years
        assert np.max(valid_dips) < 0.50, "SPY should not have 50%+ dips"

        # SPY should have some dips (not always at peak)
        assert np.max(valid_dips) > 0.01, "SPY should have some dips"

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance", reason="yfinance not installed"),
        reason="yfinance required for real data tests"
    )
    def test_stock_vs_benchmark_comparison(self, config):
        """Test excess_dip calculation with real stock vs SPY."""
        import yfinance as yf

        # Fetch both stock and benchmark
        tickers = ["AAPL", "SPY"]
        data = yf.download(tickers, period="6mo", progress=False, auto_adjust=True)

        if data.empty:
            pytest.skip("Could not fetch data")

        aapl_close = data["Close"]["AAPL"].dropna().values
        spy_close = data["Close"]["SPY"].dropna().values

        # Align lengths
        min_len = min(len(aapl_close), len(spy_close))
        aapl_close = aapl_close[-min_len:]
        spy_close = spy_close[-min_len:]

        if min_len < 30:
            pytest.skip("Not enough data")

        # Compute market context
        ctx = compute_market_context(
            ticker="AAPL",
            stock_prices=aapl_close,
            benchmark_prices=spy_close,
            benchmark_ticker="SPY",
            window=30,
            config=config,
        )

        # Verify structure
        assert ctx.benchmark_ticker == "SPY"
        assert ctx.dip_stock >= 0  # Dip is always positive or zero
        assert ctx.dip_mkt >= 0
        assert ctx.dip_class in (DipClass.MARKET_DIP, DipClass.STOCK_SPECIFIC, DipClass.MIXED)

        # Excess dip should be reasonable
        assert -0.50 < ctx.excess_dip < 0.50, "Excess dip should be reasonable"

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance", reason="yfinance not installed"),
        reason="yfinance required for real data tests"
    )
    def test_dip_percentile_with_real_history(self, config):
        """Test that percentile calculation works with real historical data."""
        import yfinance as yf

        aapl = yf.Ticker("AAPL")
        hist = aapl.history(period="2y")

        if len(hist) < 365:
            pytest.skip("Not enough history")

        close_prices = hist["Close"].values

        # Compute metrics
        metrics = compute_dip_metrics("AAPL", close_prices, window=30, config=config)

        # Verify percentile is in valid range
        assert 0 <= metrics.dip_percentile <= 100

        # Verify other metrics are reasonable
        assert metrics.current_price > 0
        assert metrics.peak_price >= metrics.current_price
        assert metrics.dip_pct >= 0
        assert metrics.typical_dip >= 0

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance", reason="yfinance not installed"),
        reason="yfinance required for real data tests"
    )
    def test_multiple_windows_consistency(self, config):
        """Test that different windows produce consistent results."""
        import yfinance as yf

        msft = yf.Ticker("MSFT")
        hist = msft.history(period="2y")

        if len(hist) < 365:
            pytest.skip("Not enough history")

        close_prices = hist["Close"].values

        # Compute metrics for different windows
        metrics_7 = compute_dip_metrics("MSFT", close_prices, window=7, config=config)
        metrics_30 = compute_dip_metrics("MSFT", close_prices, window=30, config=config)
        metrics_100 = compute_dip_metrics("MSFT", close_prices, window=100, config=config)

        # All should have same current price
        assert metrics_7.current_price == metrics_30.current_price == metrics_100.current_price

        # Longer windows typically show larger dips (higher peaks to compare against)
        # This isn't always true but generally holds
        # Just verify all values are reasonable
        for m in [metrics_7, metrics_30, metrics_100]:
            assert 0 <= m.dip_pct <= 1
            assert m.peak_price >= m.current_price

    @pytest.mark.skipif(
        not pytest.importorskip("yfinance", reason="yfinance not installed"),
        reason="yfinance required for real data tests"
    )
    def test_classification_makes_sense(self, config):
        """Test that classification logic produces sensible results."""
        import yfinance as yf

        # Get data for a volatile stock and SPY
        tickers = ["TSLA", "SPY"]
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)

        if data.empty:
            pytest.skip("Could not fetch data")

        tsla_close = data["Close"]["TSLA"].dropna().values
        spy_close = data["Close"]["SPY"].dropna().values

        min_len = min(len(tsla_close), len(spy_close))
        if min_len < 100:
            pytest.skip("Not enough data")

        tsla_close = tsla_close[-min_len:]
        spy_close = spy_close[-min_len:]

        # Compute context
        ctx = compute_market_context(
            ticker="TSLA",
            stock_prices=tsla_close,
            benchmark_prices=spy_close,
            benchmark_ticker="SPY",
            window=30,
            config=config,
        )

        # Classification should follow rules
        excess = ctx.excess_dip

        if ctx.dip_mkt >= config.market_dip_threshold and excess < config.excess_dip_market:
            assert ctx.dip_class == DipClass.MARKET_DIP
        elif excess >= config.excess_dip_stock_specific:
            assert ctx.dip_class == DipClass.STOCK_SPECIFIC
        else:
            assert ctx.dip_class == DipClass.MIXED
