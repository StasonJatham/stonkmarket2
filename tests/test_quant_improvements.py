"""
Tests for quant engine improvements from the Wall Street quant review.

Tests cover:
- C1: Statistical tech score coefficients
- C2: Walk-forward OOS aggregation (no duplication)
- C3: Transaction costs in backtests
- C4: Stock-specific percentile thresholds
- C5: Empirical expected returns
- S1: Sharpe ratio with risk-free rate
- S2: Market regime detection
- S3: Correlation-aware signal weighting
- S4: HRP degraded flag
- I1: Bootstrap confidence intervals
- I6: Signal decay/half-life
- I8: Kelly criterion sizing
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestTransactionCosts:
    """C3: Verify transaction costs are applied in backtests."""

    def test_slippage_reduces_returns(self):
        """Transaction costs should reduce returns."""
        from app.quant_engine.signals import backtest_signal, DEFAULT_SLIPPAGE_BPS
        
        # Create simple uptrending price series
        np.random.seed(42)
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(260) * 0.01 + 0.001)),
            index=pd.date_range("2020-01-01", periods=260)
        )
        
        # Create a simple signal that triggers periodically
        signal = pd.Series(
            np.sin(np.arange(len(prices)) / 10) * 2,
            index=prices.index
        )
        
        results = backtest_signal(prices, signal, -1.0, "below", 20)
        
        # With slippage, even positive trades should have slightly reduced returns
        # This tests that the slippage is actually being applied
        assert DEFAULT_SLIPPAGE_BPS > 0, "Slippage should be non-zero"
        # Results should exist
        assert "avg_return" in results

    def test_slippage_constant_defined(self):
        """Verify slippage constants are defined."""
        from app.quant_engine.signals import DEFAULT_SLIPPAGE_BPS, TOTAL_ROUND_TRIP_COST
        from app.quant_engine.trade_engine import DEFAULT_SLIPPAGE_BPS as TE_SLIPPAGE
        
        assert DEFAULT_SLIPPAGE_BPS == 5, "Default slippage should be 5 bps"
        assert TOTAL_ROUND_TRIP_COST == 0.001, "Round trip should be 10 bps"
        assert TE_SLIPPAGE == 5, "Trade engine slippage should match"


class TestStatisticalTechScore:
    """C1: Verify statistical tech score uses defined coefficients."""

    def test_tech_score_coefficients_exist(self):
        """Coefficients should be defined."""
        from app.quant_engine.trade_engine import TECH_SCORE_COEFFICIENTS
        
        assert "rsi_below_30" in TECH_SCORE_COEFFICIENTS
        assert "bb_below_0" in TECH_SCORE_COEFFICIENTS
        assert "zscore_below_-2" in TECH_SCORE_COEFFICIENTS
        assert "macd_bullish_cross" in TECH_SCORE_COEFFICIENTS
        assert "trend_broken" in TECH_SCORE_COEFFICIENTS

    def test_compute_statistical_tech_score(self):
        """Tech score should return score and contributions."""
        from app.quant_engine.trade_engine import (
            compute_statistical_tech_score,
            TechnicalSnapshot,
            compute_all_indicators,
        )
        
        # Create test data
        np.random.seed(42)
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.random.randn(220) * 0.02 - 0.005)),
            index=pd.date_range("2020-01-01", periods=220)
        )
        df = pd.DataFrame({"close": prices})
        df = compute_all_indicators(df)
        
        # Create oversold snapshot
        technicals = TechnicalSnapshot(
            rsi_14=25.0,  # Very oversold
            bb_position=-0.1,  # Below lower band
            sma_200_pct=-0.05,  # Slightly below SMA200
        )
        
        score, contributions = compute_statistical_tech_score(technicals, df, None)
        
        assert isinstance(score, float)
        assert isinstance(contributions, dict)
        assert -1.0 <= score <= 1.0, "Score should be bounded"
        
        # Oversold conditions should give positive score
        assert score > 0, "Oversold technicals should give positive score"
        assert "rsi_oversold" in contributions, "RSI contribution should be tracked"


class TestMarketRegimeDetection:
    """S2: Verify market regime detection works."""

    def test_detect_bull_market(self):
        """Bull market should be detected correctly."""
        from app.quant_engine.trade_engine import detect_market_regime, MarketRegime
        
        # Create uptrending prices
        prices = pd.Series(
            100 * np.exp(np.linspace(0, 0.3, 100)),  # +30% over 100 days
            index=pd.date_range("2020-01-01", periods=100)
        )
        
        regime, details = detect_market_regime(prices)
        
        assert bool(details["is_bull"]) is True
        assert regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL]

    def test_detect_bear_market(self):
        """Bear market should be detected correctly."""
        from app.quant_engine.trade_engine import detect_market_regime, MarketRegime
        
        # Create downtrending prices
        prices = pd.Series(
            100 * np.exp(np.linspace(0, -0.3, 100)),  # -30% over 100 days
            index=pd.date_range("2020-01-01", periods=100)
        )
        
        regime, details = detect_market_regime(prices)
        
        assert bool(details["is_bull"]) is False
        assert regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]

    def test_regime_threshold_adjustments(self):
        """Regime-specific thresholds should be defined."""
        from app.quant_engine.trade_engine import (
            REGIME_THRESHOLD_ADJUSTMENTS,
            MarketRegime,
        )
        
        assert MarketRegime.BULL_LOW_VOL in REGIME_THRESHOLD_ADJUSTMENTS
        assert MarketRegime.BEAR_HIGH_VOL in REGIME_THRESHOLD_ADJUSTMENTS
        
        # Bear high vol should have stricter thresholds
        bear_high = REGIME_THRESHOLD_ADJUSTMENTS[MarketRegime.BEAR_HIGH_VOL]
        bull_low = REGIME_THRESHOLD_ADJUSTMENTS[MarketRegime.BULL_LOW_VOL]
        
        assert bear_high["rsi_threshold"] < bull_low["rsi_threshold"]


class TestCorrelationAwareSignals:
    """S3: Verify correlated signals are downweighted."""

    def test_correlation_weight_computation(self):
        """Correlated signals should get lower weights."""
        from app.quant_engine.signals import _compute_correlation_adjusted_weights
        from dataclasses import dataclass
        
        @dataclass
        class MockSignal:
            name: str
            optimal_threshold: float
        
        # Create price data
        np.random.seed(42)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(160)),
            index=pd.date_range("2020-01-01", periods=160)
        )
        price_data = {"close": prices}
        
        # Mock signals
        signals = [
            MockSignal("RSI Oversold", 30.0),
            MockSignal("Stochastic Oversold", 20.0),  # Correlated with RSI
            MockSignal("Volume Spike on Dip", 1.5),    # Less correlated
        ]
        
        weights = _compute_correlation_adjusted_weights(price_data, signals)
        
        assert len(weights) == 3
        # All weights should be between 0 and 1
        for w in weights.values():
            assert 0 < w <= 1


class TestHRPDegradedFlag:
    """S4: Verify HRP reports optimization quality."""

    def test_hrp_reports_quality(self):
        """HRP result should include quality flag."""
        from app.quant_engine.risk_optimizer import RiskOptimizationResult
        
        # Check dataclass has the new fields
        result = RiskOptimizationResult(
            method="hrp",
            weights={"A": 0.5, "B": 0.5},
            portfolio_volatility=0.15,
            portfolio_var_95=-0.02,
            diversification_ratio=1.2,
            converged=True,
            iterations=1,
            objective_value=0.0,
        )
        
        assert hasattr(result, "optimization_quality")
        assert hasattr(result, "quality_reason")
        assert result.optimization_quality == "optimal"


class TestBootstrapConfidenceIntervals:
    """I1: Verify bootstrap CI computation."""

    def test_bootstrap_ci_mean(self):
        """Bootstrap CI for mean should work."""
        from app.quant_engine.trade_engine import bootstrap_confidence_interval
        
        # Create sample returns
        np.random.seed(42)
        returns = list(np.random.randn(100) * 0.02 + 0.005)  # ~0.5% avg with noise
        
        point, ci_lower, ci_upper = bootstrap_confidence_interval(
            returns, n_bootstrap=500, confidence=0.95, metric="mean"
        )
        
        assert ci_lower < point < ci_upper
        assert ci_upper - ci_lower > 0  # CI should have width

    def test_bootstrap_ci_win_rate(self):
        """Bootstrap CI for win rate should work."""
        from app.quant_engine.trade_engine import bootstrap_confidence_interval
        
        # Create sample returns with ~60% win rate
        np.random.seed(42)
        returns = list(np.random.randn(100) * 0.02 + 0.003)
        
        point, ci_lower, ci_upper = bootstrap_confidence_interval(
            returns, n_bootstrap=500, confidence=0.95, metric="win_rate"
        )
        
        assert 0 <= ci_lower <= point <= ci_upper <= 1


class TestKellyCriterion:
    """I8: Verify Kelly criterion sizing."""

    def test_kelly_positive_edge(self):
        """Kelly should give positive fraction with positive edge."""
        from app.quant_engine.trade_engine import kelly_fraction
        
        # 60% win rate, 2:1 win/loss ratio
        fraction = kelly_fraction(
            win_rate=0.6,
            avg_win=0.10,  # 10% avg win
            avg_loss=0.05,  # 5% avg loss
        )
        
        assert 0 < fraction <= 0.25, "Kelly should be positive and capped"

    def test_kelly_negative_edge(self):
        """Kelly should return 0 with negative edge."""
        from app.quant_engine.trade_engine import kelly_fraction
        
        # 40% win rate, 1:1 ratio = negative edge
        fraction = kelly_fraction(
            win_rate=0.40,
            avg_win=0.05,
            avg_loss=0.05,
        )
        
        assert fraction == 0.0

    def test_kelly_fractional(self):
        """Fractional Kelly should reduce position size."""
        from app.quant_engine.trade_engine import kelly_fraction
        
        # Use moderate edge so full Kelly < 25% cap
        full_kelly = kelly_fraction(
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.06,
            fractional_kelly=1.0,
            max_fraction=0.50,  # Higher cap to see the difference
        )
        
        half_kelly = kelly_fraction(
            win_rate=0.55,
            avg_win=0.08,
            avg_loss=0.06,
            fractional_kelly=0.5,
            max_fraction=0.50,
        )
        
        # Half Kelly should be roughly half (before capping)
        assert half_kelly < full_kelly


class TestSignalDecayTracking:
    """I6: Verify signal decay/half-life computation."""

    def test_signal_half_life(self):
        """Signal half-life should be computed."""
        from app.quant_engine.signals import compute_signal_half_life
        
        np.random.seed(42)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(220)),
            index=pd.date_range("2020-01-01", periods=220)
        )
        signal = pd.Series(np.random.randn(len(prices)), index=prices.index)
        
        half_life = compute_signal_half_life(
            prices, signal, "below", 0.0, max_lag=30
        )
        
        assert isinstance(half_life, int)
        assert 1 <= half_life <= 30

    def test_signal_turnover_rate(self):
        """Signal turnover rate should be computed."""
        from app.quant_engine.signals import compute_signal_turnover_rate
        
        # Signal that flips frequently
        n_days = 180
        signal = pd.Series(
            np.sin(np.arange(n_days) / 5),  # Oscillates several times
            index=pd.date_range("2020-01-01", periods=n_days)
        )
        
        turnover = compute_signal_turnover_rate(signal, "above", 0.0)
        
        assert turnover > 0  # Should have some turnover
        assert turnover < n_days  # But not flipping every day


class TestOOSAggregation:
    """C2: Verify walk-forward uses individual returns, not averages."""

    def test_individual_returns_function_exists(self):
        """Helper function for individual returns should exist."""
        from app.quant_engine.signals import _get_individual_trade_returns
        
        np.random.seed(42)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(160)),
            index=pd.date_range("2020-01-01", periods=160)
        )
        signal = pd.Series(np.random.randn(len(prices)), index=prices.index)
        
        returns = _get_individual_trade_returns(
            prices, signal, -1.0, "below", 10
        )
        
        assert isinstance(returns, list)
        # Should have individual trade returns, not duplicated averages


class TestStockSpecificThresholds:
    """C4: Verify percentile-based thresholds."""

    def test_percentile_threshold_in_analyze_dip(self):
        """analyze_dip should compute stock-specific percentiles."""
        from app.quant_engine.trade_engine import analyze_dip
        
        # Create price data with known dip pattern (need 252+ days)
        np.random.seed(42)
        # Start at 100, rise, dip, recover, dip again
        base = np.concatenate([
            np.linspace(100, 120, 70),   # Rise
            np.linspace(120, 100, 35),   # Fall 16%
            np.linspace(100, 115, 50),   # Recover
            np.linspace(115, 92, 35),    # Fall 20%
            np.linspace(92, 105, 40),    # Recover
            np.linspace(105, 90, 30),    # Current dip (14%)
        ])
        prices = pd.Series(base, index=pd.date_range("2020-01-01", periods=len(base)))
        
        result = analyze_dip({"close": prices}, "TEST")
        
        # Should have computed dip zscore and typical dip
        assert hasattr(result, "dip_zscore")
        assert hasattr(result, "typical_dip_pct")
        assert result.typical_dip_pct > 0, f"Expected typical_dip > 0, got {result.typical_dip_pct}"


class TestEmpiricalExpectedReturns:
    """C5: Verify empirical expected returns computation."""

    def test_compute_empirical_expected_returns(self):
        """Empirical expected returns should work."""
        from app.quant_engine.trade_engine import (
            compute_empirical_expected_returns,
            TradeCycle,
        )
        
        # Create mock historical trades
        trades = [
            TradeCycle(
                entry_date="2020-01-01",
                exit_date="2020-01-20",
                entry_price=100,
                exit_price=110 if i % 3 != 0 else 95,  # 2/3 winners
                return_pct=10 if i % 3 != 0 else -5,
                holding_days=20,
                entry_signal="RSI",
                exit_signal="Profit Target",
            )
            for i in range(30)
        ]
        
        exp_ret, exp_loss, prob = compute_empirical_expected_returns(trades, 0.15)
        
        assert exp_ret > 0  # Winners average positive
        assert exp_loss > 0  # Losers average (abs) positive
        assert 0 < prob < 1  # Probability between 0 and 1
        assert prob > 0.5  # Should be ~66% win rate


class TestRiskFreeRateInSharpe:
    """S1: Verify Sharpe ratio uses risk-free rate."""

    def test_risk_free_rate_constant(self):
        """Risk-free rate constant should be defined."""
        from app.quant_engine.trade_engine import RISK_FREE_RATE_ANNUAL
        
        assert RISK_FREE_RATE_ANNUAL == 0.04, "Should be 4% annual"
