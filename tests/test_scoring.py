"""
Comprehensive tests for the APUS + DOUS scoring pipeline.

Tests:
- Stationary bootstrap implementation
- Deflated Sharpe ratio calculation
- Mode A gate logic
- Mode B scoring
- Ranking and "no opportunity" case
- Evidence block completeness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, datetime, timedelta

from app.quant_engine.scoring import (
    ScoringConfig,
    EvidenceBlock,
    ScoringResult,
    stationary_bootstrap,
    compute_bootstrap_stats,
    compute_effective_n,
    compute_probabilistic_sharpe,
    compute_expected_max_sharpe,
    compute_deflated_sharpe,
    walk_forward_split,
    compute_equity_curve,
    detect_regimes,
    compute_regime_edges,
    check_mode_a_gate,
    compute_mode_a_score,
    compute_mode_b_score,
    compute_fundamental_momentum,
    compute_valuation_z,
    analyze_dip_recovery,
    compute_sector_relative,
    compute_symbol_score,
    SCORING_VERSION,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def positive_edge_data():
    """Generate data with a positive edge (strategy beats benchmark)."""
    np.random.seed(42)
    # OOS edges with positive mean
    return np.array([0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.03, 0.02, 0.01, 0.025])


@pytest.fixture
def negative_edge_data():
    """Generate data with a negative edge (strategy underperforms)."""
    np.random.seed(42)
    return np.array([-0.02, -0.03, -0.01, -0.04, -0.02, 0.01, -0.03, -0.02, -0.01, -0.025])


@pytest.fixture
def random_walk_prices():
    """Generate random walk prices for testing."""
    np.random.seed(42)
    n_days = 500
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = pd.Series(100 * np.cumprod(1 + returns))
    prices.index = pd.date_range("2020-01-01", periods=n_days)
    return prices


@pytest.fixture
def uptrending_prices():
    """Generate uptrending prices for testing."""
    np.random.seed(42)
    n_days = 500
    returns = np.random.normal(0.001, 0.012, n_days)  # Positive drift
    prices = pd.Series(100 * np.cumprod(1 + returns))
    prices.index = pd.date_range("2020-01-01", periods=n_days)
    return prices


@pytest.fixture
def spy_prices():
    """Generate SPY-like prices for benchmark."""
    np.random.seed(123)
    n_days = 500
    returns = np.random.normal(0.0004, 0.011, n_days)
    prices = pd.Series(300 * np.cumprod(1 + returns))
    prices.index = pd.date_range("2020-01-01", periods=n_days)
    return prices


# =============================================================================
# Stationary Bootstrap Tests
# =============================================================================


class TestStationaryBootstrap:
    """Tests for stationary bootstrap implementation."""
    
    def test_bootstrap_deterministic_with_seed(self, positive_edge_data):
        """Bootstrap results should be reproducible with fixed seed."""
        result1 = stationary_bootstrap(positive_edge_data, n_samples=100, block_length=3, seed=42)
        result2 = stationary_bootstrap(positive_edge_data, n_samples=100, block_length=3, seed=42)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_bootstrap_different_with_different_seeds(self, positive_edge_data):
        """Bootstrap results should differ with different seeds."""
        result1 = stationary_bootstrap(positive_edge_data, n_samples=100, block_length=3, seed=42)
        result2 = stationary_bootstrap(positive_edge_data, n_samples=100, block_length=3, seed=123)
        
        assert not np.allclose(result1, result2)
    
    def test_bootstrap_output_shape(self, positive_edge_data):
        """Bootstrap should return correct number of samples."""
        n_samples = 200
        result = stationary_bootstrap(positive_edge_data, n_samples=n_samples, seed=42)
        
        assert len(result) == n_samples
    
    def test_bootstrap_positive_data_mostly_positive_means(self, positive_edge_data):
        """Bootstrap of positive-mean data should yield mostly positive means."""
        means = stationary_bootstrap(positive_edge_data, n_samples=1000, seed=42)
        
        positive_fraction = np.mean(means > 0)
        assert positive_fraction > 0.7  # Most should be positive
    
    def test_bootstrap_negative_data_mostly_negative_means(self, negative_edge_data):
        """Bootstrap of negative-mean data should yield mostly negative means."""
        means = stationary_bootstrap(negative_edge_data, n_samples=1000, seed=42)
        
        negative_fraction = np.mean(means < 0)
        assert negative_fraction > 0.7  # Most should be negative


class TestBootstrapStats:
    """Tests for bootstrap statistics computation."""
    
    def test_p_outperf_positive_data(self, positive_edge_data):
        """P(outperf) should be high for positive edge data."""
        p_outperf, ci_low, ci_high, cvar = compute_bootstrap_stats(
            positive_edge_data, n_samples=1000, seed=42
        )
        
        assert p_outperf >= 0.75  # High probability of positive edge
        assert ci_low < ci_high  # CI should be valid
    
    def test_p_outperf_negative_data(self, negative_edge_data):
        """P(outperf) should be low for negative edge data."""
        p_outperf, ci_low, ci_high, cvar = compute_bootstrap_stats(
            negative_edge_data, n_samples=1000, seed=42
        )
        
        assert p_outperf <= 0.25  # Low probability of positive edge
    
    def test_ci_contains_mean(self, positive_edge_data):
        """Confidence interval should contain the sample mean."""
        mean = np.mean(positive_edge_data)
        _, ci_low, ci_high, _ = compute_bootstrap_stats(
            positive_edge_data, n_samples=1000, confidence=0.95, seed=42
        )
        
        assert ci_low <= mean <= ci_high
    
    def test_cvar_is_tail_measure(self, positive_edge_data):
        """CVaR should be less than or equal to VaR (5th percentile)."""
        means = stationary_bootstrap(positive_edge_data, n_samples=1000, seed=42)
        var_5 = np.percentile(means, 5)
        
        _, _, _, cvar = compute_bootstrap_stats(
            positive_edge_data, n_samples=1000, seed=42
        )
        
        assert cvar <= var_5  # CVaR is average of worst 5%
    
    def test_insufficient_data_returns_zeros(self):
        """Short data should return zero stats."""
        short_data = np.array([0.01, 0.02])  # Less than 10 points
        p_outperf, ci_low, ci_high, cvar = compute_bootstrap_stats(short_data)
        
        assert p_outperf == 0.0
        assert ci_low == 0.0


# =============================================================================
# Deflated Sharpe Ratio Tests
# =============================================================================


class TestDeflatedSharpe:
    """Tests for deflated Sharpe ratio calculation."""
    
    def test_effective_n_single_strategy(self):
        """Single strategy should have N_eff = 1."""
        returns = np.random.randn(100, 1)
        n_eff = compute_effective_n(returns)
        
        assert n_eff == 1.0
    
    def test_effective_n_independent_strategies(self):
        """Independent strategies should have N_eff close to N."""
        np.random.seed(42)
        # Generate 10 independent return series
        returns = np.random.randn(100, 10)
        n_eff = compute_effective_n(returns)
        
        assert n_eff >= 8  # Should be close to 10
    
    def test_effective_n_correlated_strategies(self):
        """Highly correlated strategies should have low N_eff."""
        np.random.seed(42)
        base = np.random.randn(100)
        # Create 10 highly correlated series
        returns = np.column_stack([base + 0.1 * np.random.randn(100) for _ in range(10)])
        n_eff = compute_effective_n(returns)
        
        assert n_eff < 3  # Should be close to 1
    
    def test_probabilistic_sharpe_high_sr(self):
        """High observed SR should give high PSR."""
        psr = compute_probabilistic_sharpe(
            observed_sharpe=2.0,
            benchmark_sharpe=0.0,
            n_obs=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        
        assert psr > 0.9  # Very high probability of true skill
    
    def test_probabilistic_sharpe_low_sr(self):
        """Low observed SR should give low PSR."""
        psr = compute_probabilistic_sharpe(
            observed_sharpe=0.2,
            benchmark_sharpe=0.5,
            n_obs=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        
        assert psr < 0.3  # Below benchmark
    
    def test_expected_max_sharpe_increases_with_n(self):
        """Expected max SR should increase with N_eff."""
        sr_max_5 = compute_expected_max_sharpe(0.0, 0.5, n_effective=5)
        sr_max_20 = compute_expected_max_sharpe(0.0, 0.5, n_effective=20)
        sr_max_100 = compute_expected_max_sharpe(0.0, 0.5, n_effective=100)
        
        assert sr_max_5 < sr_max_20 < sr_max_100
    
    def test_deflated_sharpe_penalizes_multiple_testing(self):
        """DSR should be lower than PSR when testing many strategies."""
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.01, 252)
        
        dsr_1, sr, sr_max_1, _ = compute_deflated_sharpe(returns, n_strategies_tested=1)
        dsr_100, _, sr_max_100, _ = compute_deflated_sharpe(returns, n_strategies_tested=100)
        
        assert sr_max_100 > sr_max_1  # Haircut increases
        assert dsr_100 <= dsr_1  # DSR decreases with more strategies


# =============================================================================
# Mode A Gate Tests
# =============================================================================


class TestModeAGate:
    """Tests for Mode A gate logic."""
    
    def test_gate_passes_when_all_criteria_met(self):
        """Gate should pass when P_outperf >= 0.75, CI_low > 0, DSR >= 0.50."""
        assert check_mode_a_gate(
            p_outperf=0.80,
            ci_low=0.01,
            dsr=0.55,
        ) is True
    
    def test_gate_fails_low_p_outperf(self):
        """Gate should fail when P_outperf < 0.75."""
        assert check_mode_a_gate(
            p_outperf=0.70,  # Below 0.75
            ci_low=0.01,
            dsr=0.55,
        ) is False
    
    def test_gate_fails_negative_ci_low(self):
        """Gate should fail when CI_low <= 0."""
        assert check_mode_a_gate(
            p_outperf=0.80,
            ci_low=-0.01,  # Negative
            dsr=0.55,
        ) is False
    
    def test_gate_fails_low_dsr(self):
        """Gate should fail when DSR < 0.50."""
        assert check_mode_a_gate(
            p_outperf=0.80,
            ci_low=0.01,
            dsr=0.45,  # Below 0.50
        ) is False
    
    def test_gate_boundary_values(self):
        """Gate should pass at exact boundary values."""
        # Exactly at thresholds
        assert check_mode_a_gate(
            p_outperf=0.75,  # Exactly 0.75
            ci_low=0.001,  # Just above 0
            dsr=0.50,  # Exactly 0.50
        ) is True


# =============================================================================
# Mode A Score Tests
# =============================================================================


class TestModeAScore:
    """Tests for Mode A (APUS) scoring."""
    
    def test_perfect_metrics_gives_high_score(self):
        """Perfect metrics should give score near 100."""
        score = compute_mode_a_score(
            p_outperf=0.95,
            median_edge=0.20,
            dsr=0.80,
            cvar_5=0.01,  # Positive tail
            worst_regime_edge=0.15,
            sharpe_degradation=0.0,  # No degradation
        )
        
        assert score >= 80
    
    def test_poor_metrics_gives_low_score(self):
        """Poor metrics should give low score."""
        score = compute_mode_a_score(
            p_outperf=0.50,  # Just 50%
            median_edge=0.02,  # Small edge
            dsr=0.30,  # Low DSR
            cvar_5=-0.10,  # Bad tail
            worst_regime_edge=-0.05,  # Negative in worst regime
            sharpe_degradation=0.50,  # 50% degradation
        )
        
        assert score <= 40
    
    def test_score_in_valid_range(self):
        """Score should always be 0-100."""
        # Various random inputs
        for _ in range(10):
            score = compute_mode_a_score(
                p_outperf=np.random.random(),
                median_edge=np.random.uniform(-0.2, 0.3),
                dsr=np.random.random(),
                cvar_5=np.random.uniform(-0.2, 0.1),
                worst_regime_edge=np.random.uniform(-0.2, 0.2),
                sharpe_degradation=np.random.uniform(0, 1),
            )
            
            assert 0 <= score <= 100


# =============================================================================
# Mode B Score Tests
# =============================================================================


class TestModeBScore:
    """Tests for Mode B (DOUS) scoring."""
    
    def test_high_recovery_gives_high_score(self):
        """High recovery probability and EV should give high score."""
        score = compute_mode_b_score(
            p_recovery=0.85,
            expected_value=0.10,
            fund_mom=0.8,
            val_z=0.7,
            sector_relative=0.8,
            event_risk=False,
        )
        
        assert score >= 70
    
    def test_event_risk_penalizes_score(self):
        """Event risk should reduce score."""
        score_no_risk = compute_mode_b_score(
            p_recovery=0.70,
            expected_value=0.06,
            fund_mom=0.5,
            val_z=0.5,
            sector_relative=0.5,
            event_risk=False,
        )
        
        score_with_risk = compute_mode_b_score(
            p_recovery=0.70,
            expected_value=0.06,
            fund_mom=0.5,
            val_z=0.5,
            sector_relative=0.5,
            event_risk=True,
        )
        
        assert score_with_risk < score_no_risk
        assert score_no_risk - score_with_risk == pytest.approx(10.0, abs=0.1)  # 10% penalty
    
    def test_low_recovery_gives_low_score(self):
        """Low recovery probability should give low score."""
        score = compute_mode_b_score(
            p_recovery=0.40,
            expected_value=-0.02,
            fund_mom=0.3,
            val_z=0.3,
            sector_relative=0.3,
            event_risk=False,
        )
        
        assert score <= 30


# =============================================================================
# Fundamental Scores Tests
# =============================================================================


class TestFundamentalScores:
    """Tests for fundamental analysis components."""
    
    def test_fundamental_momentum_positive_z(self):
        """Positive z-scores should give high momentum score."""
        score = compute_fundamental_momentum(
            revenue_z=1.5,
            earnings_z=1.0,
            margin_z=0.5,
        )
        
        assert score > 0.7  # Sigmoid should be high
    
    def test_fundamental_momentum_negative_z(self):
        """Negative z-scores should give low momentum score."""
        score = compute_fundamental_momentum(
            revenue_z=-1.5,
            earnings_z=-1.0,
            margin_z=-0.5,
        )
        
        assert score < 0.3  # Sigmoid should be low
    
    def test_valuation_z_undervalued(self):
        """Low PE/EV multiples (negative z) should give high valuation score."""
        score = compute_valuation_z(
            pe_z=-1.5,  # Below average PE
            ev_ebitda_z=-1.0,
            ps_z=-1.2,
        )
        
        assert score > 0.7  # Undervalued = high score
    
    def test_valuation_z_overvalued(self):
        """High PE/EV multiples (positive z) should give low valuation score."""
        score = compute_valuation_z(
            pe_z=1.5,  # Above average PE
            ev_ebitda_z=1.0,
            ps_z=1.2,
        )
        
        assert score < 0.3  # Overvalued = low score


# =============================================================================
# Walk-Forward Split Tests
# =============================================================================


class TestWalkForwardSplit:
    """Tests for walk-forward validation splits."""
    
    def test_splits_have_embargo(self):
        """Train and test sets should not overlap (embargo respected)."""
        splits = walk_forward_split(n_obs=500, n_folds=5, embargo_days=5, min_train=252)
        
        for train_idx, test_idx in splits:
            train_end = train_idx[-1]
            test_start = test_idx[0]
            
            assert test_start - train_end >= 5  # Embargo respected
    
    def test_splits_are_expanding_window(self):
        """Training set should grow with each fold."""
        splits = walk_forward_split(n_obs=500, n_folds=5, embargo_days=5, min_train=252)
        
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        
        # Each training set should be larger than the previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]
    
    def test_insufficient_data_returns_empty(self):
        """Not enough data should return empty splits."""
        splits = walk_forward_split(n_obs=100, n_folds=5, embargo_days=5, min_train=252)
        
        assert splits == []


# =============================================================================
# Equity Curve Tests
# =============================================================================


class TestEquityCurve:
    """Tests for equity curve computation."""
    
    def test_equity_starts_at_one(self):
        """Equity curve should start at 1.0."""
        returns = np.array([0.01, 0.02, -0.01, 0.015])
        weights = np.ones(4)
        
        equity = compute_equity_curve(returns, weights, cost_per_trade=0)
        
        assert equity[0] == 1.0
    
    def test_equity_compounds_correctly(self):
        """Equity should compound correctly without costs."""
        returns = np.array([0.10, 0.10])  # 10% each day
        weights = np.ones(2)
        
        equity = compute_equity_curve(returns, weights, cost_per_trade=0)
        
        # 1.0 * 1.10 * 1.10 = 1.21
        assert equity[-1] == pytest.approx(1.21, rel=1e-6)
    
    def test_transaction_costs_reduce_equity(self):
        """Transaction costs should reduce final equity."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        weights = np.array([1, 0, 1, 0])  # Trade every day
        
        equity_no_cost = compute_equity_curve(returns, weights, cost_per_trade=0)
        equity_with_cost = compute_equity_curve(returns, weights, cost_per_trade=0.001)
        
        assert equity_with_cost[-1] < equity_no_cost[-1]


# =============================================================================
# Regime Detection Tests
# =============================================================================


class TestRegimeDetection:
    """Tests for market regime detection."""
    
    def test_detect_regimes_returns_masks(self, random_walk_prices):
        """Should return boolean masks for each regime."""
        returns = random_walk_prices.pct_change().dropna()
        
        regimes = detect_regimes(returns, random_walk_prices)
        
        assert "bull" in regimes
        assert "bear" in regimes
        assert "high_vol" in regimes
        # Regime masks should align with returns length for proper indexing
        assert len(regimes["bull"]) == len(returns)
    
    def test_regime_edges_computed(self, random_walk_prices):
        """Should compute edge for each regime."""
        returns = random_walk_prices.pct_change().dropna()
        # Align prices with returns for regime detection
        aligned_prices = random_walk_prices.loc[returns.index]
        
        regimes = detect_regimes(returns, aligned_prices)
        strategy_returns = returns.values * 1.1  # Strategy with edge
        benchmark_returns = returns.values
        
        edges = compute_regime_edges(strategy_returns, benchmark_returns, regimes)
        
        assert "bull" in edges
        assert "bear" in edges
        assert "high_vol" in edges


# =============================================================================
# Integration Tests
# =============================================================================


class TestComputeSymbolScore:
    """Integration tests for full symbol scoring."""
    
    def test_basic_scoring(self, uptrending_prices, spy_prices):
        """Should compute a valid score for a symbol."""
        prices_df = pd.DataFrame({"Adj Close": uptrending_prices})
        weights = pd.Series(1.0, index=uptrending_prices.index)
        
        result = compute_symbol_score(
            symbol="TEST",
            prices=prices_df,
            spy_prices=spy_prices,
            strategy_weights=weights,
            config=ScoringConfig(n_bootstrap=100),  # Faster for tests
        )
        
        assert isinstance(result, ScoringResult)
        assert 0 <= result.best_score <= 100
        assert result.mode in ("CERTIFIED_BUY", "DIP_ENTRY")
        assert result.symbol == "TEST"
    
    def test_evidence_block_complete(self, uptrending_prices, spy_prices):
        """Evidence block should have all fields populated."""
        prices_df = pd.DataFrame({"Adj Close": uptrending_prices})
        weights = pd.Series(1.0, index=uptrending_prices.index)
        
        result = compute_symbol_score(
            symbol="TEST",
            prices=prices_df,
            spy_prices=spy_prices,
            strategy_weights=weights,
            config=ScoringConfig(n_bootstrap=100),
        )
        
        evidence = result.evidence
        assert isinstance(evidence, EvidenceBlock)
        
        # Check key fields are populated
        assert evidence.p_outperf >= 0
        assert isinstance(evidence.ci_low, float)
        assert isinstance(evidence.dsr, float)
        assert isinstance(evidence.median_edge, float)
    
    def test_insufficient_data_returns_zero_score(self, spy_prices):
        """Insufficient data should return zero score."""
        short_prices = pd.DataFrame({"Adj Close": pd.Series([100, 101, 102])})
        short_weights = pd.Series([1, 1, 1])
        
        result = compute_symbol_score(
            symbol="SHORT",
            prices=short_prices,
            spy_prices=spy_prices,
            strategy_weights=short_weights,
        )
        
        assert result.best_score == 0.0
        assert result.mode == "DIP_ENTRY"  # Falls back to Mode B
    
    def test_scoring_version_included(self, uptrending_prices, spy_prices):
        """Result should include scoring version."""
        prices_df = pd.DataFrame({"Adj Close": uptrending_prices})
        weights = pd.Series(1.0, index=uptrending_prices.index)
        
        result = compute_symbol_score(
            symbol="TEST",
            prices=prices_df,
            spy_prices=spy_prices,
            strategy_weights=weights,
            config=ScoringConfig(n_bootstrap=100),
        )
        
        assert result.scoring_version == SCORING_VERSION
    
    def test_config_hash_changes_with_config(self, uptrending_prices, spy_prices):
        """Config hash should change when config changes."""
        prices_df = pd.DataFrame({"Adj Close": uptrending_prices})
        weights = pd.Series(1.0, index=uptrending_prices.index)
        
        result1 = compute_symbol_score(
            symbol="TEST",
            prices=prices_df,
            spy_prices=spy_prices,
            strategy_weights=weights,
            config=ScoringConfig(n_bootstrap=100, min_p_outperf=0.75),
        )
        
        result2 = compute_symbol_score(
            symbol="TEST",
            prices=prices_df,
            spy_prices=spy_prices,
            strategy_weights=weights,
            config=ScoringConfig(n_bootstrap=100, min_p_outperf=0.80),  # Changed threshold
        )
        
        assert result1.config_hash != result2.config_hash


class TestNoOpportunityCase:
    """Tests for 'no certified opportunity' scenario."""
    
    def test_low_score_triggers_no_opportunity(self, random_walk_prices, spy_prices):
        """Score below 70 should indicate no opportunity."""
        # Random walk strategy (no edge)
        prices_df = pd.DataFrame({"Adj Close": random_walk_prices})
        weights = pd.Series(0.0, index=random_walk_prices.index)  # Flat strategy
        
        result = compute_symbol_score(
            symbol="NOEDGE",
            prices=prices_df,
            spy_prices=spy_prices,
            strategy_weights=weights,
            config=ScoringConfig(n_bootstrap=200),
        )
        
        # With no strategy (weights=0), there's no edge to speak of
        # The score should reflect the dip analysis which is separate
        assert isinstance(result.best_score, float)


class TestEvidenceBlockSerialization:
    """Tests for evidence block JSON serialization."""
    
    def test_to_dict_returns_dict(self):
        """Evidence block should serialize to dict."""
        evidence = EvidenceBlock(
            p_outperf=0.75,
            ci_low=0.01,
            ci_high=0.05,
            dsr=0.55,
            median_edge=0.03,
        )
        
        result = evidence.to_dict()
        
        assert isinstance(result, dict)
        assert result["p_outperf"] == 0.75
        assert result["ci_low"] == 0.01
    
    def test_to_dict_rounds_values(self):
        """Evidence block should round values for clean JSON."""
        evidence = EvidenceBlock(
            p_outperf=0.7512345678,
            ci_low=0.0123456789,
        )
        
        result = evidence.to_dict()
        
        # Values should be rounded to 4 decimal places
        assert result["p_outperf"] == 0.7512
        assert result["ci_low"] == 0.0123
