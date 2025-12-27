"""
Tests for Quantitative Portfolio Engine V2.

Tests the non-predictive, risk-based portfolio optimization system.
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Analytics Module Tests
# =============================================================================


class TestAnalytics:
    """Tests for the analytics module."""

    @pytest.fixture
    def sample_returns(self) -> pd.DataFrame:
        """Create sample return data."""
        np.random.seed(42)
        n_days = 500
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
        
        # Create correlated returns
        base = np.random.randn(n_days)
        returns_data = np.column_stack([
            base * 0.4 + np.random.randn(n_days) * 0.6,
            base * 0.3 + np.random.randn(n_days) * 0.7,
            base * 0.5 + np.random.randn(n_days) * 0.5,
            base * 0.2 + np.random.randn(n_days) * 0.8,
            base * 0.6 + np.random.randn(n_days) * 0.4,
        ]) * 0.02
        
        return pd.DataFrame(
            returns_data,
            columns=symbols,
            index=pd.date_range("2022-01-01", periods=n_days, freq="D"),
        )

    @pytest.fixture
    def sample_holdings(self) -> dict[str, float]:
        """Sample portfolio weights."""
        return {"AAPL": 0.40, "GOOG": 0.30, "MSFT": 0.20, "AMZN": 0.10}

    def test_compute_covariance_matrix(self, sample_returns):
        """Test Ledoit-Wolf shrinkage covariance estimation."""
        from app.quant_engine.analytics import compute_covariance_matrix
        
        cov = compute_covariance_matrix(sample_returns)
        
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)  # Symmetric
        assert all(np.linalg.eigvalsh(cov) > -1e-10)  # Positive semi-definite

    def test_analyze_portfolio_risk_decomposition(self, sample_returns, sample_holdings):
        """Test risk contribution calculation via analyze_portfolio."""
        from app.quant_engine.analytics import analyze_portfolio
        
        analytics = analyze_portfolio(sample_holdings, sample_returns, total_value=10000)
        risk = analytics.risk_decomposition
        
        assert risk.portfolio_volatility > 0
        assert len(risk.marginal_risk) > 0
        assert len(risk.component_risk) > 0
        assert len(risk.risk_contribution_pct) > 0
        
        # Risk contributions should sum to ~100%
        total_risk = sum(risk.risk_contribution_pct.values())
        assert abs(total_risk - 1.0) < 0.1
        
        # Largest contributor should be identified
        assert risk.largest_risk_contributor is not None

    def test_analyze_portfolio_tail_risk(self, sample_returns, sample_holdings):
        """Test tail risk metrics (VaR, CVaR, drawdown) via analyze_portfolio."""
        from app.quant_engine.analytics import analyze_portfolio
        
        analytics = analyze_portfolio(sample_holdings, sample_returns, total_value=10000)
        tail = analytics.tail_risk
        
        # VaR should be negative (loss)
        assert tail.var_95_daily < 0
        assert tail.var_99_daily < 0
        
        # CVaR should be more negative than VaR
        assert tail.cvar_95_daily <= tail.var_95_daily
        
        # Cornish-Fisher adjustment exists
        assert tail.cf_var_95 < 0
        
        # Max drawdown should be negative
        assert tail.max_drawdown <= 0
        
        # Risk level should be assigned
        assert tail.risk_level in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]

    def test_analyze_portfolio_diversification(self, sample_returns, sample_holdings):
        """Test diversification metrics calculation via analyze_portfolio."""
        from app.quant_engine.analytics import analyze_portfolio
        
        analytics = analyze_portfolio(sample_holdings, sample_returns, total_value=10000)
        div = analytics.diversification
        
        # Effective N should be between 1 and number of holdings
        assert 1.0 <= div.effective_n <= len(sample_holdings)
        
        # HHI should be between 0 and 1
        assert 0 <= div.hhi <= 1
        
        # Diversification ratio should be > 0
        assert div.diversification_ratio > 0

    def test_detect_regime(self, sample_returns):
        """Test market regime detection."""
        from app.quant_engine.analytics import analyze_portfolio
        
        holdings = {"AAPL": 0.5, "GOOG": 0.5}
        analytics = analyze_portfolio(holdings, sample_returns, total_value=10000)
        regime = analytics.regime
        
        # Should have combined regime label
        assert "_" in regime.regime  # e.g., "bull_low"
        
        # Should have is_bull and is_high_vol flags
        assert isinstance(regime.is_bull, bool)
        assert isinstance(regime.is_high_vol, bool)
        
        # Should have description
        assert len(regime.regime_description) > 0
        
        # Should have risk budget recommendation
        assert len(regime.risk_budget_recommendation) > 0

    def test_analyze_portfolio_correlation(self, sample_returns):
        """Test correlation analysis via analyze_portfolio."""
        from app.quant_engine.analytics import analyze_portfolio
        
        holdings = {"AAPL": 0.5, "GOOG": 0.3, "MSFT": 0.2}
        analytics = analyze_portfolio(holdings, sample_returns, total_value=10000)
        corr = analytics.correlations
        
        # Average correlation should be between -1 and 1
        assert -1 <= corr.avg_correlation <= 1
        
        # Should identify clusters
        assert corr.n_clusters >= 1

    def test_analyze_portfolio_full(self, sample_returns, sample_holdings):
        """Test complete portfolio analysis."""
        from app.quant_engine.analytics import analyze_portfolio
        
        analytics = analyze_portfolio(
            holdings=sample_holdings,
            returns=sample_returns,
            total_value=10000.0,
        )
        
        assert analytics.n_positions == len(sample_holdings)
        assert 1 <= analytics.overall_risk_score <= 10
        assert analytics.risk_decomposition is not None
        assert analytics.tail_risk is not None
        assert analytics.diversification is not None
        assert analytics.regime is not None
        assert analytics.correlations is not None
        assert len(analytics.key_insights) > 0
        assert len(analytics.action_items) >= 0

    def test_translate_for_user(self, sample_returns, sample_holdings):
        """Test user-friendly translation."""
        from app.quant_engine.analytics import analyze_portfolio, translate_for_user
        
        analytics = analyze_portfolio(
            holdings=sample_holdings,
            returns=sample_returns,
            total_value=10000.0,
        )
        user_output = translate_for_user(analytics)
        
        assert "summary" in user_output
        assert "risk" in user_output
        assert "diversification" in user_output
        assert "market" in user_output
        assert "action_items" in user_output
        assert "insights" in user_output
        
        # Summary should have readable content
        assert "risk_label" in user_output["summary"]
        assert "headline" in user_output["summary"]


# =============================================================================
# Risk Optimizer Module Tests
# =============================================================================


class TestRiskOptimizer:
    """Tests for the risk optimizer module."""

    @pytest.fixture
    def sample_returns(self) -> pd.DataFrame:
        """Create sample return data for optimization."""
        np.random.seed(42)
        n_days = 500
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
        
        base = np.random.randn(n_days)
        returns_data = np.column_stack([
            base * 0.4 + np.random.randn(n_days) * 0.6,
            base * 0.3 + np.random.randn(n_days) * 0.7,
            base * 0.5 + np.random.randn(n_days) * 0.5,
            base * 0.2 + np.random.randn(n_days) * 0.8,
            base * 0.6 + np.random.randn(n_days) * 0.4,
        ]) * 0.02
        
        return pd.DataFrame(
            returns_data,
            columns=symbols,
            index=pd.date_range("2022-01-01", periods=n_days, freq="D"),
        )

    def test_optimize_risk_parity(self, sample_returns):
        """Test Risk Parity optimization."""
        from app.quant_engine.risk_optimizer import optimize_risk_parity
        
        # Compute covariance matrix
        cov = sample_returns.cov().values
        
        result = optimize_risk_parity(
            cov,
            symbols=list(sample_returns.columns),
        )
        
        # Weights should sum to 1
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6
        
        # All weights should be positive
        assert all(w >= 0 for w in result.weights.values())
        
        # Should have risk metrics
        assert result.portfolio_volatility > 0
        assert result.diversification_ratio > 0
        
        # Should have converged
        assert result.converged

    def test_optimize_min_variance(self, sample_returns):
        """Test Minimum Variance optimization."""
        from app.quant_engine.risk_optimizer import optimize_min_variance
        
        # Convert returns to covariance matrix
        cov = sample_returns.cov().values
        
        result = optimize_min_variance(
            cov,
            symbols=list(sample_returns.columns),
        )
        
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
        assert result.portfolio_volatility > 0
        assert result.converged

    def test_optimize_max_diversification(self, sample_returns):
        """Test Maximum Diversification optimization."""
        from app.quant_engine.risk_optimizer import optimize_max_diversification
        
        # Convert returns to covariance matrix
        cov = sample_returns.cov().values
        
        result = optimize_max_diversification(
            cov,
            symbols=list(sample_returns.columns),
        )
        
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
        assert result.diversification_ratio >= 1.0  # Should be at least 1
        assert result.converged

    def test_optimize_cvar(self, sample_returns):
        """Test CVaR (Expected Shortfall) optimization."""
        from app.quant_engine.risk_optimizer import optimize_cvar
        
        result = optimize_cvar(
            sample_returns,
            symbols=list(sample_returns.columns),
        )
        
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
        assert result.converged

    def test_optimize_hrp(self, sample_returns):
        """Test Hierarchical Risk Parity optimization."""
        from app.quant_engine.risk_optimizer import optimize_hrp
        
        result = optimize_hrp(
            sample_returns,
            symbols=list(sample_returns.columns),
        )
        
        weights_sum = sum(result.weights.values())
        assert abs(weights_sum - 1.0) < 1e-6
        assert all(w >= 0 for w in result.weights.values())
        assert result.portfolio_volatility > 0
        # HRP doesn't use numerical optimization, so "converged" is always True
        assert result.converged

    def test_optimize_portfolio_risk_based(self, sample_returns):
        """Test main entry point with different methods."""
        from app.quant_engine.risk_optimizer import (
            optimize_portfolio_risk_based,
            RiskOptimizationMethod,
        )
        
        for method in RiskOptimizationMethod:
            result = optimize_portfolio_risk_based(
                sample_returns,
                symbols=list(sample_returns.columns),
                method=method,
            )
            
            weights_sum = sum(result.weights.values())
            assert abs(weights_sum - 1.0) < 1e-6, f"Failed for {method}"
            assert all(w >= 0 for w in result.weights.values()), f"Failed for {method}"

    def test_generate_allocation_recommendation(self, sample_returns):
        """Test allocation recommendation generation."""
        from app.quant_engine.risk_optimizer import (
            generate_allocation_recommendation,
            RiskOptimizationMethod,
        )
        
        current_weights = {
            "AAPL": 0.50,
            "GOOG": 0.20,
            "MSFT": 0.15,
            "AMZN": 0.10,
            "META": 0.05,
        }
        
        recommendation = generate_allocation_recommendation(
            returns=sample_returns,
            symbols=list(sample_returns.columns),
            current_weights=current_weights,
            inflow_eur=1000,
            portfolio_value_eur=10000,
            method=RiskOptimizationMethod.RISK_PARITY,
        )
        
        # Should have confidence rating
        assert recommendation.confidence in ["LOW", "MEDIUM", "HIGH"]
        
        # Should have explanation
        assert len(recommendation.explanation) > 0
        
        # Should have risk improvement summary
        assert len(recommendation.risk_improvement_summary) > 0
        
        # Should have trade recommendations
        assert isinstance(recommendation.recommendations, list)
        
        # Optimal portfolio should be provided
        assert len(recommendation.optimal_portfolio) == len(current_weights)


# =============================================================================
# Signals Module Tests
# =============================================================================


class TestSignals:
    """Tests for the signals module."""

    @pytest.fixture
    def sample_prices(self) -> dict[str, pd.Series]:
        """Create sample price data."""
        np.random.seed(42)
        n_days = 500
        dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
        
        # Simulate price series with trend and noise
        base_price = 100
        prices = {}
        
        for symbol in ["AAPL", "GOOG", "MSFT"]:
            trend = np.linspace(0, 20, n_days)
            noise = np.cumsum(np.random.randn(n_days) * 2)
            prices[symbol] = pd.Series(
                base_price + trend + noise,
                index=dates,
                name=symbol,
            )
            prices[symbol] = prices[symbol].clip(lower=1)
        
        return prices

    @pytest.fixture
    def symbol_names(self) -> dict[str, str]:
        """Symbol names for testing."""
        return {
            "AAPL": "Apple Inc.",
            "GOOG": "Alphabet Inc.",
            "MSFT": "Microsoft Corporation",
        }

    def test_scan_all_stocks(self, sample_prices, symbol_names):
        """Test scanning all stocks for signals."""
        from app.quant_engine.signals import scan_all_stocks
        
        opportunities = scan_all_stocks(
            price_data=sample_prices,
            stock_names=symbol_names,
            holding_days_options=[5, 10, 20],
        )
        
        # Should return list of opportunities
        assert isinstance(opportunities, list)
        assert len(opportunities) == len(sample_prices)
        
        for opp in opportunities:
            # Should have symbol and name
            assert opp.symbol in sample_prices
            assert opp.name in symbol_names.values()
            
            # Should have buy score
            assert 0 <= opp.buy_score <= 100
            
            # Should have opportunity type
            assert opp.opportunity_type in [
                "STRONG_BUY", "BUY", "WEAK_BUY", "NEUTRAL", "AVOID"
            ]
            
            # Should have signals
            assert isinstance(opp.signals, list)
            
            # If there's a best signal, verify it exists
            if opp.best_signal_name:
                assert opp.best_holding_days > 0

    def test_stock_opportunity_structure(self, sample_prices, symbol_names):
        """Test StockOpportunity dataclass structure."""
        from app.quant_engine.signals import scan_all_stocks, StockOpportunity
        
        opportunities = scan_all_stocks(sample_prices, symbol_names, [10])
        
        for opp in opportunities:
            assert isinstance(opp, StockOpportunity)
            
            # Price metrics should be present
            assert opp.current_price > 0
            
            # Z-scores should be reasonable
            assert -10 < opp.zscore_20d < 10
            assert -10 < opp.zscore_60d < 10
            
            # RSI should be in valid range
            assert 0 <= opp.rsi_14 <= 100

    def test_optimized_signal_structure(self, sample_prices, symbol_names):
        """Test OptimizedSignal dataclass structure."""
        from app.quant_engine.signals import scan_all_stocks, OptimizedSignal
        
        opportunities = scan_all_stocks(sample_prices, symbol_names, [5, 10])
        
        for opp in opportunities:
            for sig in opp.signals:
                assert isinstance(sig, OptimizedSignal)
                
                # Should have name and description
                assert len(sig.name) > 0
                
                # Win rate should be between 0 and 1
                assert 0 <= sig.win_rate <= 1
                
                # Optimal holding days should be positive
