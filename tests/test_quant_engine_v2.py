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
        n_days = 200
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
# Signals Module Tests
# =============================================================================


class TestSignals:
    """Tests for the signals module."""

    @pytest.fixture
    def sample_prices(self) -> dict[str, pd.Series]:
        """Create sample price data."""
        np.random.seed(42)
        n_days = 200
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
                "STRONG_BUY", "BUY", "WEAK_BUY", "NEUTRAL", "AVOID", "INSUFFICIENT_DATA"
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


class TestSignalEdgeDetection:
    """
    Tests for edge detection in signal triggers.
    
    CRITICAL: Signals should only fire on the FIRST day a condition becomes true,
    not every day the condition remains true. This is the "event" vs "state" distinction.
    """

    @pytest.fixture
    def oscillating_prices(self) -> dict[str, pd.Series]:
        """
        Create price data that oscillates above/below thresholds.
        This tests that we only get signals on the first crossing.
        """
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        
        # Create prices that oscillate: goes down, stays down, goes up, stays up, repeat
        # Pattern: 5 days down, 5 days down, 5 days up, 5 days up (repeat 5x)
        base_price = 100.0
        prices = []
        for cycle in range(5):
            # 5 days going down
            prices.extend([base_price * (1 - 0.01 * i) for i in range(1, 6)])
            # 5 days staying down (near the low)
            prices.extend([base_price * 0.95 for _ in range(5)])
            # 5 days going up
            prices.extend([base_price * (0.95 + 0.01 * i) for i in range(1, 6)])
            # 5 days staying up (near the high)
            prices.extend([base_price for _ in range(5)])
        
        close = pd.Series(prices[:n_days], index=dates, name="close")
        
        return {
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "open": close,
            "volume": pd.Series([1000000] * n_days, index=dates),
        }

    def test_backtest_signal_uses_edge_detection(self, oscillating_prices):
        """
        backtest_signal should only count signals on the FIRST day condition becomes true.
        
        In the oscillating_prices fixture, the price goes below a threshold ~5 times
        (once per cycle), not every single day it's below the threshold.
        """
        from app.quant_engine.signals import backtest_signal
        
        prices = oscillating_prices["close"]
        
        # Create a simple signal that's below threshold when price is low
        # The signal value is (price - 100) / 100, so negative when below $100
        signal = (prices - 100) / 100
        
        # With edge detection, we should get ~5 triggers (one per down cycle)
        # With state detection, we'd get ~50 triggers (every day below threshold)
        result = backtest_signal(
            prices=prices,
            signal=signal,
            threshold=-0.02,  # 2% below reference
            direction="below",
            holding_days=5,
        )
        
        # Should be ~5 signals (one per cycle), definitely NOT 40+
        assert result["n_signals"] <= 10, (
            f"Expected ~5 edge-triggered signals, got {result['n_signals']}. "
            "This suggests signals are firing every day condition is true (state-based) "
            "instead of only on the first crossing (edge-based)."
        )
        assert result["n_signals"] >= 3, f"Expected at least 3 signals, got {result['n_signals']}"

    def test_get_historical_triggers_uses_edge_detection(self, oscillating_prices):
        """
        get_historical_triggers should only return triggers on crossing days.
        
        This is what shows up as dots on the chart. We should see very few dots,
        not a dot on every day the price is below a threshold.
        """
        from app.quant_engine.signals import get_historical_triggers
        
        triggers = get_historical_triggers(oscillating_prices, lookback_days=100, min_signals=3)
        
        # Should get at most ~10 triggers (edge crossings), not 40+ (every day below threshold)
        assert len(triggers) <= 15, (
            f"Expected ~5-10 edge-triggered chart markers, got {len(triggers)}. "
            "Too many triggers means signals are firing every day condition is true "
            "instead of only on the first crossing."
        )

    def test_individual_trade_returns_uses_edge_detection(self, oscillating_prices):
        """
        _get_individual_trade_returns should return one trade per signal entry,
        not one trade per day condition is true.
        """
        from app.quant_engine.signals import _get_individual_trade_returns
        
        prices = oscillating_prices["close"]
        signal = (prices - 100) / 100
        
        trades = _get_individual_trade_returns(
            prices=prices,
            signal=signal,
            threshold=-0.02,
            direction="below",
            holding_days=5,
        )
        
        # Should be ~5 trades (one per cycle), not 40+
        assert len(trades) <= 10, (
            f"Expected ~5 edge-triggered trades, got {len(trades)}. "
            "This suggests each day below threshold is counted as a separate trade."
        )

    def test_above_direction_edge_detection(self, oscillating_prices):
        """Test edge detection also works for 'above' direction."""
        from app.quant_engine.signals import backtest_signal
        
        prices = oscillating_prices["close"]
        signal = (prices - 95) / 95  # Positive when above $95
        
        result = backtest_signal(
            prices=prices,
            signal=signal,
            threshold=0.02,  # 2% above reference
            direction="above",
            holding_days=5,
        )
        
        # Should be ~5 signals (once per up cycle), not 40+
        assert result["n_signals"] <= 10, (
            f"Expected ~5 edge-triggered signals for 'above' direction, got {result['n_signals']}"
        )
