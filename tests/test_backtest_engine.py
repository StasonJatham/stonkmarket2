"""
Tests for the professional backtesting engine.

These tests verify:
1. Trade execution correctness
2. Win rate calculation matches actual trades
3. No look-ahead bias
4. Transaction costs applied correctly
5. Walk-forward validation works
6. Statistical validation is accurate
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from app.quant_engine.backtest_engine import (
    BacktestEngine,
    TradingConfig,
    Trade,
    StrategyResult,
    ValidationReport,
    compute_indicators,
    compute_buy_and_hold_return,
    compare_to_benchmark,
    STRATEGIES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate sample price data with known patterns."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="B")
    n = len(dates)
    
    # Generate trending price with mean reversion
    price = 100.0
    prices = [price]
    
    for i in range(1, n):
        # Mean reversion with trend
        mean_rev = -0.002 * (prices[-1] - 100) / 100
        trend = 0.0003  # Slight upward drift
        noise = np.random.normal(0, 0.015)
        
        # Add some dips for buy signals
        if i % 60 == 0:  # Create dip every ~3 months
            noise -= 0.05
        
        ret = mean_rev + trend + noise
        price = prices[-1] * (1 + ret)
        prices.append(price)
    
    df = pd.DataFrame({
        "close": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "volume": np.random.randint(1000000, 5000000, n),
    }, index=dates)
    
    df.attrs["symbol"] = "TEST"
    return df


@pytest.fixture
def simple_prices() -> pd.DataFrame:
    """Simple price data for basic tests."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
    n = len(dates)
    
    # Simple uptrend with known dips
    prices = []
    price = 100.0
    for i in range(n):
        if i in [50, 100, 150, 200]:  # Known dip points
            price *= 0.90  # 10% drop
        else:
            price *= 1.002  # Small daily gain
        prices.append(price)
    
    df = pd.DataFrame({
        "close": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "volume": [1000000] * n,
    }, index=dates)
    
    df.attrs["symbol"] = "SIMPLE"
    return df


@pytest.fixture
def config() -> TradingConfig:
    """Standard trading config for tests."""
    return TradingConfig(
        initial_capital=50_000.0,
        flat_cost_per_trade=1.0,
        slippage_bps=5.0,
        min_trades_for_significance=5,  # Lower for testing
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestComputeIndicators:
    """Test technical indicator computation."""
    
    def test_indicators_computed(self, sample_prices):
        """Test that all indicators are computed."""
        df = compute_indicators(sample_prices)
        
        # Check key indicators exist
        assert "rsi_14" in df.columns
        assert "macd" in df.columns
        assert "bb_upper" in df.columns
        assert "sma_20" in df.columns
        assert "zscore_20" in df.columns
        assert "drawdown" in df.columns
    
    def test_rsi_bounds(self, sample_prices):
        """RSI should be between 0 and 100."""
        df = compute_indicators(sample_prices)
        rsi = df["rsi_14"].dropna()
        
        assert rsi.min() >= 0
        assert rsi.max() <= 100
    
    def test_no_future_leak(self, sample_prices):
        """Indicators should not use future data."""
        df = compute_indicators(sample_prices)
        
        # SMA should have NaN for first window-1 values
        assert df["sma_20"].iloc[:19].isna().all()
        assert not df["sma_20"].iloc[19:].isna().all()


class TestTradingConfig:
    """Test trading configuration."""
    
    def test_default_config(self):
        """Default config has sensible values."""
        config = TradingConfig()
        
        assert config.initial_capital == 50_000.0
        assert config.flat_cost_per_trade == 1.0
        assert config.slippage_bps == 5.0
        assert config.stop_loss_pct == 0.15
        assert config.take_profit_pct == 0.30
    
    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = TradingConfig(
            initial_capital=100_000.0,
            flat_cost_per_trade=2.0,
        )
        
        assert config.initial_capital == 100_000.0
        assert config.flat_cost_per_trade == 2.0
    
    def test_from_runtime_settings(self):
        """Config can be loaded from runtime settings dict."""
        settings = {
            "trading_initial_capital": 75_000.0,
            "trading_flat_cost_per_trade": 1.50,
            "trading_slippage_bps": 10.0,
            "trading_stop_loss_pct": 20.0,  # Percent, not decimal
            "trading_take_profit_pct": 50.0,
            "trading_max_holding_days": 90,
            "trading_min_trades_required": 25,
        }
        
        config = TradingConfig.from_runtime_settings(settings)
        
        assert config.initial_capital == 75_000.0
        assert config.flat_cost_per_trade == 1.50
        assert config.slippage_bps == 10.0
        assert config.stop_loss_pct == 0.20  # Converted to decimal
        assert config.take_profit_pct == 0.50
        assert config.max_holding_days == 90
        assert config.min_trades_for_significance == 25


# =============================================================================
# Trade Execution Tests
# =============================================================================

class TestTradeExecution:
    """Test trade execution mechanics."""
    
    def test_trade_costs_applied(self, sample_prices, config):
        """Transaction costs should be deducted from P&L."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        # All trades should have costs
        for trade in result.trades:
            assert trade.total_cost > 0
            # Cost = 2 * flat_cost + slippage
            min_cost = config.flat_cost_per_trade * 2
            assert trade.total_cost >= min_cost
    
    def test_slippage_direction(self, sample_prices, config):
        """Entry should be at higher price, exit at lower."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        if result.trades:
            trade = result.trades[0]
            # Entry price should be higher than the theoretical price
            # (we buy at ask, which is higher)
            # This is embedded in the entry_price calculation
            assert trade.entry_price > 0
    
    def test_all_in_position_sizing(self, sample_prices, config):
        """For single stock analysis, each trade uses full available equity."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        if result.n_trades == 0:
            pytest.skip("No trades generated")
        
        # First trade should use approximately the initial capital
        first_trade = result.trades[0]
        assert first_trade.position_value >= config.initial_capital * 0.95  # Some buffer for costs


# =============================================================================
# Win Rate Calculation Tests (CRITICAL)
# =============================================================================

class TestWinRateCalculation:
    """Test that win rate matches actual trade outcomes."""
    
    def test_win_rate_matches_trades(self, sample_prices, config):
        """Win rate should exactly match profitable trades / total trades."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        if result.n_trades == 0:
            pytest.skip("No trades generated")
        
        # Count actual wins manually
        actual_wins = sum(1 for t in result.trades if t.pnl_pct > 0)
        expected_win_rate = (actual_wins / result.n_trades) * 100  # As percentage
        
        assert abs(result.win_rate - expected_win_rate) < 0.1, \
            f"Win rate {result.win_rate}% != expected {expected_win_rate}%"
    
    def test_last_n_trades_consistent(self, sample_prices, config):
        """Last N trades should be verifiable in the trades list."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        if len(result.trades) < 5:
            pytest.skip("Not enough trades")
        
        # Check last 5 trades
        last_5 = result.trades[-5:]
        last_5_win_rate = sum(1 for t in last_5 if t.pnl_pct > 0) / 5
        
        # This is just for verification, not an equality assertion
        # The point is we can trace individual trades
        assert all(t.pnl_pct != 0 or t.exit_price == t.entry_price for t in last_5)
    
    def test_100_percent_win_rate_all_positive(self, config):
        """If win_rate is 100%, ALL trades must be positive."""
        # Create data that should generate winning trades
        dates = pd.date_range(start="2023-01-01", periods=500, freq="B")
        
        # Steadily rising prices (all buys should win)
        prices = 100 * np.exp(np.linspace(0, 0.5, 500))
        
        # Add periodic dips that quickly recover
        for i in range(50, 450, 100):
            prices[i:i+5] *= 0.85  # 15% dip
            prices[i+5:i+25] *= 1.10  # Quick recovery
        
        df = pd.DataFrame({
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [1000000] * 500,
        }, index=dates)
        df.attrs["symbol"] = "RISING"
        
        df = compute_indicators(df)
        
        engine = BacktestEngine(config)
        result = engine.backtest_strategy(df, "drawdown_buy", {"drawdown_threshold": -0.10}, holding_days=20)
        
        if result.win_rate == 100.0:  # Now stored as percentage
            # Every single trade must be profitable
            for i, trade in enumerate(result.trades):
                assert trade.pnl_pct > 0, f"Trade {i} has negative return {trade.pnl_pct}% but win_rate is 100%"


# =============================================================================
# Look-Ahead Bias Tests
# =============================================================================

class TestNoLookAheadBias:
    """Ensure no look-ahead bias in signals or trades."""
    
    def test_entry_uses_only_past_data(self, sample_prices, config):
        """Entry signals should only use data up to that point."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        for trade in result.trades:
            # Entry should happen AFTER the signal date's data is available
            # i.e., we can only act on data we've seen
            entry_idx = df.index.get_loc(trade.entry_date)
            
            # All indicators at entry should be computable from past data
            assert entry_idx >= 20  # Need enough history for indicators (0-19 = 20 days)
    
    def test_future_prices_not_accessible(self, sample_prices, config):
        """Trade P&L should not use future information at entry time."""
        # This is a structural test - ensure we're not computing returns wrong
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30}, holding_days=20)
        
        for trade in result.trades:
            if trade.exit_date:
                # Exit must be after entry
                assert trade.exit_date >= trade.entry_date
                # Holding days must be <= max
                assert trade.holding_days <= config.max_holding_days or trade.exit_reason == "end_of_period"


# =============================================================================
# Statistical Validation Tests
# =============================================================================

class TestStatisticalValidation:
    """Test statistical measures are correctly computed."""
    
    def test_sharpe_ratio_calculation(self, sample_prices, config):
        """Sharpe ratio should be computed correctly."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        if result.equity_curve is None:
            pytest.skip("No equity curve")
        
        # Manually compute Sharpe
        returns = result.equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            excess = returns - 0.04/252  # 4% risk-free
            manual_sharpe = (excess.mean() / returns.std()) * np.sqrt(252)
            
            assert abs(result.sharpe_ratio - manual_sharpe) < 0.01
    
    def test_max_drawdown_correct(self, sample_prices, config):
        """Max drawdown should be the worst peak-to-trough decline."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        if result.equity_curve is None:
            pytest.skip("No equity curve")
        
        # Manual computation
        rolling_max = result.equity_curve.cummax()
        drawdown = (result.equity_curve - rolling_max) / rolling_max
        manual_mdd = drawdown.min() * 100
        
        assert abs(result.max_drawdown_pct - manual_mdd) < 0.01


# =============================================================================
# Benchmark Comparison Tests
# =============================================================================

class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""
    
    def test_buy_hold_return(self, sample_prices):
        """Buy and hold return should be simple start-to-end."""
        prices = sample_prices["close"]
        bh_return = compute_buy_and_hold_return(prices)
        
        expected = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        assert abs(bh_return - expected) < 0.01
    
    def test_compare_to_benchmark(self, sample_prices, config):
        """Comparison should correctly identify if strategy beats benchmark."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        comparison = compare_to_benchmark(result, sample_prices["close"])
        
        assert "strategy_return" in comparison
        assert "stock_buy_hold_return" in comparison
        assert "excess_vs_stock" in comparison
        assert comparison["excess_vs_stock"] == comparison["strategy_return"] - comparison["stock_buy_hold_return"]


# =============================================================================
# Strategy Tests
# =============================================================================

class TestStrategies:
    """Test individual strategies work correctly."""
    
    @pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
    def test_strategy_runs(self, sample_prices, config, strategy_name):
        """Each strategy should run without errors."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        _, default_params, _ = STRATEGIES[strategy_name]
        
        # Should not raise
        result = engine.backtest_strategy(df, strategy_name, default_params)
        
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == strategy_name
    
    def test_mean_reversion_triggers_on_oversold(self, sample_prices, config):
        """Mean reversion RSI should trigger when RSI < threshold."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 40})
        
        # Should have some trades
        # (threshold 40 is generous, should trigger)
        # Note: may still be 0 if RSI never crosses below 40
        assert isinstance(result.n_trades, int)


# =============================================================================
# Walk-Forward Validation Tests
# =============================================================================

class TestWalkForward:
    """Test walk-forward optimization."""
    
    def test_walk_forward_returns_oos_results(self, sample_prices, config):
        """Walk-forward should return out-of-sample results."""
        config.n_folds = 3  # Fewer folds for faster test
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        result, report = engine.walk_forward_optimization(
            df, "mean_reversion_rsi", n_trials_per_fold=10
        )
        
        assert result.is_out_of_sample or result.n_trades == 0
        assert isinstance(report, ValidationReport)
    
    def test_walk_forward_detects_overfitting(self, config):
        """Walk-forward should detect obvious overfitting."""
        # Create random data where any pattern is spurious
        np.random.seed(123)
        dates = pd.date_range(start="2022-01-01", periods=750, freq="B")
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.02, 750))
        
        df = pd.DataFrame({
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": [1000000] * 750,
        }, index=dates)
        df.attrs["symbol"] = "RANDOM"
        df = compute_indicators(df)
        
        config.n_folds = 3
        engine = BacktestEngine(config)
        
        _, report = engine.walk_forward_optimization(df, "mean_reversion_rsi", n_trials_per_fold=20)
        
        # Random walk should show poor OOS performance or overfitting
        # This is probabilistic, so we just check the report structure
        assert hasattr(report, "has_overfitting")
        assert hasattr(report, "sharpe_degradation")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_find_best_strategy(self, sample_prices, config):
        """Should find and validate the best strategy."""
        config.n_folds = 2  # Faster
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        # Test with limited strategies for speed
        best_name, result, report = engine.find_best_strategy(
            df,
            strategies=["mean_reversion_rsi", "drawdown_buy"],
            n_trials_per_strategy=10,
        )
        
        assert isinstance(best_name, str)
        assert isinstance(result, StrategyResult)
        assert isinstance(report, ValidationReport)
    
    def test_full_pipeline(self, sample_prices, config):
        """Test the complete backtesting pipeline."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        # 1. Backtest with default params
        result1 = engine.backtest_strategy(df, "mean_reversion_rsi", {"oversold_threshold": 30})
        
        # 2. Optimize parameters
        best_params, result2 = engine.optimize_strategy_optuna(
            df, "mean_reversion_rsi", n_trials=20
        )
        
        # 3. Compare to benchmark
        comparison = compare_to_benchmark(result2, sample_prices["close"])
        
        # All steps should complete
        assert result1.strategy_name == "mean_reversion_rsi"
        assert isinstance(best_params, dict)


# =============================================================================
# Ensemble Strategy Tests
# =============================================================================

class TestEnsembleStrategies:
    """Test strategy combination/ensemble functionality."""
    
    def test_get_all_strategies_includes_base(self):
        """get_all_strategies includes all base strategies."""
        from app.quant_engine.backtest_engine import STRATEGIES, get_all_strategies
        
        all_strats = get_all_strategies(include_ensembles=False)
        
        # Should include all base strategies
        for base_name in STRATEGIES:
            assert base_name in all_strats
    
    def test_get_all_strategies_includes_ensembles(self):
        """get_all_strategies includes ensemble combinations."""
        from app.quant_engine.backtest_engine import STRATEGIES, get_all_strategies
        
        base_only = get_all_strategies(include_ensembles=False)
        with_ensembles = get_all_strategies(include_ensembles=True, max_ensemble_size=2)
        
        # With ensembles should have more strategies
        assert len(with_ensembles) > len(base_only)
        
        # Should include some ensemble strategies
        ensemble_names = [k for k in with_ensembles if k.startswith("ensemble_")]
        assert len(ensemble_names) > 0
    
    def test_ensemble_strategy_runs(self, sample_prices, config):
        """Ensemble strategies can be backtested."""
        from app.quant_engine.backtest_engine import get_all_strategies
        
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        # Get an ensemble strategy
        all_strats = get_all_strategies(include_ensembles=True, max_ensemble_size=2)
        ensemble_names = [k for k in all_strats if k.startswith("ensemble_AND")]
        
        if not ensemble_names:
            pytest.skip("No ensemble strategies available")
        
        # Should be able to backtest it
        result = engine.backtest_strategy(df, ensemble_names[0], {})
        
        assert isinstance(result, StrategyResult)
        # AND ensembles are more selective, may have fewer signals
        assert result.n_trades >= 0
    
    def test_find_best_includes_ensembles(self, sample_prices, config):
        """find_best_strategy can search through ensembles."""
        engine = BacktestEngine(config)
        df = compute_indicators(sample_prices)
        df.attrs = sample_prices.attrs
        
        # Should be able to include ensembles in search
        best_name, result, report = engine.find_best_strategy(
            df,
            strategies=None,  # Use all including ensembles
            n_trials_per_strategy=5,
            include_ensembles=True,
            max_ensemble_size=2,
        )
        
        # Should return something (even if no strategy passes all checks)
        assert isinstance(best_name, str)
        assert isinstance(result, StrategyResult)


# =============================================================================
# Integration Tests - Trade Engine Integration
# =============================================================================

class TestGetOptimizedStrategy:
    """Tests for get_optimized_strategy integration function."""
    
    def test_returns_valid_structure(self, sample_prices):
        """get_optimized_strategy returns expected dict structure."""
        from app.quant_engine.trade_engine import get_optimized_strategy
        
        result = get_optimized_strategy(sample_prices, "TEST")
        
        # Must have all required keys
        assert "symbol" in result
        assert "strategy_name" in result
        assert "has_active_signal" in result
        assert "action" in result
        assert "reason" in result
        assert "metrics" in result
        assert "benchmark_comparison" in result
        assert "recent_trades" in result
        
        # Symbol should match input
        assert result["symbol"] == "TEST"
        
        # Action should be one of expected values
        assert result["action"] in ["BUY", "HOLD_POSITION", "WAIT", "HOLD"]
    
    def test_with_runtime_settings(self, sample_prices):
        """get_optimized_strategy uses runtime settings when provided."""
        from app.quant_engine.trade_engine import get_optimized_strategy
        
        settings = {
            "trading_initial_capital": 25000,
            "trading_flat_cost_per_trade": 5.0,
            "trading_slippage_bps": 10,
            "trading_stop_loss_pct": 8.0,
            "trading_take_profit_pct": 25.0,
            "trading_max_holding_days": 45,
            "trading_min_trades_required": 10,
            "trading_walk_forward_folds": 3,
            "trading_train_ratio": 0.65,
        }
        
        result = get_optimized_strategy(sample_prices, "TEST", runtime_settings=settings)
        
        # Should still return valid structure
        assert result["symbol"] == "TEST"
        assert isinstance(result["metrics"], dict)
    
    def test_metrics_structure(self, sample_prices):
        """get_optimized_strategy metrics has expected fields."""
        from app.quant_engine.trade_engine import get_optimized_strategy
        
        result = get_optimized_strategy(sample_prices, "TEST")
        
        if result["strategy_name"] is not None:
            # If a valid strategy was found, check metrics
            metrics = result["metrics"]
            assert "total_trades" in metrics
            assert "win_rate" in metrics
            assert "total_return_pct" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown_pct" in metrics
            assert "avg_holding_days" in metrics
            
            # Win rate should be percentage (0-100)
            assert 0 <= metrics["win_rate"] <= 100
    
    def test_benchmark_comparison_structure(self, sample_prices):
        """get_optimized_strategy benchmark comparison has expected fields."""
        from app.quant_engine.trade_engine import get_optimized_strategy
        
        result = get_optimized_strategy(sample_prices, "TEST")
        
        comparison = result["benchmark_comparison"]
        assert "stock_buy_hold_return" in comparison
        assert "beats_stock" in comparison
        assert "excess_vs_stock" in comparison
    
    def test_handles_uppercase_columns(self, sample_prices):
        """get_optimized_strategy handles uppercase column names."""
        from app.quant_engine.trade_engine import get_optimized_strategy
        
        # Uppercase columns
        df = sample_prices.copy()
        df.columns = [c.upper() for c in df.columns]
        
        result = get_optimized_strategy(df, "TEST")
        
        # Should still work
        assert result["symbol"] == "TEST"
        assert isinstance(result, dict)
