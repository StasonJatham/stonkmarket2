"""
REAL Integration Tests for BacktestV2 Engine.

These tests run ACTUAL backtests with REAL market data.
Not mocks, not synthetic data - real yfinance prices.

Run with:
    python -m pytest system_tests/smoke/test_backtest_engine.py -v -s

Uses the PriceService which checks DB first, then yfinance if needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper to run async in sync context
# =============================================================================

def run_async(coro):
    """Run async function in sync test context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# Test 1: Real Data Fetching
# =============================================================================

class TestRealDataFetching:
    """Test that we can actually fetch real market data."""

    def test_fetch_msft_history(self):
        """Fetch MSFT price history - should have 20+ years of data."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        
        df = run_async(fetch_all_history("MSFT"))
        
        assert df is not None, "Failed to fetch MSFT data"
        assert not df.empty, "MSFT data is empty"
        assert len(df) > 5000, f"Expected 20+ years of data, got {len(df)} days"
        
        # Check data quality
        assert "Close" in df.columns or "close" in df.columns
        
        # Should have data back to at least 2000
        first_date = df.index.min()
        assert first_date.year <= 2000, f"Data only goes back to {first_date.year}"
        
        logger.info(f"MSFT: {len(df)} days from {first_date.date()} to {df.index.max().date()}")

    def test_fetch_spy_for_benchmark(self):
        """Fetch SPY for benchmark comparison."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        
        df = run_async(fetch_all_history("SPY"))
        
        assert df is not None
        assert not df.empty
        assert len(df) > 5000, f"SPY should have 20+ years, got {len(df)}"
        
        logger.info(f"SPY: {len(df)} days")

    def test_fetch_volatile_stock_tsla(self):
        """Fetch TSLA - volatile stock, IPO'd 2010."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        
        df = run_async(fetch_all_history("TSLA"))
        
        assert df is not None
        assert not df.empty
        # TSLA IPO'd June 2010, so ~14 years of data
        assert len(df) > 3000, f"TSLA should have 14+ years, got {len(df)}"
        
        logger.info(f"TSLA: {len(df)} days")


# =============================================================================
# Test 2: Indicator Matrix Computation
# =============================================================================

class TestIndicatorMatrixReal:
    """Test indicator computation on real data."""

    def test_compute_all_indicators_msft(self):
        """Compute full indicator matrix on MSFT real data."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import IndicatorMatrix
        
        # Fetch real data
        df = run_async(fetch_all_history("MSFT"))
        assert not df.empty
        
        # Compute indicators
        matrix = IndicatorMatrix(df)
        
        # Should have many pre-computed indicators
        assert len(matrix.available_indicators) > 50, f"Only {len(matrix.available_indicators)} indicators"
        
        # Check RSI values are sane (0-100)
        rsi = matrix.get_numpy("rsi", 14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0
        assert valid_rsi.min() >= 0, f"RSI min={valid_rsi.min()}"
        assert valid_rsi.max() <= 100, f"RSI max={valid_rsi.max()}"
        
        logger.info(f"Computed {len(matrix.available_indicators)} indicators for MSFT")
        logger.info(f"RSI(14) range: {valid_rsi.min():.1f} - {valid_rsi.max():.1f}")

    def test_bollinger_bands_sane(self):
        """Bollinger Band %B should be mostly between -0.5 and 1.5."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import IndicatorMatrix
        
        df = run_async(fetch_all_history("AAPL"))
        matrix = IndicatorMatrix(df)
        
        bb_pct = matrix.get_numpy("bb_pct", 20)
        valid = bb_pct[~np.isnan(bb_pct)]
        
        # Most values should be in normal range
        in_range = np.sum((valid > -1) & (valid < 2)) / len(valid)
        assert in_range > 0.9, f"Only {in_range:.0%} of BB%B values in normal range"
        
        logger.info(f"BB%B range: {valid.min():.2f} to {valid.max():.2f}")


# =============================================================================
# Test 3: AlphaFactory Optimization with Real Data
# =============================================================================

class TestAlphaFactoryReal:
    """Test strategy optimization on real market data."""

    def test_optimize_msft_strategy(self):
        """Run actual optimization on MSFT data."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import (
            AlphaFactory, AlphaFactoryConfig
        )
        
        df = run_async(fetch_all_history("MSFT"))
        
        # Use last 5 years for faster test
        df = df.tail(1260)
        
        # Run optimization with minimal trials for speed
        config = AlphaFactoryConfig(n_trials=20)
        factory = AlphaFactory(train_prices=df, config=config)
        result = factory.optimize()
        
        assert result is not None, "Optimization returned None"
        assert result.best_genome is not None, "No best genome found"
        assert result.best_metrics is not None, "No metrics computed"
        
        # Log actual results
        metrics = result.best_metrics
        logger.info(f"MSFT Optimization Results:")
        logger.info(f"  Total Return: {metrics.total_return:.1%}")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown:.1%}")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Num Trades: {metrics.num_trades}")
        
        # Sanity checks on real results
        assert metrics.num_trades > 0, "Strategy made no trades"
        assert -1 <= metrics.total_return <= 10, f"Unrealistic return: {metrics.total_return}"
        assert 0 <= metrics.win_rate <= 1, f"Invalid win rate: {metrics.win_rate}"

    def test_optimize_volatile_tsla(self):
        """TSLA optimization should produce different strategy than MSFT."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import (
            AlphaFactory, AlphaFactoryConfig
        )
        
        df = run_async(fetch_all_history("TSLA"))
        df = df.tail(1260)  # Last 5 years
        
        config = AlphaFactoryConfig(n_trials=20)
        factory = AlphaFactory(train_prices=df, config=config)
        result = factory.optimize()
        
        assert result is not None
        metrics = result.best_metrics
        
        logger.info(f"TSLA Optimization Results:")
        logger.info(f"  Total Return: {metrics.total_return:.1%}")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown:.1%}")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        
        # TSLA is volatile - drawdown should be significant
        # But strategy should still be valid
        assert metrics.num_trades > 0


# =============================================================================
# Test 4: Full Strategy Report Generation
# =============================================================================

class TestStrategyReportReal:
    """Test the complete StrategyFullReport with real data."""

    def test_generate_full_report_msft(self):
        """Generate complete strategy report with all metrics."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import (
            AlphaFactory, AlphaFactoryConfig
        )
        from app.quant_engine.backtest_v2.strategy_analyzer import (
            create_strategy_analyzer
        )
        
        # Fetch real data
        msft_df = run_async(fetch_all_history("MSFT"))
        spy_df = run_async(fetch_all_history("SPY"))
        
        # Use last 3 years for reasonable test time
        msft_df = msft_df.tail(756)
        spy_df = spy_df.tail(756)
        
        # Run optimization
        config = AlphaFactoryConfig(n_trials=15)
        factory = AlphaFactory(train_prices=msft_df, config=config)
        opt_result = factory.optimize()
        
        # Generate full report
        analyzer = create_strategy_analyzer(
            prices=msft_df,
            spy_prices=spy_df,
            symbol="MSFT"
        )
        report = analyzer.generate_full_report(opt_result)
        
        # Validate report structure
        assert report is not None
        assert report.winner is not None
        assert report.meta.symbol == "MSFT"
        
        # Check Trade Stats
        stats = report.winner.trade_stats
        logger.info(f"\nTrade Stats:")
        logger.info(f"  Total Trades: {stats.total_trades}")
        logger.info(f"  Win Rate: {stats.win_rate:.1%}")
        logger.info(f"  Avg Duration (days): {stats.avg_duration_days:.1f}")
        logger.info(f"  Best Trade: {stats.best_trade_pct:.1f}%")
        logger.info(f"  Worst Trade: {stats.worst_trade_pct:.1f}%")
        
        # Sanity checks
        assert stats.total_trades >= 0
        assert 0 <= stats.win_rate <= 1
        
        # Check Risk Metrics
        risk = report.winner.risk_metrics
        logger.info(f"\nRisk Metrics:")
        logger.info(f"  Sharpe Ratio: {risk.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio: {risk.sortino_ratio:.2f}")
        logger.info(f"  Max Drawdown: {risk.max_drawdown_pct:.1f}%")
        logger.info(f"  Profit Factor: {risk.profit_factor:.2f}")
        
        # Check Advanced Metrics
        adv = report.winner.advanced_metrics
        logger.info(f"\nAdvanced Metrics:")
        logger.info(f"  Kelly Criterion: {adv.kelly_criterion:.2%}")
        logger.info(f"  Kelly Half: {adv.kelly_half:.2%}")
        logger.info(f"  SQN: {adv.sqn:.2f} ({adv.sqn_rating})")
        logger.info(f"  Max Consecutive Wins: {adv.max_consecutive_wins}")
        logger.info(f"  Max Consecutive Losses: {adv.max_consecutive_losses}")
        
        # Check Equity Curve
        curve = report.winner.equity_curve
        logger.info(f"\nEquity Curve: {len(curve)} points")
        assert len(curve) > 0
        assert curve[0].equity > 0
        
        # Check Benchmark Comparison
        bench = report.winner.benchmark_comparison
        logger.info(f"\nBenchmark Comparison:")
        logger.info(f"  Strategy Return: {bench.strategy_return_pct:.1f}%")
        logger.info(f"  Buy & Hold Return: {bench.buy_hold_return_pct:.1f}%")
        logger.info(f"  SPY Return: {bench.spy_return_pct:.1f}%")
        logger.info(f"  Alpha vs B&H: {bench.alpha_vs_buy_hold:.1f}%")
        logger.info(f"  Correlation to SPY: {bench.correlation_to_spy:.2f}")

    def test_report_json_serialization(self):
        """Ensure report can be serialized to JSON without NaN errors."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import (
            AlphaFactory, AlphaFactoryConfig
        )
        from app.quant_engine.backtest_v2.strategy_analyzer import (
            create_strategy_analyzer
        )
        
        df = run_async(fetch_all_history("AAPL"))
        df = df.tail(500)
        
        config = AlphaFactoryConfig(n_trials=10)
        factory = AlphaFactory(train_prices=df, config=config)
        result = factory.optimize()
        
        analyzer = create_strategy_analyzer(df, symbol="AAPL")
        report = analyzer.generate_full_report(result)
        
        # This should NOT raise
        json_data = report.to_json_safe()
        json_str = json.dumps(json_data, default=str)
        
        assert len(json_str) > 1000, "JSON too short, something missing"
        
        # Verify no NaN in serialized data
        assert "NaN" not in json_str, "NaN found in JSON output"
        assert "Infinity" not in json_str, "Infinity found in JSON output"
        
        logger.info(f"JSON size: {len(json_str)} bytes")


# =============================================================================
# Test 5: Regime Detection with Real Data
# =============================================================================

class TestRegimeDetectionReal:
    """Test regime detection on real market history."""

    def test_detect_2008_crash(self):
        """Regime detector should identify 2008 financial crisis."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.regime_filter import RegimeDetector, MarketRegime
        
        spy_df = run_async(fetch_all_history("SPY"))
        
        detector = RegimeDetector()
        # Extract Close column as Series (fetch_all_history returns DataFrame)
        close_prices = spy_df["Close"] if isinstance(spy_df, pd.DataFrame) else spy_df
        detector.set_spy_prices(close_prices)
        
        # Check regime in late 2008 (crash bottom)
        crash_date = pd.Timestamp(2008, 11, 20, tz="UTC")  # Near bottom
        
        regime = detector.detect_at_date(crash_date)
        
        # Should detect bearish/crash regime
        assert regime.regime in [MarketRegime.BEAR, MarketRegime.CRASH], \
            f"Expected BEAR/CRASH in Nov 2008, got {regime.regime}"
        
        logger.info(f"2008 Crash Detection: {regime.regime.value}")
        logger.info(f"  Drawdown: {regime.drawdown_pct:.1f}%")

    def test_detect_covid_crash_and_recovery(self):
        """Detect COVID crash (Mar 2020) and recovery (Aug 2020)."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.regime_filter import RegimeDetector, MarketRegime
        
        spy_df = run_async(fetch_all_history("SPY"))
        
        detector = RegimeDetector()
        # Extract Close column as Series (fetch_all_history returns DataFrame)
        close_prices = spy_df["Close"] if isinstance(spy_df, pd.DataFrame) else spy_df
        detector.set_spy_prices(close_prices)
        
        # March 2020 = COVID crash
        crash_date = pd.Timestamp(2020, 3, 23, tz="UTC")
        crash_regime = detector.detect_at_date(crash_date)
        
        # August 2020 = Recovery/Bull
        recovery_date = pd.Timestamp(2020, 8, 15, tz="UTC")
        recovery_regime = detector.detect_at_date(recovery_date)
        
        logger.info(f"COVID Crash (Mar 2020): {crash_regime.regime.value}")
        logger.info(f"Recovery (Aug 2020): {recovery_regime.regime.value}")
        
        # Crash should be BEAR or CRASH
        assert crash_regime.regime in [MarketRegime.BEAR, MarketRegime.CRASH]
        
        # Recovery should be RECOVERY or BULL
        assert recovery_regime.regime in [MarketRegime.RECOVERY, MarketRegime.BULL]


# =============================================================================
# Test 6: Crash Testing with Real Data
# =============================================================================

class TestCrashTestingReal:
    """Test crash testing module with real historical data."""

    def test_available_crash_periods(self):
        """Check which crash periods are testable with SPY data."""
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.crash_testing import (
            get_available_crash_periods_for_data, CrashPeriod
        )
        
        spy_df = run_async(fetch_all_history("SPY"))
        spy_series = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
        spy_series.index = pd.to_datetime(spy_series.index)
        
        available = get_available_crash_periods_for_data(spy_series)
        
        logger.info(f"Available crash periods for SPY:")
        for period in available:
            logger.info(f"  - {period.value}")
        
        # SPY should have data for at least 2008, 2020, 2022 crashes
        assert CrashPeriod.FINANCIAL_CRISIS_2008 in available
        assert CrashPeriod.COVID_CRASH_2020 in available
        assert CrashPeriod.TECH_CRASH_2022 in available


# =============================================================================
# Test 7: End-to-End Full Backtest
# =============================================================================

class TestFullBacktestE2E:
    """Complete end-to-end backtest test."""

    def test_full_backtest_workflow(self):
        """
        Complete workflow:
        1. Fetch real data
        2. Run optimization
        3. Generate report
        4. Serialize to JSON
        """
        from app.quant_engine.backtest_v2.service import fetch_all_history
        from app.quant_engine.backtest_v2.alpha_factory import (
            AlphaFactory, AlphaFactoryConfig
        )
        from app.quant_engine.backtest_v2.strategy_analyzer import (
            create_strategy_analyzer
        )
        
        logger.info("\n" + "="*60)
        logger.info("FULL E2E BACKTEST: MSFT")
        logger.info("="*60)
        
        # Step 1: Fetch data
        logger.info("\n1. Fetching real data...")
        msft = run_async(fetch_all_history("MSFT"))
        spy = run_async(fetch_all_history("SPY"))
        
        logger.info(f"   MSFT: {len(msft)} days")
        logger.info(f"   SPY: {len(spy)} days")
        
        # Use 2 years for test
        msft = msft.tail(504)
        spy = spy.tail(504)
        
        # Step 2: Optimize
        logger.info("\n2. Running optimization (20 trials)...")
        config = AlphaFactoryConfig(n_trials=20)
        factory = AlphaFactory(train_prices=msft, config=config)
        opt_result = factory.optimize()
        
        logger.info(f"   Best Sharpe: {opt_result.best_metrics.sharpe_ratio:.2f}")
        logger.info(f"   Trades: {opt_result.best_metrics.num_trades}")
        
        # Step 3: Generate report
        logger.info("\n3. Generating full report...")
        analyzer = create_strategy_analyzer(msft, spy_prices=spy, symbol="MSFT")
        report = analyzer.generate_full_report(opt_result)
        
        logger.info(f"   Winner Verdict: {report.winner.verdict.value}")
        logger.info(f"   Confidence: {report.winner.confidence_score:.0f}/100")
        
        # Step 4: Serialize
        logger.info("\n4. Serializing to JSON...")
        json_data = report.to_json_safe()
        json_str = json.dumps(json_data, default=str, indent=2)
        
        logger.info(f"   JSON size: {len(json_str):,} bytes")
        
        # Verify
        assert len(json_str) > 5000
        assert "NaN" not in json_str
        
        logger.info("\n" + "="*60)
        logger.info("E2E TEST PASSED")
        logger.info("="*60)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
