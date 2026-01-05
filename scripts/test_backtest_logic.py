#!/usr/bin/env python
"""
Real data validation script for the backtest engine.
Tests the logic to ensure output makes sense with real market data.
"""

import json
import sys

import numpy as np
import pandas as pd
import yfinance as yf


def test_alpha_factory_logic():
    """Test AlphaFactory optimization with real MSFT data."""
    print("=" * 60)
    print("TESTING ALPHAFACTORY WITH REAL MSFT DATA")
    print("=" * 60)
    
    # Fetch real data
    print("\n[1] Fetching MSFT data...")
    msft = yf.Ticker("MSFT").history(period="5y")
    print(f"    Got {len(msft)} days of data")
    print(f"    Date range: {msft.index[0].date()} to {msft.index[-1].date()}")
    print(f"    Price range: ${msft['Close'].min():.2f} - ${msft['Close'].max():.2f}")
    
    # Run optimization
    print("\n[2] Running AlphaFactory optimization (30 trials)...")
    from app.quant_engine.backtest_v2.alpha_factory import (
        AlphaFactory,
        AlphaFactoryConfig,
    )
    
    config = AlphaFactoryConfig(n_trials=30)
    factory = AlphaFactory(train_prices=msft, config=config)
    result = factory.optimize()
    
    print(f"    Best strategy: {result.best_genome.name}")
    print(f"    Optimization time: {result.optimization_time_seconds:.2f}s")
    
    # Analyze metrics
    m = result.best_metrics
    print("\n[3] STRATEGY METRICS:")
    print(f"    Total Return:      {m.total_return * 100:+.2f}%")
    print(f"    Sharpe Ratio:      {m.sharpe_ratio:.3f}")
    print(f"    Max Drawdown:      {m.max_drawdown * 100:.2f}%")
    print(f"    Win Rate:          {m.win_rate * 100:.1f}%")
    print(f"    Num Trades:        {m.num_trades}")
    
    # Compare to buy & hold
    bh_return = (msft["Close"].iloc[-1] / msft["Close"].iloc[0] - 1) * 100
    print(f"\n[4] BENCHMARK COMPARISON:")
    print(f"    Buy & Hold Return: {bh_return:+.2f}%")
    print(f"    Strategy vs B&H:   {m.total_return * 100 - bh_return:+.2f}%")
    
    # Sanity checks
    print("\n[5] SANITY CHECKS:")
    issues = []
    
    if m.sharpe_ratio > 5:
        issues.append(f"Sharpe {m.sharpe_ratio:.2f} > 5 (unrealistic)")
    if m.sharpe_ratio < -5:
        issues.append(f"Sharpe {m.sharpe_ratio:.2f} < -5 (unrealistic)")
    if m.win_rate > 0.95:
        issues.append(f"Win rate {m.win_rate:.1%} too high")
    if m.num_trades < 5:
        issues.append(f"Only {m.num_trades} trades (low sample)")
    if m.num_trades > 500:
        issues.append(f"{m.num_trades} trades (overtrading)")
    if abs(m.max_drawdown) > 1:
        issues.append(f"Max DD {m.max_drawdown:.0%} > 100%")
    
    if issues:
        for issue in issues:
            print(f"    ⚠️  {issue}")
    else:
        print("    ✓ All metrics look reasonable")
    
    # Show strategy logic
    g = result.best_genome
    print("\n[6] WINNING STRATEGY LOGIC:")
    print(f"    Entry ({len(g.entry_conditions)} conditions):")
    for c in g.entry_conditions:
        print(f"      - {c.indicator_type.value}({c.period}) {c.logic_gate.value} {c.threshold:.1f}")
    print(f"    Exit ({len(g.exit_conditions)} conditions):")
    for c in g.exit_conditions:
        print(f"      - {c.indicator_type.value}({c.period}) {c.logic_gate.value} {c.threshold:.1f}")
    print(f"    Stop Loss:    {g.stop_loss_pct:.1%}")
    print(f"    Take Profit:  {g.take_profit_pct:.1%}")
    print(f"    Max Holding:  {g.holding_period_max} days")
    
    return result, msft


def test_strategy_analyzer_logic(result, msft):
    """Test StrategyAnalyzer with the optimization result."""
    print("\n" + "=" * 60)
    print("TESTING STRATEGY ANALYZER")
    print("=" * 60)
    
    # Fetch SPY for benchmark
    print("\n[1] Fetching SPY for benchmark...")
    spy = yf.Ticker("SPY").history(period="5y")
    print(f"    Got {len(spy)} days")
    
    # Create analyzer
    print("\n[2] Creating StrategyAnalyzer...")
    from app.quant_engine.backtest_v2.strategy_analyzer import create_strategy_analyzer
    
    analyzer = create_strategy_analyzer(
        prices=msft,
        spy_prices=spy,
        symbol="MSFT",
    )
    
    # Generate report
    print("[3] Generating full report...")
    report = analyzer.generate_full_report(result)
    winner = report.winner
    
    print(f"\n[4] WINNER ANALYSIS:")
    print(f"    Name:       {winner.name}")
    print(f"    Verdict:    {winner.verdict.value}")
    print(f"    Confidence: {winner.confidence_score:.1f}/100")
    
    # Trade stats
    ts = winner.trade_stats
    print(f"\n[5] TRADE STATISTICS:")
    print(f"    Total Trades:     {ts.total_trades}")
    print(f"    Winning:          {ts.winning_trades}")
    print(f"    Losing:           {ts.losing_trades}")
    print(f"    Win Rate:         {ts.win_rate:.1%}")
    print(f"    Avg Duration:     {ts.avg_duration_days:.1f} days")
    print(f"    Best Trade:       {ts.best_trade_pct:+.1f}%")
    print(f"    Worst Trade:      {ts.worst_trade_pct:+.1f}%")
    
    # Risk metrics
    rm = winner.risk_metrics
    print(f"\n[6] RISK METRICS:")
    print(f"    Sharpe Ratio:     {rm.sharpe_ratio:.3f}")
    print(f"    Sortino Ratio:    {rm.sortino_ratio:.3f}")
    print(f"    Calmar Ratio:     {rm.calmar_ratio:.3f}")
    print(f"    Max Drawdown:     {rm.max_drawdown_pct:.1f}%")
    print(f"    Profit Factor:    {rm.profit_factor:.2f}")
    print(f"    Expectancy:       ${rm.expectancy:.2f}/trade")
    
    # Advanced metrics
    am = winner.advanced_metrics
    print(f"\n[7] ADVANCED METRICS:")
    print(f"    Kelly Criterion:  {am.kelly_criterion:.1%}")
    print(f"    Kelly Half:       {am.kelly_half:.1%}")
    print(f"    SQN:              {am.sqn:.2f} ({am.sqn_rating})")
    print(f"    Payoff Ratio:     {am.payoff_ratio:.2f}")
    print(f"    Time in Market:   {am.time_in_market_pct:.1f}%")
    print(f"    Max Win Streak:   {am.max_consecutive_wins}")
    print(f"    Max Loss Streak:  {am.max_consecutive_losses}")
    
    # Benchmark comparison
    bc = winner.benchmark_comparison
    print(f"\n[8] BENCHMARK COMPARISON:")
    print(f"    Strategy Return:  {bc.strategy_return_pct:+.1f}%")
    print(f"    Buy & Hold:       {bc.buy_hold_return_pct:+.1f}%")
    print(f"    SPY Return:       {bc.spy_return_pct:+.1f}%")
    print(f"    Alpha vs B&H:     {bc.alpha_vs_buy_hold:+.1f}%")
    print(f"    Alpha vs SPY:     {bc.alpha_vs_spy:+.1f}%")
    print(f"    Beta to SPY:      {bc.beta_to_spy:.2f}")
    print(f"    Correlation:      {bc.correlation_to_spy:.2f}")
    
    # Equity curve
    ec = winner.equity_curve
    print(f"\n[9] EQUITY CURVE:")
    print(f"    Total points:     {len(ec)}")
    if ec:
        print(f"    Start:            {ec[0].timestamp.date()} @ ${ec[0].equity:.2f}")
        print(f"    End:              {ec[-1].timestamp.date()} @ ${ec[-1].equity:.2f}")
        print(f"    Final Return:     {ec[-1].equity_pct:+.1f}%")
    
    # Signals
    signals = winner.signals
    print(f"\n[10] TRADING SIGNALS:")
    print(f"    Total signals:    {len(signals)}")
    if signals:
        buys = [s for s in signals if s.signal_type.value == "BUY"]
        sells = [s for s in signals if s.signal_type.value != "BUY"]
        print(f"    Buy signals:      {len(buys)}")
        print(f"    Sell signals:     {len(sells)}")
    
    # Sanity checks
    print(f"\n[11] SANITY CHECKS:")
    issues = []
    
    if ts.total_trades == 0:
        issues.append("No trades executed")
    if ts.avg_duration_days == 0 and ts.total_trades > 0:
        issues.append("avg_duration_days is 0 but trades exist")
    if ts.win_rate > 1:
        issues.append(f"Win rate {ts.win_rate:.1%} > 100%")
    if rm.profit_factor < 0:
        issues.append(f"Profit factor {rm.profit_factor:.2f} is negative")
    if abs(rm.max_drawdown_pct) > 100:
        issues.append(f"Max DD {rm.max_drawdown_pct:.0f}% > 100%")
    if am.kelly_criterion > 1:
        issues.append(f"Kelly {am.kelly_criterion:.1%} > 100%")
    if abs(am.sqn) > 10:
        issues.append(f"SQN {am.sqn:.2f} out of valid range")
    if len(ec) == 0:
        issues.append("Empty equity curve")
    if abs(bc.correlation_to_spy) > 1:
        issues.append(f"Correlation {bc.correlation_to_spy:.2f} out of range")
    
    if issues:
        print("    ISSUES FOUND:")
        for issue in issues:
            print(f"      ❌ {issue}")
    else:
        print("    ✓ All values look valid")
    
    # JSON serialization
    print(f"\n[12] JSON SERIALIZATION:")
    try:
        json_data = report.to_json_safe()
        json_str = json.dumps(json_data, default=str)
        print(f"    ✓ Serialization OK ({len(json_str):,} bytes)")
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
    
    print(f"\n[13] AUTO-GENERATED SUMMARY:")
    print(f"    {report.summary}")
    
    # Baseline Comparison (NEW)
    baseline = report.baseline_comparison
    if baseline:
        print("\n" + "=" * 60)
        print("BASELINE STRATEGY COMPARISON (with compounding)")
        print("=" * 60)
        
        # Show optimal dip threshold
        buy_dips = baseline.buy_dips
        print(f"\n[14] DIP ANALYSIS (from DipEntryOptimizer):")
        print(f"    {buy_dips.description}")
        print(f"    Dips Detected: {buy_dips.dips_detected}")
        print(f"    Avg Dip Depth: {buy_dips.avg_dip_depth_pct:.1f}%")
        
        # Main comparison (same capital)
        print(f"\n[15] {baseline.symbol} MAIN STRATEGIES ($10k + $1k/month, compounding):")
        strategies = [
            ("DCA Monthly", baseline.dca),
            ("Buy Dips & Hold", baseline.buy_dips),
        ]
        if baseline.dip_trading:
            strategies.append(("Perfect Dip Trading", baseline.dip_trading))
        if baseline.technical_trading:
            strategies.append(("Technical Trading", baseline.technical_trading))
        strategies.append(("SPY DCA", baseline.spy_dca))
        
        print(f"\n    {'Strategy':<25} {'Invested':>10} {'Final':>12} {'Profit':>12} {'ROI %':>8} {'Sharpe':>7}")
        print("    " + "-" * 78)
        for name, s in strategies:
            print(f"    {name:<25} ${s.total_invested:>8,.0f} ${s.final_value:>10,.0f} ${s.profit:>+10,.0f} {s.total_return_pct:>+7.1f}% {s.sharpe_ratio:>7.2f}")
        
        # Reference strategies
        print(f"\n[15b] REFERENCE (different capital, for context):")
        refs = [
            ("Buy & Hold ($10k only)", baseline.buy_hold),
            ("Lump Sum (all day 1)", baseline.lump_sum),
        ]
        for name, s in refs:
            print(f"    {name:<25} ${s.total_invested:>8,.0f} ${s.final_value:>10,.0f} ${s.profit:>+10,.0f} {s.total_return_pct:>+7.1f}% {s.sharpe_ratio:>7.2f}")
        
        print(f"\n[16] RANKINGS:")
        print(f"    By Return:  {' > '.join(baseline.ranked_by_return[:5])}")
        print(f"    By Sharpe:  {' > '.join(baseline.ranked_by_sharpe[:5])}")
        
        rec = baseline.recommendation
        print(f"\n[17] INVESTMENT RECOMMENDATION:")
        print(f"    {'='*50}")
        print(f"    📌 {rec.headline}")
        print(f"    {'='*50}")
        print(f"    Type: {rec.recommendation.value}")
        print(f"    Reasoning: {rec.reasoning}")
        print(f"\n    Best Strategy: {rec.best_strategy_name} ({rec.best_strategy_return_pct:+.1f}%)")
        print(f"    SPY Return:    {rec.spy_return_pct:+.1f}%")
        print(f"    Alpha vs SPY:  {rec.alpha_vs_spy:+.1f}%")
        print(f"    Risk-Adjusted Winner: {rec.risk_adjusted_winner}")
        
        print(f"\n    📋 Action Items:")
        for item in rec.action_items:
            print(f"       • {item}")
        
        print(f"\n    ⚠️  Warnings:")
        for warn in rec.warnings:
            print(f"       • {warn}")
    else:
        print("\n[14] BASELINE COMPARISON: Not available")
    
    return report


def test_multiple_stocks():
    """Test with different stock types."""
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE STOCK TYPES")
    print("=" * 60)
    
    from app.quant_engine.backtest_v2.alpha_factory import (
        AlphaFactory,
        AlphaFactoryConfig,
    )
    from app.quant_engine.backtest_v2.baseline_strategies import BaselineEngine
    
    stocks = {
        "MSFT": "Growth/Tech",
        "XOM": "Value/Energy", 
        "TSLA": "High Volatility",
    }
    
    # Fetch SPY once for all comparisons
    print("\nFetching SPY for baseline comparisons...")
    spy = yf.Ticker("SPY").history(period="3y")
    print(f"    Got {len(spy)} days of SPY data")
    
    results = []
    for symbol, category in stocks.items():
        print(f"\n--- {symbol} ({category}) ---")
        
        df = yf.Ticker(symbol).history(period="3y")
        print(f"    Data: {len(df)} days")
        
        # Run optimization
        config = AlphaFactoryConfig(n_trials=15)
        factory = AlphaFactory(train_prices=df, config=config)
        result = factory.optimize()
        
        m = result.best_metrics
        opt_return = m.total_return * 100
        
        # Run baseline comparison
        baseline_engine = BaselineEngine(
            prices=df,
            spy_prices=spy,
            symbol=symbol,
        )
        baseline = baseline_engine.run_all(optimized_return_pct=opt_return)
        
        bh = baseline.buy_hold.total_return_pct
        dca = baseline.dca.total_return_pct
        spy_ret = baseline.spy_buy_hold.total_return_pct
        rec = baseline.recommendation
        
        print(f"    Optimized:       {opt_return:+.1f}%")
        print(f"    Buy & Hold:      {bh:+.1f}%")
        print(f"    DCA Monthly:     {dca:+.1f}%")
        print(f"    SPY B&H:         {spy_ret:+.1f}%")
        print(f"    Best Strategy:   {rec.best_strategy_name}")
        print(f"    📌 {rec.headline}")
        
        results.append({
            "symbol": symbol,
            "category": category,
            "optimized": opt_return,
            "bh": bh,
            "dca": dca,
            "spy": spy_ret,
            "recommendation": rec.recommendation.value,
            "headline": rec.headline,
        })
    
    print("\n" + "=" * 60)
    print("MULTI-STOCK SUMMARY WITH RECOMMENDATIONS")
    print("=" * 60)
    print(f"\n{'Symbol':<8} {'Optimized':>10} {'B&H':>10} {'DCA':>10} {'SPY':>10} {'Recommendation'}")
    print("-" * 80)
    for r in results:
        print(f"{r['symbol']:<8} {r['optimized']:>+9.1f}% {r['bh']:>+9.1f}% {r['dca']:>+9.1f}% {r['spy']:>+9.1f}% {r['recommendation']}")
    
    print("\n📌 HEADLINES:")
    for r in results:
        print(f"   {r['symbol']}: {r['headline']}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BACKTEST ENGINE LOGIC VALIDATION")
    print("=" * 60)
    
    # Test 1: AlphaFactory with real data
    result, msft = test_alpha_factory_logic()
    
    # Test 2: StrategyAnalyzer
    report = test_strategy_analyzer_logic(result, msft)
    
    # Test 3: Multiple stocks
    test_multiple_stocks()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
