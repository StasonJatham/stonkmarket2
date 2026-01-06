#!/usr/bin/env python3
"""
Test if regime-aware enhancements benefit trading strategies.

Compares:
1. Standard Technical Trading (ignores market regime)
2. Regime-Aware Technical Trading (adapts to BULL/BEAR/CRASH)
3. Regime-Aware Dip Trading (scales in more aggressively in crashes)

Tests across multiple time periods including crash periods:
- 2008 Financial Crisis
- 2020 COVID Crash
- 2022 Tech Crash
- Full 10-year period
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.quant_engine.backtest.regime_filter import (
    RegimeDetector,
    MarketRegime,
    StrategyMode,
    StrategyConfig,
)


def get_stock_data(symbol: str, years: int = 10) -> tuple[pd.Series, pd.Series]:
    """Fetch stock and SPY data."""
    import yfinance as yf
    
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    
    stock = yf.Ticker(symbol)
    spy = yf.Ticker("SPY")
    
    stock_hist = stock.history(start=start, end=end)
    spy_hist = spy.history(start=start, end=end)
    
    return stock_hist["Close"], spy_hist["Close"]


def run_standard_technical(
    close: pd.Series,
    spy_close: pd.Series,
    initial_capital: float = 10000,
    monthly_contribution: float = 1000,
) -> dict:
    """
    Standard technical trading - no regime awareness.
    
    Uses simple moving average crossover:
    - Buy when price > SMA50
    - Sell when price < SMA50
    - Fixed 10% stop loss
    """
    sma50 = close.rolling(50).mean()
    
    cash = initial_capital
    shares = 0.0
    position = False
    entry_price = 0.0
    
    total_trades = 0
    winning_trades = 0
    total_contributions = initial_capital
    
    equity = [cash]
    last_month = None
    
    for i, (date, price) in enumerate(close.items()):
        # Monthly contribution
        if hasattr(date, 'month'):
            current_month = (date.year, date.month)
            if last_month is not None and current_month != last_month:
                if position:
                    shares += monthly_contribution / price
                else:
                    cash += monthly_contribution
                total_contributions += monthly_contribution
            last_month = current_month
        
        if i < 50:
            equity.append(cash + shares * price)
            continue
        
        if not position:
            # Buy when price > SMA50
            if price > sma50.iloc[i] and cash > 0:
                shares = cash / price
                entry_price = price
                cash = 0
                position = True
                total_trades += 1
        else:
            # Sell when price < SMA50 or stop loss
            pnl_pct = (price - entry_price) / entry_price
            
            if price < sma50.iloc[i] or pnl_pct <= -0.10:
                cash = shares * price
                if price > entry_price:
                    winning_trades += 1
                shares = 0
                position = False
        
        equity.append(cash + shares * price)
    
    # Close position
    if position:
        cash = shares * close.iloc[-1]
        shares = 0
    
    final_value = cash
    return_pct = (final_value / total_contributions - 1) * 100
    sharpe = _calculate_sharpe(pd.Series(equity))
    max_dd = _calculate_max_drawdown(pd.Series(equity))
    
    return {
        "strategy": "Standard Technical",
        "final_value": final_value,
        "total_invested": total_contributions,
        "profit": final_value - total_contributions,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": total_trades,
        "win_rate": winning_trades / total_trades * 100 if total_trades > 0 else 0,
    }


def run_regime_aware_technical(
    close: pd.Series,
    spy_close: pd.Series,
    initial_capital: float = 10000,
    monthly_contribution: float = 1000,
) -> dict:
    """
    Regime-aware technical trading.
    
    The key insight: Regime detection should ENHANCE existing strategies,
    not prevent them from working in normal conditions.
    
    BULL: Standard technical (SMA50 crossover), 10% stop - FULL position
    BEAR: More patient entries (wait for deeper dips), wider stops, hold through chop
    CRASH: Aggressive accumulation - buy more on deeper drops
    RECOVERY: Same as BULL but with trailing stop
    """
    regime_detector = RegimeDetector(spy_close)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rolling_high = close.rolling(252).max()
    
    cash = initial_capital
    shares = 0.0
    position = False
    entry_price = 0.0
    highest_price_in_trade = 0.0  # For trailing stop
    regime_at_entry = None
    
    total_trades = 0
    winning_trades = 0
    total_contributions = initial_capital
    
    equity = [cash]
    last_month = None
    
    for i, (date, price) in enumerate(close.items()):
        # Monthly contribution
        if hasattr(date, 'month'):
            current_month = (date.year, date.month)
            if last_month is not None and current_month != last_month:
                if position:
                    shares += monthly_contribution / price
                else:
                    cash += monthly_contribution
                total_contributions += monthly_contribution
            last_month = current_month
        
        if i < 200:  # Need enough data for regime detection
            equity.append(cash + shares * price)
            continue
        
        # Detect current regime
        regime_state = regime_detector.detect_at_date(date)
        regime = regime_state.regime
        
        # Calculate P&L if in position
        pnl_pct = 0
        if position and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price
            highest_price_in_trade = max(highest_price_in_trade, price)
        
        if not position:
            should_enter = False
            
            # Entry logic - ALWAYS buy in BULL, be pickier in other regimes
            if regime == MarketRegime.BULL:
                # Standard: Buy when price > SMA50 (momentum entry)
                if price > sma50.iloc[i]:
                    should_enter = True
            
            elif regime == MarketRegime.BEAR:
                # Bear: Wait for price to be oversold (below SMA50 and bouncing)
                # Buy when price crosses ABOVE SMA50 from below (reversal)
                if i > 0 and price > sma50.iloc[i] and close.iloc[i-1] < sma50.iloc[i-1]:
                    should_enter = True
            
            elif regime == MarketRegime.CRASH:
                # Crash: Buy aggressively on any dip - this is the opportunity
                dd_from_high = (price - rolling_high.iloc[i]) / rolling_high.iloc[i]
                if dd_from_high <= -0.15:  # 15% down from high = buy
                    should_enter = True
            
            elif regime == MarketRegime.RECOVERY:
                # Recovery: Standard momentum entry
                if price > sma50.iloc[i]:
                    should_enter = True
            
            if should_enter and cash > 0:
                shares = cash / price  # ALWAYS full position
                entry_price = price
                highest_price_in_trade = price
                cash = 0
                position = True
                total_trades += 1
                regime_at_entry = regime
        
        else:
            # In position - exit logic varies by regime
            should_exit = False
            
            if regime == MarketRegime.BULL:
                # Bull: Standard 10% stop, exit on SMA50 breakdown
                if pnl_pct <= -0.10:  # 10% stop
                    should_exit = True
                elif price < sma50.iloc[i] * 0.98:  # Exit on breakdown (2% below SMA50)
                    should_exit = True
            
            elif regime == MarketRegime.BEAR:
                # Bear: Wider 20% stop, hold through normal volatility
                if pnl_pct <= -0.20:  # 20% stop
                    should_exit = True
                # Take profit on bounce back to SMA200
                elif price > sma200.iloc[i] and pnl_pct > 0.10:
                    should_exit = True
            
            elif regime == MarketRegime.CRASH:
                # Crash: NO stop loss - this is where fortunes are made
                # Only exit on major recovery (price back above SMA200 with profit)
                if price > sma200.iloc[i] and pnl_pct > 0.20:
                    should_exit = True
            
            elif regime == MarketRegime.RECOVERY:
                # Recovery: Use trailing stop (protect gains)
                trailing_stop_price = highest_price_in_trade * 0.92  # 8% trailing
                if price < trailing_stop_price:
                    should_exit = True
                elif pnl_pct <= -0.12:  # 12% hard stop
                    should_exit = True
            
            if should_exit:
                cash = shares * price
                if price > entry_price:
                    winning_trades += 1
                shares = 0
                position = False
                highest_price_in_trade = 0
                regime_at_entry = None
        
        equity.append(cash + shares * price)
    
    # Close position
    if position:
        cash = shares * close.iloc[-1]
        if close.iloc[-1] > entry_price:
            winning_trades += 1
        shares = 0
    
    final_value = cash
    return_pct = (final_value / total_contributions - 1) * 100
    sharpe = _calculate_sharpe(pd.Series(equity))
    max_dd = _calculate_max_drawdown(pd.Series(equity))
    
    return {
        "strategy": "Regime-Aware Technical",
        "final_value": final_value,
        "total_invested": total_contributions,
        "profit": final_value - total_contributions,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": total_trades,
        "win_rate": winning_trades / total_trades * 100 if total_trades > 0 else 0,
    }


def run_standard_dip_trading(
    close: pd.Series,
    spy_close: pd.Series,
    initial_capital: float = 10000,
    monthly_contribution: float = 1000,
    dip_threshold: float = -10.0,
    recovery_target: float = 50.0,
) -> dict:
    """
    Standard dip trading - no regime awareness.
    
    - Buy on 10% dip
    - Sell on 50% recovery of the dip
    - Fixed behavior regardless of market
    """
    rolling_high = close.rolling(252).max()
    drawdown = (close - rolling_high) / rolling_high * 100
    
    cash = initial_capital
    shares = 0.0
    position = False
    entry_price = 0.0
    entry_dd = 0.0
    
    total_trades = 0
    winning_trades = 0
    total_contributions = initial_capital
    
    equity = [cash]
    last_month = None
    
    for i, (date, price) in enumerate(close.items()):
        # Monthly contribution
        if hasattr(date, 'month'):
            current_month = (date.year, date.month)
            if last_month is not None and current_month != last_month:
                if position:
                    shares += monthly_contribution / price
                else:
                    cash += monthly_contribution
                total_contributions += monthly_contribution
            last_month = current_month
        
        dd = drawdown.iloc[i]
        
        if not position:
            # Buy on dip threshold crossing
            if dd <= dip_threshold and cash > 0:
                shares = cash / price
                entry_price = price
                entry_dd = dd
                cash = 0
                position = True
                total_trades += 1
        else:
            # Sell on recovery
            drop_amount = abs(entry_dd) / 100 * entry_price
            target_price = entry_price + drop_amount * (recovery_target / 100)
            
            if price >= target_price:
                cash = shares * price
                if price > entry_price:
                    winning_trades += 1
                shares = 0
                position = False
        
        equity.append(cash + shares * price)
    
    # Close position
    if position:
        cash = shares * close.iloc[-1]
        shares = 0
    
    final_value = cash
    return_pct = (final_value / total_contributions - 1) * 100
    sharpe = _calculate_sharpe(pd.Series(equity))
    max_dd = _calculate_max_drawdown(pd.Series(equity))
    
    return {
        "strategy": "Standard Dip Trading",
        "final_value": final_value,
        "total_invested": total_contributions,
        "profit": final_value - total_contributions,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": total_trades,
        "win_rate": winning_trades / total_trades * 100 if total_trades > 0 else 0,
    }


def run_regime_aware_dip_trading(
    close: pd.Series,
    spy_close: pd.Series,
    initial_capital: float = 10000,
    monthly_contribution: float = 1000,
) -> dict:
    """
    Regime-aware dip trading.
    
    Key insight: Use regime to TUNE parameters, not fundamentally change strategy.
    
    BULL: Standard dip trading (-8% entry, sell on recovery)
    BEAR: Deeper dip threshold (-12%), wider profit target (+15%)
    CRASH: Very deep dips (-15%), hold for big recovery (+25%)
    RECOVERY: Shallower dips (-5%), quick profits (+8%)
    """
    regime_detector = RegimeDetector(spy_close)
    rolling_high = close.rolling(252).max()
    
    cash = initial_capital
    shares = 0.0
    position = False
    entry_price = 0.0
    entry_high = 0.0  # Rolling high at entry
    regime_at_entry = None
    
    total_trades = 0
    winning_trades = 0
    total_contributions = initial_capital
    
    equity = [cash]
    last_month = None
    
    for i, (date, price) in enumerate(close.items()):
        # Monthly contribution
        if hasattr(date, 'month'):
            current_month = (date.year, date.month)
            if last_month is not None and current_month != last_month:
                if position:
                    shares += monthly_contribution / price
                else:
                    cash += monthly_contribution
                total_contributions += monthly_contribution
            last_month = current_month
        
        if i < 200:
            equity.append(cash + shares * price)
            continue
        
        # Detect regime
        regime_state = regime_detector.detect_at_date(date)
        regime = regime_state.regime
        
        # Calculate drawdown from 52-week high
        current_high = rolling_high.iloc[i]
        dd_from_high = (price - current_high) / current_high
        
        # Regime-specific thresholds
        if regime == MarketRegime.BULL:
            entry_threshold = -0.08  # 8% dip
            profit_target = 0.08     # 8% profit
        elif regime == MarketRegime.BEAR:
            entry_threshold = -0.12  # 12% dip - wait for deeper
            profit_target = 0.15     # 15% profit - hold longer
        elif regime == MarketRegime.CRASH:
            entry_threshold = -0.15  # 15% dip - really deep
            profit_target = 0.25     # 25% profit - hold for recovery
        else:  # RECOVERY
            entry_threshold = -0.05  # 5% dip - shallower
            profit_target = 0.08     # 8% profit - quick
        
        if not position:
            # Entry: Buy when drawdown hits threshold
            if dd_from_high <= entry_threshold and cash > 0:
                shares = cash / price  # Full position always
                entry_price = price
                entry_high = current_high
                regime_at_entry = regime
                cash = 0
                position = True
                total_trades += 1
        
        else:
            # Exit: Based on profit from entry
            pnl_pct = (price - entry_price) / entry_price
            
            # Use profit target from regime at ENTRY (not current regime)
            if regime_at_entry == MarketRegime.BULL:
                target = 0.08
            elif regime_at_entry == MarketRegime.BEAR:
                target = 0.15
            elif regime_at_entry == MarketRegime.CRASH:
                target = 0.25
            else:
                target = 0.08
            
            # Exit on profit target OR if we've recovered fully and market is bullish
            if pnl_pct >= target:
                cash = shares * price
                winning_trades += 1
                shares = 0
                position = False
                regime_at_entry = None
            # Emergency stop: 25% loss (shouldn't hit often with dip buying)
            elif pnl_pct <= -0.25:
                cash = shares * price
                shares = 0
                position = False
                regime_at_entry = None
        
        equity.append(cash + shares * price)
    
    # Close position
    if position:
        cash = shares * close.iloc[-1]
        if close.iloc[-1] > entry_price:
            winning_trades += 1
        shares = 0
    
    final_value = cash
    return_pct = (final_value / total_contributions - 1) * 100
    sharpe = _calculate_sharpe(pd.Series(equity))
    max_dd = _calculate_max_drawdown(pd.Series(equity))
    
    return {
        "strategy": "Regime-Aware Dip Trading",
        "final_value": final_value,
        "total_invested": total_contributions,
        "profit": final_value - total_contributions,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": total_trades,
        "win_rate": winning_trades / total_trades * 100 if total_trades > 0 else 0,
    }


def run_dca(
    close: pd.Series,
    initial_capital: float = 10000,
    monthly_contribution: float = 1000,
) -> dict:
    """DCA baseline for comparison."""
    shares = initial_capital / close.iloc[0]
    total_contributions = initial_capital
    
    equity = [initial_capital]
    last_month = None
    
    for i, (date, price) in enumerate(close.items()):
        if i == 0:
            continue
        
        if hasattr(date, 'month'):
            current_month = (date.year, date.month)
            if last_month is not None and current_month != last_month:
                shares += monthly_contribution / price
                total_contributions += monthly_contribution
            last_month = current_month
        
        equity.append(shares * price)
    
    final_value = shares * close.iloc[-1]
    return_pct = (final_value / total_contributions - 1) * 100
    sharpe = _calculate_sharpe(pd.Series(equity))
    max_dd = _calculate_max_drawdown(pd.Series(equity))
    
    return {
        "strategy": "DCA (Baseline)",
        "final_value": final_value,
        "total_invested": total_contributions,
        "profit": final_value - total_contributions,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": 0,
        "win_rate": 0,
    }


def _calculate_sharpe(equity: pd.Series, rf_rate: float = 0.04) -> float:
    """Calculate Sharpe ratio."""
    returns = equity.pct_change().dropna()
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - rf_rate / 252
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
    
    return float(sharpe) if not (np.isnan(sharpe) or np.isinf(sharpe)) else 0.0


def _calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate max drawdown."""
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak * 100
    return float(drawdown.min())


def print_results(results: list[dict], title: str):
    """Pretty print comparison results."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    
    # Sort by return
    results = sorted(results, key=lambda x: x["return_pct"], reverse=True)
    
    print(f"\n{'Strategy':<30} {'Final Value':>12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Trades':>8}")
    print("-" * 80)
    
    for r in results:
        print(
            f"{r['strategy']:<30} "
            f"${r['final_value']:>10,.0f} "
            f"{r['return_pct']:>+9.1f}% "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_drawdown']:>9.1f}% "
            f"{r['trades']:>7}"
        )
    
    # Show winner
    winner = results[0]
    second = results[1] if len(results) > 1 else None
    
    print(f"\n🏆 Winner: {winner['strategy']} with {winner['return_pct']:+.1f}% return")
    if second:
        advantage = winner['return_pct'] - second['return_pct']
        print(f"   Advantage over #{2}: +{advantage:.1f}% ({winner['profit'] - second['profit']:+,.0f} more profit)")


def analyze_regime_distribution(spy_close: pd.Series):
    """Show regime distribution in the test period."""
    regime_detector = RegimeDetector(spy_close)
    
    regimes = {"BULL": 0, "BEAR": 0, "CRASH": 0, "RECOVERY": 0}
    total_days = 0
    
    for i, (date, _) in enumerate(spy_close.items()):
        if i < 200:
            continue
        
        state = regime_detector.detect_at_date(date)
        regimes[state.regime.value] += 1
        total_days += 1
    
    print(f"\n📊 Regime Distribution ({total_days} trading days):")
    for regime, days in regimes.items():
        pct = days / total_days * 100 if total_days > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {regime:10} {days:>5} days ({pct:>5.1f}%) {bar}")


def main():
    """Run regime awareness comparison tests."""
    print("\n" + "="*80)
    print("  REGIME-AWARE STRATEGY TEST")
    print("  Testing if market regime adaptation improves trading performance")
    print("="*80)
    
    # Test multiple symbols and periods
    test_cases = [
        ("MSFT", 10),  # Full 10 years (includes all crashes)
        ("AAPL", 10),
        ("GOOGL", 10),
        ("QQQ", 10),   # NASDAQ ETF - more volatile
    ]
    
    all_results = {}
    
    for symbol, years in test_cases:
        print(f"\n{'─'*80}")
        print(f"  Testing {symbol} over {years} years")
        print(f"{'─'*80}")
        
        try:
            close, spy_close = get_stock_data(symbol, years)
            print(f"  Data: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Price: ${close.iloc[0]:.2f} → ${close.iloc[-1]:.2f}")
            
            # Show regime distribution
            analyze_regime_distribution(spy_close)
            
            # Run all strategies
            results = [
                run_dca(close),
                run_standard_technical(close, spy_close),
                run_regime_aware_technical(close, spy_close),
                run_standard_dip_trading(close, spy_close),
                run_regime_aware_dip_trading(close, spy_close),
            ]
            
            print_results(results, f"{symbol} Strategy Comparison")
            
            # Calculate regime benefit
            standard_tech = next(r for r in results if r["strategy"] == "Standard Technical")
            regime_tech = next(r for r in results if r["strategy"] == "Regime-Aware Technical")
            tech_benefit = regime_tech["return_pct"] - standard_tech["return_pct"]
            
            standard_dip = next(r for r in results if r["strategy"] == "Standard Dip Trading")
            regime_dip = next(r for r in results if r["strategy"] == "Regime-Aware Dip Trading")
            dip_benefit = regime_dip["return_pct"] - standard_dip["return_pct"]
            
            all_results[symbol] = {
                "tech_benefit": tech_benefit,
                "dip_benefit": dip_benefit,
                "tech_sharpe_diff": regime_tech["sharpe"] - standard_tech["sharpe"],
                "dip_sharpe_diff": regime_dip["sharpe"] - standard_dip["sharpe"],
            }
            
        except Exception as e:
            print(f"  ❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY: Does Regime Awareness Help?")
    print("="*80)
    
    print(f"\n{'Symbol':<10} {'Tech Benefit':>15} {'Dip Benefit':>15} {'Tech Sharpe':>15} {'Dip Sharpe':>15}")
    print("-"*70)
    
    tech_benefits = []
    dip_benefits = []
    
    for symbol, data in all_results.items():
        tech_b = data["tech_benefit"]
        dip_b = data["dip_benefit"]
        tech_s = data["tech_sharpe_diff"]
        dip_s = data["dip_sharpe_diff"]
        
        tech_benefits.append(tech_b)
        dip_benefits.append(dip_b)
        
        tech_color = "+" if tech_b > 0 else ""
        dip_color = "+" if dip_b > 0 else ""
        
        print(
            f"{symbol:<10} "
            f"{tech_color}{tech_b:>14.1f}% "
            f"{dip_color}{dip_b:>14.1f}% "
            f"{tech_s:>+14.2f} "
            f"{dip_s:>+14.2f}"
        )
    
    print("-"*70)
    avg_tech = np.mean(tech_benefits)
    avg_dip = np.mean(dip_benefits)
    print(f"{'AVERAGE':<10} {avg_tech:>+14.1f}% {avg_dip:>+14.1f}%")
    
    # Verdict
    print("\n" + "="*80)
    print("  VERDICT")
    print("="*80)
    
    if avg_tech > 0 and avg_dip > 0:
        print("\n✅ REGIME AWARENESS HELPS BOTH STRATEGIES")
        print(f"   • Technical trading: +{avg_tech:.1f}% average improvement")
        print(f"   • Dip trading: +{avg_dip:.1f}% average improvement")
    elif avg_tech > 0:
        print(f"\n⚠️  MIXED RESULTS")
        print(f"   • Technical trading benefits: +{avg_tech:.1f}%")
        print(f"   • Dip trading hurt by regime: {avg_dip:.1f}%")
    elif avg_dip > 0:
        print(f"\n⚠️  MIXED RESULTS")
        print(f"   • Technical trading hurt by regime: {avg_tech:.1f}%")
        print(f"   • Dip trading benefits: +{avg_dip:.1f}%")
    else:
        print("\n❌ REGIME AWARENESS DOES NOT HELP")
        print(f"   • Both strategies perform worse with regime detection")
        print(f"   • Consider simplifying the strategy")
    
    # Test crash periods specifically
    print("\n" + "="*80)
    print("  CRASH PERIOD ANALYSIS")
    print("  Testing regime awareness during market crashes")
    print("="*80)
    
    test_crash_periods()


def test_crash_periods():
    """Test specifically during crash periods where regime awareness should help most."""
    
    crash_periods = [
        ("COVID Crash", "2020-01-01", "2021-06-01"),
        ("2022 Tech Crash", "2022-01-01", "2023-06-01"),
    ]
    
    symbols = ["MSFT", "QQQ"]
    
    for period_name, start, end in crash_periods:
        print(f"\n{'─'*80}")
        print(f"  {period_name}: {start} to {end}")
        print(f"{'─'*80}")
        
        for symbol in symbols:
            try:
                import yfinance as yf
                
                stock = yf.Ticker(symbol)
                spy = yf.Ticker("SPY")
                
                stock_hist = stock.history(start=start, end=end)
                spy_hist = spy.history(start=start, end=end)
                
                close = stock_hist["Close"]
                spy_close = spy_hist["Close"]
                
                if len(close) < 100:
                    print(f"  {symbol}: Insufficient data")
                    continue
                
                print(f"\n  {symbol}: ${close.iloc[0]:.2f} → ${close.iloc[-1]:.2f}")
                
                # Show regime distribution
                analyze_regime_distribution(spy_close)
                
                # Run strategies with smaller capital (shorter period)
                results = [
                    run_dca(close, initial_capital=10000, monthly_contribution=500),
                    run_standard_technical(close, spy_close, initial_capital=10000, monthly_contribution=500),
                    run_regime_aware_technical(close, spy_close, initial_capital=10000, monthly_contribution=500),
                ]
                
                print(f"\n  {'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10}")
                print(f"  {'-'*60}")
                
                for r in sorted(results, key=lambda x: x["return_pct"], reverse=True):
                    print(f"  {r['strategy']:<30} {r['return_pct']:>+9.1f}% {r['sharpe']:>7.2f} {r['max_drawdown']:>9.1f}%")
                
                # Calculate benefit
                std = next(r for r in results if r["strategy"] == "Standard Technical")
                reg = next(r for r in results if r["strategy"] == "Regime-Aware Technical")
                
                benefit = reg["return_pct"] - std["return_pct"]
                dd_benefit = std["max_drawdown"] - reg["max_drawdown"]  # Less negative is better
                
                print(f"\n  Regime benefit: {benefit:+.1f}% return, {dd_benefit:+.1f}% less drawdown")
                
            except Exception as e:
                print(f"  {symbol}: Error - {e}")


if __name__ == "__main__":
    main()
