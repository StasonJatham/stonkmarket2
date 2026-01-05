#!/usr/bin/env python3
"""
Analyze the effect of different holding periods on dip trading strategy performance.

Uses quantstats for proper risk metrics and compares against buy-and-hold to detect
if longer holding periods just converge to B&H (which would defeat the purpose).

Tests holding periods from 30 to 200 days with focus on:
- Recovery rate (return per day held) - key metric for capital efficiency
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Strategy alpha vs buy-and-hold
- Statistical significance of improvements

Usage:
    python scripts/analyze_holding_periods.py
"""

import sys
from pathlib import Path

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yfinance as yf
from typing import NamedTuple
from scipy import stats
import quantstats as qs


# =============================================================================
# Configuration
# =============================================================================

# Symbols to test (mix of sectors)
TEST_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech
    "JPM", "BAC", "GS",  # Finance
    "JNJ", "PFE",  # Healthcare
    "XOM", "CVX",  # Energy
    "WMT", "HD",  # Retail
    "DIS", "NFLX",  # Entertainment
]

# Holding periods to test (days)
HOLDING_PERIODS = [30, 60, 90, 120, 150, 180, 200]

# Dip thresholds to test (% drop from recent high)
DIP_THRESHOLDS = [-15, -20]  # Focus on meaningful dips

# Lookback for detecting "recent high" (days)
HIGH_LOOKBACK = 52  # ~2 months of trading days

# Years of historical data
YEARS_OF_DATA = 10

# Risk-free rate for Sharpe calculation
RISK_FREE_RATE = 0.04  # 4% annual


# =============================================================================
# Data Classes
# =============================================================================

class Trade(NamedTuple):
    """A single trade record."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    exit_reason: str  # "target" or "max_hold"
    # For B&H comparison
    bh_return_pct: float  # What B&H would have returned over same period


@dataclass
class HoldingPeriodResult:
    """Results for a specific holding period with comprehensive risk metrics."""
    holding_period: int
    total_trades: int
    winning_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    avg_holding_days: float
    
    # Risk metrics from quantstats
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR (expected shortfall)
    
    # Capital efficiency metrics
    return_per_day: float  # Avg return / avg holding days
    capital_efficiency: float  # How much return per day of capital lockup
    
    # Recovery analysis
    trades_hit_max_hold: int
    pct_hit_max_hold: float
    avg_days_to_target: float  # For trades that hit target, how many days?
    
    # Buy-and-hold comparison
    avg_bh_return: float  # What B&H returned over same periods
    alpha_vs_bh: float  # Strategy return - B&H return
    pct_beating_bh: float  # % of trades that beat B&H
    
    # Statistical tests
    t_stat_vs_bh: float  # t-statistic for strategy vs B&H
    p_value_vs_bh: float  # p-value for significance


# =============================================================================
# Data Loading
# =============================================================================

def load_price_data(symbols: list[str], years: int = 10) -> dict[str, pd.DataFrame]:
    """Load historical price data for symbols."""
    print(f"\n📊 Loading {years} years of price data for {len(symbols)} symbols...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    price_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if len(df) > 252:  # At least 1 year of data
                price_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} days")
            else:
                print(f"  ✗ {symbol}: insufficient data ({len(df)} days)")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")
    
    return price_data


# =============================================================================
# Dip Detection
# =============================================================================

def detect_dips(
    prices: pd.Series, 
    threshold: float, 
    lookback: int = 52
) -> pd.DataFrame:
    """
    Detect dip entry points where price drops threshold% from recent high.
    """
    # Calculate rolling high
    rolling_high = prices.rolling(window=lookback, min_periods=1).max()
    
    # Calculate drawdown from rolling high
    drawdown = (prices - rolling_high) / rolling_high * 100
    
    # Find days where drawdown crosses threshold
    crossed_threshold = (drawdown <= threshold) & (drawdown.shift(1) > threshold)
    
    dip_dates = prices.index[crossed_threshold]
    
    return pd.DataFrame({
        "date": dip_dates,
        "entry_price": prices.loc[dip_dates].values,
        "drawdown": drawdown.loc[dip_dates].values,
    })


# =============================================================================
# Trade Simulation
# =============================================================================

def simulate_trades(
    symbol: str,
    prices: pd.Series,
    dip_threshold: float,
    max_hold: int,
    recovery_target: float = 0.8,  # Exit when recovered 80% of drop
) -> list[Trade]:
    """
    Simulate trades for a symbol with given parameters.
    Also calculates what buy-and-hold would have returned.
    """
    dips = detect_dips(prices, dip_threshold)
    trades = []
    
    for _, dip in dips.iterrows():
        entry_date = dip["date"]
        entry_price = dip["entry_price"]
        
        # Get prices from entry forward
        future_prices = prices.loc[entry_date:]
        if len(future_prices) < 5:
            continue  # Not enough data
        
        # Calculate recovery target price (entry + 80% of the way back to high)
        drop_amount = abs(dip["drawdown"]) / 100 * entry_price
        target_price = entry_price + (drop_amount * recovery_target)
        
        # Find exit: either hit target or max hold
        exit_date = None
        exit_price = None
        exit_reason = None
        
        for i, (date, price) in enumerate(future_prices.items()):
            if i == 0:
                continue  # Skip entry day
            
            days_held = i
            
            # Check if hit target
            if price >= target_price:
                exit_date = date
                exit_price = price
                exit_reason = "target"
                break
            
            # Check if hit max hold
            if days_held >= max_hold:
                exit_date = date
                exit_price = price
                exit_reason = "max_hold"
                break
        
        if exit_date is None:
            # Ran out of data - exit at last available price
            exit_date = future_prices.index[-1]
            exit_price = future_prices.iloc[-1]
            exit_reason = "end_of_data"
        
        return_pct = (exit_price - entry_price) / entry_price * 100
        holding_days = (exit_date - entry_date).days
        
        # Calculate what B&H would have returned over the SAME period
        bh_return_pct = return_pct  # Same exit price, so same return for this trade
        
        trades.append(Trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=return_pct,
            holding_days=max(holding_days, 1),  # Avoid division by zero
            exit_reason=exit_reason,
            bh_return_pct=bh_return_pct,
        ))
    
    return trades


def calculate_quantstats_metrics(returns: list[float]) -> dict:
    """Calculate comprehensive risk metrics using quantstats."""
    if len(returns) < 5:
        return {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "max_dd": 0.0, "volatility": 0.0, "var_95": 0.0, "cvar_95": 0.0,
        }
    
    # Convert to pandas Series (daily returns format for quantstats)
    returns_series = pd.Series(returns) / 100  # Convert from % to decimal
    
    try:
        # Calculate metrics
        sharpe = qs.stats.sharpe(returns_series, rf=RISK_FREE_RATE/252, periods=252)
        sortino = qs.stats.sortino(returns_series, rf=RISK_FREE_RATE/252, periods=252)
        
        # For Calmar, we need cumulative returns
        cum_returns = (1 + returns_series).cumprod()
        max_dd = qs.stats.max_drawdown(cum_returns) * 100  # Convert to %
        
        # Calmar = Annual return / Max drawdown
        total_return = (cum_returns.iloc[-1] - 1) * 100
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0
        
        volatility = qs.stats.volatility(returns_series, periods=252) * 100
        
        # VaR and CVaR (expected shortfall)
        var_95 = qs.stats.var(returns_series) * 100
        cvar_95 = qs.stats.cvar(returns_series) * 100
        
        return {
            "sharpe": float(sharpe) if not np.isnan(sharpe) else 0.0,
            "sortino": float(sortino) if not np.isnan(sortino) else 0.0,
            "calmar": float(calmar) if not np.isnan(calmar) else 0.0,
            "max_dd": float(max_dd) if not np.isnan(max_dd) else 0.0,
            "volatility": float(volatility) if not np.isnan(volatility) else 0.0,
            "var_95": float(var_95) if not np.isnan(var_95) else 0.0,
            "cvar_95": float(cvar_95) if not np.isnan(cvar_95) else 0.0,
        }
    except Exception as e:
        print(f"  Warning: quantstats calculation error: {e}")
        return {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "max_dd": 0.0, "volatility": 0.0, "var_95": 0.0, "cvar_95": 0.0,
        }


def compare_vs_buyhold(
    price_data: dict[str, pd.DataFrame],
    trades: list[Trade],
    holding_period: int,
) -> dict:
    """
    Compare strategy returns vs what B&H would have done.
    
    For a fair comparison: If we entered at dip and held for N days,
    what would a random B&H entry have returned over N days?
    """
    if not trades:
        return {"avg_bh": 0.0, "alpha": 0.0, "pct_beating": 0.0, "t_stat": 0.0, "p_value": 1.0}
    
    strategy_returns = [t.return_pct for t in trades]
    
    # Calculate average B&H return for same holding period across all data
    bh_returns = []
    for symbol, df in price_data.items():
        prices = df["Close"]
        # Sample random entry points and calculate N-day returns
        for i in range(0, len(prices) - holding_period, holding_period):
            entry = prices.iloc[i]
            exit_price = prices.iloc[i + holding_period]
            bh_returns.append((exit_price - entry) / entry * 100)
    
    if not bh_returns:
        return {"avg_bh": 0.0, "alpha": 0.0, "pct_beating": 0.0, "t_stat": 0.0, "p_value": 1.0}
    
    avg_bh = np.mean(bh_returns)
    avg_strategy = np.mean(strategy_returns)
    alpha = avg_strategy - avg_bh
    
    # What % of our trades beat B&H?
    pct_beating = sum(1 for r in strategy_returns if r > avg_bh) / len(strategy_returns) * 100
    
    # Statistical significance: paired t-test
    # Compare strategy returns to B&H average
    try:
        t_stat, p_value = stats.ttest_1samp(strategy_returns, avg_bh)
    except Exception:
        t_stat, p_value = 0.0, 1.0
    
    return {
        "avg_bh": avg_bh,
        "alpha": alpha,
        "pct_beating": pct_beating,
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
    }


# =============================================================================
# Analysis
# =============================================================================

def analyze_holding_period(
    price_data: dict[str, pd.DataFrame],
    holding_period: int,
    dip_threshold: float,
) -> HoldingPeriodResult:
    """Analyze performance for a specific holding period with full risk metrics."""
    all_trades = []
    
    for symbol, df in price_data.items():
        prices = df["Close"]
        trades = simulate_trades(symbol, prices, dip_threshold, holding_period)
        all_trades.extend(trades)
    
    if not all_trades:
        return HoldingPeriodResult(
            holding_period=holding_period,
            total_trades=0, winning_trades=0, win_rate=0.0,
            avg_return=0.0, total_return=0.0, avg_holding_days=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, volatility=0.0, var_95=0.0, cvar_95=0.0,
            return_per_day=0.0, capital_efficiency=0.0,
            trades_hit_max_hold=0, pct_hit_max_hold=0.0, avg_days_to_target=0.0,
            avg_bh_return=0.0, alpha_vs_bh=0.0, pct_beating_bh=0.0,
            t_stat_vs_bh=0.0, p_value_vs_bh=1.0,
        )
    
    returns = [t.return_pct for t in all_trades]
    holding_days = [t.holding_days for t in all_trades]
    winning = [t for t in all_trades if t.return_pct > 0]
    hit_max_hold = [t for t in all_trades if t.exit_reason == "max_hold"]
    hit_target = [t for t in all_trades if t.exit_reason == "target"]
    
    # Calculate quantstats metrics
    qs_metrics = calculate_quantstats_metrics(returns)
    
    # B&H comparison
    bh_comparison = compare_vs_buyhold(price_data, all_trades, holding_period)
    
    # Capital efficiency: return per day of capital lockup
    avg_return = np.mean(returns)
    avg_hold = np.mean(holding_days)
    return_per_day = avg_return / avg_hold if avg_hold > 0 else 0.0
    
    # Average days to hit target (for successful trades)
    avg_days_target = np.mean([t.holding_days for t in hit_target]) if hit_target else 0.0
    
    return HoldingPeriodResult(
        holding_period=holding_period,
        total_trades=len(all_trades),
        winning_trades=len(winning),
        win_rate=len(winning) / len(all_trades) * 100,
        avg_return=avg_return,
        total_return=np.sum(returns),
        avg_holding_days=avg_hold,
        sharpe_ratio=qs_metrics["sharpe"],
        sortino_ratio=qs_metrics["sortino"],
        calmar_ratio=qs_metrics["calmar"],
        max_drawdown=qs_metrics["max_dd"],
        volatility=qs_metrics["volatility"],
        var_95=qs_metrics["var_95"],
        cvar_95=qs_metrics["cvar_95"],
        return_per_day=return_per_day,
        capital_efficiency=return_per_day,  # Same metric, different name
        trades_hit_max_hold=len(hit_max_hold),
        pct_hit_max_hold=len(hit_max_hold) / len(all_trades) * 100,
        avg_days_to_target=avg_days_target,
        avg_bh_return=bh_comparison["avg_bh"],
        alpha_vs_bh=bh_comparison["alpha"],
        pct_beating_bh=bh_comparison["pct_beating"],
        t_stat_vs_bh=bh_comparison["t_stat"],
        p_value_vs_bh=bh_comparison["p_value"],
    )


def print_results_table(results: list[HoldingPeriodResult], dip_threshold: float):
    """Print formatted results table with risk metrics."""
    print(f"\n{'='*120}")
    print(f"RESULTS FOR {abs(dip_threshold)}% DIP THRESHOLD")
    print(f"{'='*120}")
    
    # Main metrics table
    print(f"\n{'Hold':>6} | {'Trades':>6} | {'Win%':>5} | {'AvgRet':>7} | {'Ret/Day':>8} | "
          f"{'Sharpe':>7} | {'Sortino':>8} | {'MaxDD':>7} | {'VaR95':>7} | {'HitMax%':>7}")
    print("-" * 120)
    
    for r in results:
        print(f"{r.holding_period:>6} | {r.total_trades:>6} | {r.win_rate:>4.1f}% | "
              f"{r.avg_return:>6.2f}% | {r.return_per_day:>7.4f}% | "
              f"{r.sharpe_ratio:>7.3f} | {r.sortino_ratio:>8.3f} | "
              f"{r.max_drawdown:>6.1f}% | {r.var_95:>6.2f}% | {r.pct_hit_max_hold:>6.1f}%")
    
    # B&H comparison table
    print(f"\n--- Buy & Hold Comparison ---")
    print(f"{'Hold':>6} | {'Strategy':>9} | {'B&H Avg':>9} | {'Alpha':>8} | {'Beat B&H':>8} | "
          f"{'t-stat':>7} | {'p-value':>8} | {'Significant':>11}")
    print("-" * 100)
    
    for r in results:
        significant = "YES ✓" if r.p_value_vs_bh < 0.05 else "NO"
        print(f"{r.holding_period:>6} | {r.avg_return:>8.2f}% | {r.avg_bh_return:>8.2f}% | "
              f"{r.alpha_vs_bh:>7.2f}% | {r.pct_beating_bh:>7.1f}% | "
              f"{r.t_stat_vs_bh:>7.2f} | {r.p_value_vs_bh:>8.4f} | {significant:>11}")


def analyze_recovery_efficiency(results: list[HoldingPeriodResult]):
    """Analyze recovery rate efficiency across holding periods."""
    print("\n" + "=" * 120)
    print("RECOVERY EFFICIENCY ANALYSIS (Key Metric: Return per Day Held)")
    print("=" * 120)
    
    # Find best by return per day (capital efficiency)
    best_efficiency = max(results, key=lambda r: r.return_per_day)
    
    print(f"\n🎯 BEST CAPITAL EFFICIENCY: {best_efficiency.holding_period} days")
    print(f"   Return per day: {best_efficiency.return_per_day:.4f}%")
    print(f"   Avg return: {best_efficiency.avg_return:.2f}%")
    print(f"   Avg hold: {best_efficiency.avg_holding_days:.1f} days")
    
    # Show efficiency trend
    print(f"\n📊 Capital Efficiency by Holding Period:")
    print(f"   (Higher = more efficient use of capital)")
    print()
    
    for r in results:
        bar_len = int(r.return_per_day * 500)  # Scale for display
        bar = "█" * max(1, bar_len)
        print(f"   {r.holding_period:>3}d: {r.return_per_day:>7.4f}%/day  {bar}")
    
    # Analysis
    print("\n💡 INTERPRETATION:")
    
    # Check if efficiency decreases with longer holds
    efficiencies = [(r.holding_period, r.return_per_day) for r in results]
    periods = [e[0] for e in efficiencies]
    effs = [e[1] for e in efficiencies]
    
    # Calculate correlation
    corr, _ = stats.pearsonr(periods, effs)
    
    if corr < -0.3:
        print(f"   ⚠️  Efficiency DECREASES with longer holds (corr: {corr:.2f})")
        print(f"      This suggests longer holds are just 'waiting' - not adding value.")
        print(f"      Shorter holds may be more capital efficient.")
    elif corr > 0.3:
        print(f"   ✓ Efficiency INCREASES with longer holds (corr: {corr:.2f})")
        print(f"      Longer holds genuinely recover more value.")
    else:
        print(f"   → Efficiency is STABLE across holding periods (corr: {corr:.2f})")
        print(f"      Holding period doesn't significantly affect efficiency.")


def analyze_statistical_significance(results: list[HoldingPeriodResult]):
    """Test if improvements are statistically significant."""
    print("\n" + "=" * 120)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 120)
    
    r90 = next((r for r in results if r.holding_period == 90), None)
    r200 = next((r for r in results if r.holding_period == 200), None)
    
    if not r90 or not r200:
        print("  Cannot compare - missing 90 or 200 day results")
        return
    
    print(f"\n🔬 90 days vs 200 days Comparison:")
    print(f"   {'Metric':<25} {'90 days':>12} {'200 days':>12} {'Change':>12} {'Better?':>10}")
    print("   " + "-" * 75)
    
    comparisons = [
        ("Average Return", r90.avg_return, r200.avg_return, "%", True),  # Higher is better
        ("Return per Day", r90.return_per_day, r200.return_per_day, "%/day", True),
        ("Sharpe Ratio", r90.sharpe_ratio, r200.sharpe_ratio, "", True),
        ("Sortino Ratio", r90.sortino_ratio, r200.sortino_ratio, "", True),
        ("Win Rate", r90.win_rate, r200.win_rate, "%", True),
        ("Max Drawdown", r90.max_drawdown, r200.max_drawdown, "%", False),  # Lower is better
        ("VaR 95%", r90.var_95, r200.var_95, "%", False),
        ("Hit Max Hold %", r90.pct_hit_max_hold, r200.pct_hit_max_hold, "%", False),
        ("Alpha vs B&H", r90.alpha_vs_bh, r200.alpha_vs_bh, "%", True),
    ]
    
    wins_200 = 0
    for name, v90, v200, unit, higher_better in comparisons:
        change = v200 - v90
        if higher_better:
            better = "200d ✓" if v200 > v90 else "90d"
            if v200 > v90:
                wins_200 += 1
        else:
            better = "200d ✓" if v200 < v90 else "90d"
            if v200 < v90:
                wins_200 += 1
        
        print(f"   {name:<25} {v90:>11.3f}{unit} {v200:>11.3f}{unit} {change:>+11.3f} {better:>10}")
    
    print(f"\n📊 VERDICT:")
    if r200.p_value_vs_bh < 0.05 and r90.p_value_vs_bh < 0.05:
        print(f"   Both strategies significantly beat B&H (p < 0.05)")
    
    # Key question: Is the IMPROVEMENT in efficiency worth the EXTRA RISK?
    efficiency_change = r200.return_per_day - r90.return_per_day
    risk_change = r200.max_drawdown - r90.max_drawdown
    
    print(f"\n🎯 KEY TRADEOFF:")
    print(f"   Efficiency change (90d→200d): {efficiency_change:+.4f}%/day")
    print(f"   Max drawdown change: {risk_change:+.1f}%")
    
    if efficiency_change < 0 and risk_change > 0:
        print(f"   ⚠️  200 days is WORSE: Lower efficiency AND higher risk")
        print(f"   RECOMMENDATION: Keep 90 days")
    elif efficiency_change > 0 and risk_change <= 0:
        print(f"   ✓ 200 days is BETTER: Higher efficiency AND lower/same risk")
        print(f"   RECOMMENDATION: Extend to 200 days")
    elif efficiency_change < 0 and risk_change <= 0:
        print(f"   → 200 days trades efficiency for safety")
        print(f"   RECOMMENDATION: Depends on risk tolerance")
    else:
        print(f"   → 200 days trades safety for efficiency")
        print(f"   RECOMMENDATION: Depends on risk tolerance")


def main():
    """Run the full analysis."""
    print("\n" + "=" * 120)
    print("HOLDING PERIOD ANALYSIS WITH QUANTSTATS RISK METRICS")
    print(f"Testing periods: {HOLDING_PERIODS} days")
    print(f"Dip thresholds: {DIP_THRESHOLDS}%")
    print(f"Symbols: {len(TEST_SYMBOLS)}")
    print("=" * 120)
    
    # Load data
    price_data = load_price_data(TEST_SYMBOLS, YEARS_OF_DATA)
    
    if not price_data:
        print("❌ No price data loaded. Exiting.")
        return
    
    print(f"\n✓ Loaded data for {len(price_data)} symbols")
    
    # Run analysis for each dip threshold
    for dip_threshold in DIP_THRESHOLDS:
        results = []
        
        print(f"\n⏳ Analyzing {abs(dip_threshold)}% dip threshold...")
        
        for holding_period in HOLDING_PERIODS:
            result = analyze_holding_period(price_data, holding_period, dip_threshold)
            results.append(result)
        
        print_results_table(results, dip_threshold)
        analyze_recovery_efficiency(results)
        analyze_statistical_significance(results)
    
    print("\n" + "=" * 120)
    print("FINAL CONCLUSIONS")
    print("=" * 120)
    print("""
Key metrics to consider:

1. RETURN PER DAY (Capital Efficiency)
   - The most important metric for comparing holding periods
   - If this DECREASES with longer holds, you're just waiting, not adding value
   - Capital locked in a trade can't be used elsewhere

2. ALPHA VS BUY-AND-HOLD
   - If alpha shrinks with longer holds, strategy converges to B&H
   - We want to BEAT B&H, not match it

3. STATISTICAL SIGNIFICANCE
   - p-value < 0.05 means the difference is real, not luck
   - High t-stat = strong evidence

4. HIT MAX HOLD %
   - If still >15-20% at 200 days, those entries may never recover
   - Consider better entry signals, not longer holds
""")


if __name__ == "__main__":
    main()
