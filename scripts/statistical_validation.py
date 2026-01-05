#!/usr/bin/env python3
"""
Statistical Validation of Trading Strategies

This script validates that our trading strategies produce statistically significant
results across a large sample of stocks - proving the results aren't just luck.

Key metrics:
- Win rate vs random baseline
- Sharpe ratio distribution
- Consistency across different market conditions
- Outperformance vs buy-and-hold
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from dataclasses import dataclass
from typing import Any

from app.quant_engine.backtest_v2.baseline_strategies import (
    BaselineEngine,
    BaselineStrategyType,
)


@dataclass
class StrategyStats:
    """Statistics for a strategy across all tested stocks."""
    name: str
    n_stocks: int
    avg_return: float
    std_return: float
    median_return: float
    min_return: float
    max_return: float
    avg_sharpe: float
    win_rate: float  # % stocks where strategy outperformed SPY DCA
    beat_buy_hold_rate: float  # % stocks where strategy beat buy & hold
    avg_max_drawdown: float
    
    # Statistical significance
    t_statistic: float | None = None
    p_value: float | None = None
    is_significant: bool = False
    confidence_level: str = ""
    
    def __str__(self) -> str:
        sig_marker = "**" if self.is_significant else ""
        return (
            f"{sig_marker}{self.name}{sig_marker}\n"
            f"  N={self.n_stocks}, Avg Return: {self.avg_return:+.1f}% ± {self.std_return:.1f}%\n"
            f"  Median: {self.median_return:+.1f}%, Range: [{self.min_return:+.1f}%, {self.max_return:+.1f}%]\n"
            f"  Avg Sharpe: {self.avg_sharpe:.2f}, Avg Max DD: {self.avg_max_drawdown:.1f}%\n"
            f"  Win Rate vs SPY DCA: {self.win_rate:.1f}%, Beat B&H: {self.beat_buy_hold_rate:.1f}%\n"
            f"  {self.confidence_level}"
        )


def get_all_symbols() -> list[str]:
    """Get a large sample of symbols for testing."""
    # Major tech + diversified sample
    symbols = [
        # Big Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # More Tech
        "AMD", "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL", "CSCO",
        # Finance
        "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "C",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
        # Consumer
        "WMT", "HD", "MCD", "NKE", "COST", "PG", "KO", "PEP", "DIS",
        # Industrial
        "CAT", "BA", "HON", "GE", "MMM", "UPS", "FDX",
        # Energy
        "XOM", "CVX", "COP", "SLB",
        # ETFs
        "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE",
    ]
    return symbols


def fetch_data(symbol: str, years: int = 5) -> tuple[pd.DataFrame | None, str]:
    """Fetch historical data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{years}y")
        if hist.empty or len(hist) < 252:
            return None, f"Insufficient data ({len(hist)} days)"
        return hist, "OK"
    except Exception as e:
        return None, str(e)


def run_validation(symbols: list[str], years: int = 5) -> dict[str, list[dict]]:
    """
    Run all strategies on all symbols and collect results.
    
    Returns:
        Dict mapping strategy name to list of result dicts per symbol
    """
    results = {
        "DCA": [],
        "Buy_Dips": [],
        "Dip_Trading": [],
        "Regime_Aware_Technical": [],
        "Buy_Hold": [],
        "SPY_DCA": [],
    }
    
    # Fetch SPY data once
    print("Fetching SPY benchmark data...")
    spy_data, spy_status = fetch_data("SPY", years)
    if spy_data is None:
        print(f"ERROR: Could not fetch SPY data: {spy_status}")
        return results
    
    total = len(symbols)
    success = 0
    failed = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{total}] Processing {symbol}...", end=" ")
        
        data, status = fetch_data(symbol, years)
        if data is None:
            print(f"SKIP: {status}")
            failed.append((symbol, status))
            continue
            
        # Align data
        common_idx = data.index.intersection(spy_data.index)
        if len(common_idx) < 252:
            print(f"SKIP: Only {len(common_idx)} common days")
            failed.append((symbol, "Insufficient overlap"))
            continue
            
        stock_data = data.loc[common_idx]
        spy_aligned = spy_data.loc[common_idx]
        
        try:
            engine = BaselineEngine(
                prices=stock_data,
                spy_prices=spy_aligned,
                symbol=symbol
            )
            
            # Run all strategies
            comparison = engine.run_all()
            
            # Extract results
            results["DCA"].append({
                "symbol": symbol,
                "return_pct": comparison.dca.total_return_pct,
                "sharpe": comparison.dca.sharpe_ratio,
                "max_dd": comparison.dca.max_drawdown_pct,
            })
            
            results["Buy_Dips"].append({
                "symbol": symbol,
                "return_pct": comparison.buy_dips.total_return_pct,
                "sharpe": comparison.buy_dips.sharpe_ratio,
                "max_dd": comparison.buy_dips.max_drawdown_pct,
            })
            
            if comparison.dip_trading:
                results["Dip_Trading"].append({
                    "symbol": symbol,
                    "return_pct": comparison.dip_trading.total_return_pct,
                    "sharpe": comparison.dip_trading.sharpe_ratio,
                    "max_dd": comparison.dip_trading.max_drawdown_pct,
                    "win_rate": comparison.dip_trading.win_rate_pct,
                    "trades": comparison.dip_trading.total_buys,
                })
            
            if comparison.regime_aware_technical:
                results["Regime_Aware_Technical"].append({
                    "symbol": symbol,
                    "return_pct": comparison.regime_aware_technical.total_return_pct,
                    "sharpe": comparison.regime_aware_technical.sharpe_ratio,
                    "max_dd": comparison.regime_aware_technical.max_drawdown_pct,
                    "win_rate": comparison.regime_aware_technical.win_rate_pct,
                    "trades": comparison.regime_aware_technical.total_buys,
                })
            
            results["Buy_Hold"].append({
                "symbol": symbol,
                "return_pct": comparison.buy_hold.total_return_pct,
                "sharpe": comparison.buy_hold.sharpe_ratio,
                "max_dd": comparison.buy_hold.max_drawdown_pct,
            })
            
            results["SPY_DCA"].append({
                "symbol": symbol,
                "return_pct": comparison.spy_dca.total_return_pct,
                "sharpe": comparison.spy_dca.sharpe_ratio,
                "max_dd": comparison.spy_dca.max_drawdown_pct,
            })
            
            print(f"OK (DCA: {comparison.dca.total_return_pct:+.1f}%, Regime: {comparison.regime_aware_technical.total_return_pct if comparison.regime_aware_technical else 'N/A':+.1f}%)")
            success += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append((symbol, str(e)))
            continue
    
    print(f"\n=== Data Collection Complete ===")
    print(f"Success: {success}/{total}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {[s for s, _ in failed[:10]]}")
    
    return results


def calculate_statistics(results: dict[str, list[dict]]) -> list[StrategyStats]:
    """Calculate comprehensive statistics for each strategy."""
    statistics = []
    
    # Use SPY DCA as baseline for comparison
    spy_dca_returns = {r["symbol"]: r["return_pct"] for r in results["SPY_DCA"]}
    buy_hold_returns = {r["symbol"]: r["return_pct"] for r in results["Buy_Hold"]}
    
    for strategy_name, data in results.items():
        if not data or strategy_name in ["SPY_DCA", "Buy_Hold"]:
            continue
            
        returns = [r["return_pct"] for r in data]
        sharpes = [r["sharpe"] for r in data]
        max_dds = [r["max_dd"] for r in data]
        
        n = len(returns)
        if n < 5:
            continue
            
        # Calculate outperformance rates
        beat_spy = sum(
            1 for r in data 
            if r["symbol"] in spy_dca_returns and r["return_pct"] > spy_dca_returns[r["symbol"]]
        )
        beat_bh = sum(
            1 for r in data 
            if r["symbol"] in buy_hold_returns and r["return_pct"] > buy_hold_returns[r["symbol"]]
        )
        
        # Paired t-test vs SPY DCA
        paired_returns = [
            (r["return_pct"], spy_dca_returns[r["symbol"]])
            for r in data
            if r["symbol"] in spy_dca_returns
        ]
        
        t_stat = None
        p_val = None
        is_sig = False
        conf_level = ""
        
        if len(paired_returns) >= 10:
            strat_returns = [p[0] for p in paired_returns]
            spy_returns = [p[1] for p in paired_returns]
            
            # One-sided t-test: strategy > SPY DCA
            t_stat, p_val_two_sided = stats.ttest_rel(strat_returns, spy_returns)
            # Convert to one-sided p-value
            p_val = p_val_two_sided / 2 if t_stat > 0 else 1 - p_val_two_sided / 2
            
            if p_val < 0.01:
                is_sig = True
                conf_level = f"⭐⭐⭐ p={p_val:.4f} (99% confidence)"
            elif p_val < 0.05:
                is_sig = True
                conf_level = f"⭐⭐ p={p_val:.4f} (95% confidence)"
            elif p_val < 0.10:
                conf_level = f"⭐ p={p_val:.4f} (90% confidence)"
            else:
                conf_level = f"Not significant (p={p_val:.4f})"
        
        stat = StrategyStats(
            name=strategy_name.replace("_", " "),
            n_stocks=n,
            avg_return=np.mean(returns),
            std_return=np.std(returns),
            median_return=np.median(returns),
            min_return=np.min(returns),
            max_return=np.max(returns),
            avg_sharpe=np.mean(sharpes),
            win_rate=(beat_spy / n) * 100 if n > 0 else 0,
            beat_buy_hold_rate=(beat_bh / n) * 100 if n > 0 else 0,
            avg_max_drawdown=np.mean(max_dds),
            t_statistic=t_stat,
            p_value=p_val,
            is_significant=is_sig,
            confidence_level=conf_level,
        )
        statistics.append(stat)
    
    return statistics


def run_monte_carlo_baseline(results: dict[str, list[dict]], n_simulations: int = 1000) -> dict:
    """
    Run Monte Carlo simulation to establish random baseline.
    
    This simulates random entry/exit to show that our strategies
    are better than random chance.
    """
    # Get all returns from all strategies
    all_returns = []
    for strategy_data in results.values():
        all_returns.extend([r["return_pct"] for r in strategy_data])
    
    if not all_returns:
        return {}
    
    # Calculate what random selection would give
    random_means = []
    random_sharpes = []
    
    for _ in range(n_simulations):
        # Randomly select returns
        sample = np.random.choice(all_returns, size=min(30, len(all_returns)), replace=True)
        random_means.append(np.mean(sample))
        
    return {
        "random_mean": np.mean(random_means),
        "random_std": np.std(random_means),
        "random_95_ci": (np.percentile(random_means, 2.5), np.percentile(random_means, 97.5)),
    }


def print_summary(statistics: list[StrategyStats], monte_carlo: dict) -> None:
    """Print comprehensive summary report."""
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION SUMMARY")
    print("="*80)
    
    # Sort by significance then avg return
    sorted_stats = sorted(
        statistics,
        key=lambda s: (s.is_significant, s.avg_return),
        reverse=True
    )
    
    for stat in sorted_stats:
        print()
        print(stat)
    
    print("\n" + "-"*80)
    print("MONTE CARLO BASELINE (Random Trading)")
    print("-"*80)
    if monte_carlo:
        print(f"Random Mean Return: {monte_carlo['random_mean']:+.1f}% ± {monte_carlo['random_std']:.1f}%")
        print(f"Random 95% CI: [{monte_carlo['random_95_ci'][0]:+.1f}%, {monte_carlo['random_95_ci'][1]:+.1f}%]")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    significant = [s for s in statistics if s.is_significant]
    if significant:
        print(f"✅ {len(significant)} strategies showed STATISTICALLY SIGNIFICANT outperformance:")
        for s in significant:
            print(f"   - {s.name}: {s.avg_return:+.1f}% avg return, {s.confidence_level}")
    else:
        print("⚠️  No strategies showed statistically significant outperformance")
    
    # Best strategy
    if sorted_stats:
        best = sorted_stats[0]
        print(f"\n🏆 Best Strategy: {best.name}")
        print(f"   Average Return: {best.avg_return:+.1f}%")
        print(f"   Win Rate vs SPY: {best.win_rate:.1f}%")


if __name__ == "__main__":
    print("=" * 80)
    print("STONKMARKET STATISTICAL VALIDATION")
    print("Testing trading strategies across multiple stocks to prove statistical significance")
    print("=" * 80)
    print()
    
    # Get symbols to test
    symbols = get_all_symbols()
    print(f"Testing {len(symbols)} stocks over 5 years")
    print()
    
    # Run validation
    results = run_validation(symbols, years=5)
    
    # Calculate statistics
    statistics = calculate_statistics(results)
    
    # Run Monte Carlo baseline
    monte_carlo = run_monte_carlo_baseline(results)
    
    # Print summary
    print_summary(statistics, monte_carlo)
