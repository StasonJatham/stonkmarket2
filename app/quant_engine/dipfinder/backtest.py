"""Dipfinder Backtest Harness.

Evaluates dipfinder signal performance using walk-forward validation.
Leverages the quant_engine backtest infrastructure.

Key features:
- Walk-forward splits to prevent overfitting
- Measures signal quality: hit rate, average return, drawdown
- Compares against buy-and-hold and random entry baselines
- Statistical significance testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging import get_logger
from app.quant_engine.dipfinder.signal import (
    compute_signal,
    AlertLevel,
)
from app.quant_engine.dipfinder.config import DipFinderConfig
from app.quant_engine.core.config import QUANT_LIMITS

logger = get_logger("dipfinder.backtest")


@dataclass
class BacktestTrade:
    """Individual trade from a dipfinder signal."""
    
    symbol: str
    signal_date: datetime
    entry_price: float
    entry_score: float
    alert_level: str
    opportunity_type: str = "NONE"  # OUTLIER, BOUNCE, BOTH, NONE
    
    # EVA metrics (Extreme Value Analysis)
    is_tail_event: bool = False  # True if dip exceeds tail threshold
    return_period_years: float | None = None  # How rare is this dip in years
    regime_dip_percentile: float | None = None  # Percentile within normal regime
    dip_pct: float = 0.0  # Current dip percentage
    
    # Exit info (filled when trade closes)
    exit_date: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""  # target, stop_loss, max_hold
    
    # Performance
    holding_days: int = 0
    return_pct: float = 0.0
    is_winner: bool = False


@dataclass
class SignalPerformance:
    """Performance metrics for a specific signal level."""
    
    alert_level: str  # "STRONG", "GOOD", "ALL", "TAIL", "NORMAL"
    n_signals: int = 0
    n_trades: int = 0  # Signals that could be traded
    
    # Hit rate
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    
    # Returns
    avg_return_pct: float = 0.0
    median_return_pct: float = 0.0
    total_return_pct: float = 0.0
    
    # Best/Worst
    best_return_pct: float = 0.0
    worst_return_pct: float = 0.0
    
    # Holding period
    avg_holding_days: float = 0.0
    
    # Risk
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Statistical significance
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False  # p < 0.05
    
    # EVA-specific metrics
    avg_return_period_years: float | None = None  # Avg rarity of traded dips
    avg_regime_percentile: float | None = None  # Avg percentile within normal regime


@dataclass
class DipfinderBacktestResult:
    """Complete dipfinder backtest results."""
    
    # Backtest parameters
    start_date: datetime
    end_date: datetime
    n_symbols: int
    n_folds: int
    
    # Performance by alert level
    strong_signals: SignalPerformance | None = None
    good_signals: SignalPerformance | None = None
    all_signals: SignalPerformance | None = None
    
    # Performance by opportunity type
    outlier_signals: SignalPerformance | None = None
    bounce_signals: SignalPerformance | None = None
    
    # Performance by EVA classification (tail events vs normal)
    tail_event_signals: SignalPerformance | None = None
    normal_event_signals: SignalPerformance | None = None
    
    # EVA summary stats
    n_tail_events: int = 0
    tail_event_pct: float = 0.0  # % of trades that were tail events
    avg_return_period_years: float | None = None
    
    # Baseline comparisons
    vs_buy_hold: float = 0.0  # Excess return vs buy-and-hold
    vs_random_entry: float = 0.0  # Excess return vs random entries
    
    # Walk-forward stability
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    sharpe_degradation_pct: float = 0.0  # Should be < 50%
    is_walk_forward_stable: bool = False
    
    # Optimal config (if tuned)
    optimal_config: dict = field(default_factory=dict)
    
    # Individual trades for analysis
    trades: list[BacktestTrade] = field(default_factory=list)
    
    # Per-fold results
    fold_results: list[dict] = field(default_factory=list)


@dataclass
class BacktestConfig:
    """Configuration for dipfinder backtesting."""
    
    # Trade parameters
    stop_loss_pct: float = 0.10  # 10% stop loss
    take_profit_pct: float = 0.20  # 20% take profit
    max_holding_days: int = QUANT_LIMITS.max_holding_days  # From central config
    
    # Walk-forward settings
    n_folds: int = 5
    train_ratio: float = 0.70  # 70% train, 30% test
    
    # Filter
    min_score_threshold: float = 60.0  # Minimum score to consider
    
    # Statistical settings
    min_trades_for_significance: int = 30
    confidence_level: float = 0.95


def _compute_walk_forward_splits(
    dates: pd.DatetimeIndex,
    n_folds: int = 5,
    train_ratio: float = 0.70,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Create walk-forward train/test splits.
    
    Uses expanding window: each fold trains on all prior data,
    tests on the next chunk.
    """
    n = len(dates)
    fold_size = n // n_folds
    
    splits = []
    for i in range(n_folds):
        # Train on first (i+1) * fold_size * train_ratio points
        train_end_idx = int((i + 1) * fold_size * train_ratio)
        train_dates = dates[:train_end_idx]
        
        # Test on next chunk
        test_start_idx = train_end_idx
        test_end_idx = min((i + 1) * fold_size, n)
        test_dates = dates[test_start_idx:test_end_idx]
        
        if len(train_dates) > 0 and len(test_dates) > 0:
            splits.append((train_dates, test_dates))
    
    return splits


def _simulate_trade(
    entry_date: datetime,
    entry_price: float,
    prices_after_entry: pd.Series,
    config: BacktestConfig,
) -> tuple[datetime, float, str, int, float]:
    """
    Simulate a single trade from entry to exit.
    
    Returns:
        (exit_date, exit_price, exit_reason, holding_days, return_pct)
    """
    stop_price = entry_price * (1 - config.stop_loss_pct)
    target_price = entry_price * (1 + config.take_profit_pct)
    
    for i, (date, price) in enumerate(prices_after_entry.items()):
        holding_days = i + 1
        
        # Check stop loss
        if price <= stop_price:
            return_pct = (price - entry_price) / entry_price * 100
            return date, price, "stop_loss", holding_days, return_pct
        
        # Check take profit
        if price >= target_price:
            return_pct = (price - entry_price) / entry_price * 100
            return date, price, "take_profit", holding_days, return_pct
        
        # Check max holding period
        if holding_days >= config.max_holding_days:
            return_pct = (price - entry_price) / entry_price * 100
            return date, price, "max_hold", holding_days, return_pct
    
    # Still holding at end of data
    if len(prices_after_entry) > 0:
        last_date = prices_after_entry.index[-1]
        last_price = prices_after_entry.iloc[-1]
        return_pct = (last_price - entry_price) / entry_price * 100
        return last_date, last_price, "end_of_data", len(prices_after_entry), return_pct
    
    return entry_date, entry_price, "no_data", 0, 0.0


def _compute_signal_performance(
    trades: list[BacktestTrade],
    alert_level: str,
) -> SignalPerformance:
    """Compute performance metrics for a set of trades."""
    if not trades:
        return SignalPerformance(alert_level=alert_level)
    
    returns = [t.return_pct for t in trades if t.exit_price is not None]
    holding_days = [t.holding_days for t in trades if t.exit_price is not None]
    
    if not returns:
        return SignalPerformance(
            alert_level=alert_level,
            n_signals=len(trades),
        )
    
    winners = sum(1 for r in returns if r > 0)
    losers = sum(1 for r in returns if r <= 0)
    
    # Statistical test: are returns significantly > 0?
    t_stat, p_value = 0.0, 1.0
    if len(returns) >= 3:
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        # One-sided test: we want returns > 0
        p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    
    # Sharpe ratio (annualized, assuming daily returns)
    returns_arr = np.array(returns)
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 0.01
    sharpe = (mean_return / std_return) * np.sqrt(252 / np.mean(holding_days)) if std_return > 0 else 0.0
    
    # Max drawdown from cumulative returns
    cumulative = np.cumprod(1 + returns_arr / 100)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    # EVA metrics
    return_periods = [t.return_period_years for t in trades if t.return_period_years is not None]
    regime_percentiles = [t.regime_dip_percentile for t in trades if t.regime_dip_percentile is not None]
    
    avg_return_period = float(np.mean(return_periods)) if return_periods else None
    avg_regime_pctl = float(np.mean(regime_percentiles)) if regime_percentiles else None
    
    return SignalPerformance(
        alert_level=alert_level,
        n_signals=len(trades),
        n_trades=len(returns),
        winners=winners,
        losers=losers,
        win_rate=(winners / len(returns) * 100) if returns else 0.0,
        avg_return_pct=float(np.mean(returns_arr)),
        median_return_pct=float(np.median(returns_arr)),
        total_return_pct=float(np.sum(returns_arr)),
        best_return_pct=float(np.max(returns_arr)),
        worst_return_pct=float(np.min(returns_arr)),
        avg_holding_days=float(np.mean(holding_days)),
        max_drawdown_pct=max_dd,
        sharpe_ratio=float(sharpe),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        is_significant=p_value < 0.05,
        avg_return_period_years=avg_return_period,
        avg_regime_percentile=avg_regime_pctl,
    )


async def backtest_dipfinder_signals(
    symbols: list[str],
    prices_by_symbol: dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    config: BacktestConfig | None = None,
    dipfinder_config: DipFinderConfig | None = None,
) -> DipfinderBacktestResult:
    """
    Backtest dipfinder signals across multiple symbols with walk-forward validation.
    
    Args:
        symbols: List of ticker symbols to backtest
        prices_by_symbol: Dict mapping symbol -> DataFrame with 'date', 'close', 'volume'
        start_date: Backtest start date
        end_date: Backtest end date
        config: Backtest configuration
        dipfinder_config: Dipfinder signal configuration
        
    Returns:
        DipfinderBacktestResult with performance metrics
    """
    if config is None:
        config = BacktestConfig()
    if dipfinder_config is None:
        dipfinder_config = DipFinderConfig()
    
    all_trades: list[BacktestTrade] = []
    fold_results: list[dict] = []
    
    for symbol in symbols:
        if symbol not in prices_by_symbol:
            continue
            
        df = prices_by_symbol[symbol]
        if df.empty or len(df) < 200:  # Need enough data for signals
            continue
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        
        # Filter to date range
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        df = df[mask]
        
        if len(df) < 50:  # Minimum data
            continue
        
        # Create walk-forward splits
        splits = _compute_walk_forward_splits(
            df.index,
            n_folds=config.n_folds,
            train_ratio=config.train_ratio,
        )
        
        for fold_idx, (train_dates, test_dates) in enumerate(splits):
            # For each test date, compute dipfinder signal and simulate trade
            for test_date in test_dates:
                # Get data up to test_date for signal computation
                historical_data = df[df.index <= test_date]
                if len(historical_data) < 50:
                    continue
                
                # Extract price arrays for compute_signal
                stock_prices = historical_data["close"].values
                volumes = historical_data.get("volume", pd.Series([1.0] * len(historical_data))).values
                
                # Compute dipfinder signal
                # Note: compute_signal is synchronous
                try:
                    signal = compute_signal(
                        ticker=symbol,
                        stock_prices=stock_prices,
                        benchmark_prices=stock_prices,  # Use self as benchmark for simplicity
                        volumes=volumes,
                        window=dipfinder_config.window,
                        config=dipfinder_config,
                    )
                except Exception as e:
                    logger.debug(f"Signal error for {symbol} on {test_date}: {e}")
                    continue
                
                # Only trade on alerts
                if signal.alert_level == AlertLevel.NONE:
                    continue
                
                if signal.final_score < config.min_score_threshold:
                    continue
                
                # Get prices after entry for trade simulation
                entry_price = float(stock_prices[-1])
                future_prices = df[df.index > test_date]["close"]
                
                if len(future_prices) < 1:
                    continue
                
                # Simulate trade
                exit_date, exit_price, exit_reason, holding_days, return_pct = _simulate_trade(
                    entry_date=test_date,
                    entry_price=entry_price,
                    prices_after_entry=future_prices,
                    config=config,
                )
                
                trade = BacktestTrade(
                    symbol=symbol,
                    signal_date=test_date,
                    entry_price=entry_price,
                    entry_score=signal.final_score,
                    alert_level=signal.alert_level.value,
                    opportunity_type=signal.opportunity_type.value,
                    # EVA metrics from dip_metrics
                    is_tail_event=signal.dip_metrics.is_tail_event,
                    return_period_years=signal.dip_metrics.return_period_years,
                    regime_dip_percentile=signal.dip_metrics.regime_dip_percentile,
                    dip_pct=signal.dip_metrics.dip_pct,
                    exit_date=exit_date,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    holding_days=holding_days,
                    return_pct=return_pct,
                    is_winner=return_pct > 0,
                )
                all_trades.append(trade)
            
            # Compute fold statistics
            fold_trades = [t for t in all_trades if t.exit_date is not None]
            fold_returns = [t.return_pct for t in fold_trades]
            
            fold_results.append({
                "fold": fold_idx,
                "n_trades": len(fold_trades),
                "avg_return": float(np.mean(fold_returns)) if fold_returns else 0.0,
                "sharpe": float(np.mean(fold_returns) / np.std(fold_returns)) if len(fold_returns) > 1 and np.std(fold_returns) > 0 else 0.0,
            })
    
    # Compute performance by alert level
    strong_trades = [t for t in all_trades if t.alert_level == "STRONG"]
    good_trades = [t for t in all_trades if t.alert_level == "GOOD"]
    
    strong_perf = _compute_signal_performance(strong_trades, "STRONG")
    good_perf = _compute_signal_performance(good_trades, "GOOD")
    all_perf = _compute_signal_performance(all_trades, "ALL")
    
    # Compute performance by opportunity type
    outlier_trades = [t for t in all_trades if t.opportunity_type in ("OUTLIER", "BOTH")]
    bounce_trades = [t for t in all_trades if t.opportunity_type in ("BOUNCE", "BOTH")]
    
    outlier_perf = _compute_signal_performance(outlier_trades, "OUTLIER")
    bounce_perf = _compute_signal_performance(bounce_trades, "BOUNCE")
    
    # Compute performance by EVA classification (tail events vs normal)
    tail_trades = [t for t in all_trades if t.is_tail_event]
    normal_trades = [t for t in all_trades if not t.is_tail_event]
    
    tail_perf = _compute_signal_performance(tail_trades, "TAIL")
    normal_perf = _compute_signal_performance(normal_trades, "NORMAL")
    
    # EVA summary statistics
    n_tail = len(tail_trades)
    tail_pct = (n_tail / len(all_trades) * 100) if all_trades else 0.0
    return_periods = [t.return_period_years for t in all_trades if t.return_period_years is not None]
    avg_return_period = float(np.mean(return_periods)) if return_periods else None
    
    # Walk-forward stability
    # In-sample = first half of folds, out-of-sample = second half
    mid = len(fold_results) // 2
    is_sharpes = [f["sharpe"] for f in fold_results[:mid] if f["sharpe"] != 0]
    oos_sharpes = [f["sharpe"] for f in fold_results[mid:] if f["sharpe"] != 0]
    
    in_sample_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
    out_of_sample_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
    
    sharpe_degradation = 0.0
    if in_sample_sharpe > 0:
        sharpe_degradation = (in_sample_sharpe - out_of_sample_sharpe) / in_sample_sharpe * 100
    
    is_stable = sharpe_degradation < 50 and out_of_sample_sharpe > 0
    
    return DipfinderBacktestResult(
        start_date=start_date,
        end_date=end_date,
        n_symbols=len(symbols),
        n_folds=config.n_folds,
        strong_signals=strong_perf,
        good_signals=good_perf,
        all_signals=all_perf,
        outlier_signals=outlier_perf,
        bounce_signals=bounce_perf,
        tail_event_signals=tail_perf,
        normal_event_signals=normal_perf,
        n_tail_events=n_tail,
        tail_event_pct=tail_pct,
        avg_return_period_years=avg_return_period,
        in_sample_sharpe=in_sample_sharpe,
        out_of_sample_sharpe=out_of_sample_sharpe,
        sharpe_degradation_pct=sharpe_degradation,
        is_walk_forward_stable=is_stable,
        trades=all_trades,
        fold_results=fold_results,
    )


def backtest_result_to_dict(result: DipfinderBacktestResult) -> dict:
    """Convert backtest result to JSON-serializable dict."""
    def perf_to_dict(perf: SignalPerformance | None) -> dict | None:
        if perf is None:
            return None
        return {
            "alert_level": perf.alert_level,
            "n_signals": perf.n_signals,
            "n_trades": perf.n_trades,
            "win_rate": round(perf.win_rate, 2),
            "avg_return_pct": round(perf.avg_return_pct, 2),
            "median_return_pct": round(perf.median_return_pct, 2),
            "total_return_pct": round(perf.total_return_pct, 2),
            "best_return_pct": round(perf.best_return_pct, 2),
            "worst_return_pct": round(perf.worst_return_pct, 2),
            "avg_holding_days": round(perf.avg_holding_days, 1),
            "max_drawdown_pct": round(perf.max_drawdown_pct, 2),
            "sharpe_ratio": round(perf.sharpe_ratio, 2),
            "p_value": round(perf.p_value, 4),
            "is_significant": perf.is_significant,
            # EVA metrics
            "avg_return_period_years": round(perf.avg_return_period_years, 1) if perf.avg_return_period_years else None,
            "avg_regime_percentile": round(perf.avg_regime_percentile, 1) if perf.avg_regime_percentile else None,
        }
    
    return {
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "n_symbols": result.n_symbols,
        "n_folds": result.n_folds,
        "strong_signals": perf_to_dict(result.strong_signals),
        "good_signals": perf_to_dict(result.good_signals),
        "all_signals": perf_to_dict(result.all_signals),
        "outlier_signals": perf_to_dict(result.outlier_signals),
        "bounce_signals": perf_to_dict(result.bounce_signals),
        # EVA performance breakdown
        "tail_event_signals": perf_to_dict(result.tail_event_signals),
        "normal_event_signals": perf_to_dict(result.normal_event_signals),
        # EVA summary
        "n_tail_events": result.n_tail_events,
        "tail_event_pct": round(result.tail_event_pct, 1),
        "avg_return_period_years": round(result.avg_return_period_years, 1) if result.avg_return_period_years else None,
        # Walk-forward
        "in_sample_sharpe": round(result.in_sample_sharpe, 2),
        "out_of_sample_sharpe": round(result.out_of_sample_sharpe, 2),
        "sharpe_degradation_pct": round(result.sharpe_degradation_pct, 1),
        "is_walk_forward_stable": result.is_walk_forward_stable,
        "n_trades": len(result.trades),
        "fold_results": result.fold_results,
    }
