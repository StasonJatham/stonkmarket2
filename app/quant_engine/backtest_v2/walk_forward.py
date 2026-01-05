"""
Walk-Forward Optimization with Strict Anti-Overfitting Measures.

This module implements proper walk-forward analysis:
- Time-series aware K-fold splits (no look-ahead bias)
- Out-of-sample validation on each fold
- Aggregate metrics across all folds
- Kill switch for failed strategies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Minimum requirements for statistical significance
MIN_TRADES_PER_FOLD = 30
MIN_FOLDS = 3
MAX_FOLDS = 10
DEFAULT_TRAIN_RATIO = 0.70


class WFOResult(str, Enum):
    """Walk-Forward Optimization result verdict."""

    PASSED = "PASSED"  # Strategy is robust, not overfit
    FAILED_INSUFFICIENT_DATA = "FAILED_INSUFFICIENT_DATA"
    FAILED_NOT_ENOUGH_TRADES = "FAILED_NOT_ENOUGH_TRADES"
    FAILED_NEGATIVE_OOS_RETURNS = "FAILED_NEGATIVE_OOS_RETURNS"
    FAILED_INCONSISTENT_PERFORMANCE = "FAILED_INCONSISTENT_PERFORMANCE"
    FAILED_KILL_SWITCH = "FAILED_KILL_SWITCH"


@dataclass
class FoldResult:
    """Result of a single walk-forward fold."""

    fold_number: int
    
    # Date ranges
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Train metrics (for comparison)
    train_return_pct: float
    train_trades: int
    train_sharpe: float
    
    # Test (out-of-sample) metrics - THIS IS WHAT MATTERS
    test_return_pct: float
    test_trades: int
    test_sharpe: float
    test_max_drawdown_pct: float
    test_win_rate: float
    
    # Parameter values used
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Did this fold pass?
    passed: bool = True
    failure_reason: str | None = None


@dataclass
class KillSwitchResult:
    """Kill switch evaluation result."""

    triggered: bool
    reasons: list[str] = field(default_factory=list)
    
    # Thresholds that were violated
    total_return_check: bool = True  # Pass if > 0%
    max_drawdown_check: bool = True  # Pass if < 40%
    calmar_ratio_check: bool = True  # Pass if > 0.5
    win_rate_check: bool = True  # Pass if > 40%
    sharpe_check: bool = True  # Pass if > 0.0


@dataclass
class WFOSummary:
    """Complete Walk-Forward Optimization summary."""

    result: WFOResult
    message: str
    
    # Aggregate out-of-sample metrics
    n_folds: int
    avg_oos_return_pct: float
    std_oos_return_pct: float
    avg_oos_sharpe: float
    avg_oos_win_rate: float
    total_oos_trades: int
    
    # Consistency metrics
    oos_return_volatility: float  # Lower is better
    pct_folds_profitable: float
    pct_folds_beat_benchmark: float
    
    # Kill switch
    kill_switch: KillSwitchResult
    
    # Individual fold results
    folds: list[FoldResult] = field(default_factory=list)
    
    # Optimal parameters (mode across folds)
    optimal_parameters: dict[str, Any] = field(default_factory=dict)


class StrategyProtocol(Protocol):
    """Protocol for strategies that can be walk-forward tested."""

    def run(
        self,
        prices: pd.Series,
        params: dict[str, Any],
    ) -> tuple[float, int, float, float, float]:
        """
        Run strategy and return metrics.
        
        Returns:
            (return_pct, n_trades, sharpe, max_drawdown_pct, win_rate)
        """
        ...

    def optimize(
        self,
        prices: pd.Series,
        param_grid: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """
        Optimize parameters on training data.
        
        Returns:
            Best parameter combination
        """
        ...


@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration."""

    n_folds: int = 5
    train_ratio: float = 0.70  # 70% train, 30% test per fold
    min_trades_per_fold: int = 30
    
    # Kill switch thresholds
    min_total_return: float = 0.0
    max_drawdown_limit: float = 40.0
    min_calmar_ratio: float = 0.5
    min_win_rate: float = 40.0
    min_sharpe_ratio: float = 0.0
    
    # Consistency requirements
    min_pct_folds_profitable: float = 50.0  # At least 50% of folds must be profitable
    max_return_volatility: float = 50.0  # Return std across folds < 50%


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization Engine.
    
    Key Features:
    - Time-series aware splits (no shuffling, no look-ahead)
    - Out-of-sample validation on each fold
    - Parameter optimization on train, validation on test
    - Kill switch for catastrophic failures
    """
    
    def __init__(self, config: WFOConfig | None = None):
        self.config = config or WFOConfig()
    
    def create_folds(
        self,
        prices: pd.Series,
        n_folds: int | None = None,
    ) -> list[tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Create time-series walk-forward folds.
        
        Unlike k-fold cross-validation, walk-forward:
        1. Always maintains chronological order
        2. Never looks ahead
        3. Each fold's test period comes AFTER its train period
        
        Returns:
            List of (train_prices, test_prices, train_start, train_end, test_start, test_end)
        """
        n_folds = n_folds or self.config.n_folds
        n_samples = len(prices)
        
        # Each fold uses all data up to a point for training, then forward for testing
        # With 5 folds, we divide the data into 6 parts:
        # Fold 1: Train on 1, Test on 2
        # Fold 2: Train on 1-2, Test on 3
        # Fold 3: Train on 1-3, Test on 4
        # etc.
        
        # Minimum data per fold
        fold_size = n_samples // (n_folds + 1)
        if fold_size < 60:  # At least 60 trading days per fold
            raise ValueError(f"Insufficient data for {n_folds} folds. Need at least {(n_folds + 1) * 60} days.")
        
        folds = []
        for fold_idx in range(n_folds):
            # Train: from start to fold boundary
            train_end_idx = (fold_idx + 1) * fold_size
            
            # Test: next fold_size samples
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + fold_size, n_samples)
            
            train_prices = prices.iloc[:train_end_idx]
            test_prices = prices.iloc[test_start_idx:test_end_idx]
            
            # Use train_ratio to further split train into train/validation
            # But for WFO, we optimize on full train and validate on test
            
            train_start = train_prices.index[0]
            train_end = train_prices.index[-1]
            test_start = test_prices.index[0]
            test_end = test_prices.index[-1]
            
            folds.append((
                train_prices,
                test_prices,
                train_start,
                train_end,
                test_start,
                test_end,
            ))
        
        return folds
    
    def run_wfo(
        self,
        prices: pd.Series,
        strategy: StrategyProtocol,
        param_grid: dict[str, list[Any]],
        benchmark_prices: pd.Series | None = None,
    ) -> WFOSummary:
        """
        Run complete walk-forward optimization.
        
        Args:
            prices: Price series for the asset
            strategy: Strategy implementing StrategyProtocol
            param_grid: Parameter grid for optimization
            benchmark_prices: Optional benchmark (e.g., SPY) for comparison
        
        Returns:
            WFOSummary with all results and verdict
        """
        try:
            folds_data = self.create_folds(prices)
        except ValueError as e:
            return WFOSummary(
                result=WFOResult.FAILED_INSUFFICIENT_DATA,
                message=str(e),
                n_folds=0,
                avg_oos_return_pct=0,
                std_oos_return_pct=0,
                avg_oos_sharpe=0,
                avg_oos_win_rate=0,
                total_oos_trades=0,
                oos_return_volatility=0,
                pct_folds_profitable=0,
                pct_folds_beat_benchmark=0,
                kill_switch=KillSwitchResult(triggered=True, reasons=["Insufficient data"]),
            )
        
        fold_results: list[FoldResult] = []
        all_oos_returns: list[float] = []
        total_trades = 0
        
        for fold_idx, (train_prices, test_prices, train_start, train_end, test_start, test_end) in enumerate(folds_data):
            # Step 1: Optimize parameters on training data
            best_params = strategy.optimize(train_prices, param_grid)
            
            # Step 2: Run strategy on training data (for comparison)
            train_return, train_trades, train_sharpe, train_dd, train_wr = strategy.run(
                train_prices, best_params
            )
            
            # Step 3: Run strategy on TEST data (out-of-sample)
            test_return, test_trades, test_sharpe, test_dd, test_wr = strategy.run(
                test_prices, best_params
            )
            
            # Check if fold passes
            passed = True
            failure_reason = None
            
            if test_trades < self.config.min_trades_per_fold:
                passed = False
                failure_reason = f"Insufficient trades in test: {test_trades} < {self.config.min_trades_per_fold}"
            
            fold_results.append(FoldResult(
                fold_number=fold_idx + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_return_pct=train_return,
                train_trades=train_trades,
                train_sharpe=train_sharpe,
                test_return_pct=test_return,
                test_trades=test_trades,
                test_sharpe=test_sharpe,
                test_max_drawdown_pct=test_dd,
                test_win_rate=test_wr,
                parameters=best_params,
                passed=passed,
                failure_reason=failure_reason,
            ))
            
            all_oos_returns.append(test_return)
            total_trades += test_trades
        
        # Aggregate metrics
        n_folds = len(fold_results)
        avg_oos_return = float(np.mean(all_oos_returns))
        std_oos_return = float(np.std(all_oos_returns))
        avg_oos_sharpe = float(np.mean([f.test_sharpe for f in fold_results]))
        avg_oos_win_rate = float(np.mean([f.test_win_rate for f in fold_results]))
        
        # Consistency metrics
        profitable_folds = sum(1 for r in all_oos_returns if r > 0)
        pct_profitable = (profitable_folds / n_folds * 100) if n_folds > 0 else 0
        
        # Compare to benchmark if provided
        pct_beat_benchmark = 0.0
        if benchmark_prices is not None:
            # Would calculate benchmark returns per fold and compare
            # Simplified: just check against 0
            pct_beat_benchmark = pct_profitable
        
        # Find optimal parameters (mode across folds)
        optimal_params = self._compute_optimal_params(fold_results)
        
        # Evaluate kill switch
        kill_switch = self._evaluate_kill_switch(
            avg_return=avg_oos_return,
            max_dd=float(max(f.test_max_drawdown_pct for f in fold_results)),
            sharpe=avg_oos_sharpe,
            win_rate=avg_oos_win_rate,
        )
        
        # Determine final result
        result, message = self._determine_result(
            fold_results=fold_results,
            avg_return=avg_oos_return,
            pct_profitable=pct_profitable,
            return_volatility=std_oos_return,
            kill_switch=kill_switch,
            total_trades=total_trades,
        )
        
        return WFOSummary(
            result=result,
            message=message,
            n_folds=n_folds,
            avg_oos_return_pct=avg_oos_return,
            std_oos_return_pct=std_oos_return,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_oos_win_rate=avg_oos_win_rate,
            total_oos_trades=total_trades,
            oos_return_volatility=std_oos_return,
            pct_folds_profitable=pct_profitable,
            pct_folds_beat_benchmark=pct_beat_benchmark,
            kill_switch=kill_switch,
            folds=fold_results,
            optimal_parameters=optimal_params,
        )
    
    def run_simple_wfo(
        self,
        train_metrics: tuple[float, int, float, float, float],
        test_metrics: tuple[float, int, float, float, float],
    ) -> WFOSummary:
        """
        Simple WFO validation with pre-computed train/test metrics.
        
        This is a convenience method when you already have metrics from
        a train/test split (e.g., from the existing BacktestService).
        
        Args:
            train_metrics: (return_pct, n_trades, sharpe, max_dd_pct, win_rate)
            test_metrics: Same for test period
        
        Returns:
            WFOSummary with single-fold validation
        """
        train_return, train_trades, train_sharpe, train_dd, train_wr = train_metrics
        test_return, test_trades, test_sharpe, test_dd, test_wr = test_metrics
        
        # Create single fold result
        fold = FoldResult(
            fold_number=1,
            train_start=pd.Timestamp.now(),  # Placeholder
            train_end=pd.Timestamp.now(),
            test_start=pd.Timestamp.now(),
            test_end=pd.Timestamp.now(),
            train_return_pct=train_return,
            train_trades=train_trades,
            train_sharpe=train_sharpe,
            test_return_pct=test_return,
            test_trades=test_trades,
            test_sharpe=test_sharpe,
            test_max_drawdown_pct=test_dd,
            test_win_rate=test_wr,
        )
        
        # Evaluate kill switch on test metrics
        kill_switch = self._evaluate_kill_switch(
            avg_return=test_return,
            max_dd=test_dd,
            sharpe=test_sharpe,
            win_rate=test_wr,
        )
        
        # Determine result
        if test_trades < self.config.min_trades_per_fold:
            result = WFOResult.FAILED_NOT_ENOUGH_TRADES
            message = f"Insufficient OOS trades: {test_trades} < {self.config.min_trades_per_fold}"
        elif kill_switch.triggered:
            result = WFOResult.FAILED_KILL_SWITCH
            message = f"Kill switch triggered: {', '.join(kill_switch.reasons)}"
        elif test_return < 0:
            result = WFOResult.FAILED_NEGATIVE_OOS_RETURNS
            message = f"Negative OOS return: {test_return:.1f}%"
        else:
            result = WFOResult.PASSED
            message = f"Passed WFO validation. OOS return: {test_return:.1f}%"
        
        return WFOSummary(
            result=result,
            message=message,
            n_folds=1,
            avg_oos_return_pct=test_return,
            std_oos_return_pct=0,
            avg_oos_sharpe=test_sharpe,
            avg_oos_win_rate=test_wr,
            total_oos_trades=test_trades,
            oos_return_volatility=0,
            pct_folds_profitable=100 if test_return > 0 else 0,
            pct_folds_beat_benchmark=100 if test_return > 0 else 0,
            kill_switch=kill_switch,
            folds=[fold],
        )
    
    def _evaluate_kill_switch(
        self,
        avg_return: float,
        max_dd: float,
        sharpe: float,
        win_rate: float,
    ) -> KillSwitchResult:
        """Evaluate kill switch conditions."""
        reasons = []
        
        total_return_check = avg_return > self.config.min_total_return
        if not total_return_check:
            reasons.append(f"Negative return: {avg_return:.1f}%")
        
        max_dd_check = max_dd < self.config.max_drawdown_limit
        if not max_dd_check:
            reasons.append(f"Excessive drawdown: {max_dd:.1f}%")
        
        calmar_check = (avg_return / max_dd) > self.config.min_calmar_ratio if max_dd > 0 else avg_return > 0
        if not calmar_check:
            calmar = avg_return / max_dd if max_dd > 0 else 0
            reasons.append(f"Low Calmar ratio: {calmar:.2f}")
        
        win_rate_check = win_rate > self.config.min_win_rate
        if not win_rate_check:
            reasons.append(f"Low win rate: {win_rate:.1f}%")
        
        sharpe_check = sharpe > self.config.min_sharpe_ratio
        if not sharpe_check:
            reasons.append(f"Low Sharpe ratio: {sharpe:.2f}")
        
        triggered = len(reasons) > 0
        
        return KillSwitchResult(
            triggered=triggered,
            reasons=reasons,
            total_return_check=total_return_check,
            max_drawdown_check=max_dd_check,
            calmar_ratio_check=calmar_check,
            win_rate_check=win_rate_check,
            sharpe_check=sharpe_check,
        )
    
    def _determine_result(
        self,
        fold_results: list[FoldResult],
        avg_return: float,
        pct_profitable: float,
        return_volatility: float,
        kill_switch: KillSwitchResult,
        total_trades: int,
    ) -> tuple[WFOResult, str]:
        """Determine final WFO result and message."""
        
        # Check for insufficient trades
        if total_trades < self.config.min_trades_per_fold * len(fold_results) // 2:
            return (
                WFOResult.FAILED_NOT_ENOUGH_TRADES,
                f"Insufficient total OOS trades: {total_trades}"
            )
        
        # Check kill switch
        if kill_switch.triggered:
            return (
                WFOResult.FAILED_KILL_SWITCH,
                f"Kill switch triggered: {', '.join(kill_switch.reasons)}"
            )
        
        # Check negative returns
        if avg_return < 0:
            return (
                WFOResult.FAILED_NEGATIVE_OOS_RETURNS,
                f"Negative average OOS return: {avg_return:.1f}%"
            )
        
        # Check consistency
        if pct_profitable < self.config.min_pct_folds_profitable:
            return (
                WFOResult.FAILED_INCONSISTENT_PERFORMANCE,
                f"Only {pct_profitable:.0f}% of folds profitable (need {self.config.min_pct_folds_profitable:.0f}%)"
            )
        
        if return_volatility > self.config.max_return_volatility:
            return (
                WFOResult.FAILED_INCONSISTENT_PERFORMANCE,
                f"High return volatility across folds: {return_volatility:.1f}%"
            )
        
        # All checks passed
        return (
            WFOResult.PASSED,
            f"Strategy passed WFO. Avg OOS return: {avg_return:.1f}%, "
            f"{pct_profitable:.0f}% folds profitable, Sharpe: {np.mean([f.test_sharpe for f in fold_results]):.2f}"
        )
    
    def _compute_optimal_params(
        self,
        fold_results: list[FoldResult],
    ) -> dict[str, Any]:
        """Compute optimal parameters across all folds."""
        if not fold_results:
            return {}
        
        # Get all parameter keys from first fold
        all_params: dict[str, list[Any]] = {}
        for fold in fold_results:
            for key, value in fold.parameters.items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)
        
        # Find mode (most common value) for each parameter
        optimal: dict[str, Any] = {}
        for key, values in all_params.items():
            if not values:
                continue
            # For numeric values, use median
            if all(isinstance(v, (int, float)) for v in values):
                optimal[key] = float(np.median(values))
            else:
                # For categorical, use mode
                from collections import Counter
                optimal[key] = Counter(values).most_common(1)[0][0]
        
        return optimal
