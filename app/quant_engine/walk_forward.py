"""
Walk-forward validation harness.

Implements expanding-window walk-forward with:
- Train/validation/test splits
- No lookahead guarantees
- Regime-sliced evaluation
- Baseline comparisons
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.types import (
    QuantConfig,
    WalkForwardFold,
    WalkForwardResult,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationMetrics:
    """Metrics from one validation fold."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Portfolio metrics
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float

    # Alpha model metrics
    mu_hat_mse: float
    mu_hat_r2: float
    hit_rate: float  # % of correct direction predictions

    # Regime breakdown
    bull_return: float | None
    bear_return: float | None
    high_vol_return: float | None


def generate_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_months: int = 36,
    validation_months: int = 6,
    test_months: int = 6,
    min_folds: int = 3,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward splits with expanding window.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Available dates.
    train_months : int
        Minimum training period in months.
    validation_months : int
        Validation period in months.
    test_months : int
        Test period in months.
    min_folds : int
        Minimum number of folds required.
    
    Returns
    -------
    list[WalkForwardFold]
        List of fold specifications.
    """
    if len(dates) == 0:
        return []

    dates_sorted = dates.sort_values()
    start = dates_sorted[0]
    end = dates_sorted[-1]

    # Convert months to approximate days
    train_days = train_months * 21
    val_days = validation_months * 21
    test_days = test_months * 21

    fold_step = test_days  # Move forward by test period each fold

    folds = []
    fold_id = 0

    # First split starts after initial training period
    current_train_end = start + pd.Timedelta(days=train_days)

    while True:
        val_end = current_train_end + pd.Timedelta(days=val_days)
        test_end = val_end + pd.Timedelta(days=test_days)

        if test_end > end:
            break

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=start,
            train_end=current_train_end,
            validation_start=current_train_end,
            validation_end=val_end,
            test_start=val_end,
            test_end=test_end,
        )
        folds.append(fold)

        # Expand training window for next fold
        current_train_end = current_train_end + pd.Timedelta(days=fold_step)
        fold_id += 1

    if len(folds) < min_folds:
        logger.warning(
            f"Only {len(folds)} folds generated, minimum {min_folds} required"
        )

    return folds


def split_data(
    data: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data according to fold specification.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with DatetimeIndex.
    fold : WalkForwardFold
        Fold specification.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, test data.
    """
    train = data.loc[fold.train_start:fold.train_end]
    val = data.loc[fold.validation_start:fold.validation_end]
    test = data.loc[fold.test_start:fold.test_end]

    return train, val, test


def compute_baseline_return(
    returns: pd.DataFrame,
    method: str = "equal_weight",
) -> pd.Series:
    """
    Compute baseline portfolio returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    method : str
        "equal_weight", "random_walk", or "momentum_only".
    
    Returns
    -------
    pd.Series
        Baseline portfolio returns.
    """
    n_assets = returns.shape[1]

    if method == "equal_weight":
        # Equal weight rebalanced monthly
        weights = np.ones(n_assets) / n_assets
        return (returns * weights).sum(axis=1)

    elif method == "random_walk":
        # No expected return (μ = 0), equal weight
        weights = np.ones(n_assets) / n_assets
        return (returns * weights).sum(axis=1)

    elif method == "momentum_only":
        # Simple momentum strategy
        portfolio_returns = []
        lookback = 63  # 3 months

        for i in range(lookback, len(returns)):
            past = returns.iloc[i - lookback:i]
            mom = past.sum()

            # Long top half, equal weight
            n_long = max(1, n_assets // 2)
            top_assets = mom.nlargest(n_long).index
            weights = pd.Series(0.0, index=returns.columns)
            weights[top_assets] = 1.0 / n_long

            portfolio_returns.append((returns.iloc[i] * weights).sum())

        return pd.Series(
            portfolio_returns,
            index=returns.index[lookback:],
        )

    else:
        raise ValueError(f"Unknown baseline method: {method}")


def compute_metrics(
    portfolio_returns: pd.Series,
    mu_hat: pd.Series | None = None,
    realized: pd.Series | None = None,
    regime_states: pd.Series | None = None,
) -> dict[str, float]:
    """
    Compute validation metrics.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns.
    mu_hat : pd.Series, optional
        Predicted returns.
    realized : pd.Series, optional
        Realized returns.
    regime_states : pd.Series, optional
        Regime states per date.
    
    Returns
    -------
    dict[str, float]
        Metrics dictionary.
    """
    ann_factor = np.sqrt(252)

    total_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * ann_factor
    sharpe = (
        portfolio_returns.mean() * 252 / (volatility + 1e-8)
        if volatility > 0 else 0.0
    )

    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())

    metrics = {
        "total_return": float(total_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
    }

    # Alpha model metrics
    if mu_hat is not None and realized is not None:
        aligned = pd.concat([mu_hat, realized], axis=1, keys=["pred", "real"])
        aligned = aligned.dropna()

        if len(aligned) > 0:
            mse = ((aligned["pred"] - aligned["real"]) ** 2).mean()
            var_real = aligned["real"].var()
            r2 = 1 - mse / (var_real + 1e-8) if var_real > 0 else 0.0

            # Hit rate: correct direction
            correct = (aligned["pred"] * aligned["real"]) > 0
            hit_rate = correct.mean()

            metrics["mu_hat_mse"] = float(mse)
            metrics["mu_hat_r2"] = float(r2)
            metrics["hit_rate"] = float(hit_rate)

    # Regime breakdown
    if regime_states is not None:
        for regime in ["bull", "bear", "high_vol"]:
            mask = regime_states.str.contains(regime, case=False, na=False)
            if mask.any():
                regime_ret = portfolio_returns[mask].mean() * 252
                metrics[f"{regime}_return"] = float(regime_ret)

    return metrics


def validate_no_lookahead(
    fold: WalkForwardFold,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> bool:
    """
    Verify no lookahead bias.
    
    Parameters
    ----------
    fold : WalkForwardFold
        Fold specification.
    train_data : pd.DataFrame
        Training data used.
    test_data : pd.DataFrame
        Test data.
    
    Returns
    -------
    bool
        True if no lookahead detected.
    """
    if len(train_data) == 0 or len(test_data) == 0:
        return True

    train_last = train_data.index.max()
    test_first = test_data.index.min()

    if train_last >= test_first:
        logger.error(
            f"Lookahead detected! Train ends {train_last}, test starts {test_first}"
        )
        return False

    return True


class WalkForwardValidator:
    """
    Walk-forward validation harness for quant engine.
    """

    def __init__(
        self,
        config: QuantConfig,
    ):
        """
        Initialize validator.
        
        Parameters
        ----------
        config : QuantConfig
            Quant engine configuration.
        """
        self.config = config

    def run_validation(
        self,
        returns: pd.DataFrame,
        model_fn: Callable[[pd.DataFrame], tuple[pd.Series, pd.Series]],
        optimize_fn: Callable[[pd.Series, np.ndarray], np.ndarray],
    ) -> WalkForwardResult:
        """
        Run full walk-forward validation.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (dates x assets).
        model_fn : Callable
            Function that takes training returns and returns (mu_hat, uncertainty).
        optimize_fn : Callable
            Function that takes (mu_hat, current_weights) and returns new weights.
        
        Returns
        -------
        WalkForwardResult
            Validation results.
        """
        folds = generate_walk_forward_splits(
            dates=returns.index,
            train_months=self.config.train_months,
            validation_months=self.config.validation_months,
            test_months=self.config.test_months,
        )

        if len(folds) == 0:
            logger.error("No valid folds generated")
            return WalkForwardResult(
                folds=folds,
                fold_metrics=[],
                aggregate_sharpe=0.0,
                aggregate_return=0.0,
                aggregate_volatility=0.0,
                aggregate_max_drawdown=0.0,
                baseline_sharpe={"equal_weight": 0.0},
                total_turnover=0.0,
                hit_rate=0.0,
                regime_performance={},
            )

        all_portfolio_returns = []
        all_mu_hat = []
        all_realized = []
        fold_metrics_list = []
        total_turnover = 0.0

        # Initial equal weight
        n_assets = returns.shape[1]
        w_current = np.ones(n_assets) / n_assets

        for fold in folds:
            train_data, val_data, test_data = split_data(returns, fold)

            # Verify no lookahead
            if not validate_no_lookahead(fold, train_data, test_data):
                continue

            # Train model on train+val (validation used for hyperparameter selection)
            train_val = pd.concat([train_data, val_data])
            mu_hat, uncertainty = model_fn(train_val)

            # Optimize
            w_new = optimize_fn(mu_hat, w_current)

            # Track turnover
            turnover = np.sum(np.abs(w_new - w_current))
            total_turnover += turnover

            # Evaluate on test period
            test_portfolio_returns = (test_data * w_new).sum(axis=1)

            all_portfolio_returns.append(test_portfolio_returns)

            # Store predictions vs realized for model evaluation
            test_realized = test_data.mean()  # Average return per asset in test
            aligned_mu = mu_hat.reindex(test_realized.index)

            all_mu_hat.append(aligned_mu)
            all_realized.append(test_realized)

            # Compute fold metrics
            fold_met = compute_metrics(
                portfolio_returns=test_portfolio_returns,
                mu_hat=aligned_mu,
                realized=test_realized,
            )
            fold_met["fold_id"] = fold.fold_id
            fold_met["turnover"] = turnover
            fold_metrics_list.append(fold_met)

            # Update weights for next fold
            w_current = w_new

        # Aggregate results
        if len(all_portfolio_returns) > 0:
            combined_returns = pd.concat(all_portfolio_returns)
            agg_metrics = compute_metrics(combined_returns)
        else:
            agg_metrics = {
                "total_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        # Compute baselines
        baselines = {}
        for method in ["equal_weight", "momentum_only"]:
            baseline_ret = compute_baseline_return(returns, method)
            baseline_met = compute_metrics(baseline_ret)
            baselines[method] = baseline_met.get("sharpe_ratio", 0.0)

        # Hit rate across all folds
        if all_mu_hat and all_realized:
            all_pred = pd.concat(all_mu_hat)
            all_real = pd.concat(all_realized)
            aligned = pd.concat([all_pred, all_real], axis=1, keys=["p", "r"]).dropna()
            if len(aligned) > 0:
                hit_rate = ((aligned["p"] * aligned["r"]) > 0).mean()
            else:
                hit_rate = 0.5
        else:
            hit_rate = 0.5

        return WalkForwardResult(
            folds=folds,
            fold_metrics=fold_metrics_list,
            aggregate_sharpe=agg_metrics.get("sharpe_ratio", 0.0),
            aggregate_return=agg_metrics.get("total_return", 0.0),
            aggregate_volatility=agg_metrics.get("volatility", 0.0),
            aggregate_max_drawdown=agg_metrics.get("max_drawdown", 0.0),
            baseline_sharpe=baselines,
            total_turnover=total_turnover,
            hit_rate=hit_rate,
            regime_performance={},  # TODO: implement regime breakdown
        )

    def evaluate_hyperparameters(
        self,
        returns: pd.DataFrame,
        hyperparams: dict[str, Any],
        model_factory: Callable[[dict], Callable],
        optimize_factory: Callable[[dict], Callable],
    ) -> tuple[float, WalkForwardResult]:
        """
        Evaluate a set of hyperparameters.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns.
        hyperparams : dict
            Hyperparameters to evaluate.
        model_factory : Callable
            Factory that creates model_fn from hyperparams.
        optimize_factory : Callable
            Factory that creates optimize_fn from hyperparams.
        
        Returns
        -------
        tuple[float, WalkForwardResult]
            Objective score and full result.
        """
        model_fn = model_factory(hyperparams)
        optimize_fn = optimize_factory(hyperparams)

        result = self.run_validation(returns, model_fn, optimize_fn)

        # Objective: Sharpe ratio minus drawdown penalty
        objective = result.aggregate_sharpe - 0.5 * result.aggregate_max_drawdown

        return objective, result
