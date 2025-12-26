"""
Hyperparameter tuner for the quant engine.

Implements nested walk-forward hyperparameter selection:
1. Outer loop: Walk-forward test folds
2. Inner loop: Walk-forward validation for hyperparameter selection

Tuned parameters:
- H (history window)
- ridge_alpha, lasso_alpha
- dip_k (dip adjustment coefficient)
- n_pca_factors
- lambda_risk
- turnover_penalty
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.types import (
    HyperparameterLog,
    QuantConfig,
)
from app.quant_engine.walk_forward import (
    WalkForwardValidator,
)


logger = logging.getLogger(__name__)


@dataclass
class HyperparameterGrid:
    """Grid of hyperparameters to search."""

    # History window (months)
    history_months: list[int] = None

    # Alpha model regularization
    ridge_alpha: list[float] = None
    lasso_alpha: list[float] = None
    ensemble_method: list[str] = None

    # Dip adjustment
    dip_k: list[float] = None

    # Risk model
    n_pca_factors: list[int] = None
    min_variance_explained: list[float] = None

    # Optimizer
    lambda_risk: list[float] = None
    turnover_penalty: list[float] = None
    max_weight: list[float] = None

    def __post_init__(self):
        """Set defaults."""
        if self.history_months is None:
            self.history_months = [24, 36, 48]
        if self.ridge_alpha is None:
            self.ridge_alpha = [0.1, 1.0, 10.0]
        if self.lasso_alpha is None:
            self.lasso_alpha = [0.001, 0.01, 0.1]
        if self.ensemble_method is None:
            self.ensemble_method = ["inverse_mse", "equal"]
        if self.dip_k is None:
            self.dip_k = [0.0, 0.05, 0.1, 0.2]
        if self.n_pca_factors is None:
            self.n_pca_factors = [3, 5, 10]
        if self.min_variance_explained is None:
            self.min_variance_explained = [0.8, 0.9]
        if self.lambda_risk is None:
            self.lambda_risk = [5.0, 10.0, 20.0]
        if self.turnover_penalty is None:
            self.turnover_penalty = [0.0005, 0.001, 0.002]
        if self.max_weight is None:
            self.max_weight = [0.10, 0.15, 0.20]

    def to_list(self) -> list[dict[str, Any]]:
        """Convert to list of parameter dictionaries."""
        params = {
            "history_months": self.history_months,
            "ridge_alpha": self.ridge_alpha,
            "lasso_alpha": self.lasso_alpha,
            "ensemble_method": self.ensemble_method,
            "dip_k": self.dip_k,
            "n_pca_factors": self.n_pca_factors,
            "lambda_risk": self.lambda_risk,
            "turnover_penalty": self.turnover_penalty,
            "max_weight": self.max_weight,
        }

        keys = list(params.keys())
        values = list(params.values())

        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]

    def get_reduced_grid(self) -> HyperparameterGrid:
        """Get reduced grid for fast search."""
        return HyperparameterGrid(
            history_months=[36],
            ridge_alpha=[1.0],
            lasso_alpha=[0.01],
            ensemble_method=["inverse_mse"],
            dip_k=[0.0, 0.1],
            n_pca_factors=[5],
            lambda_risk=[10.0],
            turnover_penalty=[0.001],
            max_weight=[0.15],
        )


@dataclass
class TuningResult:
    """Result from hyperparameter tuning."""

    best_params: dict[str, Any]
    best_score: float
    all_results: list[HyperparameterLog]
    n_evaluations: int
    tuning_time_seconds: float


class HyperparameterTuner:
    """
    Nested walk-forward hyperparameter tuner.
    """

    def __init__(
        self,
        base_config: QuantConfig,
        grid: HyperparameterGrid | None = None,
        max_evaluations: int = 50,
        random_search: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize tuner.
        
        Parameters
        ----------
        base_config : QuantConfig
            Base configuration.
        grid : HyperparameterGrid
            Parameter grid. If None, use defaults.
        max_evaluations : int
            Maximum parameter combinations to try.
        random_search : bool
            Use random search instead of grid search.
        random_seed : int
            Random seed for reproducibility.
        """
        self.base_config = base_config
        self.grid = grid or HyperparameterGrid()
        self.max_evaluations = max_evaluations
        self.random_search = random_search
        self.rng = np.random.default_rng(random_seed)

    def _sample_params(self, n: int) -> list[dict[str, Any]]:
        """Sample n parameter combinations."""
        all_params = self.grid.to_list()

        if len(all_params) <= n:
            return all_params

        if self.random_search:
            indices = self.rng.choice(len(all_params), size=n, replace=False)
            return [all_params[i] for i in indices]
        else:
            return all_params[:n]

    def _update_config(
        self,
        params: dict[str, Any],
    ) -> QuantConfig:
        """Create new config with updated parameters."""
        # Get current config as dict
        config_dict = {
            "base_currency": self.base_config.base_currency,
            "monthly_inflow_min": self.base_config.monthly_inflow_min,
            "monthly_inflow_max": self.base_config.monthly_inflow_max,
            "max_weight": params.get("max_weight", self.base_config.max_weight),
            "max_turnover": self.base_config.max_turnover,
            "fixed_cost_eur": self.base_config.fixed_cost_eur,
            "min_trade_eur": self.base_config.min_trade_eur,
            "history_days": params.get("history_months", 36) * 21,
            "train_months": self.base_config.train_months,
            "validation_months": self.base_config.validation_months,
            "test_months": self.base_config.test_months,
            "ridge_alpha": params.get("ridge_alpha", self.base_config.ridge_alpha),
            "lasso_alpha": params.get("lasso_alpha", self.base_config.lasso_alpha),
            "ensemble_method": params.get(
                "ensemble_method", self.base_config.ensemble_method
            ),
            "dip_k": params.get("dip_k", self.base_config.dip_k),
            "n_pca_factors": params.get(
                "n_pca_factors", self.base_config.n_pca_factors
            ),
            "min_variance_explained": self.base_config.min_variance_explained,
            "lambda_risk": params.get("lambda_risk", self.base_config.lambda_risk),
            "turnover_penalty": params.get(
                "turnover_penalty", self.base_config.turnover_penalty
            ),
            "shrinkage_factor": self.base_config.shrinkage_factor,
        }

        return QuantConfig(**config_dict)

    def tune(
        self,
        returns: pd.DataFrame,
        model_factory: Callable[[QuantConfig], Callable],
        optimize_factory: Callable[[QuantConfig], Callable],
    ) -> TuningResult:
        """
        Run hyperparameter tuning.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (dates x assets).
        model_factory : Callable
            Factory that creates model_fn from config.
        optimize_factory : Callable
            Factory that creates optimize_fn from config.
        
        Returns
        -------
        TuningResult
            Tuning results.
        """
        import time

        start_time = time.time()

        param_combinations = self._sample_params(self.max_evaluations)
        logger.info(f"Tuning {len(param_combinations)} parameter combinations")

        all_results = []
        best_score = float("-inf")
        best_params = None

        for i, params in enumerate(param_combinations):
            logger.info(f"Evaluating params {i + 1}/{len(param_combinations)}: {params}")

            try:
                config = self._update_config(params)

                validator = WalkForwardValidator(config)

                model_fn = model_factory(config)
                optimize_fn = optimize_factory(config)

                result = validator.run_validation(
                    returns=returns,
                    model_fn=model_fn,
                    optimize_fn=optimize_fn,
                )

                # Objective: Sharpe - drawdown penalty - turnover penalty
                score = (
                    result.aggregate_sharpe
                    - 0.5 * result.aggregate_max_drawdown
                    - 0.1 * result.total_turnover
                )

                log = HyperparameterLog(
                    timestamp=datetime.now(),
                    parameters=params,
                    validation_sharpe=result.aggregate_sharpe,
                    validation_return=result.aggregate_return,
                    validation_volatility=result.aggregate_volatility,
                    validation_max_drawdown=result.aggregate_max_drawdown,
                    baseline_sharpe=result.baseline_sharpe.get("equal_weight", 0.0),
                    selected=False,  # Will update for best
                )
                all_results.append(log)

                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"New best score: {score:.4f}")

            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {e}")
                continue

        # Mark best as selected
        for log in all_results:
            if log.parameters == best_params:
                # Create new log with selected=True
                idx = all_results.index(log)
                all_results[idx] = HyperparameterLog(
                    timestamp=log.timestamp,
                    parameters=log.parameters,
                    validation_sharpe=log.validation_sharpe,
                    validation_return=log.validation_return,
                    validation_volatility=log.validation_volatility,
                    validation_max_drawdown=log.validation_max_drawdown,
                    baseline_sharpe=log.baseline_sharpe,
                    selected=True,
                )

        elapsed = time.time() - start_time

        logger.info(
            f"Tuning complete: best_score={best_score:.4f}, "
            f"best_params={best_params}, elapsed={elapsed:.1f}s"
        )

        return TuningResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=all_results,
            n_evaluations=len(param_combinations),
            tuning_time_seconds=elapsed,
        )

    def tune_dip_k_only(
        self,
        returns: pd.DataFrame,
        dip_scores: pd.DataFrame,
        mu_hat_base: pd.Series,
        k_values: list[float] | None = None,
    ) -> tuple[float, dict[float, float]]:
        """
        Tune only the dip adjustment coefficient k.
        
        Fast tuning for just the dip_k parameter.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns.
        dip_scores : pd.DataFrame
            DipScore values per asset per date.
        mu_hat_base : pd.Series
            Base expected returns (without dip adjustment).
        k_values : list[float]
            Values of k to try.
        
        Returns
        -------
        tuple[float, dict[float, float]]
            Best k and score for each k.
        """
        if k_values is None:
            k_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

        scores = {}

        for k in k_values:
            # Adjust mu_hat by dip score
            # μ_hat_adj = μ_hat + k * max(0, -DipScore)
            if k == 0:
                mu_hat_adj = mu_hat_base
            else:
                latest_dip = dip_scores.iloc[-1] if len(dip_scores) > 0 else pd.Series(dtype=float)
                dip_boost = k * (-latest_dip).clip(lower=0)
                mu_hat_adj = mu_hat_base + dip_boost.reindex(mu_hat_base.index, fill_value=0)

            # Simple forward return correlation
            forward_returns = returns.mean()  # Average return
            corr = mu_hat_adj.corr(forward_returns)

            scores[k] = corr if not np.isnan(corr) else 0.0

        best_k = max(scores, key=scores.get)

        logger.info(f"Dip k tuning: best_k={best_k}, scores={scores}")

        return best_k, scores


def create_default_grid() -> HyperparameterGrid:
    """Create default hyperparameter grid."""
    return HyperparameterGrid()


def create_fast_grid() -> HyperparameterGrid:
    """Create fast hyperparameter grid for testing."""
    return HyperparameterGrid(
        history_months=[36],
        ridge_alpha=[1.0],
        lasso_alpha=[0.01],
        ensemble_method=["inverse_mse"],
        dip_k=[0.0, 0.1],
        n_pca_factors=[5],
        lambda_risk=[10.0],
        turnover_penalty=[0.001],
        max_weight=[0.15],
    )
