"""
Main Quant Engine service orchestration.

This is the primary entry point for the quantitative portfolio engine.
It coordinates:
- Data fetching
- Feature engineering
- Alpha model training/prediction
- Dip score computation
- Risk model fitting
- Portfolio optimization
- Recommendation generation

All decisions come from explicit mathematics, never ad-hoc triggers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.alpha_models import (
    AlphaModelEnsemble,
    apply_dip_adjustment,
    shrink_mu_hat,
)
from app.quant_engine.dip import (
    compute_dip_score,
    get_dip_bucket,
)
from app.quant_engine.features import (
    compute_all_features,
    prepare_alpha_training_data,
)
from app.quant_engine.optimizer import (
    optimize_portfolio,
)
from app.quant_engine.regimes import (
    compute_regime_state,
)
from app.quant_engine.risk import (
    fit_pca_risk_model,
)
from app.quant_engine.tuner import (
    HyperparameterGrid,
    HyperparameterTuner,
    create_fast_grid,
)
from app.quant_engine.types import (
    ActionType,
    AuditBlock,
    DipAnnotation,
    DipArtifacts,
    EngineOutput,
    MomentumCondition,
    MuHatUncertainty,
    OptimizationResult,
    QuantConfig,
    RecommendationRow,
    RegimeState,
    RiskInfo,
    RiskModel,
)
from app.quant_engine.walk_forward import (
    WalkForwardValidator,
)


logger = logging.getLogger(__name__)


def get_default_config() -> QuantConfig:
    """Get default quant engine configuration."""
    return QuantConfig(
        base_currency="EUR",
        inflow_min_eur=1000.0,
        inflow_max_eur=1500.0,
        max_weight=0.15,
        max_turnover=0.20,
        fixed_cost_eur=1.0,
        min_trade_eur=10.0,
        train_months=12,  # Reduced from 36 to work with ~1 year of data
        validation_months=6,
        test_months=6,
        ridge_alpha=1.0,
        lasso_alpha=0.01,
        ensemble_method="inverse_mse",
        dip_k=0.05,
        n_pca_factors=5,
        lambda_risk=10.0,
        turnover_penalty=0.001,
    )


class QuantEngineService:
    """
    Main quant engine service.
    
    Orchestrates all components to generate portfolio recommendations.
    """

    def __init__(
        self,
        config: QuantConfig | None = None,
    ):
        """
        Initialize the quant engine.
        
        Parameters
        ----------
        config : QuantConfig, optional
            Engine configuration. Uses defaults if not provided.
        """
        self.config = config or get_default_config()
        self.alpha_model: AlphaModelEnsemble | None = None
        self.risk_model: RiskModel | None = None
        self._last_training_date: datetime | None = None

    def _needs_retraining(self, current_date: datetime) -> bool:
        """Check if models need retraining."""
        if self._last_training_date is None:
            return True

        # Retrain monthly
        days_since_training = (current_date - self._last_training_date).days
        return days_since_training >= 30

    def train(
        self,
        prices: pd.DataFrame,
        market_prices: pd.Series | None = None,
        as_of_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Train alpha and risk models.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices (dates x assets).
        market_prices : pd.Series, optional
            Market benchmark prices for dip calculation.
        as_of_date : datetime, optional
            Training cutoff date.
        
        Returns
        -------
        dict
            Training summary.
        """
        if as_of_date is None:
            as_of_date = prices.index.max()

        # Use data up to as_of_date only
        prices_train = prices.loc[:as_of_date]

        # Compute returns
        returns = prices_train.pct_change().dropna()

        # Minimum history: train_months * 21 trading days
        min_history = self.config.train_months * 21 // 2
        if len(returns) < min_history:
            logger.warning("Insufficient history for training")
            return {"status": "error", "message": "Insufficient history"}

        # Feature engineering
        features = compute_all_features(
            prices=prices_train,
            returns=returns,
            momentum_windows=self.config.momentum_windows,
            volatility_window=self.config.volatility_window,
            reversal_window=self.config.reversal_window,
        )

        # Prepare training data
        X, y = prepare_alpha_training_data(
            features=features,
            returns=returns,
            forecast_horizon_months=self.config.forecast_horizon_months,
        )

        if len(X) < 100:
            logger.warning("Insufficient training samples")
            return {"status": "error", "message": "Insufficient training samples"}

        # Split into train and validation
        val_split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:val_split], X.iloc[val_split:]
        y_train, y_val = y.iloc[:val_split], y.iloc[val_split:]

        # Train alpha model
        self.alpha_model = AlphaModelEnsemble(
            ridge_alpha=self.config.ridge_alpha,
            lasso_alpha=self.config.lasso_alpha,
            use_lasso=self.config.use_lasso,
            ensemble_method=self.config.ensemble_method,
        )

        oos_scores = self.alpha_model.fit(X_train, y_train, X_val, y_val)

        # Fit risk model
        self.risk_model = fit_pca_risk_model(
            returns=returns.iloc[-min(252, len(returns)):],
            n_factors=self.config.n_pca_factors,
        )

        self._last_training_date = as_of_date

        logger.info(
            f"Training complete: {len(X)} samples, "
            f"weights={self.alpha_model.weights}"
        )

        return {
            "status": "success",
            "training_samples": len(X),
            "oos_scores": oos_scores,
            "risk_model_factors": self.risk_model.n_factors,
            "as_of_date": as_of_date.isoformat(),
        }

    def predict(
        self,
        prices: pd.DataFrame,
        market_prices: pd.Series | None = None,
        as_of_date: datetime | None = None,
    ) -> tuple[pd.Series, pd.Series, DipArtifacts | None]:
        """
        Generate expected return predictions.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices.
        market_prices : pd.Series, optional
            Market benchmark for dip calculation.
        as_of_date : datetime, optional
            Prediction date.
        
        Returns
        -------
        tuple[pd.Series, pd.Series, DipArtifacts]
            Expected returns, uncertainty, dip artifacts.
        """
        if self.alpha_model is None:
            raise ValueError("Model not trained. Call train() first.")

        if as_of_date is None:
            as_of_date = prices.index.max()

        # Use data up to as_of_date
        prices_pred = prices.loc[:as_of_date]
        returns = prices_pred.pct_change().dropna()

        # Compute features for latest date
        features = compute_all_features(
            prices=prices_pred,
            returns=returns,
            momentum_windows=self.config.momentum_windows,
            volatility_window=self.config.volatility_window,
            reversal_window=self.config.reversal_window,
        )

        # Get feature values for latest date (per asset)
        feature_df = features.to_dataframe()
        latest_date = feature_df.index.get_level_values("date").max()
        X_latest = feature_df.loc[latest_date].dropna()

        if X_latest.empty:
            raise ValueError("No valid features available for prediction")

        # Predict
        alpha_result = self.alpha_model.predict(X_latest)

        # Convert to Series indexed by asset
        mu_hat_series = alpha_result.mu_hat
        uncertainty_series = alpha_result.uncertainty

        # Compute dip scores
        dip_artifacts = None
        if market_prices is not None and len(market_prices) > 0:
            try:
                market_returns = market_prices.pct_change().dropna()
                # Convert market_returns to DataFrame if it's a Series
                if isinstance(market_returns, pd.Series):
                    market_returns = market_returns.to_frame(name="market")
                    
                dip_artifacts = compute_dip_score(
                    asset_returns=returns,
                    factor_returns=market_returns,
                    resid_vol_window=self.config.dip_resid_vol_window,
                    min_obs=self.config.dip_min_obs,
                )

                # Apply dip adjustment to mu_hat
                if self.config.dip_k > 0 and len(dip_artifacts.dip_score) > 0:
                    latest_dip = dip_artifacts.dip_score.iloc[-1]
                    mu_hat_series = apply_dip_adjustment(
                        mu_hat=mu_hat_series,
                        dip_scores=latest_dip,
                        dip_k=self.config.dip_k,
                    )
            except Exception as e:
                logger.warning(f"Failed to compute dip scores: {e}")

        # Note: Shrinkage is already applied in AlphaModelEnsemble.predict()

        return mu_hat_series, uncertainty_series, dip_artifacts

    def optimize(
        self,
        w_current: np.ndarray,
        mu_hat: pd.Series,
        portfolio_value_eur: float,
        inflow_eur: float,
    ) -> OptimizationResult:
        """
        Optimize portfolio allocation.
        
        Parameters
        ----------
        w_current : np.ndarray
            Current weights.
        mu_hat : pd.Series
            Expected returns.
        portfolio_value_eur : float
            Portfolio value in EUR.
        inflow_eur : float
            Monthly inflow in EUR.
        
        Returns
        -------
        OptimizationResult
            Optimization result.
        """
        if self.risk_model is None:
            raise ValueError("Risk model not fitted. Call train() first.")

        return optimize_portfolio(
            w_current=w_current,
            mu_hat=mu_hat,
            risk_model=self.risk_model,
            inflow_eur=inflow_eur,
            portfolio_value_eur=portfolio_value_eur,
            lambda_risk=self.config.lambda_risk,
            max_weight=self.config.max_weight,
            max_turnover=self.config.max_turnover,
            turnover_penalty=self.config.turnover_penalty,
            fixed_cost_eur=self.config.fixed_cost_eur,
            min_trade_eur=self.config.min_trade_eur,
        )

    def generate_recommendations(
        self,
        prices: pd.DataFrame,
        w_current: np.ndarray,
        portfolio_value_eur: float,
        inflow_eur: float,
        market_prices: pd.Series | None = None,
        as_of_date: datetime | None = None,
        retrain: bool = False,
    ) -> EngineOutput:
        """
        Generate full portfolio recommendations.
        
        This is the main entry point for the API.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices (dates x assets).
        w_current : np.ndarray
            Current portfolio weights.
        portfolio_value_eur : float
            Current portfolio value in EUR.
        inflow_eur : float
            Monthly inflow amount in EUR.
        market_prices : pd.Series, optional
            Market benchmark prices.
        as_of_date : datetime, optional
            Date for recommendations.
        retrain : bool
            Force model retraining.
        
        Returns
        -------
        EngineOutput
            Complete recommendation output with audit block.
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        # Check if retraining needed
        if retrain or self._needs_retraining(as_of_date):
            train_result = self.train(
                prices=prices,
                market_prices=market_prices,
                as_of_date=as_of_date,
            )

            if train_result.get("status") != "success":
                # Return empty recommendations on training failure
                return self._create_error_output(
                    message=train_result.get("message", "Training failed"),
                    as_of_date=as_of_date,
                )

        # Generate predictions
        mu_hat, uncertainty, dip_artifacts = self.predict(
            prices=prices,
            market_prices=market_prices,
            as_of_date=as_of_date,
        )

        # Optimize
        opt_result = self.optimize(
            w_current=w_current,
            mu_hat=mu_hat,
            portfolio_value_eur=portfolio_value_eur,
            inflow_eur=inflow_eur,
        )

        # Compute regime
        returns = prices.pct_change().dropna()
        if market_prices is not None and len(market_prices) > 0:
            market_returns = market_prices.pct_change().dropna()
        else:
            market_returns = returns.mean(axis=1)

        regime = compute_regime_state(
            market_returns=market_returns,
            as_of=as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date,
        )

        # Generate recommendations from Δw
        recommendations = self._create_recommendations(
            opt_result=opt_result,
            mu_hat=mu_hat,
            uncertainty=uncertainty,
            dip_artifacts=dip_artifacts,
            portfolio_value_eur=portfolio_value_eur,
            regime=regime,
        )

        # Create audit block
        audit = self._create_audit_block(
            opt_result=opt_result,
            mu_hat=mu_hat,
            dip_artifacts=dip_artifacts,
            regime=regime,
            as_of_date=as_of_date,
        )

        return EngineOutput(
            recommendations=recommendations,
            as_of=as_of_date.date() if isinstance(as_of_date, datetime) else as_of_date,
            portfolio_value_eur=portfolio_value_eur,
            inflow_eur=inflow_eur,
            solver_status=opt_result.status,
            total_trades=len([r for r in recommendations if r.action != ActionType.HOLD]),
            total_transaction_cost_eur=opt_result.transaction_cost_eur,
            expected_portfolio_return=float(
                (mu_hat * pd.Series(opt_result.w_new, index=opt_result.assets)).sum()
            ),
            expected_portfolio_risk=float(np.sqrt(
                self.risk_model.portfolio_variance(opt_result.w_new)
            )) if self.risk_model else 0.0,
            audit=audit,
        )

    def _create_recommendations(
        self,
        opt_result: OptimizationResult,
        mu_hat: pd.Series,
        uncertainty: pd.Series,
        dip_artifacts: DipArtifacts | None,
        portfolio_value_eur: float,
        regime: RegimeState | None = None,
    ) -> list[RecommendationRow]:
        """Create recommendation rows from optimization result."""
        recommendations = []

        assets = opt_result.assets
        dw = opt_result.dw
        w_new = opt_result.w_new

        # Get dip scores for annotation
        dip_scores = {}
        dip_buckets = {}
        if dip_artifacts is not None and len(dip_artifacts.dip_score) > 0:
            latest_dip = dip_artifacts.dip_score.iloc[-1]
            for asset in assets:
                if asset in latest_dip.index:
                    dip_scores[asset] = latest_dip[asset]
                    dip_buckets[asset] = get_dip_bucket(latest_dip[asset])

        # Get MCRs once if risk model exists
        mcrs = None
        if self.risk_model is not None:
            mcrs = self.risk_model.marginal_contribution_to_risk(w_new)

        for i, asset in enumerate(assets):
            delta_w = dw[i]
            notional_eur = delta_w * portfolio_value_eur

            # Determine action
            if abs(notional_eur) < self.config.min_trade_eur:
                action = ActionType.HOLD
                notional_eur = 0.0
            elif delta_w > 0:
                action = ActionType.BUY
            else:
                action = ActionType.SELL
                notional_eur = abs(notional_eur)  # Report as positive

            # Get asset-level stats
            asset_mu = mu_hat.get(asset, 0.0) if isinstance(mu_hat, pd.Series) else mu_hat[i]
            asset_uncertainty = (
                uncertainty.get(asset, 0.0)
                if isinstance(uncertainty, pd.Series)
                else uncertainty[i]
            )

            # Risk info
            if mcrs is not None:
                mcr_val = mcrs[i]
                marginal_vol = mcr_val / max(self.risk_model.portfolio_volatility(w_new), 1e-8)
            else:
                mcr_val = 0.0
                marginal_vol = 0.0

            risk_info = RiskInfo(
                marginal_vol=marginal_vol,
                mcr=mcr_val,
            )

            # Uncertainty info
            mu_uncertainty = MuHatUncertainty(
                ci_low=asset_mu - 1.96 * asset_uncertainty,
                ci_high=asset_mu + 1.96 * asset_uncertainty,
                oos_rmse=asset_uncertainty,
            )

            # Transaction cost
            trade_cost = self.config.fixed_cost_eur if action != ActionType.HOLD else 0.0

            # Delta utility net of costs
            delta_utility_net = opt_result.marginal_utilities[i] - trade_cost / max(portfolio_value_eur, 1.0)

            # Dip annotation
            dip_annotation = None
            if asset in dip_scores:
                dip_annotation = DipAnnotation(
                    dip_score=dip_scores[asset],
                    bucket=dip_buckets.get(asset, "neutral"),
                    regime=regime if regime else RegimeState.neutral_medium(),
                    momentum_12m=MomentumCondition.NEUTRAL,  # TODO: compute from data
                )

            rec = RecommendationRow(
                ticker=asset,
                name=None,  # TODO: get from symbol service
                action=action,
                notional_eur=notional_eur,
                delta_weight=delta_w,
                mu_hat=asset_mu,
                mu_hat_uncertainty=mu_uncertainty,
                risk=risk_info,
                delta_utility_net=delta_utility_net,
                trade_cost_eur=trade_cost,
                constraints=[],  # TODO: populate from optimizer
                dip=dip_annotation,
            )
            recommendations.append(rec)

        # Sort by absolute notional (most impactful first)
        recommendations.sort(key=lambda r: abs(r.notional_eur), reverse=True)

        return recommendations

    def _create_audit_block(
        self,
        opt_result: OptimizationResult,
        mu_hat: pd.Series,
        dip_artifacts: DipArtifacts | None,
        regime: RegimeState,
        as_of_date: datetime,
    ) -> AuditBlock:
        """Create audit block for transparency."""
        return AuditBlock(
            timestamp=as_of_date,
            config_hash=hash(str(self.config)),
            mu_hat_summary={
                "mean": float(mu_hat.mean()),
                "std": float(mu_hat.std()),
                "min": float(mu_hat.min()),
                "max": float(mu_hat.max()),
            },
            risk_model_summary={
                "n_factors": self.risk_model.n_factors if self.risk_model else 0,
                "explained_variance": (
                    float(self.risk_model.explained_variance.sum())
                    if self.risk_model else 0.0
                ),
            },
            optimizer_status=opt_result.status.value,
            constraint_binding=opt_result.constraint_status.max_weight_binding,
            turnover_realized=float(np.sum(np.abs(opt_result.dw))),
            regime_state=f"{regime.trend.value}_{regime.volatility.value}",
            dip_stats={
                "n_dipped": sum(
                    1 for r in dip_artifacts.dip_score.iloc[-1].values
                    if r < -1.0
                ) if dip_artifacts and len(dip_artifacts.dip_score) > 0 else 0,
            } if dip_artifacts else None,
        )

    def _create_error_output(
        self,
        message: str,
        as_of_date: datetime,
    ) -> EngineOutput:
        """Create error output."""
        return EngineOutput(
            recommendations=[],
            as_of_date=as_of_date,
            portfolio_value_eur=0.0,
            inflow_eur=0.0,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=AuditBlock(
                timestamp=as_of_date,
                config_hash=0,
                mu_hat_summary={},
                risk_model_summary={},
                optimizer_status="error",
                constraint_binding=[],
                turnover_realized=0.0,
                regime_state="unknown",
                dip_stats=None,
                error_message=message,
            ),
        )

    def validate_walk_forward(
        self,
        prices: pd.DataFrame,
        market_prices: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Run walk-forward validation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices.
        market_prices : pd.Series, optional
            Market benchmark.
        
        Returns
        -------
        dict
            Validation results.
        """
        returns = prices.pct_change().dropna()

        def model_fn(train_returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
            """Create model function for validation."""
            # Fit on training data - construct prices from cumulative returns
            train_prices = (1 + train_returns).cumprod()
            features = compute_all_features(
                prices=train_prices,
                returns=train_returns,
                momentum_windows=self.config.momentum_windows,
                volatility_window=self.config.volatility_window,
                reversal_window=self.config.reversal_window,
            )
            X, y = prepare_alpha_training_data(
                features=features,
                returns=train_returns,
                forecast_horizon_months=self.config.forecast_horizon_months,
            )

            if len(X) < 50:
                # Not enough data, return zeros
                return (
                    pd.Series(0.0, index=train_returns.columns),
                    pd.Series(1.0, index=train_returns.columns),
                )

            model = AlphaModelEnsemble(
                ridge_alpha=self.config.ridge_alpha,
                lasso_alpha=self.config.lasso_alpha,
            )
            model.fit(X, y)

            # Predict on latest
            feature_df = features.to_dataframe()
            latest_date = feature_df.index.get_level_values("date").max()
            X_latest = feature_df.loc[latest_date].dropna()
            alpha_result = model.predict(X_latest)

            return (alpha_result.mu_hat, alpha_result.uncertainty)

        def optimize_fn(
            mu_hat: pd.Series,
            w_current: np.ndarray,
        ) -> np.ndarray:
            """Create optimize function for validation."""
            n = len(mu_hat)

            # Simple equal risk optimization for validation
            # (full optimization would need covariance)

            # Score by mu_hat, allocate proportionally
            scores = mu_hat.clip(lower=0)
            if scores.sum() > 0:
                weights = scores / scores.sum()
            else:
                weights = pd.Series(1.0 / n, index=mu_hat.index)

            # Apply max weight constraint
            weights = weights.clip(upper=self.config.max_weight)
            weights = weights / weights.sum()  # Renormalize

            return weights.values

        validator = WalkForwardValidator(self.config)
        result = validator.run_validation(returns, model_fn, optimize_fn)

        return {
            "n_folds": len(result.folds),
            "aggregate_sharpe": result.aggregate_sharpe,
            "aggregate_return": result.aggregate_return,
            "aggregate_volatility": result.aggregate_volatility,
            "max_drawdown": result.aggregate_max_drawdown,
            "baseline_sharpe": result.baseline_sharpe,
            "hit_rate": result.hit_rate,
            "total_turnover": result.total_turnover,
        }

    def tune_hyperparameters(
        self,
        prices: pd.DataFrame,
        fast: bool = True,
    ) -> dict[str, Any]:
        """
        Tune hyperparameters.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices.
        fast : bool
            Use reduced grid for fast tuning.
        
        Returns
        -------
        dict
            Tuning results with best parameters.
        """
        returns = prices.pct_change().dropna()

        grid = create_fast_grid() if fast else HyperparameterGrid()

        tuner = HyperparameterTuner(
            base_config=self.config,
            grid=grid,
            max_evaluations=10 if fast else 50,
        )

        def model_factory(config: QuantConfig) -> callable:
            def model_fn(train_returns: pd.DataFrame):
                # Construct prices from cumulative returns
                train_prices = (1 + train_returns).cumprod()
                features = compute_all_features(
                    prices=train_prices,
                    returns=train_returns,
                    momentum_windows=config.momentum_windows,
                    volatility_window=config.volatility_window,
                    reversal_window=config.reversal_window,
                )
                X, y = prepare_alpha_training_data(
                    features=features,
                    returns=train_returns,
                    forecast_horizon_months=config.forecast_horizon_months,
                )

                if len(X) < 50:
                    return (
                        pd.Series(0.0, index=train_returns.columns),
                        pd.Series(1.0, index=train_returns.columns),
                    )

                model = AlphaModelEnsemble(
                    ridge_alpha=config.ridge_alpha,
                    lasso_alpha=config.lasso_alpha,
                    ensemble_method=config.ensemble_method,
                )
                model.fit(X, y)

                feature_df = features.to_dataframe()
                latest_date = feature_df.index.get_level_values("date").max()
                X_latest = feature_df.loc[latest_date].dropna()
                alpha_result = model.predict(X_latest)

                return (alpha_result.mu_hat, alpha_result.uncertainty)

            return model_fn

        def optimize_factory(config: QuantConfig) -> callable:
            def optimize_fn(mu_hat: pd.Series, w_current: np.ndarray):
                scores = mu_hat.clip(lower=0)
                if scores.sum() > 0:
                    weights = scores / scores.sum()
                else:
                    weights = pd.Series(1.0 / len(mu_hat), index=mu_hat.index)

                weights = weights.clip(upper=config.max_weight)
                weights = weights / weights.sum()

                return weights.values

            return optimize_fn

        result = tuner.tune(returns, model_factory, optimize_factory)

        return {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_evaluations": result.n_evaluations,
            "tuning_time_seconds": result.tuning_time_seconds,
        }
