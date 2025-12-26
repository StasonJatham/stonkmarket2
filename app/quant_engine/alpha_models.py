"""
Alpha models for expected return estimation.

Implements Ridge/Lasso ensemble with uncertainty quantification.
All model selection is done via out-of-sample validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app.quant_engine.types import AlphaResult, AlphaModelScore

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Container for a trained alpha model."""
    name: str
    model: Any
    scaler: StandardScaler
    feature_names: list[str]
    train_end_date: str
    oos_score: AlphaModelScore | None = None


def train_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 10.0,
    model_name: str | None = None,
) -> TrainedModel:
    """
    Train Ridge regression model.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × features).
    y : pd.Series
        Target variable (forward returns).
    alpha : float
        Regularization strength.
    model_name : str, optional
        Model name for logging.
    
    Returns
    -------
    TrainedModel
        Trained model container.
    """
    name = model_name or f"ridge:{alpha}"
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    # Train model
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_scaled, y.values)
    
    # Get training end date
    if isinstance(X.index, pd.MultiIndex):
        dates = X.index.get_level_values("date")
        train_end = str(max(dates))
    else:
        train_end = str(X.index.max())
    
    return TrainedModel(
        name=name,
        model=model,
        scaler=scaler,
        feature_names=list(X.columns),
        train_end_date=train_end,
    )


def train_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.01,
    model_name: str | None = None,
) -> TrainedModel:
    """
    Train Lasso regression model.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    alpha : float
        Regularization strength.
    model_name : str, optional
        Model name for logging.
    
    Returns
    -------
    TrainedModel
        Trained model container.
    """
    name = model_name or f"lasso:{alpha}"
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
    model.fit(X_scaled, y.values)
    
    if isinstance(X.index, pd.MultiIndex):
        dates = X.index.get_level_values("date")
        train_end = str(max(dates))
    else:
        train_end = str(X.index.max())
    
    return TrainedModel(
        name=name,
        model=model,
        scaler=scaler,
        feature_names=list(X.columns),
        train_end_date=train_end,
    )


def predict(
    model: TrainedModel,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Generate predictions from a trained model.
    
    Parameters
    ----------
    model : TrainedModel
        Trained model.
    X : pd.DataFrame
        Feature matrix for prediction.
    
    Returns
    -------
    pd.Series
        Predictions indexed by X.index.
    """
    # Ensure feature alignment
    X_aligned = X[model.feature_names].copy()
    
    # Scale features
    X_scaled = model.scaler.transform(X_aligned.values)
    
    # Predict
    preds = model.model.predict(X_scaled)
    
    return pd.Series(preds, index=X.index, name=f"pred_{model.name}")


def compute_oos_score(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> AlphaModelScore:
    """
    Compute out-of-sample performance metrics.
    
    Parameters
    ----------
    y_true : pd.Series
        Actual forward returns.
    y_pred : pd.Series
        Predicted returns.
    model_name : str
        Model name.
    
    Returns
    -------
    AlphaModelScore
        Performance metrics.
    """
    # Align
    aligned = pd.concat([y_true, y_pred], axis=1, join="inner")
    aligned.columns = ["true", "pred"]
    aligned = aligned.dropna()
    
    if len(aligned) < 10:
        return AlphaModelScore(
            model_name=model_name,
            mse=np.nan,
            rmse=np.nan,
            r2=np.nan,
            n_samples=len(aligned),
        )
    
    residuals = aligned["true"] - aligned["pred"]
    mse = float((residuals ** 2).mean())
    rmse = np.sqrt(mse)
    
    ss_tot = ((aligned["true"] - aligned["true"].mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
    
    # Sharpe of forecast (correlation-based approximation)
    if aligned["pred"].std() > 1e-12:
        corr = aligned["true"].corr(aligned["pred"])
        sharpe = corr * np.sqrt(252 / 21)  # Approximate monthly to annual
    else:
        sharpe = None
    
    return AlphaModelScore(
        model_name=model_name,
        mse=mse,
        rmse=rmse,
        r2=r2,
        n_samples=len(aligned),
        sharpe_forecast=float(sharpe) if sharpe is not None else None,
    )


def train_and_validate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    alpha: float,
) -> tuple[TrainedModel, AlphaModelScore]:
    """
    Train model and compute OOS validation score.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation target.
    model_type : str
        "ridge" or "lasso".
    alpha : float
        Regularization strength.
    
    Returns
    -------
    tuple[TrainedModel, AlphaModelScore]
        Trained model and OOS score.
    """
    if model_type == "ridge":
        trained = train_ridge(X_train, y_train, alpha=alpha)
    elif model_type == "lasso":
        trained = train_lasso(X_train, y_train, alpha=alpha)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Validate
    y_pred = predict(trained, X_val)
    score = compute_oos_score(y_val, y_pred, trained.name)
    trained.oos_score = score
    
    return trained, score


def compute_ensemble_weights(
    scores: dict[str, AlphaModelScore],
    method: str = "inverse_mse",
) -> dict[str, float]:
    """
    Compute ensemble weights from OOS scores.
    
    Parameters
    ----------
    scores : dict[str, AlphaModelScore]
        OOS scores per model.
    method : str
        "inverse_mse" or "equal".
    
    Returns
    -------
    dict[str, float]
        Ensemble weights (sum to 1).
    """
    if method == "equal":
        n = len(scores)
        return {name: 1.0 / n for name in scores}
    
    elif method == "inverse_mse":
        # Use inverse MSE as weights (lower MSE = higher weight)
        inv_mse = {}
        for name, score in scores.items():
            if np.isfinite(score.mse) and score.mse > 1e-12:
                inv_mse[name] = 1.0 / score.mse
            else:
                inv_mse[name] = 1e-6  # Small weight for invalid scores
        
        total = sum(inv_mse.values())
        if total < 1e-12:
            # Fallback to equal weights
            n = len(scores)
            return {name: 1.0 / n for name in scores}
        
        return {name: w / total for name, w in inv_mse.items()}
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def compute_uncertainty(
    models: list[TrainedModel],
    X: pd.DataFrame,
    oos_residuals: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """
    Compute forecast uncertainty per asset.
    
    Uses OOS residual standard deviation as uncertainty estimate.
    
    Parameters
    ----------
    models : list[TrainedModel]
        Trained models.
    X : pd.DataFrame
        Feature matrix (for index alignment).
    oos_residuals : dict[str, pd.Series], optional
        OOS residuals per model.
    
    Returns
    -------
    pd.Series
        Uncertainty estimate per asset.
    """
    if oos_residuals is None:
        # Use model OOS scores as fallback
        avg_rmse = np.mean([
            m.oos_score.rmse 
            for m in models 
            if m.oos_score and np.isfinite(m.oos_score.rmse)
        ])
        if not np.isfinite(avg_rmse):
            avg_rmse = 0.1  # Default 10% uncertainty
        
        if isinstance(X.index, pd.MultiIndex):
            assets = X.index.get_level_values("asset").unique()
        else:
            assets = X.index
        
        return pd.Series(avg_rmse, index=assets, name="uncertainty")
    
    # Combine residuals from all models
    all_resid = pd.concat(list(oos_residuals.values()), axis=1)
    uncertainty = all_resid.std(axis=1)
    
    # Group by asset if multi-indexed
    if isinstance(uncertainty.index, pd.MultiIndex):
        uncertainty = uncertainty.groupby(level="asset").mean()
    
    return uncertainty


def shrink_mu_hat(
    mu_hat: pd.Series,
    uncertainty: pd.Series,
    shrinkage_strength: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Shrink expected returns toward zero based on uncertainty.
    
    Higher uncertainty = more shrinkage toward zero.
    
    μ_shrunk = μ_hat * (1 - shrinkage_factor)
    shrinkage_factor = min(1, shrinkage_strength * uncertainty / max(uncertainty))
    
    Parameters
    ----------
    mu_hat : pd.Series
        Raw expected return estimates.
    uncertainty : pd.Series
        Uncertainty per asset.
    shrinkage_strength : float
        How aggressively to shrink.
    
    Returns
    -------
    tuple[pd.Series, pd.Series]
        Shrunk mu_hat and shrinkage factors applied.
    """
    # Align
    mu_hat, uncertainty = mu_hat.align(uncertainty, join="inner")
    
    max_unc = uncertainty.max()
    if max_unc < 1e-12:
        return mu_hat, pd.Series(0.0, index=mu_hat.index)
    
    shrinkage_factor = (shrinkage_strength * uncertainty / max_unc).clip(upper=1.0)
    mu_shrunk = mu_hat * (1 - shrinkage_factor)
    
    return mu_shrunk, shrinkage_factor


def apply_dip_adjustment(
    mu_hat: pd.Series,
    dip_scores: pd.Series,
    dip_k: float,
) -> pd.Series:
    """
    Adjust expected returns based on DipScore.
    
    μ_hat_adjusted = μ_hat + k * max(0, -DipScore)
    
    Negative DipScore = underperformance vs factor model = potential mean reversion.
    
    Parameters
    ----------
    mu_hat : pd.Series
        Expected returns.
    dip_scores : pd.Series
        DipScore per asset.
    dip_k : float
        Dip coefficient (tuned OOS).
    
    Returns
    -------
    pd.Series
        Adjusted expected returns.
    """
    if dip_k <= 0:
        return mu_hat
    
    mu_hat, dip_scores = mu_hat.align(dip_scores, join="left", fill_value=0)
    
    # Only negative DipScore contributes (underperformance)
    dip_contribution = dip_k * np.maximum(0, -dip_scores)
    
    return mu_hat + dip_contribution


class AlphaModelEnsemble:
    """
    Ensemble of alpha models for expected return estimation.
    
    Combines Ridge and optionally Lasso models with weights
    determined by out-of-sample performance.
    """
    
    def __init__(
        self,
        ridge_alpha: float = 10.0,
        lasso_alpha: float | None = None,
        use_lasso: bool = False,
        ensemble_method: str = "inverse_mse",
        shrinkage_strength: float = 2.0,
    ):
        """
        Initialize ensemble.
        
        Parameters
        ----------
        ridge_alpha : float
            Ridge regularization strength.
        lasso_alpha : float, optional
            Lasso regularization strength.
        use_lasso : bool
            Whether to include Lasso in ensemble.
        ensemble_method : str
            "inverse_mse" or "equal".
        shrinkage_strength : float
            Shrinkage toward zero strength.
        """
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.use_lasso = use_lasso
        self.ensemble_method = ensemble_method
        self.shrinkage_strength = shrinkage_strength
        
        self.models: dict[str, TrainedModel] = {}
        self.weights: dict[str, float] = {}
        self.oos_residuals: dict[str, pd.Series] = {}
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, AlphaModelScore]:
        """
        Fit all models and compute ensemble weights.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_val : pd.DataFrame
            Validation features.
        y_val : pd.Series
            Validation target.
        
        Returns
        -------
        dict[str, AlphaModelScore]
            OOS scores per model.
        """
        scores = {}
        
        # Train Ridge
        ridge, ridge_score = train_and_validate_model(
            X_train, y_train, X_val, y_val,
            model_type="ridge",
            alpha=self.ridge_alpha,
        )
        self.models[ridge.name] = ridge
        scores[ridge.name] = ridge_score
        
        # Compute residuals for uncertainty
        y_pred_ridge = predict(ridge, X_val)
        self.oos_residuals[ridge.name] = y_val - y_pred_ridge
        
        # Train Lasso if enabled
        if self.use_lasso and self.lasso_alpha is not None:
            lasso, lasso_score = train_and_validate_model(
                X_train, y_train, X_val, y_val,
                model_type="lasso",
                alpha=self.lasso_alpha,
            )
            self.models[lasso.name] = lasso
            scores[lasso.name] = lasso_score
            
            y_pred_lasso = predict(lasso, X_val)
            self.oos_residuals[lasso.name] = y_val - y_pred_lasso
        
        # Compute ensemble weights
        self.weights = compute_ensemble_weights(scores, self.ensemble_method)
        
        logger.info(f"Trained {len(self.models)} alpha models")
        logger.info(f"Ensemble weights: {self.weights}")
        for name, score in scores.items():
            logger.info(f"  {name}: MSE={score.mse:.6f}, R2={score.r2:.4f}")
        
        return scores
    
    def predict(
        self,
        X: pd.DataFrame,
        dip_scores: pd.Series | None = None,
        dip_k: float = 0.0,
    ) -> AlphaResult:
        """
        Generate ensemble prediction with uncertainty.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.
        dip_scores : pd.Series, optional
            DipScore per asset for adjustment.
        dip_k : float
            Dip coefficient.
        
        Returns
        -------
        AlphaResult
            Complete alpha prediction with uncertainty.
        """
        if not self.models:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = predict(model, X)
        
        # Compute weighted ensemble
        mu_hat = pd.Series(0.0, index=predictions[list(predictions.keys())[0]].index)
        for name, pred in predictions.items():
            mu_hat += self.weights[name] * pred
        
        # If multi-indexed (date, asset), get latest date per asset
        if isinstance(mu_hat.index, pd.MultiIndex):
            # Group by asset, take latest date
            mu_hat = mu_hat.groupby(level="asset").last()
            for name in predictions:
                predictions[name] = predictions[name].groupby(level="asset").last()
        
        # Compute uncertainty
        uncertainty = compute_uncertainty(
            list(self.models.values()),
            X,
            self.oos_residuals,
        )
        
        # Apply shrinkage
        mu_hat, shrinkage = shrink_mu_hat(
            mu_hat, uncertainty, self.shrinkage_strength
        )
        
        # Apply dip adjustment (if enabled)
        if dip_scores is not None and dip_k > 0:
            mu_hat = apply_dip_adjustment(mu_hat, dip_scores, dip_k)
        
        oos_scores = {
            name: model.oos_score 
            for name, model in self.models.items() 
            if model.oos_score is not None
        }
        
        return AlphaResult(
            mu_hat=mu_hat,
            mu_hat_raw=predictions,
            uncertainty=uncertainty,
            model_weights=self.weights,
            oos_scores=oos_scores,
            shrinkage_applied=shrinkage,
        )
