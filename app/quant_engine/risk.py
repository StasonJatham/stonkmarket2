"""
Risk model implementation.

Implements PCA-based factor covariance model:
Σ ≈ B Σ_F B^T + D

Where:
- B: Factor loadings (n_assets × n_factors)
- Σ_F: Factor covariance (n_factors × n_factors)
- D: Diagonal idiosyncratic variance
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from app.quant_engine.types import RiskModel


logger = logging.getLogger(__name__)


def fit_pca_risk_model(
    returns: pd.DataFrame,
    n_factors: int = 5,
    min_obs: int = 120,
) -> RiskModel:
    """
    Fit PCA-based factor risk model.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix (dates × assets).
    n_factors : int
        Number of PCA factors (K).
    min_obs : int
        Minimum observations required.
    
    Returns
    -------
    RiskModel
        Fitted risk model.
    
    Raises
    ------
    ValueError
        If insufficient data.
    """
    # Drop assets with insufficient data
    valid_assets = returns.dropna(axis=1, thresh=min_obs).columns.tolist()

    if len(valid_assets) < n_factors + 1:
        raise ValueError(
            f"Insufficient valid assets: {len(valid_assets)} < {n_factors + 1}"
        )

    returns_clean = returns[valid_assets].dropna()

    if len(returns_clean) < min_obs:
        raise ValueError(
            f"Insufficient observations: {len(returns_clean)} < {min_obs}"
        )

    n_assets = len(valid_assets)
    n_factors = min(n_factors, n_assets - 1, len(returns_clean) - 1)

    logger.info(f"Fitting PCA risk model: {n_assets} assets, {n_factors} factors")

    # Demean returns
    returns_demean = returns_clean - returns_clean.mean()
    X = returns_demean.values

    # Fit PCA
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X)  # (n_obs × n_factors)

    # Factor loadings B (n_assets × n_factors)
    B = pca.components_.T  # Transpose to get (n_assets × n_factors)

    # Factor covariance Σ_F (n_factors × n_factors)
    # Factors are orthogonal by construction, so Σ_F is diagonal
    factor_var = np.var(factors, axis=0)
    sigma_f = np.diag(factor_var)

    # Idiosyncratic variance D
    # Residuals = X - factors @ B^T
    X_fitted = factors @ B.T
    residuals = X - X_fitted
    D = np.var(residuals, axis=0)

    # Explained variance
    explained_var = pca.explained_variance_ratio_

    logger.info(
        f"PCA risk model fitted: "
        f"{sum(explained_var):.1%} variance explained by {n_factors} factors"
    )

    return RiskModel(
        B=B,
        sigma_f=sigma_f,
        D=D,
        explained_variance=explained_var,
        n_factors=n_factors,
        assets=valid_assets,
    )


def fit_ledoit_wolf_shrinkage(
    returns: pd.DataFrame,
    min_obs: int = 120,
) -> np.ndarray:
    """
    Fit Ledoit-Wolf shrinkage estimator for full covariance matrix.
    
    Alternative to PCA factor model for smaller universes.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix (dates × assets).
    min_obs : int
        Minimum observations.
    
    Returns
    -------
    np.ndarray
        Shrunk covariance matrix.
    """
    from sklearn.covariance import LedoitWolf

    returns_clean = returns.dropna()

    if len(returns_clean) < min_obs:
        raise ValueError(
            f"Insufficient observations: {len(returns_clean)} < {min_obs}"
        )

    X = returns_clean.values

    lw = LedoitWolf()
    lw.fit(X)

    logger.info(f"Ledoit-Wolf shrinkage: intensity={lw.shrinkage_:.4f}")

    return lw.covariance_


def compute_portfolio_risk_metrics(
    risk_model: RiskModel,
    weights: np.ndarray,
) -> dict[str, Any]:
    """
    Compute portfolio risk metrics from risk model.
    
    Parameters
    ----------
    risk_model : RiskModel
        Fitted risk model.
    weights : np.ndarray
        Portfolio weights.
    
    Returns
    -------
    dict[str, Any]
        Risk metrics including variance, volatility, MCR.
    """
    sigma = risk_model.get_covariance()

    variance = float(weights @ sigma @ weights)
    volatility = np.sqrt(variance)

    # Marginal contribution to risk
    if volatility > 1e-12:
        mcr = (weights * (sigma @ weights)) / volatility
    else:
        mcr = np.zeros_like(weights)

    # Factor contribution
    factor_var = weights @ risk_model.B @ risk_model.sigma_f @ risk_model.B.T @ weights
    idio_var = weights @ np.diag(risk_model.D) @ weights

    return {
        "variance": variance,
        "volatility": volatility,
        "volatility_annualized": volatility * np.sqrt(252),
        "mcr": mcr,
        "factor_variance_contribution": float(factor_var / variance) if variance > 1e-12 else 0.0,
        "idiosyncratic_variance_contribution": float(idio_var / variance) if variance > 1e-12 else 0.0,
    }


def tune_n_factors(
    returns: pd.DataFrame,
    k_candidates: list[int] = [3, 5, 8],
    min_explained_variance: float = 0.50,
    max_explained_variance: float = 0.95,
) -> tuple[int, dict[int, float]]:
    """
    Tune number of PCA factors via explained variance.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return matrix.
    k_candidates : list[int]
        Candidate K values.
    min_explained_variance : float
        Minimum acceptable explained variance.
    max_explained_variance : float
        Maximum (to avoid overfitting).
    
    Returns
    -------
    tuple[int, dict[int, float]]
        Best K and explained variance for each candidate.
    """
    returns_clean = returns.dropna()
    n_assets = len(returns.columns)
    n_obs = len(returns_clean)

    results = {}

    for k in k_candidates:
        if k >= n_assets or k >= n_obs:
            continue

        try:
            pca = PCA(n_components=k)
            pca.fit(returns_clean.values)
            explained = sum(pca.explained_variance_ratio_)
            results[k] = explained
        except Exception as e:
            logger.warning(f"PCA failed for k={k}: {e}")
            results[k] = 0.0

    # Select K with explained variance in target range
    # Prefer smaller K if multiple satisfy constraints
    best_k = k_candidates[0]

    for k in sorted(results.keys()):
        exp_var = results[k]
        if min_explained_variance <= exp_var <= max_explained_variance:
            best_k = k
            break

    logger.info(f"Selected n_factors={best_k}, explained_variance={results.get(best_k, 0):.2%}")

    return best_k, results
