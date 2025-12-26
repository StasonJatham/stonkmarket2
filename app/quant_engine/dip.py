"""
Statistical DipScore computation.

DipScore = (r_i,t - E[r_i,t | market, factors, regime]) / σ_i,t

This is a factor-residual z-score that measures how much an asset
has underperformed relative to its factor model expectation.

CRITICAL: DipScore is ONLY for diagnostics and informational purposes.
It may ONLY influence μ_hat or uncertainty, NEVER direct order generation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from app.quant_engine.types import DipArtifacts, MomentumCondition


logger = logging.getLogger(__name__)


def compute_dip_score(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    resid_vol_window: int = 20,
    min_obs: int = 120,
) -> DipArtifacts:
    """
    Compute DipScore as factor-residual z-score.
    
    For each asset i at time t:
    1. Estimate factor regression up to time t-1 (expanding window, no lookahead)
    2. Compute expected return E[r_i,t | factors]
    3. Compute residual e_i,t = r_i,t - E[r_i,t | factors]
    4. Compute conditional volatility σ_i,t (rolling std of residuals)
    5. DipScore_i,t = e_i,t / σ_i,t
    
    Parameters
    ----------
    asset_returns : pd.DataFrame
        Asset return matrix (dates × assets).
    factor_returns : pd.DataFrame
        Factor return matrix (dates × factors).
        Typical factors: market, sector ETFs, rates, FX, commodities.
    resid_vol_window : int
        Rolling window for residual volatility.
    min_obs : int
        Minimum observations before computing DipScore.
    
    Returns
    -------
    DipArtifacts
        Contains dip_score, residuals, residual volatility, and factor betas.
    """
    # Align data
    r, f = asset_returns.align(factor_returns, join="inner", axis=0)

    if len(r) < min_obs:
        logger.warning(f"Insufficient data for DipScore: {len(r)} < {min_obs}")
        return DipArtifacts(
            dip_score=pd.DataFrame(index=r.index, columns=r.columns, dtype=float),
            resid=pd.DataFrame(index=r.index, columns=r.columns, dtype=float),
            resid_sigma=pd.DataFrame(index=r.index, columns=r.columns, dtype=float),
            factor_betas={},
        )

    # Add constant to factors
    X = sm.add_constant(f.values, has_constant="add")

    # Initialize output arrays
    resid = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    factor_betas: dict[str, pd.Series] = {}

    # Compute residuals using expanding window regression (no lookahead)
    for tkr in r.columns:
        y = r[tkr].values
        e = np.full_like(y, np.nan, dtype=float)
        betas_last = None

        for t in range(min_obs, len(y)):
            try:
                # Regression on data up to t-1 (exclusive of t)
                mdl = sm.OLS(y[:t], X[:t]).fit()

                # Predict for time t
                yhat = float(mdl.predict(X[t:t+1])[0])
                e[t] = y[t] - yhat
                betas_last = mdl.params
            except Exception:
                # Skip on regression failure
                e[t] = np.nan

        resid[tkr] = e

        # Store latest betas
        if betas_last is not None:
            factor_betas[tkr] = pd.Series(
                betas_last[1:],  # Exclude constant
                index=f.columns,
                name=tkr,
            )

    # Compute rolling residual volatility
    sigma = resid.rolling(resid_vol_window, min_periods=resid_vol_window // 2).std()

    # Compute DipScore
    dip_score = (resid / sigma).replace([np.inf, -np.inf], np.nan)

    logger.info(f"Computed DipScore for {len(r.columns)} assets, {len(r)} dates")

    return DipArtifacts(
        dip_score=dip_score,
        resid=resid,
        resid_sigma=sigma,
        factor_betas=factor_betas,
    )


def get_dip_bucket(score: float) -> str:
    """
    Convert DipScore to bucket string for analysis.
    
    Parameters
    ----------
    score : float
        DipScore value.
    
    Returns
    -------
    str
        Bucket label.
    """
    if np.isnan(score):
        return "nan"
    elif score <= -2:
        return "<=-2"
    elif score <= -1:
        return "(-2,-1]"
    elif score <= 0:
        return "(-1,0]"
    elif score <= 1:
        return "(0,1]"
    else:
        return ">1"


def get_momentum_condition(
    returns: pd.Series,
    window: int = 252,
) -> MomentumCondition:
    """
    Determine momentum condition for an asset.
    
    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int
        Lookback window in trading days.
    
    Returns
    -------
    MomentumCondition
        POSITIVE, NEGATIVE, or NEUTRAL.
    """
    if len(returns) < window:
        return MomentumCondition.NEUTRAL

    cumret = (1 + returns.iloc[-window:]).prod() - 1

    if cumret > 0.05:  # > 5% over period
        return MomentumCondition.POSITIVE
    elif cumret < -0.05:  # < -5% over period
        return MomentumCondition.NEGATIVE
    else:
        return MomentumCondition.NEUTRAL


def verify_dip_effectiveness(
    dip_artifacts: DipArtifacts,
    forward_returns: pd.DataFrame,
    regime_labels: pd.Series | None = None,
    momentum_labels: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Verify dip effectiveness via bucket analysis.
    
    Computes E[forward returns | DipScore bucket] and tests for significance.
    
    Parameters
    ----------
    dip_artifacts : DipArtifacts
        Computed DipScore artifacts.
    forward_returns : pd.DataFrame
        Forward returns (same shape as dip_score).
    regime_labels : pd.Series, optional
        Regime labels per date.
    momentum_labels : pd.DataFrame, optional
        Momentum labels per asset per date.
    
    Returns
    -------
    dict[str, Any]
        Verification results with bucket returns, regime splits, etc.
    """
    dip_score = dip_artifacts.dip_score

    # Stack to long format
    dip_stack = dip_score.stack()
    dip_stack.name = "dip_score"

    fwd_stack = forward_returns.stack()
    fwd_stack.name = "fwd_ret"

    # Combine
    df = pd.concat([dip_stack, fwd_stack], axis=1, join="inner").dropna()

    if len(df) < 100:
        logger.warning("Insufficient data for dip verification")
        return {"valid": False, "reason": "insufficient_data", "n": len(df)}

    # Add buckets
    df["bucket"] = df["dip_score"].apply(get_dip_bucket)

    # Bucket analysis
    bucket_stats = df.groupby("bucket")["fwd_ret"].agg(["mean", "std", "count"])
    bucket_stats["t_stat"] = bucket_stats["mean"] / (
        bucket_stats["std"] / np.sqrt(bucket_stats["count"])
    )

    result = {
        "valid": True,
        "n": len(df),
        "bucket_stats": bucket_stats.to_dict(),
    }

    # Check if extreme negative dip has higher returns (mean reversion)
    if "<=-2" in bucket_stats.index:
        extreme_dip_ret = bucket_stats.loc["<=-2", "mean"]
        overall_mean = df["fwd_ret"].mean()
        result["extreme_dip_excess"] = float(extreme_dip_ret - overall_mean)
        result["extreme_dip_significant"] = (
            abs(bucket_stats.loc["<=-2", "t_stat"]) > 1.96
        )

    # Regime dependence (if available)
    if regime_labels is not None:
        df_regime = df.copy()
        df_regime["regime"] = regime_labels.reindex(
            df.index.get_level_values(0)
        ).values

        regime_bucket = df_regime.groupby(["regime", "bucket"])["fwd_ret"].mean()
        result["regime_bucket_returns"] = regime_bucket.to_dict()

    # Momentum interaction (if available)
    if momentum_labels is not None:
        mom_stack = momentum_labels.stack()
        mom_stack.name = "momentum"

        df_mom = df.join(mom_stack, how="left")
        df_mom = df_mom.dropna(subset=["momentum"])

        if len(df_mom) > 50:
            mom_bucket = df_mom.groupby(["momentum", "bucket"])["fwd_ret"].mean()
            result["momentum_bucket_returns"] = mom_bucket.to_dict()

    logger.info(f"Dip verification complete: {len(df)} samples")

    return result


def compute_dip_adjustment_k(
    dip_artifacts: DipArtifacts,
    forward_returns: pd.DataFrame,
    k_grid: list[float] = [0.0, 0.001, 0.002, 0.005, 0.01],
) -> tuple[float, dict[str, float]]:
    """
    Tune dip coefficient k via OOS validation.
    
    For each k, compute mu_hat_adjusted = mu_hat + k * max(0, -DipScore)
    and evaluate forecasting performance.
    
    Parameters
    ----------
    dip_artifacts : DipArtifacts
        Computed DipScore.
    forward_returns : pd.DataFrame
        Forward returns for validation.
    k_grid : list[float]
        Candidate k values.
    
    Returns
    -------
    tuple[float, dict[str, float]]
        Best k and scores for all candidates.
    """
    dip_score = dip_artifacts.dip_score

    # Align
    dip_stack = dip_score.stack()
    fwd_stack = forward_returns.stack()

    df = pd.concat([dip_stack, fwd_stack], axis=1, join="inner")
    df.columns = ["dip", "fwd"]
    df = df.dropna()

    if len(df) < 50:
        logger.warning("Insufficient data for dip k tuning, using k=0")
        return 0.0, {"0.0": np.nan}

    scores = {}

    for k in k_grid:
        # Dip contribution to mu_hat
        dip_contrib = k * np.maximum(0, -df["dip"])

        # Simple evaluation: correlation of dip_contrib with fwd returns
        # Higher correlation = k is adding signal
        if k > 0 and dip_contrib.std() > 1e-12:
            corr = dip_contrib.corr(df["fwd"])
        else:
            corr = 0.0

        scores[str(k)] = float(corr) if np.isfinite(corr) else 0.0

    # Select k with highest correlation (but penalize for overfitting)
    # If no k improves, use k=0
    best_k = 0.0
    best_score = scores.get("0.0", 0.0)

    for k in k_grid:
        if k > 0 and scores[str(k)] > best_score + 0.01:  # Require improvement
            best_k = k
            best_score = scores[str(k)]

    logger.info(f"Selected dip_k={best_k}, scores={scores}")

    return best_k, scores
