"""
Core type definitions for the Quantitative Portfolio Engine.

All dataclasses are frozen and use strict typing for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd


class RegimeTrend(str, Enum):
    """Market trend regime classification."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class RegimeVolatility(str, Enum):
    """Market volatility regime classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MomentumCondition(str, Enum):
    """Momentum state for dip effectiveness conditioning."""
    POSITIVE = "pos"
    NEGATIVE = "neg"
    NEUTRAL = "neutral"


class ActionType(str, Enum):
    """Trade action type."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SolverStatus(str, Enum):
    """Optimization solver status."""
    OPTIMAL = "optimal"
    OPTIMAL_INACCURATE = "optimal_inaccurate"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    ERROR = "error"


@dataclass(frozen=True)
class QuantConfig:
    """
    Configuration for the quantitative engine.
    
    All hyperparameters are candidates for OOS tuning unless marked as fixed.
    """
    # Fixed constraints (not tuned)
    base_currency: str = "EUR"
    long_only: bool = True
    max_leverage: float = 1.0  # No leverage
    fixed_cost_eur: float = 1.0  # €1 per trade (TradeRepublic)
    min_trade_eur: float = 10.0  # Minimum trade size
    
    # Monthly inflow range (for planning)
    inflow_min_eur: float = 1000.0
    inflow_max_eur: float = 1500.0
    
    # Hyperparameter candidates (tuned OOS)
    forecast_horizon_months: int = 2  # H ∈ {1, 2, 3}
    
    # Alpha model hyperparameters
    ridge_alpha: float = 10.0  # Ridge regularization
    lasso_alpha: float = 0.01  # Lasso regularization (if used)
    use_lasso: bool = False  # Whether to include Lasso in ensemble
    ensemble_method: Literal["inverse_mse", "equal"] = "inverse_mse"
    
    # Dip integration
    dip_k: float = 0.002  # Dip coefficient for μ_hat adjustment
    dip_uncertainty_scale: float = 0.0  # Scale uncertainty by |DipScore|
    
    # Risk model
    n_pca_factors: int = 5  # K ∈ {3, 5, 8}
    
    # Optimizer
    lambda_risk: float = 10.0  # Risk aversion λ
    max_weight: float = 0.15  # Maximum position weight
    max_turnover: float = 0.20  # Maximum monthly turnover
    turnover_penalty: float = 0.001  # L1 penalty on turnover
    allow_cash: bool = True  # Whether to allow cash position
    
    # Feature engineering
    momentum_windows: tuple[int, ...] = (21, 63, 126, 252)  # Trading days
    volatility_window: int = 21
    reversal_window: int = 5
    
    # Walk-forward validation
    train_months: int = 36  # Minimum training window
    validation_months: int = 6  # Validation window for HP selection
    test_months: int = 6  # Test window for final evaluation
    retrain_frequency_months: int = 3  # How often to retune
    
    # DipScore computation
    dip_resid_vol_window: int = 20  # Rolling residual volatility window
    dip_min_obs: int = 120  # Minimum observations for factor regression


@dataclass(frozen=True)
class AlphaModelScore:
    """Out-of-sample score for a single alpha model."""
    model_name: str
    mse: float
    rmse: float
    r2: float
    n_samples: int
    sharpe_forecast: float | None = None


@dataclass(frozen=True)
class AlphaResult:
    """
    Result from alpha model ensemble.
    
    Contains expected returns and uncertainty quantification.
    """
    mu_hat: pd.Series  # Expected returns per asset
    mu_hat_raw: dict[str, pd.Series]  # Per-model predictions
    uncertainty: pd.Series  # Forecast error estimate per asset
    model_weights: dict[str, float]  # Ensemble weights α_k
    oos_scores: dict[str, AlphaModelScore]  # OOS performance
    shrinkage_applied: pd.Series  # How much shrinkage toward zero
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mu_hat": self.mu_hat.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "model_weights": self.model_weights,
            "oos_scores": {k: {
                "mse": v.mse,
                "rmse": v.rmse,
                "r2": v.r2,
                "n": v.n_samples,
            } for k, v in self.oos_scores.items()},
        }


@dataclass(frozen=True)
class DipArtifacts:
    """
    DipScore computation artifacts.
    
    DipScore = (r_i,t - E[r_i,t | market, factors, regime]) / σ_i,t
    
    This is ONLY for diagnostics/annotation, never for direct order generation.
    """
    dip_score: pd.DataFrame  # Z-score per asset per date
    resid: pd.DataFrame  # Factor-adjusted residuals
    resid_sigma: pd.DataFrame  # Conditional volatility of residuals
    factor_betas: dict[str, pd.Series]  # Factor loadings per asset
    bucket: pd.Series | None = None  # Bucketed DipScore for current date
    
    def get_current(self, as_of: date) -> pd.Series:
        """Get DipScore as of a specific date."""
        if as_of in self.dip_score.index:
            return self.dip_score.loc[as_of]
        # Find most recent date
        valid_dates = self.dip_score.index[self.dip_score.index <= pd.Timestamp(as_of)]
        if len(valid_dates) == 0:
            return pd.Series(dtype=float)
        return self.dip_score.loc[valid_dates[-1]]
    
    def bucketize(self, score: float) -> str:
        """Convert DipScore to bucket string."""
        if score <= -2:
            return "<=-2"
        elif score <= -1:
            return "(-2,-1]"
        elif score <= 0:
            return "(-1,0]"
        elif score <= 1:
            return "(0,1]"
        else:
            return ">1"


@dataclass(frozen=True)
class RegimeState:
    """Current market regime state."""
    trend: RegimeTrend
    volatility: RegimeVolatility
    trend_score: float  # Underlying continuous score
    vol_score: float  # Underlying continuous score
    as_of: date
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "trend": self.trend.value,
            "vol": self.volatility.value,
            "trend_score": self.trend_score,
            "vol_score": self.vol_score,
        }


@dataclass(frozen=True)
class RiskModel:
    """
    Factor-based risk model.
    
    Σ ≈ B Σ_F B^T + D
    
    Where:
    - B: Factor loadings (n_assets × n_factors)
    - Σ_F: Factor covariance (n_factors × n_factors)
    - D: Diagonal idiosyncratic variance
    """
    B: np.ndarray  # Factor loadings
    sigma_f: np.ndarray  # Factor covariance
    D: np.ndarray  # Idiosyncratic variance (diagonal)
    explained_variance: np.ndarray  # Per-factor explained variance
    n_factors: int
    assets: list[str]
    
    def get_covariance(self) -> np.ndarray:
        """Compute full covariance matrix Σ."""
        return self.B @ self.sigma_f @ self.B.T + np.diag(self.D)
    
    def portfolio_variance(self, w: np.ndarray) -> float:
        """Compute portfolio variance w' Σ w."""
        sigma = self.get_covariance()
        return float(w @ sigma @ w)
    
    def portfolio_volatility(self, w: np.ndarray) -> float:
        """Compute portfolio volatility sqrt(w' Σ w)."""
        return np.sqrt(self.portfolio_variance(w))
    
    def marginal_contribution_to_risk(self, w: np.ndarray) -> np.ndarray:
        """
        Compute marginal contribution to risk (MCR).
        
        MCR_i = w_i * (Σw)_i / σ_p
        """
        sigma = self.get_covariance()
        sigma_w = sigma @ w
        sigma_p = self.portfolio_volatility(w)
        if sigma_p < 1e-12:
            return np.zeros_like(w)
        return (w * sigma_w) / sigma_p


@dataclass(frozen=True)
class ConstraintStatus:
    """Status of optimization constraints."""
    max_weight_binding: list[str]  # Assets at max weight
    turnover_binding: bool  # Turnover constraint binding
    min_trade_filtered: list[str]  # Assets filtered by min trade
    budget_slack: float  # Unused budget (cash)


@dataclass(frozen=True)
class OptimizationResult:
    """
    Result from portfolio optimization.
    
    Contains the optimal Δw and all auxiliary information.
    """
    w_current: np.ndarray  # Current weights before optimization
    w_new: np.ndarray  # Optimal weights after optimization
    dw: np.ndarray  # Weight changes Δw = w_new - w_current
    assets: list[str]
    status: SolverStatus
    objective_value: float
    constraint_status: ConstraintStatus
    transaction_cost_eur: float
    marginal_utilities: np.ndarray  # Per-asset marginal utility
    
    def get_trades(self, portfolio_value_eur: float) -> dict[str, float]:
        """Convert Δw to EUR trade amounts."""
        return {
            asset: float(self.dw[i] * portfolio_value_eur)
            for i, asset in enumerate(self.assets)
            if abs(self.dw[i]) > 1e-8
        }


@dataclass(frozen=True)
class DipAnnotation:
    """Dip annotation for a single asset (informational only)."""
    dip_score: float
    bucket: str
    regime: RegimeState
    momentum_12m: MomentumCondition
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "dip_score": round(self.dip_score, 3),
            "bucket": self.bucket,
            "regime": self.regime.to_dict(),
            "momentum_12m": self.momentum_12m.value,
        }


@dataclass(frozen=True)
class MuHatUncertainty:
    """Uncertainty quantification for expected return estimate."""
    ci_low: float  # Lower confidence interval bound
    ci_high: float  # Upper confidence interval bound
    oos_rmse: float  # Out-of-sample RMSE
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "ci": [round(self.ci_low, 5), round(self.ci_high, 5)],
            "oos_rmse": round(self.oos_rmse, 5),
        }


@dataclass(frozen=True)
class RiskInfo:
    """Risk information for a single asset."""
    marginal_vol: float  # Marginal contribution to volatility
    mcr: float  # Marginal contribution to risk
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "marginal_vol": round(self.marginal_vol, 5),
            "mcr": round(self.mcr, 4),
        }


@dataclass(frozen=True)
class RecommendationRow:
    """
    Single recommendation from the optimizer.
    
    This is what the frontend displays for "Next Allocation Recommendations".
    """
    ticker: str
    name: str | None
    action: ActionType
    notional_eur: float
    delta_weight: float
    mu_hat: float
    mu_hat_uncertainty: MuHatUncertainty
    risk: RiskInfo
    delta_utility_net: float  # Net of transaction costs
    trade_cost_eur: float
    constraints: list[str]
    dip: DipAnnotation | None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "action": self.action.value,
            "notional_eur": round(self.notional_eur, 2),
            "delta_weight": round(self.delta_weight, 5),
            "mu_hat": round(self.mu_hat, 5),
            "mu_hat_uncertainty": self.mu_hat_uncertainty.to_dict(),
            "risk": self.risk.to_dict(),
            "delta_utility_net": round(self.delta_utility_net, 6),
            "trade_cost_eur": self.trade_cost_eur,
            "constraints": self.constraints,
            "dip": self.dip.to_dict() if self.dip else None,
        }


@dataclass(frozen=True)
class AuditBlock:
    """
    Audit information for reproducibility and transparency.
    
    Contains all model artifacts needed to reproduce the decision.
    """
    model_weights: dict[str, float]
    oos_scores: dict[str, dict[str, float]]
    risk_model: dict[str, Any]
    hyperparams: dict[str, Any]
    data_hash: str  # SHA-256 of input data for reproducibility
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "model_weights": self.model_weights,
            "oos_scores": self.oos_scores,
            "risk_model": self.risk_model,
            "hyperparams": self.hyperparams,
            "data_hash": self.data_hash,
        }


@dataclass(frozen=True)
class EngineOutput:
    """
    Complete output from the quantitative engine.
    
    This is the full response for the API endpoint.
    """
    as_of: date
    portfolio_value_eur: float
    inflow_eur: float
    solver_status: SolverStatus
    recommendations: list[RecommendationRow]
    audit: AuditBlock
    diagnostics: dict[str, Any] | None = None  # Optional dip/regime info
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "as_of": self.as_of.isoformat(),
            "portfolio_value_eur": round(self.portfolio_value_eur, 2),
            "inflow_eur": round(self.inflow_eur, 2),
            "solver_status": self.solver_status.value,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "audit": self.audit.to_dict(),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class HyperparameterCandidate:
    """A single hyperparameter configuration to evaluate."""
    forecast_horizon_months: int
    ridge_alpha: float
    lasso_alpha: float | None
    use_lasso: bool
    dip_k: float
    n_pca_factors: int
    lambda_risk: float
    turnover_penalty: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "H": self.forecast_horizon_months,
            "ridge_alpha": self.ridge_alpha,
            "lasso_alpha": self.lasso_alpha,
            "use_lasso": self.use_lasso,
            "dip_k": self.dip_k,
            "n_pca_factors": self.n_pca_factors,
            "lambda_risk": self.lambda_risk,
            "turnover_penalty": self.turnover_penalty,
        }


@dataclass(frozen=True)
class HyperparameterLog:
    """
    Log of hyperparameter tuning for a single evaluation.
    
    Required for audit and reproducibility.
    """
    timestamp: datetime
    parameters: dict[str, Any]
    validation_sharpe: float
    validation_return: float
    validation_volatility: float
    validation_max_drawdown: float
    baseline_sharpe: float
    selected: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "validation_sharpe": round(self.validation_sharpe, 4),
            "validation_return": round(self.validation_return, 4),
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "selected": self.selected,
        }


@dataclass(frozen=True)
class WalkForwardFold:
    """Specification for one walk-forward fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True)
class WalkForwardResult:
    """Results from walk-forward validation."""
    folds: list[WalkForwardFold]
    fold_metrics: list[dict[str, float]]
    aggregate_sharpe: float
    aggregate_return: float
    aggregate_volatility: float
    aggregate_max_drawdown: float
    baseline_sharpe: dict[str, float]
    total_turnover: float
    hit_rate: float
    regime_performance: dict[str, float]
