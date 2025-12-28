"""
Risk-Based Portfolio Optimization.

This module implements portfolio optimization that does NOT require return forecasts:
- Risk Parity: Equal risk contribution per asset
- Minimum Variance: Minimize total portfolio risk
- Maximum Diversification: Maximize diversification benefit
- CVaR Minimization: Minimize tail risk (expected shortfall)
- Hierarchical Risk Parity: Cluster-based allocation

All methods are grounded in observed risk properties, not return predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class RiskOptimizationMethod(str, Enum):
    """Available risk-based optimization methods."""
    
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"
    CVAR = "cvar"
    HIERARCHICAL_RISK_PARITY = "hrp"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class RiskOptimizationConstraints:
    """Constraints for portfolio optimization."""
    
    min_weight: float = 0.01  # Minimum 1% per position
    max_weight: float = 0.40  # Maximum 40% per position
    max_turnover: float | None = None  # Maximum weight change (optional)
    
    # Position limits per asset (optional)
    position_limits: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class RiskOptimizationResult:
    """Result from portfolio optimization."""
    
    method: str
    weights: dict[str, float]
    
    # Risk metrics of optimal portfolio
    portfolio_volatility: float
    portfolio_var_95: float
    diversification_ratio: float
    
    # Optimization diagnostics
    converged: bool
    iterations: int
    objective_value: float
    
    # Comparison to current (if provided)
    current_weights: dict[str, float] | None = None
    weight_changes: dict[str, float] | None = None
    turnover: float = 0.0
    
    # Risk changes
    vol_change_pct: float = 0.0
    
    # Optimization quality indicators (new)
    optimization_quality: str = "optimal"  # "optimal", "degraded", "fallback"
    quality_reason: str = ""  # Explanation if not optimal


@dataclass
class AllocationRecommendation:
    """User-facing allocation recommendation."""
    
    # What to do
    recommendations: list[dict]  # [{"symbol": "AAPL", "action": "BUY", "amount_eur": 500, ...}]
    
    # Current vs optimal
    current_portfolio: dict[str, float]
    optimal_portfolio: dict[str, float]
    
    # Risk improvement
    current_risk: dict
    optimal_risk: dict
    risk_improvement_summary: str
    
    # Confidence and explanation
    confidence: str  # HIGH, MEDIUM, LOW
    explanation: str
    warnings: list[str]


# =============================================================================
# Risk Parity Optimization
# =============================================================================


def risk_parity_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Risk parity objective: minimize deviation from equal risk contribution.
    
    Each asset should contribute 1/N of total portfolio risk.
    """
    n = len(weights)
    target_risk = 1 / n
    
    portfolio_var = weights @ cov @ weights
    if portfolio_var <= 0:
        return 1e10
    
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Marginal risk contribution
    marginal = (cov @ weights) / portfolio_vol
    
    # Risk contribution per asset
    risk_contrib = weights * marginal / portfolio_vol
    
    # Sum of squared deviations from target
    return float(np.sum((risk_contrib - target_risk) ** 2))


def optimize_risk_parity(
    cov: np.ndarray,
    symbols: list[str],
    constraints: RiskOptimizationConstraints = RiskOptimizationConstraints(),
    current_weights: np.ndarray | None = None,
) -> RiskOptimizationResult:
    """
    Find risk parity portfolio.
    
    Risk parity allocates such that each asset contributes equally to total risk.
    This is robust because it doesn't require return forecasts.
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n  # Start with equal weights
    
    # Bounds
    bounds = []
    for i, s in enumerate(symbols):
        if s in constraints.position_limits:
            bounds.append(constraints.position_limits[s])
        else:
            bounds.append((constraints.min_weight, constraints.max_weight))
    
    # Constraints
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Sum to 1
    
    # Turnover constraint
    if constraints.max_turnover is not None and current_weights is not None:
        cons.append({
            "type": "ineq",
            "fun": lambda w, cw=current_weights, mt=constraints.max_turnover: mt - np.sum(np.abs(w - cw)),
        })
    
    # Optimize
    result = optimize.minimize(
        risk_parity_objective,
        x0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    
    weights = result.x
    weights = np.maximum(weights, 0)  # Ensure non-negative
    weights = weights / weights.sum()  # Normalize
    
    # Compute portfolio metrics
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((weights @ asset_vols) / np.sqrt(weights @ cov @ weights))
    
    # VaR (assuming normal distribution for simplicity)
    port_var_95 = float(-1.645 * np.sqrt(weights @ cov @ weights))
    
    # Weight changes
    weight_changes = None
    turnover = 0.0
    if current_weights is not None:
        weight_changes = {s: float(weights[i] - current_weights[i]) for i, s in enumerate(symbols)}
        turnover = float(np.sum(np.abs(weights - current_weights)))
    
    return RiskOptimizationResult(
        method=RiskOptimizationMethod.RISK_PARITY.value,
        weights={s: float(weights[i]) for i, s in enumerate(symbols)},
        portfolio_volatility=port_vol,
        portfolio_var_95=port_var_95,
        diversification_ratio=div_ratio,
        converged=result.success,
        iterations=result.nit,
        objective_value=float(result.fun),
        current_weights={s: float(current_weights[i]) for i, s in enumerate(symbols)} if current_weights is not None else None,
        weight_changes=weight_changes,
        turnover=turnover,
    )


# =============================================================================
# Minimum Variance Optimization
# =============================================================================


def min_variance_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """Minimize portfolio variance."""
    return float(weights @ cov @ weights)


def optimize_min_variance(
    cov: np.ndarray,
    symbols: list[str],
    constraints: RiskOptimizationConstraints = RiskOptimizationConstraints(),
    current_weights: np.ndarray | None = None,
) -> RiskOptimizationResult:
    """
    Find minimum variance portfolio.
    
    Minimizes total portfolio risk without requiring return forecasts.
    Tends to concentrate in low-volatility assets.
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n
    
    bounds = []
    for i, s in enumerate(symbols):
        if s in constraints.position_limits:
            bounds.append(constraints.position_limits[s])
        else:
            bounds.append((constraints.min_weight, constraints.max_weight))
    
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if constraints.max_turnover is not None and current_weights is not None:
        cons.append({
            "type": "ineq",
            "fun": lambda w, cw=current_weights, mt=constraints.max_turnover: mt - np.sum(np.abs(w - cw)),
        })
    
    result = optimize.minimize(
        min_variance_objective,
        x0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    
    weights = result.x
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((weights @ asset_vols) / np.sqrt(weights @ cov @ weights))
    port_var_95 = float(-1.645 * np.sqrt(weights @ cov @ weights))
    
    weight_changes = None
    turnover = 0.0
    if current_weights is not None:
        weight_changes = {s: float(weights[i] - current_weights[i]) for i, s in enumerate(symbols)}
        turnover = float(np.sum(np.abs(weights - current_weights)))
    
    return RiskOptimizationResult(
        method=RiskOptimizationMethod.MIN_VARIANCE.value,
        weights={s: float(weights[i]) for i, s in enumerate(symbols)},
        portfolio_volatility=port_vol,
        portfolio_var_95=port_var_95,
        diversification_ratio=div_ratio,
        converged=result.success,
        iterations=result.nit,
        objective_value=float(result.fun),
        current_weights={s: float(current_weights[i]) for i, s in enumerate(symbols)} if current_weights is not None else None,
        weight_changes=weight_changes,
        turnover=turnover,
    )


# =============================================================================
# Maximum Diversification Optimization
# =============================================================================


def max_diversification_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Maximize diversification ratio.
    
    DR = (weighted avg volatility) / (portfolio volatility)
    Higher = more diversification benefit captured.
    """
    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = weights @ asset_vols
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    
    if portfolio_vol <= 0:
        return 0
    
    # Return negative because we want to maximize
    return float(-weighted_avg_vol / portfolio_vol)


def optimize_max_diversification(
    cov: np.ndarray,
    symbols: list[str],
    constraints: RiskOptimizationConstraints = RiskOptimizationConstraints(),
    current_weights: np.ndarray | None = None,
) -> RiskOptimizationResult:
    """
    Find maximum diversification portfolio.
    
    Maximizes the diversification ratio - captures maximum diversification benefit.
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n
    
    bounds = []
    for i, s in enumerate(symbols):
        if s in constraints.position_limits:
            bounds.append(constraints.position_limits[s])
        else:
            bounds.append((constraints.min_weight, constraints.max_weight))
    
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if constraints.max_turnover is not None and current_weights is not None:
        cons.append({
            "type": "ineq",
            "fun": lambda w, cw=current_weights, mt=constraints.max_turnover: mt - np.sum(np.abs(w - cw)),
        })
    
    result = optimize.minimize(
        max_diversification_objective,
        x0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    
    weights = result.x
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((weights @ asset_vols) / np.sqrt(weights @ cov @ weights))
    port_var_95 = float(-1.645 * np.sqrt(weights @ cov @ weights))
    
    weight_changes = None
    turnover = 0.0
    if current_weights is not None:
        weight_changes = {s: float(weights[i] - current_weights[i]) for i, s in enumerate(symbols)}
        turnover = float(np.sum(np.abs(weights - current_weights)))
    
    return RiskOptimizationResult(
        method=RiskOptimizationMethod.MAX_DIVERSIFICATION.value,
        weights={s: float(weights[i]) for i, s in enumerate(symbols)},
        portfolio_volatility=port_vol,
        portfolio_var_95=port_var_95,
        diversification_ratio=div_ratio,
        converged=result.success,
        iterations=result.nit,
        objective_value=float(-result.fun),  # Convert back to positive
        current_weights={s: float(current_weights[i]) for i, s in enumerate(symbols)} if current_weights is not None else None,
        weight_changes=weight_changes,
        turnover=turnover,
    )


# =============================================================================
# CVaR (Expected Shortfall) Minimization
# =============================================================================


def optimize_cvar(
    returns: pd.DataFrame,
    symbols: list[str],
    alpha: float = 0.05,
    constraints: RiskOptimizationConstraints = RiskOptimizationConstraints(),
    current_weights: np.ndarray | None = None,
) -> RiskOptimizationResult:
    """
    Minimize Conditional Value at Risk (Expected Shortfall).
    
    CVaR = average loss in worst alpha% of scenarios.
    This focuses on tail risk rather than just volatility.
    
    Uses historical simulation approach (no distribution assumptions).
    """
    returns_arr = returns[symbols].values
    T, n = returns_arr.shape
    
    def cvar_objective(weights: np.ndarray) -> float:
        portfolio_returns = returns_arr @ weights
        var = np.percentile(portfolio_returns, alpha * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return float(-cvar)  # Negative because cvar is negative (loss)
    
    x0 = np.ones(n) / n
    
    bounds = []
    for i, s in enumerate(symbols):
        if s in constraints.position_limits:
            bounds.append(constraints.position_limits[s])
        else:
            bounds.append((constraints.min_weight, constraints.max_weight))
    
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if constraints.max_turnover is not None and current_weights is not None:
        cons.append({
            "type": "ineq",
            "fun": lambda w, cw=current_weights, mt=constraints.max_turnover: mt - np.sum(np.abs(w - cw)),
        })
    
    result = optimize.minimize(
        cvar_objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    
    weights = result.x
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    
    # Compute metrics
    cov = returns[symbols].cov().values
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((weights @ asset_vols) / np.sqrt(weights @ cov @ weights))
    
    # Actual CVaR at optimal weights
    port_returns = returns_arr @ weights
    port_var_95 = float(np.percentile(port_returns, 5))
    
    weight_changes = None
    turnover = 0.0
    if current_weights is not None:
        weight_changes = {s: float(weights[i] - current_weights[i]) for i, s in enumerate(symbols)}
        turnover = float(np.sum(np.abs(weights - current_weights)))
    
    return RiskOptimizationResult(
        method=RiskOptimizationMethod.CVAR.value,
        weights={s: float(weights[i]) for i, s in enumerate(symbols)},
        portfolio_volatility=port_vol,
        portfolio_var_95=port_var_95,
        diversification_ratio=div_ratio,
        converged=result.success,
        iterations=result.nit,
        objective_value=float(result.fun),
        current_weights={s: float(current_weights[i]) for i, s in enumerate(symbols)} if current_weights is not None else None,
        weight_changes=weight_changes,
        turnover=turnover,
    )


# =============================================================================
# Hierarchical Risk Parity (HRP)
# =============================================================================


def optimize_hrp(
    returns: pd.DataFrame,
    symbols: list[str],
    constraints: RiskOptimizationConstraints = RiskOptimizationConstraints(),
    current_weights: np.ndarray | None = None,
) -> RiskOptimizationResult:
    """
    Hierarchical Risk Parity (López de Prado, 2016).
    
    Benefits:
    - No matrix inversion (more stable than traditional optimization)
    - Handles correlated assets naturally
    - Less sensitive to estimation error
    
    Process:
    1. Tree clustering based on correlation distance
    2. Quasi-diagonalization of covariance matrix
    3. Recursive bisection to allocate weights
    """
    returns_filtered = returns[symbols].dropna()
    n = len(symbols)
    
    # Track optimization quality
    opt_quality = "optimal"
    quality_reason = ""
    
    if len(returns_filtered) < 30:
        # Fallback to equal weight
        weights = np.ones(n) / n
        opt_quality = "fallback"
        quality_reason = f"Insufficient history ({len(returns_filtered)} days < 30 required)"
        logger.warning(f"HRP fallback: {quality_reason}")
    else:
        cov = returns_filtered.cov().values
        corr = returns_filtered.corr().values
        
        # 1. Correlation distance
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0)
        
        try:
            # 2. Hierarchical clustering
            dist_condensed = squareform(dist, checks=False)
            link = linkage(dist_condensed, method="ward")
            sort_ix = leaves_list(link)
            
            # 3. Recursive bisection
            weights = _recursive_bisection(cov, sort_ix)
        except Exception as e:
            logger.warning(f"HRP clustering failed: {e}, falling back to equal weight")
            weights = np.ones(n) / n
            opt_quality = "degraded"
            quality_reason = f"Clustering failed: {str(e)[:100]}"
    
    # Apply constraints
    weights = np.maximum(weights, constraints.min_weight)
    weights = np.minimum(weights, constraints.max_weight)
    weights = weights / weights.sum()
    
    # Compute metrics
    cov = returns_filtered.cov().values
    port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
    asset_vols = np.sqrt(np.diag(cov))
    div_ratio = float((weights @ asset_vols) / np.sqrt(weights @ cov @ weights))
    
    port_returns = (returns_filtered.values @ weights)
    port_var_95 = float(np.percentile(port_returns, 5))
    
    weight_changes = None
    turnover = 0.0
    if current_weights is not None:
        weight_changes = {s: float(weights[i] - current_weights[i]) for i, s in enumerate(symbols)}
        turnover = float(np.sum(np.abs(weights - current_weights)))
    
    return RiskOptimizationResult(
        method=RiskOptimizationMethod.HIERARCHICAL_RISK_PARITY.value,
        weights={s: float(weights[i]) for i, s in enumerate(symbols)},
        portfolio_volatility=port_vol,
        portfolio_var_95=port_var_95,
        diversification_ratio=div_ratio,
        converged=True,  # HRP always converges
        iterations=1,
        objective_value=0.0,  # HRP doesn't have explicit objective
        current_weights={s: float(current_weights[i]) for i, s in enumerate(symbols)} if current_weights is not None else None,
        weight_changes=weight_changes,
        turnover=turnover,
        optimization_quality=opt_quality,
        quality_reason=quality_reason,
    )


def _recursive_bisection(cov: np.ndarray, sort_ix: np.ndarray) -> np.ndarray:
    """
    Recursive bisection for HRP.
    
    Allocates weights by recursively splitting the sorted assets
    and distributing inverse-variance weighted allocations.
    """
    n = len(sort_ix)
    weights = np.ones(n)
    
    # Recursive bisection
    clusters = [list(range(n))]
    
    while len(clusters) > 0:
        clusters_new = []
        
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            
            # Split cluster in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            
            # Get cluster covariance
            left_cov = cov[np.ix_(sort_ix[left], sort_ix[left])]
            right_cov = cov[np.ix_(sort_ix[right], sort_ix[right])]
            
            # Cluster variance (using inverse-variance weighting within cluster)
            left_var = _cluster_variance(left_cov)
            right_var = _cluster_variance(right_cov)
            
            # Allocate inversely proportional to variance
            total_var = left_var + right_var
            if total_var > 0:
                left_weight = 1 - left_var / total_var
            else:
                left_weight = 0.5
            
            right_weight = 1 - left_weight
            
            # Apply to weights
            weights[left] *= left_weight
            weights[right] *= right_weight
            
            # Continue bisection
            if len(left) > 1:
                clusters_new.append(left)
            if len(right) > 1:
                clusters_new.append(right)
        
        clusters = clusters_new
    
    # Reorder weights back to original order
    final_weights = np.zeros(n)
    for i, ix in enumerate(sort_ix):
        final_weights[ix] = weights[i]
    
    return final_weights


def _cluster_variance(cov: np.ndarray) -> float:
    """Compute cluster variance using inverse-variance weighting."""
    n = cov.shape[0]
    if n == 1:
        return float(cov[0, 0])
    
    # Inverse variance weights
    inv_diag = 1 / np.diag(cov)
    inv_diag = inv_diag / inv_diag.sum()
    
    return float(inv_diag @ cov @ inv_diag)


# =============================================================================
# Main Optimization Interface
# =============================================================================


def optimize_portfolio_risk_based(
    returns: pd.DataFrame,
    symbols: list[str],
    method: RiskOptimizationMethod = RiskOptimizationMethod.RISK_PARITY,
    constraints: RiskOptimizationConstraints = RiskOptimizationConstraints(),
    current_weights: dict[str, float] | None = None,
) -> RiskOptimizationResult:
    """
    Main entry point for risk-based portfolio optimization.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for all assets
    symbols : list[str]
        Assets to include in optimization
    method : RiskOptimizationMethod
        Optimization method to use
    constraints : RiskOptimizationConstraints
        Position and turnover constraints
    current_weights : dict[str, float], optional
        Current portfolio weights (for turnover calculation)
    
    Returns
    -------
    RiskOptimizationResult
        Optimal weights and diagnostics
    """
    # Filter returns to symbols
    available = [s for s in symbols if s in returns.columns]
    if len(available) == 0:
        raise ValueError("No price data available for specified symbols")
    
    returns_filtered = returns[available].dropna()
    
    if len(returns_filtered) < 60:
        raise ValueError("Insufficient data for optimization (need at least 60 days)")
    
    # Current weights array
    current_arr = None
    if current_weights is not None:
        current_arr = np.array([current_weights.get(s, 0) for s in available])
        if current_arr.sum() > 0:
            current_arr = current_arr / current_arr.sum()
    
    # Compute covariance (with shrinkage for stability)
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(returns_filtered)
    cov = lw.covariance_
    
    # Run optimization
    if method == RiskOptimizationMethod.RISK_PARITY:
        return optimize_risk_parity(cov, available, constraints, current_arr)
    
    elif method == RiskOptimizationMethod.MIN_VARIANCE:
        return optimize_min_variance(cov, available, constraints, current_arr)
    
    elif method == RiskOptimizationMethod.MAX_DIVERSIFICATION:
        return optimize_max_diversification(cov, available, constraints, current_arr)
    
    elif method == RiskOptimizationMethod.CVAR:
        return optimize_cvar(returns_filtered, available, 0.05, constraints, current_arr)
    
    elif method == RiskOptimizationMethod.HIERARCHICAL_RISK_PARITY:
        return optimize_hrp(returns_filtered, available, constraints, current_arr)
    
    elif method == RiskOptimizationMethod.EQUAL_WEIGHT:
        n = len(available)
        weights = np.ones(n) / n
        port_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252)
        asset_vols = np.sqrt(np.diag(cov))
        div_ratio = float((weights @ asset_vols) / np.sqrt(weights @ cov @ weights))
        
        return RiskOptimizationResult(
            method=method.value,
            weights={s: float(weights[i]) for i, s in enumerate(available)},
            portfolio_volatility=port_vol,
            portfolio_var_95=float(-1.645 * np.sqrt(weights @ cov @ weights)),
            diversification_ratio=div_ratio,
            converged=True,
            iterations=0,
            objective_value=0.0,
        )
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def generate_allocation_recommendation(
    returns: pd.DataFrame,
    symbols: list[str],
    current_weights: dict[str, float],
    inflow_eur: float,
    portfolio_value_eur: float,
    method: RiskOptimizationMethod = RiskOptimizationMethod.RISK_PARITY,
) -> AllocationRecommendation:
    """
    Generate user-facing allocation recommendation.
    
    Answers: "Where should my next €X go?"
    """
    # Optimize portfolio
    constraints = RiskOptimizationConstraints(
        min_weight=0.02,
        max_weight=0.35,
        max_turnover=0.20,  # Limit turnover to 20% of portfolio
    )
    
    result = optimize_portfolio_risk_based(
        returns, symbols, method, constraints, current_weights
    )
    
    # Generate trade recommendations
    total_value = portfolio_value_eur + inflow_eur
    recommendations = []
    
    for symbol in result.weights:
        current = current_weights.get(symbol, 0)
        target = result.weights[symbol]
        
        current_value = current * portfolio_value_eur
        target_value = target * total_value
        trade_value = target_value - current_value
        
        if abs(trade_value) < 50:  # Minimum trade size
            continue
        
        action = "BUY" if trade_value > 0 else "SELL"
        
        # Reason for trade
        if current == 0:
            reason = "New position for diversification"
        elif action == "BUY":
            reason = "Increase position to improve risk balance"
        else:
            reason = "Reduce overweight position"
        
        recommendations.append({
            "symbol": symbol,
            "action": action,
            "amount_eur": abs(trade_value),
            "current_weight_pct": current * 100,
            "target_weight_pct": target * 100,
            "reason": reason,
        })
    
    # Sort by absolute trade size
    recommendations.sort(key=lambda x: x["amount_eur"], reverse=True)
    
    # Current risk
    current_weights_arr = np.array([current_weights.get(s, 0) for s in result.weights.keys()])
    if current_weights_arr.sum() > 0:
        current_weights_arr = current_weights_arr / current_weights_arr.sum()
    else:
        current_weights_arr = np.ones(len(result.weights)) / len(result.weights)
    
    avail_cols = [s for s in result.weights.keys() if s in returns.columns]
    if avail_cols:
        cov = returns[avail_cols].cov().values
        current_vol = float(np.sqrt(current_weights_arr @ cov @ current_weights_arr)) * np.sqrt(252)
    else:
        current_vol = result.portfolio_volatility
    
    current_risk = {
        "volatility": current_vol,
        "volatility_label": _vol_to_label(current_vol),
    }
    
    optimal_risk = {
        "volatility": result.portfolio_volatility,
        "volatility_label": _vol_to_label(result.portfolio_volatility),
        "diversification_ratio": result.diversification_ratio,
    }
    
    # Risk improvement summary
    vol_change = (result.portfolio_volatility - current_vol) / current_vol * 100 if current_vol > 0 else 0
    if vol_change < -5:
        risk_summary = f"Risk reduced by {abs(vol_change):.0f}%"
    elif vol_change > 5:
        risk_summary = f"Risk increased by {vol_change:.0f}%"
    else:
        risk_summary = "Risk approximately unchanged"
    
    # Confidence
    if result.converged and result.diversification_ratio > 1.2:
        confidence = "HIGH"
    elif result.converged:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Explanation
    method_explanations = {
        RiskOptimizationMethod.RISK_PARITY.value: "Allocates so each position contributes equally to risk",
        RiskOptimizationMethod.MIN_VARIANCE.value: "Minimizes total portfolio volatility",
        RiskOptimizationMethod.MAX_DIVERSIFICATION.value: "Maximizes diversification benefit",
        RiskOptimizationMethod.CVAR.value: "Minimizes expected loss in worst scenarios",
        RiskOptimizationMethod.HIERARCHICAL_RISK_PARITY.value: "Uses correlation clustering for robust allocation",
    }
    explanation = method_explanations.get(result.method, "Risk-based optimization")
    
    # Warnings
    warnings = []
    if result.turnover > 0.15:
        warnings.append(f"High turnover ({result.turnover*100:.0f}%) - consider phasing trades")
    if result.diversification_ratio < 1.1:
        warnings.append("Limited diversification benefit - assets are highly correlated")
    if not result.converged:
        warnings.append("Optimization may not have found optimal solution")
    
    return AllocationRecommendation(
        recommendations=recommendations,
        current_portfolio=current_weights,
        optimal_portfolio=result.weights,
        current_risk=current_risk,
        optimal_risk=optimal_risk,
        risk_improvement_summary=risk_summary,
        confidence=confidence,
        explanation=explanation,
        warnings=warnings,
    )


def _vol_to_label(vol: float) -> str:
    """Convert volatility to user-friendly label."""
    if vol < 0.10:
        return "Low"
    elif vol < 0.18:
        return "Moderate"
    elif vol < 0.25:
        return "High"
    else:
        return "Very High"
