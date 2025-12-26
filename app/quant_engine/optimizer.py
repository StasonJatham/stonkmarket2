"""
Portfolio optimizer with constraints and transaction cost modeling.

Solves the incremental mean-variance optimization problem:

max_{Δw} (w+Δw)' μ_hat - λ (w+Δw)' Σ (w+Δw) - TC(Δw)

Subject to:
- Long-only: w+Δw ≥ 0
- Position limits: w+Δw ≤ w_max
- Turnover: ||Δw||_1 ≤ T_max + inflow/portfolio_value
- Budget: sum(w+Δw) ≤ 1 + inflow/portfolio_value

Transaction costs are modeled as fixed €1 per trade using
utility-based pruning.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from app.quant_engine.types import (
    ConstraintStatus,
    OptimizationResult,
    RiskModel,
    SolverStatus,
)


logger = logging.getLogger(__name__)


def solve_qp_incremental(
    w_current: np.ndarray,
    mu_hat: np.ndarray,
    sigma: np.ndarray,
    inflow_w: float,
    lambda_risk: float,
    max_weight: float,
    max_turnover: float,
    turnover_penalty: float,
    allow_cash: bool = True,
    assets: list[str] | None = None,
) -> OptimizationResult:
    """
    Solve incremental mean-variance QP.
    
    Parameters
    ----------
    w_current : np.ndarray
        Current portfolio weights.
    mu_hat : np.ndarray
        Expected returns.
    sigma : np.ndarray
        Covariance matrix.
    inflow_w : float
        Inflow as fraction of portfolio value.
    lambda_risk : float
        Risk aversion parameter.
    max_weight : float
        Maximum position weight.
    max_turnover : float
        Maximum turnover (excluding inflow).
    turnover_penalty : float
        L1 penalty on turnover.
    allow_cash : bool
        Whether to allow cash position.
    assets : list[str], optional
        Asset names.
    
    Returns
    -------
    OptimizationResult
        Optimization result.
    """
    n = len(w_current)

    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]

    # Decision variable: weight change
    dw = cp.Variable(n)
    w_new = w_current + dw

    # Objective: maximize return - risk - turnover penalty
    expected_return = mu_hat @ w_new
    risk = cp.quad_form(w_new, sigma)
    turnover = cp.norm1(dw)

    objective = cp.Maximize(
        expected_return - lambda_risk * risk - turnover_penalty * turnover
    )

    # Constraints
    constraints = [
        w_new >= 0,  # Long-only
        w_new <= max_weight,  # Position limits
        turnover <= max_turnover + max(0.0, inflow_w),  # Turnover limit
    ]

    # Budget constraint
    if allow_cash:
        # Allow holding cash (weights sum to less than 1 + inflow)
        constraints.append(cp.sum(w_new) <= 1.0 + inflow_w)
    else:
        # Fully invested
        constraints.append(cp.sum(w_new) == 1.0 + inflow_w)

    # Solve
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception as e:
        logger.warning(f"OSQP failed: {e}, trying ECOS")
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as e2:
            logger.error(f"All solvers failed: {e2}")
            return OptimizationResult(
                w_current=w_current,
                w_new=w_current.copy(),
                dw=np.zeros_like(w_current),
                assets=assets,
                status=SolverStatus.ERROR,
                objective_value=0.0,
                constraint_status=ConstraintStatus([], False, [], 1.0),
                transaction_cost_eur=0.0,
                marginal_utilities=np.zeros_like(w_current),
            )

    # Parse status
    status_map = {
        "optimal": SolverStatus.OPTIMAL,
        "optimal_inaccurate": SolverStatus.OPTIMAL_INACCURATE,
        "infeasible": SolverStatus.INFEASIBLE,
        "unbounded": SolverStatus.UNBOUNDED,
    }
    status = status_map.get(problem.status, SolverStatus.ERROR)

    if w_new.value is None or dw.value is None:
        logger.warning(f"Solver returned no solution: {problem.status}")
        return OptimizationResult(
            w_current=w_current,
            w_new=w_current.copy(),
            dw=np.zeros_like(w_current),
            assets=assets,
            status=status,
            objective_value=0.0,
            constraint_status=ConstraintStatus([], False, [], 1.0),
            transaction_cost_eur=0.0,
            marginal_utilities=np.zeros_like(w_current),
        )

    w_new_val = np.asarray(w_new.value).reshape(-1)
    dw_val = np.asarray(dw.value).reshape(-1)

    # Identify binding constraints
    max_weight_binding = [
        assets[i] for i in range(n)
        if abs(w_new_val[i] - max_weight) < 1e-6
    ]

    actual_turnover = float(np.sum(np.abs(dw_val)))
    turnover_binding = abs(actual_turnover - (max_turnover + inflow_w)) < 1e-6

    budget_slack = 1.0 + inflow_w - np.sum(w_new_val)

    # Compute marginal utilities (gradient of objective wrt w)
    # ∂obj/∂w = μ - 2λΣw
    marginal_u = mu_hat - 2 * lambda_risk * (sigma @ w_new_val)

    logger.info(
        f"Optimization: status={status.value}, "
        f"obj={problem.value:.6f}, turnover={actual_turnover:.4f}"
    )

    return OptimizationResult(
        w_current=w_current,
        w_new=w_new_val,
        dw=dw_val,
        assets=assets,
        status=status,
        objective_value=float(problem.value) if problem.value else 0.0,
        constraint_status=ConstraintStatus(
            max_weight_binding=max_weight_binding,
            turnover_binding=turnover_binding,
            min_trade_filtered=[],  # Filled after cost pruning
            budget_slack=float(budget_slack),
        ),
        transaction_cost_eur=0.0,  # Filled after cost pruning
        marginal_utilities=marginal_u,
    )


def mean_variance_utility(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float,
) -> float:
    """
    Compute mean-variance utility.
    
    U(w) = w'μ - λ w'Σw
    
    Parameters
    ----------
    w : np.ndarray
        Portfolio weights.
    mu : np.ndarray
        Expected returns.
    sigma : np.ndarray
        Covariance matrix.
    lambda_risk : float
        Risk aversion.
    
    Returns
    -------
    float
        Utility value.
    """
    return float(w @ mu - lambda_risk * (w @ sigma @ w))


def apply_fixed_cost_pruning(
    w_current: np.ndarray,
    dw: np.ndarray,
    mu_hat: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float,
    portfolio_value_eur: float,
    fixed_cost_eur: float,
    min_trade_eur: float,
) -> tuple[np.ndarray, list[str], float]:
    """
    Apply fixed cost pruning to remove trades that don't justify €1 cost.
    
    Iteratively removes trades where marginal utility < fixed cost.
    
    Parameters
    ----------
    w_current : np.ndarray
        Current weights.
    dw : np.ndarray
        Proposed weight changes.
    mu_hat : np.ndarray
        Expected returns.
    sigma : np.ndarray
        Covariance matrix.
    lambda_risk : float
        Risk aversion.
    portfolio_value_eur : float
        Portfolio value in EUR.
    fixed_cost_eur : float
        Fixed cost per trade (€1).
    min_trade_eur : float
        Minimum trade size in EUR.
    
    Returns
    -------
    tuple[np.ndarray, list[str], float]
        Pruned dw, list of filtered assets, total transaction cost.
    """
    dw_pruned = dw.copy()

    # Convert fixed cost to utility units
    cost_utility = fixed_cost_eur / max(portfolio_value_eur, 1.0)

    def is_active_trade(dw_vec: np.ndarray, idx: int) -> bool:
        """Check if trade at idx meets minimum size."""
        trade_eur = abs(dw_vec[idx]) * portfolio_value_eur
        return trade_eur >= min_trade_eur

    def get_active_mask(dw_vec: np.ndarray) -> np.ndarray:
        """Get mask of active trades."""
        return np.array([
            is_active_trade(dw_vec, i) for i in range(len(dw_vec))
        ])

    filtered_indices = []

    # Iterative pruning
    for _ in range(50):  # Max iterations
        active = get_active_mask(dw_pruned)

        if not np.any(active):
            # No active trades remain
            return np.zeros_like(dw_pruned), list(range(len(dw))), 0.0

        # Compute utility with all active trades
        dw_active = dw_pruned.copy()
        dw_active[~active] = 0.0

        base_utility = mean_variance_utility(
            w_current + dw_active, mu_hat, sigma, lambda_risk
        ) - mean_variance_utility(w_current, mu_hat, sigma, lambda_risk)

        # Check each active trade
        dropped = False

        for i in np.where(active)[0]:
            # Utility without this trade
            dw_drop = dw_active.copy()
            dw_drop[i] = 0.0

            utility_without = mean_variance_utility(
                w_current + dw_drop, mu_hat, sigma, lambda_risk
            ) - mean_variance_utility(w_current, mu_hat, sigma, lambda_risk)

            marginal_utility = base_utility - utility_without

            # If marginal utility < cost, drop the trade
            if marginal_utility < cost_utility:
                dw_pruned[i] = 0.0
                filtered_indices.append(i)
                dropped = True

        if not dropped:
            break

    # Final cleanup: zero out sub-minimum trades
    final_active = get_active_mask(dw_pruned)
    dw_pruned[~final_active] = 0.0

    # Count active trades for cost
    n_trades = int(np.sum(np.abs(dw_pruned) > 1e-8))
    total_cost = n_trades * fixed_cost_eur

    logger.info(
        f"Cost pruning: {n_trades} trades, €{total_cost:.2f} total cost, "
        f"{len(filtered_indices)} trades filtered"
    )

    return dw_pruned, filtered_indices, total_cost


def optimize_portfolio(
    w_current: np.ndarray,
    mu_hat: pd.Series,
    risk_model: RiskModel,
    inflow_eur: float,
    portfolio_value_eur: float,
    lambda_risk: float = 10.0,
    max_weight: float = 0.15,
    max_turnover: float = 0.20,
    turnover_penalty: float = 0.001,
    fixed_cost_eur: float = 1.0,
    min_trade_eur: float = 10.0,
    allow_cash: bool = True,
) -> OptimizationResult:
    """
    Full portfolio optimization pipeline with cost pruning.
    
    Parameters
    ----------
    w_current : np.ndarray
        Current weights (aligned with risk_model.assets).
    mu_hat : pd.Series
        Expected returns (indexed by asset).
    risk_model : RiskModel
        Fitted risk model.
    inflow_eur : float
        Monthly inflow in EUR.
    portfolio_value_eur : float
        Current portfolio value in EUR.
    lambda_risk : float
        Risk aversion.
    max_weight : float
        Maximum position weight.
    max_turnover : float
        Maximum monthly turnover.
    turnover_penalty : float
        L1 turnover penalty.
    fixed_cost_eur : float
        Fixed cost per trade.
    min_trade_eur : float
        Minimum trade size.
    allow_cash : bool
        Allow cash position.
    
    Returns
    -------
    OptimizationResult
        Complete optimization result.
    """
    assets = risk_model.assets
    n = len(assets)

    # Align mu_hat with assets
    mu_aligned = mu_hat.reindex(assets).fillna(0.0).values

    # Get covariance matrix
    sigma = risk_model.get_covariance()

    # Inflow as fraction of portfolio
    inflow_w = inflow_eur / max(portfolio_value_eur, 1.0)

    # Solve QP
    result = solve_qp_incremental(
        w_current=w_current,
        mu_hat=mu_aligned,
        sigma=sigma,
        inflow_w=inflow_w,
        lambda_risk=lambda_risk,
        max_weight=max_weight,
        max_turnover=max_turnover,
        turnover_penalty=turnover_penalty,
        allow_cash=allow_cash,
        assets=assets,
    )

    if result.status not in (SolverStatus.OPTIMAL, SolverStatus.OPTIMAL_INACCURATE):
        logger.warning(f"Optimization failed: {result.status}")
        return result

    # Apply fixed cost pruning
    dw_pruned, filtered_idx, total_cost = apply_fixed_cost_pruning(
        w_current=w_current,
        dw=result.dw,
        mu_hat=mu_aligned,
        sigma=sigma,
        lambda_risk=lambda_risk,
        portfolio_value_eur=portfolio_value_eur,
        fixed_cost_eur=fixed_cost_eur,
        min_trade_eur=min_trade_eur,
    )

    w_new_pruned = w_current + dw_pruned

    # Update constraint status
    min_trade_filtered = [assets[i] for i in filtered_idx]

    constraint_status = ConstraintStatus(
        max_weight_binding=result.constraint_status.max_weight_binding,
        turnover_binding=result.constraint_status.turnover_binding,
        min_trade_filtered=min_trade_filtered,
        budget_slack=result.constraint_status.budget_slack,
    )

    # Recompute marginal utilities with pruned weights
    marginal_u = mu_aligned - 2 * lambda_risk * (sigma @ w_new_pruned)

    return OptimizationResult(
        w_current=w_current,
        w_new=w_new_pruned,
        dw=dw_pruned,
        assets=assets,
        status=result.status,
        objective_value=result.objective_value,
        constraint_status=constraint_status,
        transaction_cost_eur=total_cost,
        marginal_utilities=marginal_u,
    )


def compute_delta_utility(
    w_current: np.ndarray,
    dw: float,
    idx: int,
    mu_hat: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float,
    fixed_cost_eur: float,
    portfolio_value_eur: float,
) -> float:
    """
    Compute net delta utility for a single trade.
    
    Parameters
    ----------
    w_current : np.ndarray
        Current weights.
    dw : float
        Weight change for this asset.
    idx : int
        Asset index.
    mu_hat : np.ndarray
        Expected returns.
    sigma : np.ndarray
        Covariance matrix.
    lambda_risk : float
        Risk aversion.
    fixed_cost_eur : float
        Fixed cost per trade.
    portfolio_value_eur : float
        Portfolio value.
    
    Returns
    -------
    float
        Net utility change after costs.
    """
    dw_vec = np.zeros_like(w_current)
    dw_vec[idx] = dw

    w_new = w_current + dw_vec

    utility_before = mean_variance_utility(w_current, mu_hat, sigma, lambda_risk)
    utility_after = mean_variance_utility(w_new, mu_hat, sigma, lambda_risk)

    gross_delta = utility_after - utility_before

    # Subtract fixed cost (in utility units)
    cost_utility = fixed_cost_eur / max(portfolio_value_eur, 1.0)

    return gross_delta - cost_utility
