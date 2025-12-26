"""
Quantitative Portfolio Engine API routes.

Provides endpoints for generating recommendations from the quant engine.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.quant_engine import (
    QuantEngineService,
    get_default_config,
)
from app.repositories import auth_user_orm as auth_repo
from app.repositories import portfolios_orm as portfolios_repo
from app.repositories import price_history_orm as price_history_repo
from app.schemas.quant_engine import (
    AuditBlockResponse,
    EngineOutputResponse,
    GenerateRecommendationsRequest,
    RecommendationRowResponse,
    TuningResultResponse,
    ValidationResultResponse,
)


if TYPE_CHECKING:
    from app.core.security import TokenData


router = APIRouter(prefix="/portfolios", tags=["Quant Engine"])
logger = get_logger("routes.quant_engine")


async def _get_user_id(user: TokenData) -> int:
    """Get user ID from token data."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


async def _fetch_prices_for_symbols(
    symbols: list[str],
    lookback_days: int = 400,
) -> pd.DataFrame:
    """
    Fetch price history for multiple symbols.

    Returns a DataFrame with dates as index and symbols as columns.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    price_dfs = {}
    for symbol in symbols:
        df = await price_history_repo.get_prices_as_dataframe(
            symbol, start_date, end_date
        )
        if df is not None and "Close" in df.columns:
            price_dfs[symbol] = df["Close"]

    if not price_dfs:
        return pd.DataFrame()

    # Combine into single DataFrame
    prices = pd.DataFrame(price_dfs)
    prices = prices.dropna(how="all")
    prices = prices.ffill()  # Forward fill missing

    return prices


async def _fetch_market_benchmark(lookback_days: int = 400) -> pd.Series | None:
    """Fetch market benchmark prices (SPY)."""
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    df = await price_history_repo.get_prices_as_dataframe(
        "SPY", start_date, end_date
    )
    if df is not None and "Close" in df.columns:
        return df["Close"]
    return None


def _holdings_to_weights(
    holdings: list[dict[str, Any]],
    portfolio_value_eur: float,
    current_prices: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert holdings to weight array.

    Returns (weights, assets) tuple.
    """
    if portfolio_value_eur <= 0:
        return np.array([]), []

    assets = []
    weights = []

    for h in holdings:
        symbol = h["symbol"]
        quantity = float(h.get("quantity", 0))

        if symbol in current_prices.index:
            price = current_prices[symbol]
            value = quantity * price
            weight = value / portfolio_value_eur
            assets.append(symbol)
            weights.append(weight)

    if not assets:
        return np.array([]), []

    # Normalize weights
    weight_arr = np.array(weights)
    total = weight_arr.sum()
    if total > 0:
        weight_arr = weight_arr / total

    return weight_arr, assets


def _engine_output_to_response(output: Any) -> EngineOutputResponse:
    """Convert EngineOutput dataclass to Pydantic response."""
    recommendations = []
    for r in output.recommendations:
        rec_dict = r.to_dict() if hasattr(r, "to_dict") else vars(r)
        recommendations.append(RecommendationRowResponse(
            ticker=rec_dict.get("ticker", ""),
            action=rec_dict.get("action", "HOLD"),
            notional_eur=rec_dict.get("notional_eur", 0.0),
            delta_weight=rec_dict.get("delta_weight", 0.0),
            target_weight=rec_dict.get("target_weight", 0.0),
            mu_hat=rec_dict.get("mu_hat", 0.0),
            uncertainty=rec_dict.get("mu_hat_uncertainty", {}).get("oos_rmse", 0.0)
            if isinstance(rec_dict.get("mu_hat_uncertainty"), dict)
            else 0.0,
            risk_contribution=rec_dict.get("risk", {}).get("mcr", 0.0)
            if isinstance(rec_dict.get("risk"), dict)
            else 0.0,
            dip_score=rec_dict.get("dip", {}).get("dip_score")
            if isinstance(rec_dict.get("dip"), dict)
            else None,
            dip_bucket=rec_dict.get("dip", {}).get("bucket")
            if isinstance(rec_dict.get("dip"), dict)
            else None,
            marginal_utility=rec_dict.get("delta_utility_net", 0.0),
        ))

    # Build audit block
    audit_dict = output.audit.to_dict() if hasattr(output.audit, "to_dict") else {}
    audit = AuditBlockResponse(
        timestamp=datetime.now(),
        config_hash=hash(str(audit_dict.get("hyperparams", {}))),
        mu_hat_summary={
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
        risk_model_summary=audit_dict.get("risk_model", {}),
        optimizer_status=output.solver_status.value
        if hasattr(output.solver_status, "value")
        else str(output.solver_status),
        constraint_binding=audit_dict.get("constraint_binding", []),
        turnover_realized=audit_dict.get("turnover", 0.0),
        regime_state=audit_dict.get("regime", "neutral_medium"),
        dip_stats=audit_dict.get("dip_stats"),
        error_message=None,
    )

    return EngineOutputResponse(
        recommendations=recommendations,
        as_of_date=output.as_of
        if isinstance(output.as_of, datetime)
        else datetime.combine(output.as_of, datetime.min.time()),
        portfolio_value_eur=output.portfolio_value_eur,
        inflow_eur=output.inflow_eur,
        total_trades=len([r for r in recommendations if r.action != "HOLD"]),
        total_transaction_cost_eur=output.total_transaction_cost_eur,
        expected_portfolio_return=output.expected_portfolio_return,
        expected_portfolio_risk=output.expected_portfolio_risk,
        audit=audit,
    )


@router.post(
    "/{portfolio_id}/recommendations",
    response_model=EngineOutputResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Quant Engine Recommendations",
    description="""
    Generate portfolio recommendations using the quantitative engine.

    The engine uses:
    - Ridge/Lasso ensemble alpha model with OOS validation
    - PCA-based factor risk model
    - Incremental mean-variance optimization with transaction costs
    - DipScore adjustments (informational only, never generates orders)

    Returns ranked recommendations with full audit trail.
    """,
)
async def generate_recommendations(
    portfolio_id: int,
    payload: GenerateRecommendationsRequest,
    user: TokenData = Depends(require_user),
) -> EngineOutputResponse:
    """Generate quant engine recommendations for a portfolio."""
    user_id = await _get_user_id(user)

    # Verify portfolio ownership
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    # Get holdings
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    # Extract symbols from holdings
    symbols = [h["symbol"] for h in holdings]

    # Fetch price history
    prices = await _fetch_prices_for_symbols(symbols)
    if prices.empty:
        raise ValidationError(
            message="No price history available for portfolio symbols"
        )

    # Fetch market benchmark
    market_prices = await _fetch_market_benchmark()

    # Calculate current weights
    if prices.columns.tolist():
        current_prices = prices.iloc[-1]
        w_current, assets = _holdings_to_weights(
            holdings, payload.portfolio_value_eur, current_prices
        )
    else:
        w_current = np.array([])
        assets = []

    if len(assets) == 0:
        raise ValidationError(
            message="Unable to compute weights for portfolio holdings"
        )

    # Filter prices to only assets we have weights for
    prices = prices[assets]

    # Use provided weights if available
    if payload.current_weights:
        w_current = np.array([
            payload.current_weights.get(a, 0.0) for a in assets
        ])
        # Normalize
        total = w_current.sum()
        if total > 0:
            w_current = w_current / total

    # Initialize engine with default config
    config = get_default_config()
    engine = QuantEngineService(config=config)

    # Generate recommendations
    output = engine.generate_recommendations(
        prices=prices,
        w_current=w_current,
        portfolio_value_eur=payload.portfolio_value_eur,
        inflow_eur=payload.inflow_eur,
        market_prices=market_prices,
        retrain=payload.force_retrain,
    )

    return _engine_output_to_response(output)


@router.post(
    "/{portfolio_id}/quant/validate",
    response_model=ValidationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Walk-Forward Validation",
    description="""
    Run walk-forward validation on the quant engine for a portfolio.

    This validates the model using historical data with proper train/test splits.
    """,
)
async def validate_walk_forward(
    portfolio_id: int,
    n_folds: int = Query(5, ge=2, le=20),
    user: TokenData = Depends(require_user),
) -> ValidationResultResponse:
    """Run walk-forward validation for a portfolio."""
    user_id = await _get_user_id(user)

    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    # Fetch longer history for walk-forward
    prices = await _fetch_prices_for_symbols(symbols, lookback_days=1500)
    if prices.empty:
        raise ValidationError(
            message="Insufficient price history for validation"
        )

    market_prices = await _fetch_market_benchmark(lookback_days=1500)

    config = get_default_config()
    engine = QuantEngineService(config=config)

    result = engine.validate_walk_forward(
        prices=prices,
        market_prices=market_prices,
        n_folds=n_folds,
    )

    return ValidationResultResponse(
        n_folds=len(result.folds),
        aggregate_sharpe=result.aggregate_sharpe,
        aggregate_return=result.aggregate_return,
        aggregate_volatility=result.aggregate_volatility,
        max_drawdown=result.aggregate_max_drawdown,
        baseline_sharpe=result.baseline_sharpe,
        hit_rate=result.hit_rate,
        total_turnover=result.total_turnover,
    )


@router.post(
    "/{portfolio_id}/quant/tune",
    response_model=TuningResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Tune Hyperparameters",
    description="""
    Tune quant engine hyperparameters using nested walk-forward validation.

    This is a computationally expensive operation.
    """,
)
async def tune_hyperparameters(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> TuningResultResponse:
    """Tune hyperparameters for a portfolio."""
    import time

    user_id = await _get_user_id(user)

    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    prices = await _fetch_prices_for_symbols(symbols, lookback_days=1500)
    if prices.empty:
        raise ValidationError(
            message="Insufficient price history for tuning"
        )

    market_prices = await _fetch_market_benchmark(lookback_days=1500)

    start_time = time.time()

    config = get_default_config()
    engine = QuantEngineService(config=config)

    best_params, best_score, logs = engine.tune_hyperparameters(
        prices=prices,
        market_prices=market_prices,
    )

    elapsed = time.time() - start_time

    return TuningResultResponse(
        best_params=best_params,
        best_score=best_score,
        n_evaluations=len(logs),
        tuning_time_seconds=elapsed,
    )


# =============================================================================
# GLOBAL RECOMMENDATIONS (for Dashboard - all tracked symbols)
# =============================================================================

# Separate router for global recommendations (not under /portfolios)
global_router = APIRouter(prefix="/recommendations", tags=["Quant Engine"])


async def _get_cached_recommendations(inflow_eur: float) -> dict | None:
    """Try to get cached recommendations for the given inflow amount."""
    from app.cache.cache import Cache

    cache = Cache(prefix="quant", default_ttl=86400)

    # Round to nearest cached inflow amount
    cached_inflows = [500, 1000, 2000, 5000]
    closest = min(cached_inflows, key=lambda x: abs(x - inflow_eur))

    # Only use cache if within 20% of a cached amount
    if abs(closest - inflow_eur) / closest <= 0.2:
        cached = await cache.get(f"recommendations:{closest}")
        if cached:
            logger.info(f"Cache hit for recommendations (inflow={closest})")
            return cached

    return None


@global_router.get(
    "",
    response_model=EngineOutputResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Global Quant Recommendations",
    description="""
    Generate quant engine recommendations for ALL tracked symbols.

    This endpoint is used by the Dashboard to show the "best buys" list
    ranked by statistical expected utility (μ_hat - λ*risk - costs).

    The ranking comes directly from the optimizer's Δw* output, not from
    any heuristic dip triggers.

    No portfolio or authentication required - shows global opportunities.

    Results are cached daily for common inflow amounts (500, 1000, 2000, 5000 EUR).
    """,
)
async def get_global_recommendations(
    inflow_eur: float = Query(
        default=1000.0,
        ge=100.0,
        le=10000.0,
        description="Monthly inflow amount to optimize for",
    ),
    limit: int = Query(
        default=40,
        ge=1,
        le=100,
        description="Max number of recommendations to return",
    ),
    skip_cache: bool = Query(
        default=False,
        description="Skip cache and compute fresh recommendations",
    ),
) -> EngineOutputResponse:
    """
    Get global quant engine recommendations for all tracked symbols.

    This is the primary data source for the Dashboard. Stocks are ranked
    by their marginal utility from the optimizer (expected return adjusted
    for risk and costs), NOT by dip score.

    Checks cache first (populated by daily job), falls back to live computation.
    """
    from app.repositories import dips_orm as dips_repo
    from app.repositories import symbols_orm as symbols_repo

    # Check cache first (unless skip_cache is True)
    if not skip_cache:
        cached = await _get_cached_recommendations(inflow_eur)
        if cached:
            # Convert cached dict to response
            recs = [
                RecommendationRowResponse(
                    ticker=r["ticker"],
                    name=r.get("name"),
                    action=r["action"],
                    notional_eur=r["notional_eur"],
                    delta_weight=r["delta_weight"],
                    target_weight=r["target_weight"],
                    mu_hat=r["mu_hat"],
                    uncertainty=r["uncertainty"],
                    risk_contribution=r["risk_contribution"],
                    dip_score=r.get("dip_score"),
                    dip_bucket=r.get("dip_bucket"),
                    marginal_utility=r["marginal_utility"],
                    legacy_dip_pct=r.get("legacy_dip_pct"),
                    legacy_days_in_dip=r.get("legacy_days_in_dip"),
                    legacy_domain_score=r.get("legacy_domain_score"),
                )
                for r in cached["recommendations"][:limit]
            ]
            return EngineOutputResponse(
                recommendations=recs,
                as_of_date=datetime.fromisoformat(cached["as_of_date"]),
                portfolio_value_eur=cached["portfolio_value_eur"],
                inflow_eur=cached["inflow_eur"],
                total_trades=cached["total_trades"],
                total_transaction_cost_eur=cached["total_transaction_cost_eur"],
                expected_portfolio_return=cached["expected_portfolio_return"],
                expected_portfolio_risk=cached["expected_portfolio_risk"],
                audit=AuditBlockResponse(
                    timestamp=datetime.fromisoformat(cached["audit"]["timestamp"]),
                    config_hash=cached["audit"]["config_hash"],
                    mu_hat_summary=cached["audit"]["mu_hat_summary"],
                    risk_model_summary=cached["audit"]["risk_model_summary"],
                    optimizer_status=cached["audit"]["optimizer_status"],
                    constraint_binding=cached["audit"]["constraint_binding"],
                    turnover_realized=cached["audit"]["turnover_realized"],
                    regime_state=cached["audit"]["regime_state"],
                    dip_stats=cached["audit"].get("dip_stats"),
                    error_message=cached["audit"].get("error_message"),
                ),
            )

    # Fallback to live computation
    # Get all tracked symbols
    symbols_list = await symbols_repo.list_symbols()
    if not symbols_list:
        # Return empty response
        return EngineOutputResponse(
            recommendations=[],
            as_of_date=datetime.now(),
            portfolio_value_eur=0.0,
            inflow_eur=inflow_eur,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=AuditBlockResponse(
                timestamp=datetime.now(),
                config_hash=0,
                mu_hat_summary={"mean": 0, "std": 0, "min": 0, "max": 0},
                risk_model_summary={},
                optimizer_status="no_data",
                constraint_binding=[],
                turnover_realized=0.0,
                regime_state="neutral_medium",
                dip_stats=None,
                error_message="No symbols tracked",
            ),
        )

    symbols = [s.symbol for s in symbols_list]

    # Fetch price history
    prices = await _fetch_prices_for_symbols(symbols, lookback_days=400)
    if prices.empty:
        return EngineOutputResponse(
            recommendations=[],
            as_of_date=datetime.now(),
            portfolio_value_eur=0.0,
            inflow_eur=inflow_eur,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=AuditBlockResponse(
                timestamp=datetime.now(),
                config_hash=0,
                mu_hat_summary={"mean": 0, "std": 0, "min": 0, "max": 0},
                risk_model_summary={},
                optimizer_status="no_price_data",
                constraint_binding=[],
                turnover_realized=0.0,
                regime_state="neutral_medium",
                dip_stats=None,
                error_message="No price data available",
            ),
        )

    # Fetch market benchmark
    market_prices = await _fetch_market_benchmark()

    # Get symbol info for names
    symbol_info = {}
    for s in symbols_list:
        symbol_info[s.symbol] = {"name": getattr(s, "name", s.symbol)}

    # Add dip state info (legacy compatibility)
    dip_states = await dips_repo.get_all_dip_states()
    for ds in dip_states:
        if ds.symbol in symbol_info:
            symbol_info[ds.symbol]["dip_pct"] = ds.dip_pct
            symbol_info[ds.symbol]["days_in_dip"] = ds.days_in_dip
            symbol_info[ds.symbol]["domain_score"] = ds.domain_score

    # Create quant engine
    config = get_default_config()
    engine = QuantEngineService(config=config)

    # For global recommendations, assume starting from scratch (w=0)
    # The optimizer will allocate the inflow as if building a new position
    assets = prices.columns.tolist()
    n_assets = len(assets)
    w_current = np.zeros(n_assets)

    # Inflow weight = inflow / assumed portfolio value
    # For global view, assume €10k base portfolio to get reasonable weights
    assumed_portfolio = max(10000.0, inflow_eur * 10)
    inflow_weight = inflow_eur / assumed_portfolio

    # Train the engine
    train_result = engine.train(
        prices=prices,
        market_prices=market_prices,
        as_of_date=prices.index.max().to_pydatetime()
        if hasattr(prices.index.max(), "to_pydatetime")
        else datetime.now(),
    )

    if train_result.get("status") == "error":
        return EngineOutputResponse(
            recommendations=[],
            as_of_date=datetime.now(),
            portfolio_value_eur=assumed_portfolio,
            inflow_eur=inflow_eur,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=AuditBlockResponse(
                timestamp=datetime.now(),
                config_hash=0,
                mu_hat_summary={"mean": 0, "std": 0, "min": 0, "max": 0},
                risk_model_summary={},
                optimizer_status="training_failed",
                constraint_binding=[],
                turnover_realized=0.0,
                regime_state="neutral_medium",
                dip_stats=None,
                error_message=train_result.get("message", "Training failed"),
            ),
        )

    # Generate recommendations
    try:
        output = engine.generate_recommendations(
            prices=prices,
            market_prices=market_prices,
            w_current=w_current,
            inflow_weight=inflow_weight,
            portfolio_value_eur=assumed_portfolio,
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return EngineOutputResponse(
            recommendations=[],
            as_of_date=datetime.now(),
            portfolio_value_eur=assumed_portfolio,
            inflow_eur=inflow_eur,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=AuditBlockResponse(
                timestamp=datetime.now(),
                config_hash=0,
                mu_hat_summary={"mean": 0, "std": 0, "min": 0, "max": 0},
                risk_model_summary={},
                optimizer_status="optimization_failed",
                constraint_binding=[],
                turnover_realized=0.0,
                regime_state="neutral_medium",
                dip_stats=None,
                error_message=str(e),
            ),
        )

    # Convert to response and add symbol names
    response = _engine_output_to_response(output)

    # Enrich recommendations with names and legacy dip info
    for rec in response.recommendations:
        info = symbol_info.get(rec.ticker, {})
        rec.name = info.get("name", rec.ticker)
        rec.legacy_dip_pct = info.get("dip_pct")
        rec.legacy_days_in_dip = info.get("days_in_dip")
        rec.legacy_domain_score = info.get("domain_score")

    # Sort by marginal utility (descending) and limit
    response.recommendations.sort(
        key=lambda r: r.marginal_utility, reverse=True
    )
    response.recommendations = response.recommendations[:limit]

    return response
