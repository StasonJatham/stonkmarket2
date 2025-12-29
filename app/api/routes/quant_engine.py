"""
Quantitative Portfolio Engine V2 API Routes.

Risk-based portfolio optimization - NO return forecasting.

Provides endpoints for:
- Portfolio risk analytics (risk decomposition, tail risk, diversification)
- Risk-based allocation recommendations (Risk Parity, CVaR, HRP, etc.)
- Technical signal scanning with per-stock optimization
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.dependencies import require_user
from app.cache.cache import Cache
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.quant_engine import (
    analyze_portfolio,
    generate_allocation_recommendation,
    RiskOptimizationMethod,
    scan_all_stocks,
    translate_for_user,
    perform_domain_analysis,
    domain_analysis_to_dict,
)
from app.repositories import auth_user_orm as auth_repo
from app.repositories import portfolios_orm as portfolios_repo
from app.repositories import price_history_orm as price_history_repo
from app.schemas.quant_engine import (
    SignalResultResponse,
    SignalScanResponse,
    StockSignalResponse,
)


if TYPE_CHECKING:
    from app.core.security import TokenData


router = APIRouter(prefix="/portfolios", tags=["Quant Engine"])
logger = get_logger("routes.quant_engine")


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_user_id(user: TokenData) -> int:
    """Get user ID from token data."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


async def _fetch_prices_for_symbols(
    symbols: list[str],
    lookback_days: int = 1260,  # 5 years
) -> pd.DataFrame:
    """
    Fetch price history for multiple symbols.

    Returns a DataFrame with dates as index and symbols as columns.
    Falls back to yfinance if data not in database.
    """
    from app.services.data_providers.yfinance_service import get_yfinance_service
    
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    price_dfs = {}
    missing_symbols = []
    
    for symbol in symbols:
        df = await price_history_repo.get_prices_as_dataframe(
            symbol, start_date, end_date
        )
        if df is not None and "Close" in df.columns and len(df) >= 20:
            price_dfs[symbol] = df["Close"]
        else:
            missing_symbols.append(symbol)
    
    # Fetch missing symbols from yfinance (common for sector ETFs)
    if missing_symbols:
        yf_service = get_yfinance_service()
        yf_results = await yf_service.get_price_history_batch(
            missing_symbols, start_date, end_date
        )
        for symbol, (df, _version) in yf_results.items():
            if df is not None and not df.empty and "Close" in df.columns:
                price_dfs[symbol] = df["Close"]

    if not price_dfs:
        return pd.DataFrame()

    # Combine into single DataFrame
    prices = pd.DataFrame(price_dfs)
    prices = prices.dropna(how="all")
    prices = prices.ffill()

    return prices


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from prices."""
    return prices.pct_change().dropna()


def _safe_float(value: float | None, default: float = 0.0) -> float:
    """Convert a float to JSON-safe value (handle NaN/inf)."""
    import math
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return float(value)


# =============================================================================
# Portfolio Analytics Endpoint
# =============================================================================


@router.get(
    "/{portfolio_id}/analytics",
    status_code=status.HTTP_200_OK,
    summary="Portfolio Risk Analytics",
    description="""
    Get comprehensive risk analytics for a portfolio.
    
    This endpoint provides non-predictive risk diagnostics:
    - Risk decomposition (which positions contribute most to risk)
    - Tail risk analysis (VaR, CVaR, max drawdown)
    - Diversification metrics (effective N, diversification ratio)
    - Market regime detection
    - Correlation analysis
    
    All analytics are translated into user-friendly language.
    """,
)
async def get_portfolio_analytics(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> dict[str, Any]:
    """Get risk analytics for a portfolio."""
    user_id = await _get_user_id(user)

    # Verify portfolio ownership
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    # Get holdings
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    # Fetch price history
    prices = await _fetch_prices_for_symbols(symbols)
    if prices.empty or len(prices) < 60:
        raise ValidationError(
            message="Insufficient price history for analysis (need at least 60 days)"
        )

    # Compute returns
    returns = _compute_returns(prices)

    # Calculate current weights from holdings
    current_prices = prices.iloc[-1]
    weights = {}
    total_value = 0.0
    
    for h in holdings:
        symbol = h["symbol"]
        quantity = float(h.get("quantity", 0))
        if symbol in current_prices.index:
            value = quantity * current_prices[symbol]
            weights[symbol] = value
            total_value += value
    
    if total_value > 0:
        weights = {s: v / total_value for s, v in weights.items()}
    else:
        # Equal weight fallback
        weights = {s: 1.0 / len(symbols) for s in symbols}

    # Fetch SPY returns for market regime detection
    spy_prices = await _fetch_prices_for_symbols(["SPY"])
    market_returns = None
    if not spy_prices.empty and "SPY" in spy_prices.columns:
        market_returns = spy_prices["SPY"].pct_change().dropna()

    # Run analytics
    analytics = analyze_portfolio(
        holdings=weights,
        returns=returns,
        market_returns=market_returns,
        total_value=total_value,
    )

    # Translate to user-friendly format
    user_friendly = translate_for_user(analytics)

    return {
        "portfolio_id": portfolio_id,
        "analyzed_at": datetime.now().isoformat(),
        "total_value_eur": _safe_float(total_value),
        "n_positions": analytics.n_positions,
        "risk_score": analytics.overall_risk_score,
        "summary": user_friendly["summary"],
        "risk": user_friendly["risk"],
        "diversification": user_friendly["diversification"],
        "market": user_friendly["market"],
        "insights": user_friendly["insights"],
        "action_items": user_friendly["action_items"],
        # Raw analytics for power users
        "raw": {
            "portfolio_volatility": _safe_float(analytics.risk_decomposition.portfolio_volatility),
            "var_95": _safe_float(analytics.tail_risk.var_95_daily),
            "cvar_95": _safe_float(analytics.tail_risk.cvar_95_daily),
            "max_drawdown": _safe_float(analytics.tail_risk.max_drawdown),
            "effective_n": _safe_float(analytics.diversification.effective_n),
            "diversification_ratio": _safe_float(analytics.diversification.diversification_ratio),
            "regime": analytics.regime.regime,
            "risk_contributions": {
                s: _safe_float(v) 
                for s, v in analytics.risk_decomposition.risk_contribution_pct.items()
            },
        },
    }


# =============================================================================
# Allocation Recommendation Endpoint
# =============================================================================


@router.post(
    "/{portfolio_id}/allocate",
    status_code=status.HTTP_200_OK,
    summary="Get Allocation Recommendation",
    description="""
    Get a risk-based allocation recommendation for new investment.
    
    This answers: "Where should my next €X go?"
    
    Uses risk-based optimization (no return forecasting):
    - RISK_PARITY: Equal risk contribution from each position
    - MIN_VARIANCE: Minimize total portfolio volatility  
    - MAX_DIVERSIFICATION: Maximize diversification ratio
    - CVAR: Minimize expected loss in worst scenarios
    - HRP: Hierarchical Risk Parity (correlation-based clustering)
    
    Returns specific trade recommendations with rationale.
    """,
)
async def get_allocation_recommendation(
    portfolio_id: int,
    inflow_eur: float = Query(1000.0, ge=50, le=100000, description="Amount to invest"),
    method: str = Query(
        "risk_parity",
        description="Optimization method: risk_parity, min_variance, max_diversification, cvar, hrp"
    ),
    user: TokenData = Depends(require_user),
) -> dict[str, Any]:
    """Get allocation recommendation for a portfolio."""
    user_id = await _get_user_id(user)

    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    prices = await _fetch_prices_for_symbols(symbols)
    if prices.empty or len(prices) < 60:
        raise ValidationError(
            message="Insufficient price history for optimization (need at least 60 days)"
        )

    returns = _compute_returns(prices)

    # Calculate current weights and portfolio value
    current_prices = prices.iloc[-1]
    weights = {}
    total_value = 0.0
    
    for h in holdings:
        symbol = h["symbol"]
        quantity = float(h.get("quantity", 0))
        if symbol in current_prices.index:
            value = quantity * current_prices[symbol]
            weights[symbol] = value
            total_value += value
    
    if total_value > 0:
        weights = {s: v / total_value for s, v in weights.items()}
    else:
        weights = {s: 1.0 / len(symbols) for s in symbols}

    # Parse optimization method
    method_map = {
        "risk_parity": RiskOptimizationMethod.RISK_PARITY,
        "min_variance": RiskOptimizationMethod.MIN_VARIANCE,
        "max_diversification": RiskOptimizationMethod.MAX_DIVERSIFICATION,
        "cvar": RiskOptimizationMethod.CVAR,
        "hrp": RiskOptimizationMethod.HIERARCHICAL_RISK_PARITY,
        "equal_weight": RiskOptimizationMethod.EQUAL_WEIGHT,
    }
    opt_method = method_map.get(method.lower(), RiskOptimizationMethod.RISK_PARITY)

    # Generate recommendation
    recommendation = generate_allocation_recommendation(
        returns=returns,
        symbols=list(returns.columns),
        current_weights=weights,
        inflow_eur=inflow_eur,
        portfolio_value_eur=total_value,
        method=opt_method,
    )

    return {
        "portfolio_id": portfolio_id,
        "method": recommendation.method if hasattr(recommendation, 'method') else method,
        "inflow_eur": inflow_eur,
        "portfolio_value_eur": _safe_float(total_value),
        "confidence": recommendation.confidence,
        "explanation": recommendation.explanation,
        "risk_improvement": recommendation.risk_improvement_summary,
        "current_risk": {
            "volatility": _safe_float(recommendation.current_risk.get("volatility", 0)),
            "label": recommendation.current_risk.get("volatility_label", "Unknown"),
        },
        "optimal_risk": {
            "volatility": _safe_float(recommendation.optimal_risk.get("volatility", 0)),
            "label": recommendation.optimal_risk.get("volatility_label", "Unknown"),
            "diversification_ratio": _safe_float(
                recommendation.optimal_risk.get("diversification_ratio", 0)
            ),
        },
        "trades": recommendation.recommendations,
        "current_weights": {s: _safe_float(v) for s, v in recommendation.current_portfolio.items()},
        "target_weights": {s: _safe_float(v) for s, v in recommendation.optimal_portfolio.items()},
        "warnings": recommendation.warnings,
    }


# =============================================================================
# Global Recommendations Router (No Auth Required)
# =============================================================================

global_router = APIRouter(prefix="/recommendations", tags=["Quant Engine"])


# ============================================================================
# Main Recommendations Endpoint (Global - for Landing/Dashboard)
# ============================================================================

from app.schemas.quant_engine import (
    QuantAuditBlock,
    QuantEngineResponse,
    QuantRecommendation,
)


@global_router.get(
    "",
    response_model=QuantEngineResponse,
    summary="Get stock recommendations",
    description="""
    Get ranked stock recommendations combining signal analysis with dip metrics.
    
    This is the main endpoint for the landing page and dashboard.
    Returns stocks ranked by buy opportunity, combining:
    - Technical signal analysis (RSI, MACD, Bollinger, etc.)
    - Dip metrics (percentage from ATH, days in dip)
    - AI analysis snippets (if available)
    
    Each stock includes:
    - action: BUY, SELL, or HOLD recommendation
    - mu_hat: Expected return estimate
    - dip_score: Z-score based dip signal
    - marginal_utility: Ranking score
    - AI summary/rating when available
    """,
)
async def get_recommendations(
    inflow_eur: float = Query(
        1000.0,
        ge=100,
        le=1000000,
        description="Notional investment amount in EUR"
    ),
    limit: int = Query(
        40,
        ge=1,
        le=100,
        description="Maximum recommendations to return"
    ),
) -> QuantEngineResponse:
    """
    Get ranked stock recommendations for landing page and dashboard.
    
    PERFORMANCE OPTIMIZED: Uses pre-computed data from background jobs.
    No expensive signal optimization is done per-request.
    """
    import asyncio
    from datetime import date
    
    from app.repositories import dip_state_orm as dip_state_repo
    from app.repositories import dip_votes_orm as dip_votes_repo
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import dipfinder_orm as dipfinder_repo
    from sqlalchemy import select
    from app.database.connection import get_session
    from app.database.orm import StrategySignal, QuantScore
    from app.cache.cache import Cache
    from app.schemas.quant_engine import EvidenceBlockResponse
    
    # Check cache first (5 minute TTL)
    cache = Cache(prefix="recommendations", default_ttl=300)
    cache_key = f"recs_{inflow_eur}_{limit}"
    cached = await cache.get(cache_key)
    if cached:
        # Reconstruct response from cached dict
        return QuantEngineResponse(**cached)
    
    # Parallel fetch all data sources
    symbols_task = symbols_repo.list_symbols()
    dip_states_task = dip_state_repo.get_all_dip_states()
    ai_analyses_task = dip_votes_repo.get_all_ai_analyses()
    dipfinder_task = dipfinder_repo.get_latest_signals(limit=500)
    
    symbols, dip_states, ai_analyses, dipfinder_signals = await asyncio.gather(
        symbols_task, dip_states_task, ai_analyses_task, dipfinder_task
    )
    
    if not symbols:
        return QuantEngineResponse(
            recommendations=[],
            as_of_date=date.today().isoformat(),
            portfolio_value_eur=0.0,
            inflow_eur=inflow_eur,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=QuantAuditBlock(
                timestamp=datetime.now().isoformat(),
                config_hash=0,
                optimizer_status="no_data",
                regime_state="unknown",
            ),
        )
    
    # Build lookup maps
    dip_state_map = {d.symbol: d for d in dip_states}
    ai_map = {a.symbol: a for a in ai_analyses}
    # Build dipfinder_map correctly
    dipfinder_map = {}
    for sig in (dipfinder_signals or []):
        if sig.ticker not in dipfinder_map:
            dipfinder_map[sig.ticker] = sig
    
    # Get pre-computed strategy signals and quant scores from database
    strategy_map: dict[str, Any] = {}
    quant_score_map: dict[str, Any] = {}
    try:
        async with get_session() as session:
            result = await session.execute(select(StrategySignal))
            strategy_signals = result.scalars().all()
            for sig in strategy_signals:
                strategy_map[sig.symbol] = sig
            
            # Load quant scores (APUS + DOUS)
            result = await session.execute(select(QuantScore))
            quant_scores = result.scalars().all()
            for qs in quant_scores:
                quant_score_map[qs.symbol] = qs
    except Exception as e:
        logger.warning(f"Failed to fetch strategy signals or quant scores: {e}")
    
    # Build recommendations from pre-computed data
    symbol_names = {s.symbol: s.name or s.symbol for s in symbols}
    symbol_sector_map = {s.symbol: s.sector for s in symbols}
    symbol_mcap_map = {s.symbol: s.market_cap for s in symbols}
    symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]
    
    recommendations: list[QuantRecommendation] = []
    
    for symbol in symbol_list:
        dip = dip_state_map.get(symbol)
        ai = ai_map.get(symbol)
        dipfinder = dipfinder_map.get(symbol)
        strategy = strategy_map.get(symbol)
        quant_score = quant_score_map.get(symbol)
        name = symbol_names.get(symbol, symbol)
        sector = symbol_sector_map.get(symbol)
        market_cap = symbol_mcap_map.get(symbol)
        
        # Calculate metrics from dip state (already in DB)
        dip_pct = float(dip.dip_percentage) if dip and dip.dip_percentage is not None else None
        current_price = float(dip.current_price) if dip and dip.current_price else 0
        days_in_dip = None
        if dip and dip.dip_start_date:
            days_in_dip = (date.today() - dip.dip_start_date).days
        
        # Note: DipState doesn't have change_1d, would need to compute from price history
        change_percent = None
        
        # Extract strategy signal data (pre-computed, NOT computed per-request)
        strategy_beats_bh = False
        strategy_beats_spy = False
        strategy_signal_type = None
        strategy_win_rate_val = None
        strategy_vs_bh_pct = None
        expected_recovery_days = None
        win_rate = None
        
        strategy_n_trades = None
        if strategy:
            try:
                # StrategySignal has flat columns, not nested dicts
                strategy_signal_type = strategy.signal_type
                strategy_vs_bh_pct = float(strategy.vs_buy_hold_pct) if strategy.vs_buy_hold_pct else None
                strategy_beats_bh = strategy.beats_buy_hold
                strategy_beats_spy = getattr(strategy, 'beats_spy', False) or False
                strategy_win_rate_val = float(strategy.win_rate) if strategy.win_rate else None
                win_rate = float(strategy.win_rate) / 100 if strategy.win_rate else None  # Convert from pct
                strategy_n_trades = strategy.n_trades if strategy.n_trades else 0
                expected_recovery_days = None  # Not stored in StrategySignal
            except Exception:
                pass
        
        # Get typical dip from dipfinder (pre-computed)
        typical_dip_pct = None
        dip_vs_typical = None
        is_unusual_dip = False
        
        if dipfinder:
            if dipfinder.dip_vs_typical is not None:
                dip_vs_typical = float(dipfinder.dip_vs_typical)
                is_unusual_dip = dip_vs_typical >= 1.5
            if dipfinder.stability_factors:
                stability = dipfinder.stability_factors
                if isinstance(stability, str):
                    import json
                    try:
                        stability = json.loads(stability)
                    except (json.JSONDecodeError, TypeError):
                        stability = {}
                if isinstance(stability, dict) and stability.get("typical_dip_365"):
                    typical_dip_pct = float(stability["typical_dip_365"]) * 100
        
        # Determine action - MUST be backed by proven math (strategy signals)
        # NO recommendations based on "dip + AI" alone anymore
        action = "HOLD"
        buy_score = 0.0
        
        # Minimum trades for statistical validity
        MIN_TRADES = 5
        has_enough_trades = strategy_n_trades is not None and strategy_n_trades >= MIN_TRADES
        
        # QUALITY GATE: Only BUY if strategy beats BOTH buy-and-hold AND SPY benchmark
        # AND has enough trades for statistical validity
        # Must beat SPY because otherwise just buy SPY index fund
        beats_benchmarks = strategy_beats_bh and strategy_beats_spy
        if beats_benchmarks and strategy_signal_type == "BUY" and has_enough_trades:
            action = "BUY"
            buy_score = 80.0
        # Note: We no longer give BUY based on dip_pct or win_rate alone - that's not math-backed
        
        # Simple expected return estimate - ONLY from proven strategies
        mu_hat = 0.0
        if beats_benchmarks and strategy_vs_bh_pct:
            mu_hat = strategy_vs_bh_pct / 100
        # Note: No longer estimating returns from dip_pct alone - that's not proven
        
        # AI analysis (informational only, not for ranking)
        ai_summary = ai.rating_reasoning if ai else None
        ai_rating = ai.ai_rating if ai else None
        
        # Calculate best chance score - MATH-BACKED ONLY
        # Start at 0, only add points for proven factors
        best_chance_score = 0.0
        best_chance_reasons = []
        
        # Primary factor: Strategy beats BOTH buy-and-hold AND SPY (proven)
        if beats_benchmarks:
            best_chance_score += 40
            best_chance_reasons.append("Beats B&H + SPY")
            if strategy_vs_bh_pct and strategy_vs_bh_pct > 10:
                best_chance_score += min(strategy_vs_bh_pct / 2, 20)
                best_chance_reasons.append(f"+{strategy_vs_bh_pct:.0f}% vs B&H")
        
        # Secondary factor: Win rate > 55% (proven)
        if strategy_win_rate_val and strategy_win_rate_val >= 55:
            best_chance_score += 15
            best_chance_reasons.append(f"{strategy_win_rate_val:.0f}% win rate")
        
        # Tertiary factor: Unusual dip (math-computed)
        if is_unusual_dip and dip_vs_typical and dip_vs_typical >= 1.5:
            best_chance_score += 15
            best_chance_reasons.append(f"{dip_vs_typical:.1f}x typical dip")
        elif dip_pct and dip_pct > 20:
            # Significant dip but not compared to typical - small boost
            best_chance_score += 5
            best_chance_reasons.append(f"{dip_pct:.0f}% dip")
        
        # AI is informational only - minor boost for confirmation
        if ai_rating in ("strong_buy", "buy") and best_chance_score > 30:
            best_chance_score += 5
            best_chance_reasons.append(f"AI: {ai_rating}")
        
        # NEW: Override best_chance_score with quant_score if available
        # The quant_score is computed using stationary bootstrap and deflated Sharpe
        quant_mode = None
        quant_score_a = None
        quant_score_b = None
        quant_gate_pass = False
        evidence_response = None
        
        if quant_score:
            # Use the pre-computed quant score as the primary ranking metric
            best_chance_score = float(quant_score.best_score)
            quant_mode = quant_score.mode
            quant_score_a = float(quant_score.score_a) if quant_score.score_a else None
            quant_score_b = float(quant_score.score_b) if quant_score.score_b else None
            quant_gate_pass = quant_score.gate_pass
            
            # Build evidence block from stored data
            if quant_score.evidence:
                evidence_response = EvidenceBlockResponse(**quant_score.evidence)
            else:
                # Build from individual columns if JSONB is missing
                evidence_response = EvidenceBlockResponse(
                    p_outperf=float(quant_score.p_outperf or 0),
                    ci_low=float(quant_score.ci_low or 0),
                    ci_high=float(quant_score.ci_high or 0),
                    dsr=float(quant_score.dsr or 0),
                    median_edge=float(quant_score.median_edge or 0),
                    edge_vs_stock=float(quant_score.edge_vs_stock or 0),
                    edge_vs_spy=float(quant_score.edge_vs_spy or 0),
                    worst_regime_edge=float(quant_score.worst_regime_edge or 0),
                    cvar_5=float(quant_score.cvar_5 or 0),
                    fund_mom=float(quant_score.fund_mom or 0),
                    val_z=float(quant_score.val_z or 0),
                    event_risk=quant_score.event_risk or False,
                    p_recovery=float(quant_score.p_recovery or 0),
                    expected_value=float(quant_score.expected_value or 0),
                    sector_relative=float(quant_score.sector_relative or 0),
                )
            
            # Update best_chance_reasons with quant mode
            if quant_gate_pass:
                best_chance_reasons = [f"Mode A: {best_chance_score:.0f}"]
                if evidence_response and evidence_response.p_outperf >= 0.75:
                    best_chance_reasons.append(f"P(edge>0)={evidence_response.p_outperf:.0%}")
            else:
                best_chance_reasons = [f"Mode B: {best_chance_score:.0f}"]
                if evidence_response and evidence_response.p_recovery > 0.5:
                    best_chance_reasons.append(f"P(rec)={evidence_response.p_recovery:.0%}")
            
            # Upgrade action based on quant gate
            if quant_gate_pass and quant_score.best_score >= 70:
                action = "BUY"
                buy_score = float(quant_score.best_score)
        
        best_chance_score = max(0, min(100, best_chance_score))
        
        # Opportunity rating
        if best_chance_score >= 75:
            opportunity_rating = "strong_buy"
        elif best_chance_score >= 60:
            opportunity_rating = "buy"
        elif best_chance_score >= 40:
            opportunity_rating = "hold"
        else:
            opportunity_rating = "avoid"
        
        recommendations.append(QuantRecommendation(
            ticker=symbol,
            name=name,
            action=action,
            notional_eur=inflow_eur / len(symbol_list) if action == "BUY" else 0,
            delta_weight=0.01 if action == "BUY" else 0,
            target_weight=0.05 if action == "BUY" else 0,
            last_price=current_price if current_price else None,
            change_percent=change_percent,
            market_cap=float(market_cap) if market_cap else None,
            sector=sector,
            mu_hat=mu_hat,
            uncertainty=1.0 - (buy_score / 100),
            risk_contribution=0.0,
            opportunity_score=best_chance_score,
            opportunity_rating=opportunity_rating,
            expected_recovery_days=expected_recovery_days,
            typical_dip_pct=typical_dip_pct,
            dip_vs_typical=dip_vs_typical,
            is_unusual_dip=is_unusual_dip,
            win_rate=win_rate,
            dip_bucket=_dip_bucket(dip_pct) if dip_pct else None,
            marginal_utility=buy_score / 100,
            legacy_dip_pct=dip_pct,
            legacy_days_in_dip=days_in_dip,
            ai_summary=ai_summary,
            ai_rating=ai_rating,
            strategy_beats_bh=strategy_beats_bh,
            strategy_signal=strategy_signal_type,
            strategy_win_rate=strategy_win_rate_val,
            strategy_vs_bh_pct=strategy_vs_bh_pct,
            best_chance_score=best_chance_score,
            best_chance_reason=" • ".join(best_chance_reasons[:3]) if best_chance_reasons else None,
            # APUS + DOUS Dual-Mode Scoring
            quant_mode=quant_mode,
            quant_score_a=quant_score_a,
            quant_score_b=quant_score_b,
            quant_gate_pass=quant_gate_pass,
            quant_evidence=evidence_response,
        ))
    
    # Sort by best_chance_score
    recommendations.sort(key=lambda r: r.best_chance_score, reverse=True)
    recommendations = recommendations[:limit]
    
    buy_count = sum(1 for r in recommendations if r.action == "BUY")
    high_conviction_count = sum(1 for r in recommendations if r.best_chance_score >= 60)
    avg_mu_hat = sum(r.mu_hat for r in recommendations) / len(recommendations) if recommendations else 0
    
    # Generate market message based on opportunity quality
    certified_count = sum(1 for r in recommendations if r.quant_gate_pass)
    dip_entry_count = sum(1 for r in recommendations if r.quant_mode == "DIP_ENTRY")
    hold_count = sum(1 for r in recommendations if r.quant_mode == "HOLD")
    top_score = recommendations[0].best_chance_score if recommendations else 0
    
    # Simplified market message - no Mode A/B jargon
    market_message = None
    if certified_count > 0:
        if certified_count >= 3:
            market_message = None  # Plenty of opportunities, no message needed
        else:
            market_message = f"{certified_count} certified opportunity(ies) meet all quality criteria."
    elif dip_entry_count > 0:
        # Dip entries available but no certified buys
        market_message = f"{dip_entry_count} dip recovery opportunity(ies) available. No certified buys today."
    elif top_score > 0:
        market_message = f"No actionable opportunities today. Top score: {top_score:.0f}/100."
    # If all stocks are HOLD, show nothing special - this is normal
    
    # Legacy fallback for when quant_scores are not yet computed
    if not quant_score_map and buy_count == 0:
        market_message = "No BUY signals today. Quant scoring job has not run yet."
    
    response = QuantEngineResponse(
        recommendations=recommendations,
        as_of_date=date.today().isoformat(),
        portfolio_value_eur=10000.0,
        inflow_eur=inflow_eur,
        total_trades=buy_count,
        total_transaction_cost_eur=buy_count * 5.0,
        expected_portfolio_return=avg_mu_hat,
        expected_portfolio_risk=0.15,
        audit=QuantAuditBlock(
            timestamp=datetime.now().isoformat(),
            config_hash=hash(f"{inflow_eur}-{limit}"),
            optimizer_status="success",
            regime_state="unknown",
        ),
        market_message=market_message,
    )
    
    # Cache the response
    await cache.set(cache_key, response.model_dump())
    
    return response


# ============================================================================
# Hero Stock Endpoint (Rotating featured stock for landing page)
# ============================================================================


class HeroStockResponse(BaseModel):
    """Hero stock for landing page display."""
    symbol: str
    company_name: str | None = None
    action: str
    mu_hat: float
    dip_pct: float | None = None
    days_in_dip: int | None = None
    best_chance_score: float = 0.0
    best_chance_reason: str | None = None
    ai_summary: str | None = None
    ai_rating: str | None = None
    cache_expires_at: str  # ISO timestamp when this hero expires


# Cache for hero stock rotation
_hero_cache = Cache(prefix="hero_stock", default_ttl=900)  # 15 minutes


@global_router.get(
    "/hero",
    response_model=HeroStockResponse,
    summary="Get rotating hero stock",
    description="""
    Get the current featured stock for landing page hero section.
    
    This endpoint returns a rotating stock from the top recommendations,
    cached for 15 minutes. Each rotation picks a different stock from
    the top 5 BUY recommendations to provide variety.
    """,
)
async def get_hero_stock() -> HeroStockResponse:
    """Get rotating hero stock for landing page."""
    import hashlib
    from datetime import datetime, timezone
    
    # Check cache first
    cached = await _hero_cache.get("current")
    if cached:
        return HeroStockResponse(**cached)
    
    # Get recommendations (uses its own cache)
    recommendations_response = await get_recommendations(inflow_eur=1000.0, limit=20)
    recommendations = recommendations_response.recommendations
    
    # Filter to BUY recommendations with best_chance_score > 50
    buy_recs = [
        r for r in recommendations 
        if r.action == "BUY" and r.best_chance_score >= 50
    ]
    
    if not buy_recs:
        # Fallback to any BUY, or first recommendation
        buy_recs = [r for r in recommendations if r.action == "BUY"]
        if not buy_recs:
            buy_recs = recommendations[:5] if recommendations else []
    
    if not buy_recs:
        # No recommendations at all - return a sensible error state
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No recommendations available for hero display"
        )
    
    # Rotate among top 5: Use time-based hash to pick a consistent one per 15-min window
    now = datetime.now(timezone.utc)
    window_id = int(now.timestamp()) // 900  # 15-minute windows
    hash_input = f"hero-{window_id}"
    index = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % min(len(buy_recs), 5)
    
    hero_rec = buy_recs[index]
    
    # Calculate when this hero expires (next 15-minute boundary)
    next_window = (window_id + 1) * 900
    expires_at = datetime.fromtimestamp(next_window, tz=timezone.utc)
    
    # Get company name from stock info
    from app.repositories import symbols_orm as symbols_repo
    symbol_data = await symbols_repo.get_symbol(hero_rec.ticker)
    company_name = symbol_data.name if symbol_data else None
    
    response = HeroStockResponse(
        symbol=hero_rec.ticker,
        company_name=company_name,
        action=hero_rec.action,
        mu_hat=hero_rec.mu_hat,
        dip_pct=hero_rec.legacy_dip_pct,
        days_in_dip=hero_rec.legacy_days_in_dip,
        best_chance_score=hero_rec.best_chance_score,
        best_chance_reason=hero_rec.best_chance_reason,
        ai_summary=hero_rec.ai_summary,
        ai_rating=hero_rec.ai_rating,
        cache_expires_at=expires_at.isoformat(),
    )
    
    # Cache for remaining time in this window
    remaining_seconds = next_window - int(now.timestamp())
    if remaining_seconds > 0:
        await _hero_cache.set("current", response.model_dump(), ttl=remaining_seconds)
    
    return response


def _dip_bucket(dip_pct: float | None) -> str | None:
    """Classify dip percentage into buckets."""
    if dip_pct is None:
        return None
    if dip_pct < 10:
        return "shallow"
    if dip_pct < 20:
        return "moderate"
    if dip_pct < 35:
        return "deep"
    if dip_pct < 50:
        return "very_deep"
    return "extreme"


# ============================================================================
# Signal Triggers Endpoint (Historical Buy Signals for Chart Markers)
# ============================================================================


class SignalTriggerResponse(BaseModel):
    """A historical signal trigger for chart display."""
    date: str
    signal_name: str
    price: float
    win_rate: float
    avg_return_pct: float
    holding_days: int
    drawdown_pct: float = 0.0  # The threshold that triggered (for drawdown signals)
    signal_type: str = "entry"  # "entry" for buy signals, "exit" for sell signals


class SignalTriggersResponse(BaseModel):
    """Response containing historical signal triggers with benchmark comparison."""
    symbol: str
    signal_name: str | None = None  # The signal being tracked
    triggers: list[SignalTriggerResponse]
    
    # Benchmark comparison - how the signal performed vs simply holding
    buy_hold_return_pct: float = 0.0  # Return from buying and holding over the lookback period
    signal_return_pct: float = 0.0  # Aggregate return from signal-based trading
    edge_vs_buy_hold_pct: float = 0.0  # Signal return - buy hold return (the "alpha")
    n_trades: int = 0  # Number of signal triggers in the period
    beats_buy_hold: bool = False  # Whether signal strategy outperformed buy-and-hold
    actual_win_rate: float = 0.0  # Actual win rate from visible trades (not backtest estimate)


@global_router.get(
    "/{symbol}/signal-triggers",
    response_model=SignalTriggersResponse,
    summary="Get historical signal triggers for chart markers",
    description="""
    Get historical buy signal trigger points for a stock.
    
    Returns triggers for the TOP/BEST signal for this stock (highest expected value).
    These are dates when buy signals were triggered based on PAST DATA ONLY.
    There is no look-ahead bias - each trigger uses only data available at that time.
    
    Use these to overlay buy markers on price charts.
    """,
)
async def get_signal_triggers(
    symbol: str,
    lookback_days: int = Query(default=365, ge=30, le=730, description="Days to look back"),
) -> SignalTriggersResponse:
    """Get historical signal triggers for chart markers with benchmark comparison."""
    from app.quant_engine.signals import get_historical_triggers
    
    symbol = symbol.strip().upper()
    
    # Fetch price data - use same 5-year window as the signals scanner for consistent optimization
    prices_df = await _fetch_prices_for_symbols([symbol], lookback_days=1260)
    
    if prices_df.empty or symbol not in prices_df.columns:
        return SignalTriggersResponse(symbol=symbol, signal_name=None, triggers=[])
    
    # Convert to dict format expected by signals module
    close_prices = prices_df[symbol].dropna()
    price_data = {
        "close": close_prices,
    }
    
    # Get historical triggers for the BEST signal
    triggers = get_historical_triggers(price_data, lookback_days=lookback_days)
    
    # Get the signal name from the triggers (they're all from the same signal now)
    signal_name = triggers[0].signal_name if triggers else None
    
    # Calculate benchmark comparison
    # Buy-and-hold return over the lookback period
    buy_hold_return_pct = 0.0
    signal_return_pct = 0.0
    edge_vs_buy_hold_pct = 0.0
    
    # Count entry triggers only (not exits)
    entry_triggers = [t for t in triggers if t.signal_type == "entry"]
    n_trades = len(entry_triggers)
    
    if len(close_prices) >= lookback_days and lookback_days > 0:
        lookback_slice = close_prices.iloc[-lookback_days:]
        if len(lookback_slice) >= 2:
            first_price = lookback_slice.iloc[0]
            last_price = lookback_slice.iloc[-1]
            buy_hold_return_pct = ((last_price / first_price) - 1) * 100
    
    # Signal aggregate return - sum of all actual trade returns (exits show individual trade returns)
    # This represents total return from signal-based trading
    exit_triggers = [t for t in triggers if t.signal_type == "exit"]
    if exit_triggers:
        # Sum the actual returns from all completed trades
        signal_return_pct = sum(t.avg_return_pct for t in exit_triggers)
        edge_vs_buy_hold_pct = signal_return_pct - buy_hold_return_pct
    
    # Get actual win rate from the triggers (now properly calculated)
    actual_win_rate = entry_triggers[0].win_rate if entry_triggers else 0.0
    
    # Strategy beats buy-and-hold if it has positive edge
    beats_bh = edge_vs_buy_hold_pct > 0
    
    return SignalTriggersResponse(
        symbol=symbol,
        signal_name=signal_name,
        triggers=[
            SignalTriggerResponse(
                date=t.date,
                signal_name=t.signal_name,
                price=t.price,
                win_rate=t.win_rate,
                avg_return_pct=t.avg_return_pct,
                holding_days=t.holding_days,
                drawdown_pct=t.drawdown_pct,
                signal_type=t.signal_type,
            )
            for t in triggers
        ],
        buy_hold_return_pct=round(buy_hold_return_pct, 2),
        signal_return_pct=round(signal_return_pct, 2),
        edge_vs_buy_hold_pct=round(edge_vs_buy_hold_pct, 2),
        n_trades=n_trades,
        beats_buy_hold=beats_bh,
        actual_win_rate=round(actual_win_rate, 4),
    )


# ============================================================================
# Signal Backtest Endpoint (Compare Signal Buys vs Buy-and-Hold)
# ============================================================================


class SignalTradeResponse(BaseModel):
    """A single trade in the backtest."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    signal_name: str
    won: bool


class SignalBacktestResponse(BaseModel):
    """Backtest comparison: signal buys vs buy-and-hold."""
    symbol: str
    period_start: str
    period_end: str
    
    # Signal strategy results
    signal_name: str
    n_trades: int
    win_rate: float
    total_return_pct: float
    avg_return_per_trade: float
    holding_days_per_trade: int
    trades: list[SignalTradeResponse]
    
    # Buy-and-hold comparison
    buy_hold_return_pct: float
    buy_hold_start_price: float
    buy_hold_end_price: float
    
    # Edge metrics
    signal_edge_pct: float  # signal return - buy&hold return
    signal_outperformed: bool


@global_router.get(
    "/{symbol}/signal-backtest",
    response_model=SignalBacktestResponse,
    summary="Backtest signal strategy vs buy-and-hold",
    description="""
    Compare the performance of buying on signal triggers vs simple buy-and-hold.
    
    Shows each trade made by the signal strategy and calculates total returns
    to compare against buying at the start and holding until the end.
    """,
)
async def get_signal_backtest(
    symbol: str,
    lookback_days: int = Query(default=730, ge=90, le=1825, description="Days to backtest"),
) -> SignalBacktestResponse:
    """Backtest signal strategy vs buy-and-hold."""
    from app.quant_engine.signals import get_historical_triggers
    
    symbol = symbol.strip().upper()
    
    # Fetch price data - use same 5-year window as signals scanner for consistent optimization
    prices_df = await _fetch_prices_for_symbols([symbol], lookback_days=1260)
    
    if prices_df.empty or symbol not in prices_df.columns:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")
    
    prices = prices_df[symbol].dropna()
    
    if len(prices) < 60:
        raise HTTPException(status_code=400, detail=f"Insufficient price data for {symbol}")
    
    # Convert to dict format expected by signals module
    price_data = {"close": prices}
    
    # Get historical triggers - only return triggers within the backtest period
    triggers = get_historical_triggers(price_data, lookback_days=lookback_days)
    
    if not triggers:
        # No signals triggered, just return buy-and-hold
        start_price = float(prices.iloc[-lookback_days] if len(prices) > lookback_days else prices.iloc[0])
        end_price = float(prices.iloc[-1])
        buy_hold_return = ((end_price / start_price) - 1) * 100
        
        return SignalBacktestResponse(
            symbol=symbol,
            period_start=str(prices.index[-lookback_days if len(prices) > lookback_days else 0].date()),
            period_end=str(prices.index[-1].date()),
            signal_name="No signals",
            n_trades=0,
            win_rate=0.0,
            total_return_pct=0.0,
            avg_return_per_trade=0.0,
            holding_days_per_trade=0,
            trades=[],
            buy_hold_return_pct=buy_hold_return,
            buy_hold_start_price=start_price,
            buy_hold_end_price=end_price,
            signal_edge_pct=-buy_hold_return,
            signal_outperformed=False,
        )
    
    # Get the best signal's holding period
    holding_days = triggers[0].holding_days if triggers else 20
    signal_name = triggers[0].signal_name if triggers else "Unknown"
    
    # Simulate trades
    trades: list[SignalTradeResponse] = []
    total_signal_return = 0.0
    
    for trigger in triggers:
        entry_date = trigger.date
        entry_price = trigger.price
        
        # Find exit date (holding_days later)
        try:
            entry_idx = prices.index.get_loc(entry_date)
        except KeyError:
            # Try to find closest date
            try:
                entry_idx = prices.index.get_indexer([entry_date], method='nearest')[0]
            except Exception:
                continue
        
        exit_idx = min(entry_idx + holding_days, len(prices) - 1)
        exit_date = str(prices.index[exit_idx].date())
        exit_price = float(prices.iloc[exit_idx])
        
        return_pct = ((exit_price / entry_price) - 1) * 100
        won = return_pct > 0
        
        trades.append(SignalTradeResponse(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=return_pct,
            signal_name=trigger.signal_name,
            won=won,
        ))
        
        total_signal_return += return_pct
    
    # Calculate aggregate stats
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t.won) / n_trades if n_trades > 0 else 0.0
    avg_return = total_signal_return / n_trades if n_trades > 0 else 0.0
    
    # Buy-and-hold comparison
    start_idx = max(0, len(prices) - lookback_days)
    start_price = float(prices.iloc[start_idx])
    end_price = float(prices.iloc[-1])
    buy_hold_return = ((end_price / start_price) - 1) * 100
    
    signal_edge = total_signal_return - buy_hold_return
    
    return SignalBacktestResponse(
        symbol=symbol,
        period_start=str(prices.index[start_idx].date()),
        period_end=str(prices.index[-1].date()),
        signal_name=signal_name,
        n_trades=n_trades,
        win_rate=win_rate,
        total_return_pct=total_signal_return,
        avg_return_per_trade=avg_return,
        holding_days_per_trade=holding_days,
        trades=trades,
        buy_hold_return_pct=buy_hold_return,
        buy_hold_start_price=start_price,
        buy_hold_end_price=end_price,
        signal_edge_pct=signal_edge,
        signal_outperformed=signal_edge > 0,
    )


# ============================================================================
# Full Trade Engine Endpoints (Entry + Exit Optimization)
# ============================================================================


class ExitStrategyResponse(BaseModel):
    """Exit strategy details."""
    name: str
    threshold: float
    description: str = ""


class TradeCycleResponse(BaseModel):
    """A complete trade cycle."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    entry_signal: str
    exit_signal: str


class FullTradeResponse(BaseModel):
    """Complete trade analysis with optimized entry AND exit, benchmarked vs SPY."""
    symbol: str
    entry_signal_name: str
    entry_threshold: float
    entry_description: str = ""
    exit_strategy_name: str
    exit_threshold: float
    exit_description: str = ""
    n_complete_trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    max_return_pct: float
    max_drawdown_pct: float
    avg_holding_days: float
    sharpe_ratio: float = 0.0
    # Benchmark comparison
    buy_hold_return_pct: float
    spy_return_pct: float = 0.0
    edge_vs_buy_hold_pct: float
    edge_vs_spy_pct: float = 0.0
    beats_both_benchmarks: bool = False
    # Exit timing analysis
    exit_predictability: float = 0.0  # How consistent is exit timing?
    upside_captured_pct: float = 0.0  # What % of potential gain was captured?
    trades: list[TradeCycleResponse]
    current_buy_signal: bool
    current_sell_signal: bool
    days_since_last_signal: int = 0


class CombinedSignalResponse(BaseModel):
    """Signal combination result."""
    name: str
    component_signals: list[str]
    logic: str
    win_rate: float
    avg_return_pct: float
    n_signals: int
    improvement_vs_best_single: float


class DipAnalysisResponse(BaseModel):
    """Analysis of whether a dip is overreaction or real decline (falling knife)."""
    symbol: str
    current_drawdown_pct: float
    typical_dip_pct: float
    max_historical_dip_pct: float
    dip_zscore: float = 0.0  # How many std devs from typical
    is_unusually_deep: bool
    deviation_from_typical: float
    # Technical analysis
    technical_score: float = 0.0  # -1 (bearish) to +1 (bullish)
    trend_broken: bool = False  # Below SMA 200?
    volume_confirmation: bool = False  # High volume = capitulation
    momentum_divergence: bool = False  # Price down but RSI up
    # Classification
    dip_type: str  # OVERREACTION, NORMAL_VOLATILITY, FUNDAMENTAL_DECLINE
    confidence: float
    action: str  # STRONG_BUY, BUY, WAIT, AVOID
    reasoning: str
    # Probability estimates
    recovery_probability: float = 0.5
    expected_return_if_buy: float = 0.0
    expected_loss_if_knife: float = 0.0


class CurrentSignalsResponse(BaseModel):
    """Current real-time buy and sell signals."""
    symbol: str
    buy_signals: list[dict]
    sell_signals: list[dict]
    overall_action: str
    reasoning: str


@global_router.get(
    "/{symbol}/full-trade",
    response_model=FullTradeResponse,
    summary="Get optimized full trade strategy",
    description="""
    Get the best complete trade strategy for a stock: optimized entry signal + exit signal.
    
    This answers:
    - "What's the best entry signal for this stock?"
    - "What's the best exit strategy (RSI overbought, profit target, trailing stop)?"
    - "Does this strategy beat buy-and-hold AND SPY?"
    
    The system tests all entry signals against all exit strategies and finds
    the combination that maximizes total return while beating both benchmarks.
    
    **Mathematical Proof of Exit Timing:**
    - `exit_predictability`: How consistent is the holding period (low std dev = predictable)
    - `upside_captured_pct`: What % of potential gain was captured by the exit signal
    - If exit_predictability > 0.5 and upside_captured > 80%, the exit timing is proven reliable
    """,
)
async def get_full_trade_strategy(
    symbol: str,
) -> FullTradeResponse:
    """Get optimized full trade strategy with entry AND exit signals, benchmarked vs SPY."""
    from app.quant_engine.trade_engine import get_best_trade_strategy
    
    symbol = symbol.strip().upper()
    
    # Fetch stock prices AND SPY for benchmarking
    prices_df = await _fetch_prices_for_symbols([symbol, "SPY"], lookback_days=1260)
    
    if prices_df.empty or symbol not in prices_df.columns:
        raise NotFoundError(f"No price data for {symbol}")
    
    price_data = {"close": prices_df[symbol].dropna()}
    
    # Get SPY prices for benchmark comparison
    spy_prices = prices_df["SPY"].dropna() if "SPY" in prices_df.columns else None
    
    result, _ = get_best_trade_strategy(
        price_data, symbol, spy_prices=spy_prices, test_combinations=False
    )
    
    if result is None:
        raise NotFoundError(f"Insufficient data for trade analysis of {symbol}")
    
    return FullTradeResponse(
        symbol=result.symbol,
        entry_signal_name=result.entry_signal_name,
        entry_threshold=result.entry_threshold,
        entry_description=result.entry_description,
        exit_strategy_name=result.exit_strategy_name,
        exit_threshold=result.exit_threshold,
        exit_description=result.exit_description,
        n_complete_trades=result.n_complete_trades,
        win_rate=result.win_rate,
        avg_return_pct=result.avg_return_pct,
        total_return_pct=result.total_return_pct,
        max_return_pct=result.max_return_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        avg_holding_days=result.avg_holding_days,
        sharpe_ratio=result.sharpe_ratio,
        buy_hold_return_pct=result.buy_hold_return_pct,
        spy_return_pct=result.spy_return_pct,
        edge_vs_buy_hold_pct=result.edge_vs_buy_hold_pct,
        edge_vs_spy_pct=result.edge_vs_spy_pct,
        beats_both_benchmarks=result.beats_both_benchmarks,
        exit_predictability=result.exit_predictability,
        upside_captured_pct=result.upside_captured_pct,
        trades=[
            TradeCycleResponse(
                entry_date=t.entry_date,
                exit_date=t.exit_date,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                return_pct=t.return_pct,  # Already in percentage from trade_engine
                holding_days=t.holding_days,
                entry_signal=t.entry_signal,
                exit_signal=t.exit_signal,
            )
            for t in result.trades[-20:]  # Last 20 trades
        ],
        current_buy_signal=result.current_buy_signal,
        current_sell_signal=result.current_sell_signal,
        days_since_last_signal=result.days_since_last_signal,
    )


@global_router.get(
    "/{symbol}/signal-combinations",
    response_model=list[CombinedSignalResponse],
    summary="Test signal combinations",
    description="""
    Test combinations of signals (e.g., RSI + Drawdown together).
    
    This answers:
    - "Do multiple signals together work better than individual signals?"
    - "What's the best signal combination for this stock?"
    
    Tests AND/OR combinations of indicators.
    """,
)
async def get_signal_combinations(
    symbol: str,
) -> list[CombinedSignalResponse]:
    """Test signal combinations for a stock."""
    from app.quant_engine.trade_engine import test_signal_combination, SIGNAL_COMBINATIONS
    
    symbol = symbol.strip().upper()
    
    prices_df = await _fetch_prices_for_symbols([symbol], lookback_days=1260)
    
    if prices_df.empty or symbol not in prices_df.columns:
        return []
    
    price_data = {"close": prices_df[symbol].dropna()}
    
    results = []
    for combo_cfg in SIGNAL_COMBINATIONS:
        combo = test_signal_combination(price_data, combo_cfg, holding_days=20, min_signals=3)
        if combo is not None:
            results.append(CombinedSignalResponse(
                name=combo.name,
                component_signals=combo.component_signals,
                logic=combo.logic,
                win_rate=combo.win_rate,
                avg_return_pct=combo.avg_return_pct,
                n_signals=combo.n_signals,
                improvement_vs_best_single=combo.improvement_vs_best_single,
            ))
    
    # Sort by EV
    results.sort(key=lambda c: c.win_rate * c.avg_return_pct, reverse=True)
    
    return results


@global_router.get(
    "/{symbol}/dip-analysis",
    response_model=DipAnalysisResponse,
    summary="Analyze if dip is overreaction or falling knife",
    description="""
    Analyze whether the current dip is an overreaction (buy opportunity) or a falling knife (avoid!).
    
    This answers the critical question: **"Should I buy this dip or am I catching a falling knife?"**
    
    Analysis includes:
    - **Historical comparison**: Is this a 20% dip on a stock that typically dips 15%? (unusual)
    - **Technical score**: RSI, MACD, Bollinger Bands, Z-Score - are we oversold?
    - **Trend analysis**: Is the long-term trend broken? (below SMA 200)
    - **Volume confirmation**: High volume on dip = capitulation (bullish)
    - **Momentum divergence**: Price down but RSI up = potential reversal
    
    Classification:
    - **OVERREACTION** → BUY/STRONG_BUY - Dip is larger than typical, technicals supportive
    - **NORMAL_VOLATILITY** → WAIT - Dip is within normal range
    - **FUNDAMENTAL_DECLINE** → AVOID - This is a falling knife!
    """,
)
async def get_dip_analysis(
    symbol: str,
) -> DipAnalysisResponse:
    """Analyze if current dip is overreaction or falling knife."""
    from app.quant_engine.trade_engine import analyze_dip
    
    symbol = symbol.strip().upper()
    
    prices_df = await _fetch_prices_for_symbols([symbol], lookback_days=1260)
    
    if prices_df.empty or symbol not in prices_df.columns:
        raise NotFoundError(f"No price data for {symbol}")
    
    price_data = {"close": prices_df[symbol].dropna()}
    
    analysis = analyze_dip(price_data, symbol)
    
    return DipAnalysisResponse(
        symbol=analysis.symbol,
        current_drawdown_pct=analysis.current_drawdown_pct,
        typical_dip_pct=analysis.typical_dip_pct,
        max_historical_dip_pct=analysis.max_historical_dip_pct,
        dip_zscore=analysis.dip_zscore,
        is_unusually_deep=analysis.is_unusually_deep,
        deviation_from_typical=analysis.deviation_from_typical,
        technical_score=analysis.technical_score,
        trend_broken=analysis.trend_broken,
        volume_confirmation=analysis.volume_confirmation,
        momentum_divergence=analysis.momentum_divergence,
        dip_type=analysis.dip_type.value if hasattr(analysis.dip_type, 'value') else str(analysis.dip_type),
        confidence=analysis.confidence,
        action=analysis.action,
        reasoning=analysis.reasoning,
        recovery_probability=analysis.recovery_probability,
        expected_return_if_buy=analysis.expected_return_if_buy,
        expected_loss_if_knife=analysis.expected_loss_if_knife,
    )


@global_router.get(
    "/{symbol}/current-signals",
    response_model=CurrentSignalsResponse,
    summary="Get current real-time buy and sell signals",
    description="""
    Get current actionable buy and sell signals for a stock.
    
    This answers:
    - "Should I buy or sell right now?"
    - "Which buy signals are currently active?"
    - "Are there any sell signals (RSI overbought, etc.)?"
    
    Returns an overall action recommendation: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL.
    """,
)
async def get_current_signals(
    symbol: str,
) -> CurrentSignalsResponse:
    """Get current real-time buy and sell signals."""
    from app.quant_engine.trade_engine import get_current_signals
    
    symbol = symbol.strip().upper()
    
    prices_df = await _fetch_prices_for_symbols([symbol], lookback_days=252)
    
    if prices_df.empty or symbol not in prices_df.columns:
        return CurrentSignalsResponse(
            symbol=symbol,
            buy_signals=[],
            sell_signals=[],
            overall_action="HOLD",
            reasoning="No price data",
        )
    
    price_data = {"close": prices_df[symbol].dropna()}
    
    signals = get_current_signals(price_data, symbol)
    
    return CurrentSignalsResponse(
        symbol=symbol,
        buy_signals=signals["buy_signals"],
        sell_signals=signals["sell_signals"],
        overall_action=signals["overall_action"],
        reasoning=signals["reasoning"],
    )


# ============================================================================
# Signal Scanner Endpoint (Global - No Auth Required)
# ============================================================================


@global_router.get(
    "/signals",
    response_model=SignalScanResponse,
    summary="Scan stocks for buy opportunities",
    description="""
    Scan all tracked stocks for technical buy signals with per-stock optimization.
    
    This endpoint answers:
    - "Which stock should I buy next?"
    - "What's the optimal holding period after this signal?"
    - "Which dip is a statistical overreaction?"
    
    It tests 14 technical signals (RSI, Z-score, MACD, Bollinger Bands, SMA, etc.)
    for each stock, optimizes the threshold and holding period parameters,
    and ranks stocks by their current buy opportunity score.
    
    Each signal includes:
    - optimal_threshold: Best signal threshold for this specific stock
    - optimal_holding_days: Best holding period after signal triggers
    - win_rate: Historical success rate at optimal parameters
    - avg_return_pct: Average return when buying on this signal
    """,
)
async def scan_signals() -> SignalScanResponse:
    """Scan all stocks for technical buy signals with per-stock optimization."""
    from app.repositories import symbols_orm as symbols_repo
    
    holding_days_options = [5, 10, 20, 40, 60]
    
    symbols = await symbols_repo.list_symbols()
    
    if not symbols:
        return SignalScanResponse(
            scanned_at=datetime.now(),
            holding_days_tested=holding_days_options,
            stocks=[],
            top_opportunities=[],
            n_active_signals=0,
        )
    
    # Build symbol list (exclude benchmarks)
    symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]
    symbol_names = {s.symbol: s.name or s.symbol for s in symbols}
    
    # Fetch price data
    prices_df = await _fetch_prices_for_symbols(symbol_list, lookback_days=1260)
    
    if prices_df.empty:
        return SignalScanResponse(
            scanned_at=datetime.now(),
            holding_days_tested=holding_days_options,
            stocks=[],
            top_opportunities=[],
            n_active_signals=0,
        )
    
    # Convert DataFrame to dict of Series
    price_data = {col: prices_df[col].dropna() for col in prices_df.columns}
    
    # Run signal scanner with optimization
    opportunities = scan_all_stocks(price_data, symbol_names, holding_days_options)
    
    # Convert to response
    stocks = []
    total_active_signals = 0
    
    for opp in opportunities:
        signal_responses = [
            SignalResultResponse(
                name=sig.name,
                description=sig.description,
                value=sig.current_value,
                is_buy_signal=sig.is_buy_signal,
                strength=sig.signal_strength,
                optimal_threshold=sig.optimal_threshold,
                optimal_holding_days=sig.optimal_holding_days,
                win_rate=sig.win_rate,
                avg_return_pct=sig.avg_return_pct,
                max_return_pct=sig.max_return_pct,
                min_return_pct=sig.min_return_pct,
                n_signals=sig.n_signals,
                improvement_pct=sig.improvement_pct,
            )
            for sig in opp.signals
        ]
        
        active_signal_responses = [
            SignalResultResponse(
                name=sig.name,
                description=sig.description,
                value=sig.current_value,
                is_buy_signal=sig.is_buy_signal,
                strength=sig.signal_strength,
                optimal_threshold=sig.optimal_threshold,
                optimal_holding_days=sig.optimal_holding_days,
                win_rate=sig.win_rate,
                avg_return_pct=sig.avg_return_pct,
                max_return_pct=sig.max_return_pct,
                min_return_pct=sig.min_return_pct,
                n_signals=sig.n_signals,
                improvement_pct=sig.improvement_pct,
            )
            for sig in opp.active_signals
        ]
        
        total_active_signals += len(opp.active_signals)
        
        stocks.append(StockSignalResponse(
            symbol=opp.symbol,
            name=opp.name,
            buy_score=opp.buy_score,
            opportunity_type=opp.opportunity_type,
            opportunity_reason=opp.opportunity_reason,
            current_price=opp.current_price,
            price_vs_52w_high_pct=opp.price_vs_52w_high_pct,
            price_vs_52w_low_pct=opp.price_vs_52w_low_pct,
            zscore_20d=opp.zscore_20d,
            zscore_60d=opp.zscore_60d,
            rsi_14=opp.rsi_14,
            best_signal_name=opp.best_signal_name,
            best_holding_days=opp.best_holding_days,
            best_expected_return=opp.best_expected_return,
            signals=signal_responses,
            active_buy_signals=active_signal_responses,
        ))
    
    top_opportunities = [s.symbol for s in stocks[:3] if s.buy_score > 0]
    
    return SignalScanResponse(
        scanned_at=datetime.now(),
        holding_days_tested=holding_days_options,
        stocks=stocks,
        top_opportunities=top_opportunities,
        n_active_signals=total_active_signals,
    )


# ============================================================================
# Global Analytics Endpoint (No Auth - for Dashboard)
# ============================================================================


@global_router.get(
    "/market-analysis",
    summary="Global Market Analysis",
    description="""
    Get market-wide risk analysis for all tracked symbols.
    
    Provides:
    - Current market regime (bull/bear, high/low volatility)
    - Average correlation across assets
    - Diversification opportunities
    - Sector concentration warnings
    """,
)
async def get_market_analysis() -> dict[str, Any]:
    """Get global market analysis."""
    from app.repositories import symbols_orm as symbols_repo
    from app.quant_engine.analytics import detect_regime, compute_correlation_analysis
    
    symbols = await symbols_repo.list_symbols()
    
    if not symbols:
        return {
            "analyzed_at": datetime.now().isoformat(),
            "status": "no_data",
            "message": "No symbols tracked",
        }
    
    symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]
    
    prices = await _fetch_prices_for_symbols(symbol_list, lookback_days=400)
    
    if prices.empty or len(prices) < 60:
        return {
            "analyzed_at": datetime.now().isoformat(),
            "status": "insufficient_data",
            "message": "Insufficient price history",
        }
    
    returns = _compute_returns(prices)
    
    # Detect regime
    regime = detect_regime(returns)
    
    # Compute correlation analysis
    corr_analysis = compute_correlation_analysis(returns)
    
    return {
        "analyzed_at": datetime.now().isoformat(),
        "n_symbols": len(returns.columns),
        "regime": {
            "current": regime.regime,
            "trend": regime.trend,
            "volatility": regime.volatility,
            "description": regime.description,
            "recommendation": regime.risk_budget_recommendation,
        },
        "correlations": {
            "average": _safe_float(corr_analysis.average_correlation),
            "n_clusters": corr_analysis.n_clusters,
            "clusters": corr_analysis.clusters,
            "stress_correlation": _safe_float(corr_analysis.stress_correlation),
        },
        "insights": [
            f"Market is in a {regime.regime} regime",
            f"Average correlation: {corr_analysis.average_correlation:.0%}",
            f"Found {corr_analysis.n_clusters} correlation clusters",
        ],
    }
