"""Dip entry API routes.

Returns optimal dip entry levels for adding to existing positions.
This helps answer: "How much should the stock drop before I buy more?"
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Path
from pydantic import BaseModel, Field

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.repositories import symbols_orm as symbols_repo


router = APIRouter()
_cache = Cache(prefix="dip_entry", default_ttl=3600)  # 1 hour cache
logger = get_logger("routes.dip_entry")


# =============================================================================
# Response Schemas
# =============================================================================


class DipFrequency(BaseModel):
    """Annual dip frequency at different thresholds."""
    
    ten_pct: float = Field(..., alias="10_pct", description="Average 10%+ dips per year")
    fifteen_pct: float = Field(..., alias="15_pct", description="Average 15%+ dips per year")
    twenty_pct: float = Field(..., alias="20_pct", description="Average 20%+ dips per year")

    model_config = {"populate_by_name": True}


class ThresholdStats(BaseModel):
    """Statistics for a specific dip threshold level."""
    
    threshold: float = Field(..., description="Dip threshold percentage (e.g., -10)")
    occurrences: int = Field(..., description="Number of times this dip occurred")
    per_year: float = Field(..., description="Average occurrences per year")
    # V2: Multi-period metrics (primary = 90 days)
    win_rate: float = Field(default=0.0, description="Win rate after 90 days")
    avg_return: float = Field(default=0.0, description="Average return after 90 days")
    total_profit: float = Field(default=0.0, description="Total expected profit (N × avg_return)")
    total_profit_compounded: float = Field(default=0.0, description="Compounded total profit")
    sharpe_ratio: float = Field(default=0.0, description="Risk-adjusted return")
    sortino_ratio: float = Field(default=0.0, description="Downside risk-adjusted return")
    cvar: float = Field(default=0.0, description="Tail risk (worst 5% of returns)")
    # Recovery metrics
    recovery_rate: float = Field(..., description="Recovery rate to previous high")
    recovery_threshold_rate: float = Field(default=0.0, description="Recovery to entry threshold")
    avg_recovery_days: float = Field(..., description="Average days to recover")
    avg_days_to_threshold: float = Field(default=0.0, description="Days to recover to entry price")
    avg_recovery_velocity: float = Field(default=0.0, description="Recovery speed (higher = faster)")
    # Recovery-based profit metrics (sell at recovery strategy)
    avg_return_at_recovery: float = Field(default=0.0, description="Avg return selling at recovery")
    total_profit_at_recovery: float = Field(default=0.0, description="Compounded profit at recovery")
    avg_days_at_recovery: float = Field(default=0.0, description="Avg days held until recovery")
    # Risk metrics
    max_further_drawdown: float = Field(default=0.0, description="Worst drawdown after entry")
    avg_further_drawdown: float = Field(default=0.0, description="Average MAE (pain)")
    prob_further_drop: float = Field(default=0.0, description="P(drops another 10%)")
    continuation_risk: Literal["low", "medium", "high"] = Field(default="medium")
    # Scores
    entry_score: float = Field(..., description="V2 entry quality score")
    legacy_entry_score: float = Field(default=0.0, description="Legacy entry score")
    confidence: Literal["low", "medium", "high"] = Field(default="medium")
    # Legacy (backward compat)
    win_rate_60d: float = Field(default=0.0, description="Win rate after 60 days")
    avg_return_60d: float = Field(default=0.0, description="Average return after 60 days")


class DipEntryBacktest(BaseModel):
    """Backtest results for dip-based strategy."""
    
    strategy_return_pct: float = Field(default=0.0, description="Total return from dip strategy")
    buy_hold_return_pct: float = Field(default=0.0, description="Buy-and-hold return over same period")
    vs_buy_hold_pct: float = Field(default=0.0, description="Edge vs buy-and-hold (strategy - B&H)")
    n_trades: int = Field(default=0, description="Number of dip trades executed")
    win_rate: float = Field(default=0.0, description="Percentage of winning trades")
    avg_return_per_trade: float = Field(default=0.0, description="Average return per trade")
    sharpe_ratio: float = Field(default=0.0, description="Risk-adjusted return")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown during backtest")
    years_tested: float = Field(default=0.0, description="Years of historical data used")


class DipEntryResponse(BaseModel):
    """Dip entry analysis response."""
    
    symbol: str = Field(..., description="Stock ticker")
    
    # Current state
    current_price: float = Field(..., description="Current stock price")
    recent_high: float = Field(..., description="Recent high price")
    current_drawdown_pct: float = Field(..., description="Current drawdown from high (%)")
    volatility_regime: Literal["low", "normal", "high"] = Field(
        ..., description="Current volatility regime"
    )
    
    # Risk-adjusted optimal entry (less pain, better timing)
    optimal_dip_threshold: float = Field(
        ..., description="Risk-adjusted optimal dip threshold (e.g., -15 for 15% dip)"
    )
    optimal_entry_price: float = Field(
        ..., description="Risk-adjusted price for limit buy order"
    )
    
    # Max profit optimal entry (more opportunities, higher total return)
    max_profit_threshold: float = Field(
        default=0.0, description="Max profit dip threshold"
    )
    max_profit_entry_price: float = Field(
        default=0.0, description="Max profit price for limit buy order"
    )
    max_profit_total_return: float = Field(
        default=0.0, description="Expected total return at max profit threshold"
    )
    
    # Backtest results - max profit strategy vs buy-and-hold
    backtest: DipEntryBacktest | None = Field(
        default=None, description="Backtest results for max profit dip strategy"
    )
    
    # Current signal
    is_buy_now: bool = Field(..., description="Whether current dip is a buy opportunity")
    buy_signal_strength: float = Field(..., description="Buy signal strength (0-100)")
    signal_reason: str = Field(..., description="Explanation for the signal")
    
    # Historical stats
    typical_recovery_days: float = Field(
        ..., description="Typical days to recover after buying at optimal dip"
    )
    avg_dips_per_year: DipFrequency = Field(
        ..., description="Average dip frequency per year"
    )
    
    # Fundamentals
    fundamentals_healthy: bool = Field(..., description="Whether fundamentals pass filters")
    fundamental_notes: list[str] = Field(
        default_factory=list, description="Fundamental analysis notes"
    )
    
    # V2 Analysis metadata
    continuation_risk: Literal["low", "medium", "high"] = Field(
        default="medium", description="Risk of continued decline"
    )
    data_years: float = Field(default=0.0, description="Years of data analyzed")
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium", description="Analysis confidence level"
    )
    outlier_events: int = Field(default=0, description="Number of outlier events filtered")
    
    # Detailed threshold analysis
    threshold_analysis: list[ThresholdStats] = Field(
        default_factory=list, description="Analysis for each threshold level"
    )
    
    analyzed_at: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

    model_config = {"from_attributes": True}


# =============================================================================
# Routes
# =============================================================================


@router.get(
    "/{symbol}",
    response_model=DipEntryResponse,
    summary="Get dip entry analysis for a symbol",
    description="""
    Analyze historical dips to find the optimal entry point for adding to a position.
    
    This answers: "I already own this stock. How much should it drop before I buy more?"
    
    Returns:
    - Optimal dip threshold (e.g., -15% = wait for 15% pullback)
    - Limit order price to set
    - Historical win rates and recovery times at each threshold
    - Current drawdown and whether it's a buy opportunity now
    
    Results are pre-computed nightly for tracked symbols.
    If data is not available, a computation job is queued (returns 202 Accepted).
    """,
    responses={
        200: {"description": "Dip entry analysis available"},
        202: {"description": "Analysis is being computed, try again later"},
    },
)
async def get_dip_entry(
    symbol: str = Path(..., min_length=1, max_length=10, description="Stock ticker"),
) -> DipEntryResponse:
    """Get dip entry analysis for a symbol."""
    from fastapi import Response
    from fastapi.responses import JSONResponse
    
    from app.celery_app import celery_app
    from app.repositories import dip_state_orm as dip_state_repo
    from app.repositories import quant_precomputed_orm as quant_repo
    
    symbol = symbol.upper().strip()
    
    # Check precomputed cache first
    precomputed = await quant_repo.get_precomputed(symbol)
    if precomputed and precomputed.dip_entry_optimal_threshold is not None:
        # Get current price from dip_state
        dip_state = await dip_state_repo.get_dip_state(symbol)
        current_price = float(dip_state.current_price) if dip_state and dip_state.current_price else 0.0
        recent_high = float(dip_state.ref_high) if dip_state and dip_state.ref_high else 0.0
        current_drawdown = float(dip_state.dip_percentage) if dip_state and dip_state.dip_percentage else 0.0
        
        # Parse threshold analysis to build backtest data
        threshold_analysis = [ThresholdStats(**t) for t in (precomputed.dip_entry_threshold_analysis or [])]
        
        # Compute years tested from data_start and data_end
        years_tested = 5.0  # Default
        if precomputed.data_start and precomputed.data_end:
            days_diff = (precomputed.data_end - precomputed.data_start).days
            years_tested = round(days_diff / 365.25, 1)
        
        # Build backtest from max profit threshold data
        backtest_data = None
        max_profit_threshold = float(precomputed.dip_entry_max_profit_threshold) if precomputed.dip_entry_max_profit_threshold else None
        if max_profit_threshold is not None and threshold_analysis:
            # Find the threshold stats for max profit threshold
            max_profit_stats = next(
                (t for t in threshold_analysis if abs(t.threshold - max_profit_threshold) < 0.1),
                None
            )
            if max_profit_stats:
                # Compute buy-and-hold return from precomputed or estimate from data
                # Use the buy_hold_return from quant_precomputed if available
                buy_hold_return = float(precomputed.backtest_buy_hold_return_pct) if precomputed.backtest_buy_hold_return_pct else 0.0
                strategy_return = max_profit_stats.total_profit_compounded if max_profit_stats.total_profit_compounded else max_profit_stats.total_profit
                
                backtest_data = DipEntryBacktest(
                    strategy_return_pct=strategy_return,
                    buy_hold_return_pct=buy_hold_return,
                    vs_buy_hold_pct=strategy_return - buy_hold_return,
                    n_trades=max_profit_stats.occurrences,
                    win_rate=max_profit_stats.win_rate,
                    avg_return_per_trade=max_profit_stats.avg_return,
                    sharpe_ratio=max_profit_stats.sharpe_ratio,
                    max_drawdown=max_profit_stats.max_further_drawdown,
                    years_tested=years_tested,
                )
        
        return DipEntryResponse(
            symbol=symbol,
            current_price=current_price,
            recent_high=recent_high,
            current_drawdown_pct=current_drawdown,
            volatility_regime="normal",  # Could be enhanced
            optimal_dip_threshold=float(precomputed.dip_entry_optimal_threshold),
            optimal_entry_price=float(precomputed.dip_entry_optimal_price) if precomputed.dip_entry_optimal_price else current_price,
            max_profit_threshold=max_profit_threshold or 0.0,
            max_profit_entry_price=float(precomputed.dip_entry_max_profit_price) if precomputed.dip_entry_max_profit_price else 0.0,
            max_profit_total_return=float(precomputed.dip_entry_max_profit_total_return) if precomputed.dip_entry_max_profit_total_return else 0.0,
            backtest=backtest_data,
            is_buy_now=precomputed.dip_entry_is_buy_now,
            buy_signal_strength=float(precomputed.dip_entry_signal_strength) if precomputed.dip_entry_signal_strength else 0.0,
            signal_reason=precomputed.dip_entry_signal_reason or "",
            typical_recovery_days=float(precomputed.dip_entry_recovery_days) if precomputed.dip_entry_recovery_days else 30.0,
            avg_dips_per_year=DipFrequency(**{"10_pct": 2.0, "15_pct": 1.0, "20_pct": 0.5}),
            fundamentals_healthy=True,
            fundamental_notes=[],
            data_years=years_tested,
            threshold_analysis=threshold_analysis,
            analyzed_at=precomputed.computed_at or datetime.now(),
        )
    
    # Check legacy cache
    cache_key = f"dip_entry:{symbol}"
    cached = await _cache.get(cache_key)
    if cached:
        return DipEntryResponse(**cached)
    
    # Verify symbol exists
    db_symbol = await symbols_repo.get_symbol(symbol)
    if not db_symbol:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(message=f"Symbol {symbol} not found")
    
    # Queue background computation and return 202 Accepted
    celery_app.send_task("jobs.precompute_dip_entry", args=[symbol])
    logger.info(f"Queued dip entry precomputation for {symbol}")
    
    return JSONResponse(
        status_code=202,
        content={
            "status": "pending",
            "message": f"Dip entry analysis for {symbol} is being computed. Please try again in a few seconds.",
            "symbol": symbol,
        },
    )
