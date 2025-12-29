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
    win_rate_60d: float = Field(..., description="Win rate after 60 days")
    avg_return_60d: float = Field(..., description="Average return after 60 days")
    recovery_rate: float = Field(..., description="Recovery rate to previous high")
    avg_recovery_days: float = Field(..., description="Average days to recover")
    entry_score: float = Field(..., description="Entry quality score (0-100)")


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
    
    # Optimal entry
    optimal_dip_threshold: float = Field(
        ..., description="Optimal dip threshold to buy (e.g., -15 for 15% dip)"
    )
    optimal_entry_price: float = Field(
        ..., description="Price at which to set limit buy order"
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
        recent_high = float(dip_state.peak_price) if dip_state and dip_state.peak_price else 0.0
        current_drawdown = float(dip_state.dip_pct) if dip_state and dip_state.dip_pct else 0.0
        
        return DipEntryResponse(
            symbol=symbol,
            current_price=current_price,
            recent_high=recent_high,
            current_drawdown_pct=current_drawdown,
            volatility_regime="normal",  # Could be enhanced
            optimal_dip_threshold=float(precomputed.dip_entry_optimal_threshold),
            optimal_entry_price=float(precomputed.dip_entry_optimal_price) if precomputed.dip_entry_optimal_price else current_price,
            is_buy_now=precomputed.dip_entry_is_buy_now,
            buy_signal_strength=float(precomputed.dip_entry_signal_strength) if precomputed.dip_entry_signal_strength else 0.0,
            signal_reason=precomputed.dip_entry_signal_reason or "",
            typical_recovery_days=float(precomputed.dip_entry_recovery_days) if precomputed.dip_entry_recovery_days else 30.0,
            avg_dips_per_year=DipFrequency(**{"10_pct": 2.0, "15_pct": 1.0, "20_pct": 0.5}),
            fundamentals_healthy=True,
            fundamental_notes=[],
            threshold_analysis=[ThresholdStats(**t) for t in (precomputed.dip_entry_threshold_analysis or [])],
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
