"""Dip entry API routes.

Returns optimal dip entry levels for adding to existing positions.
This helps answer: "How much should the stock drop before I buy more?"
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Path
from pydantic import BaseModel, Field

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer, get_dip_summary
from app.repositories import price_history_orm as price_history_repo
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
    """,
)
async def get_dip_entry(
    symbol: str = Path(..., min_length=1, max_length=10, description="Stock ticker"),
) -> DipEntryResponse:
    """Get dip entry analysis for a symbol."""
    symbol = symbol.upper().strip()
    
    # Check cache
    cache_key = f"dip_entry:{symbol}"
    cached = await _cache.get(cache_key)
    if cached:
        return DipEntryResponse(**cached)
    
    # Verify symbol exists
    db_symbol = await symbols_repo.get_symbol(symbol)
    if not db_symbol:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(message=f"Symbol {symbol} not found")
    
    # Fetch price history (5 years)
    end_date = date.today()
    start_date = end_date - timedelta(days=1825)  # 5 years
    
    df = await price_history_repo.get_prices_as_dataframe(symbol, start_date, end_date)
    
    if df is None or df.empty or len(df) < 252:  # Need at least 1 year
        from app.core.exceptions import ValidationError
        raise ValidationError(
            message=f"Insufficient price history for {symbol} (need at least 1 year)"
        )
    
    # Get fundamentals if available
    fundamentals = None
    if db_symbol.pe_ratio or db_symbol.debt_to_equity:
        fundamentals = {
            "pe_ratio": db_symbol.pe_ratio,
            "debt_to_equity": db_symbol.debt_to_equity,
            # Add more if available
        }
    
    # Run analysis
    optimizer = DipEntryOptimizer()
    result = optimizer.analyze(df, symbol, fundamentals)
    
    # Convert to response
    summary = get_dip_summary(result)
    
    response_data = {
        "symbol": summary["symbol"],
        "current_price": summary["current_price"],
        "recent_high": summary["recent_high"],
        "current_drawdown_pct": summary["current_drawdown_pct"],
        "volatility_regime": summary["volatility_regime"],
        "optimal_dip_threshold": summary["optimal_dip_threshold"],
        "optimal_entry_price": summary["optimal_entry_price"],
        "is_buy_now": summary["is_buy_now"],
        "buy_signal_strength": summary["buy_signal_strength"],
        "signal_reason": summary["signal_reason"],
        "typical_recovery_days": summary["typical_recovery_days"],
        "avg_dips_per_year": summary["avg_dips_per_year"],
        "fundamentals_healthy": summary["fundamentals_healthy"],
        "fundamental_notes": summary["fundamental_notes"],
        "threshold_analysis": summary["threshold_analysis"],
        "analyzed_at": datetime.now(),
    }
    
    # Cache result
    await _cache.set(cache_key, response_data, ttl=3600)
    
    return DipEntryResponse(**response_data)
