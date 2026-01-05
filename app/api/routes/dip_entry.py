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


def _to_decimal(value: float) -> float:
    """Convert percentage (85.7) to decimal (0.857)."""
    return value / 100 if value else 0.0


def _compute_optimal_period(raw: dict) -> tuple[int, float, float, float, float]:
    """Compute optimal holding period from multi-period metrics.
    
    Returns (optimal_days, avg_return, win_rate, total_profit, sharpe).
    Selects the period with best capital efficiency (return per day).
    
    Note: Analysis shows 60 days is optimal for capital efficiency.
    Beyond 60-90 days, alpha vs buy-and-hold goes negative.
    """
    # Holding periods are now DYNAMICALLY DISCOVERED per symbol
    # The optimizer tests range(5, max+1, 5) and picks best by Sharpe
    
    # Get metrics for primary period (60d)
    sharpe_60 = raw.get("sharpe_ratio", 0)  # Primary is now 60d
    sortino_60 = raw.get("sortino_ratio", 0)
    
    avg_return_60 = raw.get("avg_return", 0)
    win_rate_60 = raw.get("win_rate", 0)
    total_profit_60 = raw.get("total_profit_compounded", 0)
    
    # Check if 40d data exists (secondary period)
    avg_return_40 = raw.get("avg_return_40d", raw.get("avg_return_60d", 0))
    win_rate_40 = raw.get("win_rate_40d", raw.get("win_rate_60d", 0))
    
    # Legacy: check for old 90d data
    avg_return_90 = raw.get("avg_return_90d", 0)
    
    # If we only have 60d data, return 60d
    if not avg_return_40:
        return 60, avg_return_60, win_rate_60, total_profit_60, sharpe_60
    
    # Compare based on capital efficiency (return / days)
    efficiency_60 = avg_return_60 / 60 if avg_return_60 > 0 else 0
    efficiency_40 = avg_return_40 / 40 if avg_return_40 > 0 else 0
    
    # Pick the period with better efficiency
    # 40d needs 10% better efficiency to overcome time preference (give dips more time)
    if efficiency_40 > efficiency_60 * 1.1:
        return 40, avg_return_40, win_rate_40, avg_return_40 * raw.get("occurrences", 1), sharpe_60 * 0.95
    else:
        return 60, avg_return_60, win_rate_60, total_profit_60, sharpe_60


def _convert_threshold_stats_to_decimal(raw: dict) -> dict:
    """Convert raw threshold stats from percentage format to decimal format.
    
    All percentage values (rates, returns, drawdowns) are converted from
    percentage format (85.7) to decimal format (0.857).
    """
    # Compute optimal period from existing data if not present
    if raw.get("optimal_avg_return", 0) == 0 and raw.get("avg_return", 0) > 0:
        # Fallback: compute optimal from existing multi-period data
        opt_days, opt_return, opt_win, opt_profit, opt_sharpe = _compute_optimal_period(raw)
    else:
        # Use precomputed optimal values
        opt_days = raw.get("optimal_holding_days", 60)
        opt_return = raw.get("optimal_avg_return", 0)
        opt_win = raw.get("optimal_win_rate", 0)
        opt_profit = raw.get("optimal_total_profit", 0)
        opt_sharpe = raw.get("optimal_sharpe", 0)
    
    return {
        "threshold": raw.get("threshold", 0) / 100,  # -15% -> -0.15
        "occurrences": raw.get("occurrences", 0),
        "per_year": raw.get("per_year", 0),
        # OPTIMAL HOLDING PERIOD - computed dynamically or from cache
        "optimal_holding_days": opt_days,
        "optimal_avg_return": _to_decimal(opt_return),
        "optimal_win_rate": _to_decimal(opt_win),
        "optimal_total_profit": _to_decimal(opt_profit),
        "optimal_sharpe": opt_sharpe,
        # Returns and rates (90-day defaults)
        "win_rate": _to_decimal(raw.get("win_rate", 0)),
        "avg_return": _to_decimal(raw.get("avg_return", 0)),
        "total_profit": _to_decimal(raw.get("total_profit", 0)),
        "total_profit_compounded": _to_decimal(raw.get("total_profit_compounded", 0)),
        "sharpe_ratio": raw.get("sharpe_ratio", 0),  # Not a percentage
        "sortino_ratio": raw.get("sortino_ratio", 0),  # Not a percentage
        "cvar": _to_decimal(raw.get("cvar", 0)),
        # Recovery metrics
        "recovery_rate": _to_decimal(raw.get("recovery_rate", 0)),
        "recovery_threshold_rate": _to_decimal(raw.get("recovery_threshold_rate", 0)),
        "avg_recovery_days": raw.get("avg_recovery_days", 0),  # Days, not percentage
        "avg_days_to_threshold": raw.get("avg_days_to_threshold", 0),
        "avg_recovery_velocity": raw.get("avg_recovery_velocity", 0),
        # Recovery-based profit
        "avg_return_at_recovery": _to_decimal(raw.get("avg_return_at_recovery", 0)),
        "total_profit_at_recovery": _to_decimal(raw.get("total_profit_at_recovery", 0)),
        "avg_days_at_recovery": raw.get("avg_days_at_recovery", 0),
        # Risk metrics
        "max_further_drawdown": _to_decimal(raw.get("max_further_drawdown", 0)),
        "avg_further_drawdown": _to_decimal(raw.get("avg_further_drawdown", 0)),
        "prob_further_drop": _to_decimal(raw.get("prob_further_drop", 0)),
        "continuation_risk": raw.get("continuation_risk", "medium"),
        # Scores
        "entry_score": raw.get("entry_score", 0),  # Score, not percentage
        "confidence": raw.get("confidence", "medium"),
        # Legacy
        "win_rate_60d": _to_decimal(raw.get("win_rate_60d", 0)),
        "avg_return_60d": _to_decimal(raw.get("avg_return_60d", 0)),
    }


# =============================================================================
# Response Schemas
# =============================================================================


class DipSignalTrigger(BaseModel):
    """A historical dip trade signal for chart display."""
    
    date: str = Field(..., description="Date of the signal (YYYY-MM-DD)")
    signal_type: Literal["entry", "exit"] = Field(..., description="Whether entry or exit signal")
    price: float = Field(..., description="Price at which signal triggered")
    threshold_pct: float = Field(..., description="Dip threshold that triggered (decimal, e.g. -0.15)")
    return_pct: float = Field(default=0.0, description="Trade return (for exit signals)")
    holding_days: int = Field(default=0, description="Days held (for exit signals)")
    
    model_config = {"populate_by_name": True}


class DipSignalTriggersResponse(BaseModel):
    """Response containing historical dip trade signals for chart overlay."""
    
    symbol: str
    threshold_pct: float = Field(..., description="Optimal dip threshold used (decimal)")
    triggers: list[DipSignalTrigger] = Field(default_factory=list)
    n_trades: int = 0
    win_rate: float = 0.0  # Decimal
    total_return_pct: float = 0.0  # Decimal
    
    model_config = {"populate_by_name": True}


class DipFrequency(BaseModel):
    """Annual dip frequency at different thresholds."""
    
    ten_pct: float = Field(..., alias="10_pct", description="Average 10%+ dips per year")
    fifteen_pct: float = Field(..., alias="15_pct", description="Average 15%+ dips per year")
    twenty_pct: float = Field(..., alias="20_pct", description="Average 20%+ dips per year")

    model_config = {"populate_by_name": True}


class ThresholdStats(BaseModel):
    """Statistics for a specific dip threshold level.
    
    All percentage values are in decimal format (0.10 = 10%).
    """
    
    threshold: float = Field(..., description="Dip threshold (e.g., -0.10 for 10% dip)")
    occurrences: int = Field(..., description="Number of times this dip occurred")
    per_year: float = Field(..., description="Average occurrences per year")
    # OPTIMAL HOLDING PERIOD - dynamically computed for best risk-adjusted returns
    optimal_holding_days: int = Field(default=60, description="Statistically optimal holding period (20, 40, or 60 days)")
    optimal_avg_return: float = Field(default=0.0, description="Avg return at optimal holding period (decimal)")
    optimal_win_rate: float = Field(default=0.0, description="Win rate at optimal holding period (decimal)")
    optimal_total_profit: float = Field(default=0.0, description="Total compounded profit at optimal period (decimal)")
    optimal_sharpe: float = Field(default=0.0, description="Sharpe ratio at optimal period")
    # V2: Multi-period metrics (primary = 60 days)
    win_rate: float = Field(default=0.0, description="Win rate after 60 days (decimal)")
    avg_return: float = Field(default=0.0, description="Average return after 60 days (decimal)")
    total_profit: float = Field(default=0.0, description="Total expected profit (decimal)")
    total_profit_compounded: float = Field(default=0.0, description="Compounded total profit (decimal)")
    sharpe_ratio: float = Field(default=0.0, description="Risk-adjusted return")
    sortino_ratio: float = Field(default=0.0, description="Downside risk-adjusted return")
    cvar: float = Field(default=0.0, description="Tail risk (worst 5% of returns, decimal)")
    # Recovery metrics
    recovery_rate: float = Field(..., description="Recovery rate to previous high (decimal)")
    recovery_threshold_rate: float = Field(default=0.0, description="Recovery to entry threshold (decimal)")
    avg_recovery_days: float = Field(..., description="Average days to recover")
    avg_days_to_threshold: float = Field(default=0.0, description="Days to recover to entry price")
    avg_recovery_velocity: float = Field(default=0.0, description="Recovery speed (higher = faster)")
    # Recovery-based profit metrics (sell at recovery strategy)
    avg_return_at_recovery: float = Field(default=0.0, description="Avg return selling at recovery (decimal)")
    total_profit_at_recovery: float = Field(default=0.0, description="Compounded profit at recovery (decimal)")
    avg_days_at_recovery: float = Field(default=0.0, description="Avg days held until recovery")
    # Risk metrics
    max_further_drawdown: float = Field(default=0.0, description="Worst drawdown after entry (decimal)")
    avg_further_drawdown: float = Field(default=0.0, description="Average MAE (decimal)")
    prob_further_drop: float = Field(default=0.0, description="P(drops another 10%) (decimal)")
    continuation_risk: Literal["low", "medium", "high"] = Field(default="medium")
    # Scores
    entry_score: float = Field(..., description="V2 entry quality score")
    confidence: Literal["low", "medium", "high"] = Field(default="medium")
    # Legacy (backward compat)
    win_rate_60d: float = Field(default=0.0, description="Win rate after 60 days (decimal)")
    avg_return_60d: float = Field(default=0.0, description="Average return after 60 days (decimal)")


class DipEntryBacktest(BaseModel):
    """Backtest results for optimized dip strategy.
    
    All percentage values are in decimal format (0.10 = 10%).
    Uses statistically optimized dip threshold AND holding period.
    """
    
    optimal_dip_threshold: float = Field(default=0.0, description="Optimal dip depth (e.g., -0.16 for 16% dip)")
    optimal_holding_days: int = Field(default=60, description="Statistically optimal holding period (20, 40, or 60 days)")
    strategy_return: float = Field(default=0.0, description="Total compounded return from optimized strategy (decimal)")
    buy_hold_return: float = Field(default=0.0, description="Buy-and-hold return over same period (decimal)")
    vs_buy_hold: float = Field(default=0.0, description="Edge vs buy-and-hold (strategy - B&H, decimal)")
    n_trades: int = Field(default=0, description="Number of dip trades executed")
    win_rate: float = Field(default=0.0, description="Win rate at optimal holding period (decimal)")
    avg_return_per_trade: float = Field(default=0.0, description="Average return per trade at optimal period (decimal)")
    sharpe_ratio: float = Field(default=0.0, description="Risk-adjusted return at optimal period")
    max_drawdown: float = Field(default=0.0, description="Maximum adverse excursion after entry (decimal, negative)")
    years_tested: float = Field(default=0.0, description="Years of historical data used")


class DipEntryResponse(BaseModel):
    """Dip entry analysis response.
    
    All percentage values are in decimal format (0.10 = 10%).
    """
    
    symbol: str = Field(..., description="Stock ticker")
    
    # Current state
    current_price: float = Field(..., description="Current stock price")
    recent_high: float = Field(..., description="Recent high price")
    current_drawdown_pct: float = Field(..., description="Current drawdown from high (decimal, -0.15 = 15% dip)")
    volatility_regime: Literal["low", "normal", "high"] = Field(
        ..., description="Current volatility regime"
    )
    
    # Risk-adjusted optimal entry (less pain, better timing)
    optimal_dip_threshold: float = Field(
        ..., description="Risk-adjusted optimal dip threshold (decimal, -0.15 = 15% dip)"
    )
    optimal_entry_price: float = Field(
        ..., description="Risk-adjusted price for limit buy order"
    )
    
    # Max profit optimal entry (more opportunities, higher total return)
    max_profit_threshold: float = Field(
        default=0.0, description="Max profit dip threshold (decimal)"
    )
    max_profit_entry_price: float = Field(
        default=0.0, description="Max profit price for limit buy order"
    )
    max_profit_total_return: float = Field(
        default=0.0, description="Expected total return at max profit threshold (decimal)"
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
    
    # Historical dip trade signals for chart overlay
    signal_triggers: list[DipSignalTrigger] = Field(
        default_factory=list, description="Historical dip entry/exit points for chart markers"
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
        
        # Get symbol's min_dip_pct to filter threshold analysis
        db_symbol = await symbols_repo.get_symbol(symbol)
        min_dip_pct = float(db_symbol.min_dip_pct) if db_symbol and db_symbol.min_dip_pct else 0.10
        min_dip_threshold = -min_dip_pct  # In decimal format (e.g., -0.15)
        
        # Parse threshold analysis, convert to decimal, and filter to meaningful thresholds
        raw_thresholds = precomputed.dip_entry_threshold_analysis or []
        all_thresholds = [ThresholdStats(**_convert_threshold_stats_to_decimal(t)) for t in raw_thresholds]
        # Filter to only show thresholds at or deeper than symbol's minimum
        threshold_analysis = [t for t in all_thresholds if t.threshold <= min_dip_threshold]
        
        # Compute years tested from data_start and data_end
        years_tested = 5.0  # Default
        if precomputed.data_start and precomputed.data_end:
            days_diff = (precomputed.data_end - precomputed.data_start).days
            years_tested = round(days_diff / 365.25, 1)
        
        # Build backtest from max profit threshold data
        backtest_data = None
        max_profit_threshold_raw = float(precomputed.dip_entry_max_profit_threshold) if precomputed.dip_entry_max_profit_threshold else None
        max_profit_threshold_decimal = max_profit_threshold_raw / 100 if max_profit_threshold_raw else None  # -16 -> -0.16
        if max_profit_threshold_decimal is not None and all_thresholds:
            # Find the threshold stats for max profit threshold (comparing in decimal format)
            max_profit_stats = next(
                (t for t in all_thresholds if abs(t.threshold - max_profit_threshold_decimal) < 0.01),
                None
            )
            if max_profit_stats:
                # Get buy-and-hold return from precomputed (in % format, need to convert)
                buy_hold_return_pct = float(precomputed.backtest_buy_hold_return_pct) if precomputed.backtest_buy_hold_return_pct else 0.0
                buy_hold_return = buy_hold_return_pct / 100  # 989% -> 9.89
                
                # Use OPTIMIZED holding period metrics (dynamically computed per threshold)
                # optimal_holding_days is 20, 40, or 60 based on best Sharpe ratio
                optimal_days = max_profit_stats.optimal_holding_days
                strategy_return = max_profit_stats.optimal_total_profit
                avg_return = max_profit_stats.optimal_avg_return
                win_rate = max_profit_stats.optimal_win_rate
                sharpe = max_profit_stats.optimal_sharpe
                
                backtest_data = DipEntryBacktest(
                    optimal_dip_threshold=max_profit_threshold_decimal,  # Already in decimal
                    optimal_holding_days=optimal_days,  # Statistically optimal period (20, 40, or 60)
                    strategy_return=strategy_return,  # Return at optimal holding period
                    buy_hold_return=buy_hold_return,  # Converted to decimal
                    vs_buy_hold=strategy_return - buy_hold_return,
                    n_trades=max_profit_stats.occurrences,
                    win_rate=win_rate,  # Win rate at optimal period
                    avg_return_per_trade=avg_return,  # Avg return at optimal period
                    sharpe_ratio=sharpe,  # Sharpe at optimal period
                    max_drawdown=max_profit_stats.max_further_drawdown,  # Already in decimal
                    years_tested=years_tested,
                )
        
        # Parse signal triggers for chart overlay
        raw_triggers = precomputed.dip_entry_signal_triggers or {}
        signal_triggers = [
            DipSignalTrigger(
                date=t.get("date", ""),
                signal_type=t.get("signal_type", "entry"),
                price=t.get("price", 0.0),
                threshold_pct=t.get("threshold_pct", 0.0),
                return_pct=t.get("return_pct", 0.0),
                holding_days=t.get("holding_days", 0),
            )
            for t in raw_triggers.get("triggers", [])
        ]
        
        return DipEntryResponse(
            symbol=symbol,
            current_price=current_price,
            recent_high=recent_high,
            current_drawdown_pct=_to_decimal(current_drawdown),  # Convert to decimal
            volatility_regime="normal",  # Could be enhanced
            optimal_dip_threshold=_to_decimal(float(precomputed.dip_entry_optimal_threshold)),  # Convert to decimal
            optimal_entry_price=float(precomputed.dip_entry_optimal_price) if precomputed.dip_entry_optimal_price else current_price,
            max_profit_threshold=max_profit_threshold_decimal or 0.0,  # Already in decimal
            max_profit_entry_price=float(precomputed.dip_entry_max_profit_price) if precomputed.dip_entry_max_profit_price else 0.0,
            max_profit_total_return=_to_decimal(float(precomputed.dip_entry_max_profit_total_return)) if precomputed.dip_entry_max_profit_total_return else 0.0,
            backtest=backtest_data,
            is_buy_now=precomputed.dip_entry_is_buy_now,
            buy_signal_strength=_to_decimal(float(precomputed.dip_entry_signal_strength)) if precomputed.dip_entry_signal_strength else 0.0,
            signal_reason=precomputed.dip_entry_signal_reason or "",
            typical_recovery_days=float(precomputed.dip_entry_recovery_days) if precomputed.dip_entry_recovery_days else 30.0,
            avg_dips_per_year=DipFrequency(**{"10_pct": 2.0, "15_pct": 1.0, "20_pct": 0.5}),
            fundamentals_healthy=True,
            fundamental_notes=[],
            data_years=years_tested,
            threshold_analysis=threshold_analysis,
            signal_triggers=signal_triggers,
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
