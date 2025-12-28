"""Strategy signals API routes.

Returns optimized trading signals for tracked symbols.
Data is computed nightly by the strategy_optimize_nightly job.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.dependencies import require_user
from app.cache.cache import Cache
from app.core.security import TokenData
from app.database.connection import get_session
from app.database.orm import StrategySignal


router = APIRouter()
_cache = Cache(prefix="strategy_signals", default_ttl=300)


# =============================================================================
# Response Schemas
# =============================================================================


class StrategyMetrics(BaseModel):
    """Strategy performance metrics."""
    
    total_return_pct: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Risk-adjusted return (Sharpe ratio)")
    win_rate: float = Field(..., description="Percentage of winning trades")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    n_trades: int = Field(..., description="Total number of trades")


class RecencyMetrics(BaseModel):
    """Recency-weighted performance metrics."""
    
    weighted_return: float = Field(..., description="Recency-weighted average return per trade")
    current_year_return_pct: float = Field(..., description="Return in current year (2025)")
    current_year_win_rate: float = Field(..., description="Win rate in current year")
    current_year_trades: int = Field(..., description="Number of trades in current year")


class BenchmarkComparison(BaseModel):
    """Comparison to benchmarks."""
    
    vs_buy_hold: float = Field(..., description="Excess return vs buy-and-hold")
    vs_spy: float | None = Field(None, description="Excess return vs SPY")
    beats_buy_hold: bool = Field(..., description="Whether strategy beats buy-and-hold")


class SignalInfo(BaseModel):
    """Current signal information."""
    
    type: Literal["BUY", "SELL", "HOLD", "WAIT", "WATCH"] = Field(
        ..., description="Signal type"
    )
    reason: str = Field(..., description="Explanation for the signal")
    has_active: bool = Field(..., description="Whether there's an active entry signal")


class FundamentalStatus(BaseModel):
    """Fundamental health status."""
    
    healthy: bool = Field(..., description="Whether fundamentals pass quality filters")
    concerns: list[str] = Field(default_factory=list, description="List of fundamental concerns")


class TradeRecord(BaseModel):
    """Recent trade record."""
    
    entry_date: str
    exit_date: str | None
    entry_price: float
    exit_price: float | None
    pnl_pct: float
    exit_reason: str
    holding_days: int | None = None


class StrategySignalResponse(BaseModel):
    """Full strategy signal response for a symbol."""
    
    symbol: str = Field(..., description="Stock ticker")
    strategy_name: str = Field(..., description="Best strategy name")
    strategy_params: dict = Field(default_factory=dict, description="Strategy parameters")
    
    signal: SignalInfo = Field(..., description="Current signal")
    metrics: StrategyMetrics = Field(..., description="Performance metrics")
    recency: RecencyMetrics = Field(..., description="Recency-weighted metrics")
    benchmarks: BenchmarkComparison = Field(..., description="Benchmark comparison")
    fundamentals: FundamentalStatus = Field(..., description="Fundamental status")
    
    is_statistically_valid: bool = Field(..., description="Passes statistical tests")
    indicators_used: list[str] = Field(default_factory=list, description="Technical indicators used")
    recent_trades: list[TradeRecord] = Field(default_factory=list, description="Last 5 trades")
    
    optimized_at: datetime | None = Field(None, description="When optimization was run")
    
    model_config = {"from_attributes": True}


class StrategySignalSummary(BaseModel):
    """Summary for listing multiple signals."""
    
    symbol: str
    strategy_name: str
    signal_type: str
    has_active_signal: bool
    win_rate: float
    current_year_return_pct: float
    beats_buy_hold: bool
    fundamentals_healthy: bool
    optimized_at: datetime | None


class StrategySignalsListResponse(BaseModel):
    """Response for listing all signals."""
    
    signals: list[StrategySignalSummary]
    total: int
    active_buy_signals: int
    beating_market: int


# =============================================================================
# Routes
# =============================================================================


@router.get(
    "/{symbol}",
    response_model=StrategySignalResponse,
    summary="Get strategy signal for a symbol",
    description="Returns the optimized trading strategy and current signal for a stock.",
)
async def get_strategy_signal(
    symbol: str = Path(..., min_length=1, max_length=10, description="Stock ticker"),
    _user: TokenData = Depends(require_user),
) -> StrategySignalResponse:
    """Get the optimized strategy signal for a specific symbol."""
    symbol = symbol.upper().strip()
    
    # Check cache first
    cached = await _cache.get(symbol)
    if cached:
        return StrategySignalResponse(**cached)
    
    async with get_session() as session:
        result = await session.execute(
            select(StrategySignal).where(StrategySignal.symbol == symbol)
        )
        signal = result.scalar_one_or_none()
        
        if signal is None:
            raise HTTPException(
                status_code=404,
                detail=f"No strategy signal found for {symbol}. "
                "Signals are computed nightly after market close."
            )
        
        response = _build_response(signal)
        
        # Cache for 5 minutes
        await _cache.set(symbol, response.model_dump(mode="json"), ttl=300)
        
        return response


@router.get(
    "",
    response_model=StrategySignalsListResponse,
    summary="List all strategy signals",
    description="Returns a summary of all optimized trading signals.",
)
async def list_strategy_signals(
    signal_type: str | None = Query(None, description="Filter by signal type"),
    active_only: bool = Query(False, description="Only show active buy signals"),
    beating_market: bool = Query(False, description="Only show strategies beating buy-and-hold"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    _user: TokenData = Depends(require_user),
) -> StrategySignalsListResponse:
    """List all strategy signals with optional filters."""
    async with get_session() as session:
        query = select(StrategySignal)
        
        if signal_type:
            query = query.where(StrategySignal.signal_type == signal_type.upper())
        
        if active_only:
            query = query.where(StrategySignal.has_active_signal.is_(True))
        
        if beating_market:
            query = query.where(StrategySignal.beats_buy_hold.is_(True))
        
        query = query.order_by(
            StrategySignal.has_active_signal.desc(),
            StrategySignal.beats_buy_hold.desc(),
            StrategySignal.current_year_return_pct.desc().nullslast(),
        ).limit(limit)
        
        result = await session.execute(query)
        signals = result.scalars().all()
        
        # Count totals
        total_result = await session.execute(select(StrategySignal))
        all_signals = total_result.scalars().all()
        
        total = len(all_signals)
        active_buy = sum(1 for s in all_signals if s.has_active_signal)
        beating = sum(1 for s in all_signals if s.beats_buy_hold)
        
        summaries = [
            StrategySignalSummary(
                symbol=s.symbol,
                strategy_name=s.strategy_name,
                signal_type=s.signal_type,
                has_active_signal=s.has_active_signal,
                win_rate=float(s.win_rate) if s.win_rate else 0,
                current_year_return_pct=float(s.current_year_return_pct) if s.current_year_return_pct else 0,
                beats_buy_hold=s.beats_buy_hold,
                fundamentals_healthy=s.fundamentals_healthy,
                optimized_at=s.optimized_at,
            )
            for s in signals
        ]
        
        return StrategySignalsListResponse(
            signals=summaries,
            total=total,
            active_buy_signals=active_buy,
            beating_market=beating,
        )


@router.get(
    "/{symbol}/chart-data",
    summary="Get strategy data for chart overlay",
    description="Returns data formatted for overlaying on stock charts.",
)
async def get_strategy_chart_data(
    symbol: str = Path(..., min_length=1, max_length=10, description="Stock ticker"),
    _user: TokenData = Depends(require_user),
) -> dict:
    """
    Get strategy data formatted for chart overlay.
    
    Returns entry/exit points, performance metrics, and comparison bands.
    """
    symbol = symbol.upper().strip()
    
    async with get_session() as session:
        result = await session.execute(
            select(StrategySignal).where(StrategySignal.symbol == symbol)
        )
        signal = result.scalar_one_or_none()
        
        if signal is None:
            return {
                "symbol": symbol,
                "has_data": False,
                "message": "No strategy data available",
            }
        
        # Build chart-friendly data
        trades = signal.recent_trades or []
        
        return {
            "symbol": symbol,
            "has_data": True,
            "strategy_name": signal.strategy_name,
            "signal": {
                "type": signal.signal_type,
                "has_active": signal.has_active_signal,
                "reason": signal.signal_reason,
            },
            "performance": {
                "total_return_pct": float(signal.total_return_pct) if signal.total_return_pct else 0,
                "win_rate": float(signal.win_rate) if signal.win_rate else 0,
                "sharpe_ratio": float(signal.sharpe_ratio) if signal.sharpe_ratio else 0,
                "vs_buy_hold": float(signal.vs_buy_hold_pct) if signal.vs_buy_hold_pct else 0,
                "beats_buy_hold": signal.beats_buy_hold,
            },
            "current_year": {
                "return_pct": float(signal.current_year_return_pct) if signal.current_year_return_pct else 0,
                "win_rate": float(signal.current_year_win_rate) if signal.current_year_win_rate else 0,
                "trades": signal.current_year_trades,
            },
            "fundamentals": {
                "healthy": signal.fundamentals_healthy,
                "concerns": signal.fundamental_concerns or [],
            },
            "trade_markers": [
                {
                    "date": t.get("entry_date"),
                    "type": "entry",
                    "price": t.get("entry_price"),
                }
                for t in trades
            ] + [
                {
                    "date": t.get("exit_date"),
                    "type": "exit",
                    "price": t.get("exit_price"),
                    "pnl_pct": t.get("pnl_pct"),
                }
                for t in trades
                if t.get("exit_date")
            ],
            "optimized_at": signal.optimized_at.isoformat() if signal.optimized_at else None,
        }


def _build_response(signal: StrategySignal) -> StrategySignalResponse:
    """Build response from database model."""
    return StrategySignalResponse(
        symbol=signal.symbol,
        strategy_name=signal.strategy_name,
        strategy_params=signal.strategy_params or {},
        signal=SignalInfo(
            type=signal.signal_type,  # type: ignore
            reason=signal.signal_reason or "",
            has_active=signal.has_active_signal,
        ),
        metrics=StrategyMetrics(
            total_return_pct=float(signal.total_return_pct) if signal.total_return_pct else 0,
            sharpe_ratio=float(signal.sharpe_ratio) if signal.sharpe_ratio else 0,
            win_rate=float(signal.win_rate) if signal.win_rate else 0,
            max_drawdown_pct=float(signal.max_drawdown_pct) if signal.max_drawdown_pct else 0,
            n_trades=signal.n_trades,
        ),
        recency=RecencyMetrics(
            weighted_return=float(signal.recency_weighted_return) if signal.recency_weighted_return else 0,
            current_year_return_pct=float(signal.current_year_return_pct) if signal.current_year_return_pct else 0,
            current_year_win_rate=float(signal.current_year_win_rate) if signal.current_year_win_rate else 0,
            current_year_trades=signal.current_year_trades,
        ),
        benchmarks=BenchmarkComparison(
            vs_buy_hold=float(signal.vs_buy_hold_pct) if signal.vs_buy_hold_pct else 0,
            vs_spy=float(signal.vs_spy_pct) if signal.vs_spy_pct else None,
            beats_buy_hold=signal.beats_buy_hold,
        ),
        fundamentals=FundamentalStatus(
            healthy=signal.fundamentals_healthy,
            concerns=signal.fundamental_concerns or [],
        ),
        is_statistically_valid=signal.is_statistically_valid,
        indicators_used=signal.indicators_used or [],
        recent_trades=[
            TradeRecord(**t) for t in (signal.recent_trades or [])
        ],
        optimized_at=signal.optimized_at,
    )
