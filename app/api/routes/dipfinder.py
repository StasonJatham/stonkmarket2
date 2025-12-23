"""DipFinder API routes.

Endpoints for computing, retrieving, and managing dip signals.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Dict

from fastapi import APIRouter, Depends, Query, Path, BackgroundTasks

from app.api.dependencies import require_user, require_admin
from app.core.exceptions import NotFoundError, BadRequestError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.database.connection import fetch_all
from app.dipfinder.config import get_dipfinder_config
from app.dipfinder.service import get_dipfinder_service
from app.schemas.dipfinder import (
    DipSignalResponse,
    DipSignalListResponse,
    DipFinderRunRequest,
    DipFinderRunResponse,
    DipHistoryResponse,
    DipHistoryEntry,
    DipFinderConfigResponse,
    QualityFactorsResponse,
    StabilityFactorsResponse,
)

logger = get_logger("api.dipfinder")

router = APIRouter()


async def _get_dip_state_map(tickers: list[str]) -> Dict[str, dict]:
    """Get dip_state data for multiple tickers (ATH-based source of truth)."""
    if not tickers:
        return {}
    
    placeholders = ", ".join(f"${i+1}" for i in range(len(tickers)))
    rows = await fetch_all(
        f"""
        SELECT symbol, current_price, ath_price, dip_percentage, dip_start_date
        FROM dip_state
        WHERE symbol IN ({placeholders})
        """,
        *tickers
    )
    return {row["symbol"]: dict(row) for row in rows}


def _signal_to_response(signal, include_factors: bool = False, dip_state: dict = None) -> DipSignalResponse:
    """Convert DipSignal to response schema.
    
    Args:
        signal: The computed DipSignal
        include_factors: Whether to include quality/stability factors
        dip_state: Optional dip_state data (ATH-based source of truth)
    """
    # Default to signal's computed values
    dip_pct = signal.dip_metrics.dip_pct
    peak_price = signal.dip_metrics.peak_price
    current_price = signal.dip_metrics.current_price
    persist_days = signal.dip_metrics.persist_days
    dip_score = signal.dip_score
    final_score = signal.final_score
    
    # Override with ATH-based values from dip_state (source of truth)
    if dip_state:
        if dip_state.get("dip_percentage"):
            dip_pct = float(dip_state["dip_percentage"]) / 100
        if dip_state.get("ath_price"):
            peak_price = float(dip_state["ath_price"])
        if dip_state.get("current_price"):
            current_price = float(dip_state["current_price"])
        if dip_state.get("dip_start_date"):
            from datetime import date
            persist_days = (date.today() - dip_state["dip_start_date"]).days
        # Recalculate scores based on ATH dip
        dip_score = min(100.0, dip_pct * 100 * 5)
        final_score = (signal.quality_metrics.score + signal.stability_metrics.score + dip_score) / 3
    
    response = DipSignalResponse(
        ticker=signal.ticker,
        window=signal.window,
        benchmark=signal.benchmark,
        as_of_date=signal.as_of_date,
        dip_stock=dip_pct,
        peak_stock=peak_price,
        current_price=current_price,
        dip_pctl=signal.dip_metrics.dip_percentile,
        dip_vs_typical=signal.dip_metrics.dip_vs_typical,
        persist_days=persist_days,
        dip_mkt=signal.market_context.dip_mkt,
        excess_dip=signal.market_context.excess_dip,
        dip_class=signal.market_context.dip_class.value,
        quality_score=signal.quality_metrics.score,
        stability_score=signal.stability_metrics.score,
        dip_score=dip_score,
        final_score=final_score,
        alert_level=signal.alert_level.value,
        should_alert=signal.should_alert,
        reason=signal.reason,
    )

    if include_factors:
        response.quality_factors = QualityFactorsResponse(
            **signal.quality_metrics.to_dict()
        )
        response.stability_factors = StabilityFactorsResponse(
            **signal.stability_metrics.to_dict()
        )

    return response


@router.get(
    "/signals",
    response_model=DipSignalListResponse,
    summary="Get dip signals",
    description="Compute and return dip signals for specified tickers.",
)
async def get_signals(
    tickers: str = Query(
        ...,
        description="Comma-separated list of ticker symbols",
        examples=["NVDA,AAPL,MSFT"],
    ),
    benchmark: Optional[str] = Query(
        default=None,
        description="Benchmark ticker (default: SPY)",
    ),
    window: Optional[int] = Query(
        default=30,
        ge=7,
        le=365,
        description="Window for dip calculation in days",
    ),
    include_factors: bool = Query(
        default=False,
        description="Include detailed quality/stability factors",
    ),
) -> DipSignalListResponse:
    """
    Get dip signals for specified tickers.

    Returns computed signals with dip metrics, market context,
    quality/stability scores, and alert decisions.
    """
    # Parse tickers
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    if not ticker_list:
        raise BadRequestError(
            message="No valid tickers provided",
            details={"tickers": tickers},
        )

    if len(ticker_list) > 20:
        raise BadRequestError(
            message="Maximum 20 tickers per request",
            details={"count": len(ticker_list)},
        )

    config = get_dipfinder_config()
    service = get_dipfinder_service()

    benchmark = benchmark or config.default_benchmark
    window = window or config.windows[1]

    signals = await service.get_signals(
        tickers=ticker_list,
        benchmark=benchmark,
        window=window,
    )
    
    # Get dip_state data for ATH-based dip percentages
    dip_state_map = await _get_dip_state_map(ticker_list)

    return DipSignalListResponse(
        signals=[_signal_to_response(s, include_factors, dip_state_map.get(s.ticker)) for s in signals],
        count=len(signals),
        benchmark=benchmark,
        window=window,
        as_of_date=date.today().isoformat(),
    )


@router.get(
    "/signals/{ticker}",
    response_model=DipSignalResponse,
    summary="Get signal for a single ticker",
    description="Get detailed dip signal for a specific ticker.",
    responses={
        404: {"description": "Ticker not found or no data available"},
    },
)
async def get_ticker_signal(
    ticker: str = Path(..., min_length=1, max_length=10, description="Ticker symbol"),
    benchmark: Optional[str] = Query(default=None),
    window: Optional[int] = Query(default=30, ge=7, le=365),
    force_refresh: bool = Query(default=False, description="Force recomputation"),
) -> DipSignalResponse:
    """Get dip signal for a single ticker with full details."""
    config = get_dipfinder_config()
    service = get_dipfinder_service()

    signal = await service.get_signal(
        ticker=ticker.upper(),
        benchmark=benchmark or config.default_benchmark,
        window=window or config.windows[1],
        force_refresh=force_refresh,
    )

    if signal is None:
        raise NotFoundError(
            message=f"Could not compute signal for {ticker}",
            details={"ticker": ticker},
        )

    # Get dip_state for ATH-based dip percentage
    dip_state_map = await _get_dip_state_map([ticker.upper()])
    return _signal_to_response(signal, include_factors=True, dip_state=dip_state_map.get(ticker.upper()))


@router.post(
    "/run",
    response_model=DipFinderRunResponse,
    summary="Run DipFinder computation",
    description="Kick off signal computation for a universe of tickers.",
)
async def run_dipfinder(
    request: DipFinderRunRequest,
    background_tasks: BackgroundTasks,
    user: TokenData = Depends(require_user),
) -> DipFinderRunResponse:
    """
    Run DipFinder computation for specified or user's tickers.

    - If tickers are provided, computes for those tickers.
    - If not, uses the user's tracked symbols from the database.

    Computation runs synchronously for small sets (<10 tickers)
    or in background for larger sets.
    """
    config = get_dipfinder_config()
    service = get_dipfinder_service()

    # Get tickers to process
    if request.tickers:
        tickers = [t.upper() for t in request.tickers]
    else:
        # Get user's tracked symbols from database
        from app.database.connection import fetch_all

        rows = await fetch_all("SELECT symbol FROM symbols ORDER BY symbol")
        tickers = [r["symbol"] for r in rows]

    if not tickers:
        return DipFinderRunResponse(
            status="completed",
            message="No tickers to process",
            tickers_processed=0,
            signals_generated=0,
            alerts_triggered=0,
        )

    benchmark = request.benchmark or config.default_benchmark
    windows = request.windows or config.windows

    # For small sets, run synchronously
    if len(tickers) <= 10:
        signals = []
        errors = []

        for window in windows:
            for ticker in tickers:
                try:
                    signal = await service.get_signal(
                        ticker=ticker,
                        benchmark=benchmark,
                        window=window,
                        force_refresh=True,
                    )
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)}")

        alerts = [s for s in signals if s.should_alert]

        return DipFinderRunResponse(
            status="completed",
            message=f"Processed {len(tickers)} tickers across {len(windows)} windows",
            tickers_processed=len(tickers),
            signals_generated=len(signals),
            alerts_triggered=len(alerts),
            errors=errors[:10],  # Limit errors in response
        )

    # For larger sets, queue for background processing
    async def _run_in_background():
        for window in windows:
            await service.get_signals(tickers, benchmark, window, force_refresh=True)

    background_tasks.add_task(_run_in_background)

    return DipFinderRunResponse(
        status="started",
        message=f"Background computation started for {len(tickers)} tickers",
        tickers_processed=len(tickers),
    )


@router.get(
    "/latest",
    response_model=DipSignalListResponse,
    summary="Get latest computed signals",
    description="Retrieve latest signals from the database.",
)
async def get_latest_signals(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    min_score: Optional[float] = Query(
        default=None, ge=0, le=100, description="Minimum final score"
    ),
    only_alerts: bool = Query(default=False, description="Only return alerts"),
) -> DipSignalListResponse:
    """Get latest computed signals from database."""
    service = get_dipfinder_service()
    config = get_dipfinder_config()

    signals = await service.get_latest_signals(
        limit=limit,
        min_final_score=min_score,
        only_alerts=only_alerts,
    )

    # Get dip_state data for ATH-based values
    tickers = [s.ticker for s in signals if s]
    dip_state_map = await _get_dip_state_map(tickers)

    return DipSignalListResponse(
        signals=[_signal_to_response(s, dip_state=dip_state_map.get(s.ticker)) for s in signals if s],
        count=len(signals),
        benchmark=config.default_benchmark,
        window=config.windows[1],
        as_of_date=date.today().isoformat(),
    )


@router.get(
    "/alerts",
    response_model=DipSignalListResponse,
    summary="Get active alerts",
    description="Get signals that meet alert criteria.",
)
async def get_alerts(
    limit: int = Query(default=20, ge=1, le=50),
) -> DipSignalListResponse:
    """Get active dip alerts."""
    service = get_dipfinder_service()
    config = get_dipfinder_config()

    signals = await service.get_latest_signals(
        limit=limit,
        only_alerts=True,
    )

    # Get dip_state data for ATH-based values
    tickers = [s.ticker for s in signals if s]
    dip_state_map = await _get_dip_state_map(tickers)

    return DipSignalListResponse(
        signals=[_signal_to_response(s, dip_state=dip_state_map.get(s.ticker)) for s in signals if s],
        count=len(signals),
        benchmark=config.default_benchmark,
        window=config.windows[1],
        as_of_date=date.today().isoformat(),
    )


@router.get(
    "/history/{ticker}",
    response_model=DipHistoryResponse,
    summary="Get dip history for a ticker",
    description="Get history of dip events for a ticker.",
)
async def get_ticker_history(
    ticker: str = Path(..., min_length=1, max_length=10),
    days: int = Query(default=90, ge=7, le=365),
) -> DipHistoryResponse:
    """Get dip history for a ticker."""
    service = get_dipfinder_service()

    history = await service.get_dip_history(ticker.upper(), days)

    entries = [
        DipHistoryEntry(
            id=h["id"],
            ticker=h["ticker"],
            event_type=h["event_type"],
            window_days=h["window_days"],
            dip_pct=float(h["dip_pct"]) if h.get("dip_pct") else None,
            final_score=float(h["final_score"]) if h.get("final_score") else None,
            dip_class=h.get("dip_class"),
            recorded_at=str(h["recorded_at"]),
        )
        for h in history
    ]

    return DipHistoryResponse(
        ticker=ticker.upper(),
        history=entries,
        count=len(entries),
    )


@router.get(
    "/config",
    response_model=DipFinderConfigResponse,
    summary="Get DipFinder configuration",
    description="Get current thresholds and settings.",
)
async def get_config() -> DipFinderConfigResponse:
    """Get current DipFinder configuration."""
    config = get_dipfinder_config()

    return DipFinderConfigResponse(
        windows=config.windows,
        min_dip_abs=config.min_dip_abs,
        min_persist_days=config.min_persist_days,
        dip_percentile_threshold=config.dip_percentile_threshold,
        dip_vs_typical_threshold=config.dip_vs_typical_threshold,
        market_dip_threshold=config.market_dip_threshold,
        excess_dip_stock_specific=config.excess_dip_stock_specific,
        excess_dip_market=config.excess_dip_market,
        quality_gate=config.quality_gate,
        stability_gate=config.stability_gate,
        alert_good=config.alert_good,
        alert_strong=config.alert_strong,
        weight_dip=config.weight_dip,
        weight_quality=config.weight_quality,
        weight_stability=config.weight_stability,
        default_benchmark=config.default_benchmark,
    )


# === Admin Endpoints ===


@router.post(
    "/admin/refresh-all",
    response_model=DipFinderRunResponse,
    summary="Refresh all signals (Admin)",
    description="Force refresh signals for all tracked symbols.",
)
async def admin_refresh_all(
    background_tasks: BackgroundTasks,
    benchmark: Optional[str] = Query(default=None),
    admin: TokenData = Depends(require_admin),
) -> DipFinderRunResponse:
    """Admin endpoint to refresh all tracked symbols."""
    from app.database.connection import fetch_all

    config = get_dipfinder_config()
    service = get_dipfinder_service()

    # Get all tracked symbols
    rows = await fetch_all("SELECT symbol FROM symbols ORDER BY symbol")
    tickers = [r["symbol"] for r in rows]

    if not tickers:
        return DipFinderRunResponse(
            status="completed",
            message="No symbols to refresh",
        )

    benchmark = benchmark or config.default_benchmark

    async def _refresh_all():
        for window in config.windows:
            await service.get_signals(tickers, benchmark, window, force_refresh=True)

    background_tasks.add_task(_refresh_all)

    return DipFinderRunResponse(
        status="started",
        message=f"Refreshing {len(tickers)} symbols across {len(config.windows)} windows",
        tickers_processed=len(tickers),
    )


@router.post(
    "/admin/cleanup",
    summary="Cleanup expired signals (Admin)",
    description="Remove expired signals and cache entries.",
)
async def admin_cleanup(
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Admin endpoint to cleanup expired data."""
    from app.database.connection import execute

    # Delete expired signals
    await execute("DELETE FROM dipfinder_signals WHERE expires_at < NOW()")

    # Delete old yfinance cache
    await execute("DELETE FROM yfinance_info_cache WHERE expires_at < NOW()")

    # Delete old price history (keep 2 years)
    await execute(
        "DELETE FROM price_history WHERE date < CURRENT_DATE - INTERVAL '730 days'"
    )

    return {
        "status": "completed",
        "message": "Cleanup completed",
    }
