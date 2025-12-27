"""Symbol CRUD routes - PostgreSQL async."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, status
from pydantic import BaseModel

from app.api.dependencies import require_admin, require_user
from app.cache.cache import Cache
from app.celery_app import celery_app
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.repositories import symbols_orm as symbol_repo
from app.schemas.symbols import SymbolCreate, SymbolResponse, SymbolUpdate
from app.services.stock_info import get_stock_info


logger = get_logger("api.routes.symbols")

router = APIRouter()

# Cache for symbol validations - 30 days TTL
_validation_cache = Cache(prefix="validation", default_ttl=30 * 24 * 60 * 60)  # 30 days


class SymbolValidationResponse(BaseModel):
    """Response for symbol validation."""

    valid: bool
    symbol: str
    name: str | None = None
    sector: str | None = None
    summary: str | None = None
    error: str | None = None


def _validate_symbol_path(symbol: str = Path(..., min_length=1, max_length=10)) -> str:
    """Validate and normalize symbol from path parameter."""
    return symbol.strip().upper()


@router.get(
    "/validate/{symbol}",
    response_model=SymbolValidationResponse,
    summary="Validate a stock symbol",
    description="Check if a stock symbol exists on Yahoo Finance without requiring it to be in the database. Results are cached for 30 days.",
)
async def validate_symbol(
    symbol: str = Path(
        ..., description="Stock symbol to validate", min_length=1, max_length=10
    ),
) -> SymbolValidationResponse:
    """Validate a stock symbol against Yahoo Finance (cached for 30 days)."""
    symbol_upper = symbol.upper().strip()
    cache_key = f"symbol:{symbol_upper}"

    # Try to get from cache first
    cached_result = await _validation_cache.get(cache_key)
    if cached_result is not None:
        return SymbolValidationResponse(**cached_result)

    # Check local symbols table before hitting external API
    local_symbol = await symbol_repo.get_symbol(symbol_upper)
    if local_symbol:
        result = SymbolValidationResponse(
            valid=True,
            symbol=symbol_upper,
            name=local_symbol.name,
            sector=local_symbol.sector,
            summary=local_symbol.summary_ai,
        )
        await _validation_cache.set(cache_key, result.model_dump(), ttl=30 * 24 * 60 * 60)
        return result

    # Not in cache, validate against Yahoo Finance
    try:
        # get_stock_info is async
        info = await get_stock_info(symbol_upper)
        if info and info.name:
            result = SymbolValidationResponse(
                valid=True,
                symbol=symbol_upper,
                name=info.name,
                sector=info.sector,
                summary=info.summary,
            )
        else:
            result = SymbolValidationResponse(
                valid=False,
                symbol=symbol_upper,
                error="Symbol not found on Yahoo Finance",
            )
    except Exception as e:
        result = SymbolValidationResponse(
            valid=False,
            symbol=symbol_upper,
            error=str(e) if str(e) else "Failed to validate symbol",
        )

    # Cache the result (valid results for 30 days, invalid for 1 day)
    ttl = 30 * 24 * 60 * 60 if result.valid else 24 * 60 * 60
    await _validation_cache.set(cache_key, result.model_dump(), ttl=ttl)

    return result


class SymbolSearchResult(BaseModel):
    """Result item from symbol search."""
    symbol: str
    name: str | None = None
    sector: str | None = None
    quote_type: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    source: str | None = None  # "local", "cached", or "api"
    score: float | None = None  # Relevance score (0-1)


class SymbolSearchResponse(BaseModel):
    """Response for symbol search."""
    query: str
    results: list[SymbolSearchResult]
    count: int
    suggest_fresh_search: bool = False
    search_type: str = "local"  # "local" or "api"
    next_cursor: str | None = None  # Cursor for pagination


@router.get(
    "/search/{query}",
    response_model=SymbolSearchResponse,
    summary="Search for symbols",
    description="Search for stock symbols by name or ticker. Local-first with option for fresh API search.",
)
async def search_symbols(
    query: str = Path(..., min_length=2, max_length=50, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    fresh: bool = Query(False, description="Force fresh search from yfinance API"),
    cursor: str | None = Query(None, description="Pagination cursor from previous response"),
) -> SymbolSearchResponse:
    """
    Search for stock symbols.
    
    LOCAL-FIRST STRATEGY:
    1. Search local DB first (symbols + cached results) - instant
    2. Use trigram similarity for fuzzy name matching
    3. If not enough results, suggest_fresh_search=True in response
    4. Client can re-request with fresh=True to query API
    
    PAGINATION:
    - Pass next_cursor from previous response to get next page
    - Results are sorted by relevance score descending
    
    When fresh=True:
    - Queries yfinance API directly
    - Saves ALL results to DB for future local searches
    - Pagination not supported for fresh API search
    """
    from app.services.symbol_search import search_symbols as do_search

    result = await do_search(
        query,
        max_results=limit,
        force_api=fresh,
        cursor=cursor,
    )

    return SymbolSearchResponse(
        query=query,
        results=[SymbolSearchResult(**r) for r in result["results"]],
        count=result["count"],
        suggest_fresh_search=result["suggest_fresh_search"],
        search_type=result["search_type"],
        next_cursor=result.get("next_cursor"),
    )


class SymbolAutocompleteResult(BaseModel):
    """Simple result for autocomplete."""
    symbol: str
    name: str | None = None


@router.get(
    "/autocomplete/{partial}",
    response_model=list[SymbolAutocompleteResult],
    summary="Autocomplete symbols",
    description="Get autocomplete suggestions for partial symbol/name input. Optimized for speed.",
)
async def autocomplete_symbols(
    partial: str = Path(..., min_length=1, max_length=20, description="Partial input"),
    limit: int = Query(5, ge=1, le=10, description="Maximum suggestions"),
) -> list[SymbolAutocompleteResult]:
    """Get autocomplete suggestions for symbol input."""
    from app.services.symbol_search import get_symbol_suggestions

    results = await get_symbol_suggestions(partial, limit=limit)
    return [SymbolAutocompleteResult(**r) for r in results]


class SymbolFundamentalsResponse(BaseModel):
    """Fundamentals data for a symbol."""
    symbol: str
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    profit_margin: str | None = None
    gross_margin: str | None = None
    return_on_equity: str | None = None
    debt_to_equity: str | None = None
    current_ratio: str | None = None
    revenue_growth: str | None = None
    earnings_growth: str | None = None
    free_cash_flow: str | None = None
    recommendation: str | None = None
    target_mean_price: float | None = None
    num_analyst_opinions: int | None = None
    beta: str | None = None
    next_earnings_date: str | None = None
    source: str = "database"  # "database" or "api"
    fetched_at: str | None = None
    expires_at: str | None = None
    is_stale: bool = False
    refresh_task_id: str | None = None


@router.get(
    "/fundamentals/{symbol}",
    response_model=SymbolFundamentalsResponse,
    summary="Get symbol fundamentals",
    description="Get fundamental financial data for a symbol. Uses cached data from database when available, otherwise fetches from yfinance.",
)
async def get_symbol_fundamentals(
    symbol: str = Path(..., min_length=1, max_length=10, description="Stock symbol"),
) -> SymbolFundamentalsResponse:
    """Get fundamental data for a symbol."""
    from app.services.fundamentals import (
        fetch_fundamentals_live,
        get_fundamentals_with_status,
    )

    symbol_upper = symbol.upper().strip()
    source = "database"

    # Try database first (for tracked symbols)
    db_data, is_stale = await get_fundamentals_with_status(
        symbol_upper, allow_stale=True
    )
    refresh_task_id = None

    if db_data:
        if is_stale:
            task = celery_app.send_task(
                "jobs.refresh_fundamentals_symbol", args=[symbol_upper]
            )
            refresh_task_id = task.id

        # Format the database data
        def fmt_pct(val):
            if val is None:
                return None
            return f"{float(val) * 100:.1f}%"

        def fmt_ratio(val):
            if val is None:
                return None
            return f"{float(val):.2f}"

        def fmt_large_num(val):
            if val is None:
                return None
            val = int(val)
            if val >= 1e12:
                return f"${val / 1e12:.1f}T"
            if val >= 1e9:
                return f"${val / 1e9:.1f}B"
            if val >= 1e6:
                return f"${val / 1e6:.1f}M"
            return f"${val:,.0f}"

        data = {
            "pe_ratio": db_data.get("pe_ratio"),
            "forward_pe": db_data.get("forward_pe"),
            "peg_ratio": fmt_ratio(db_data.get("peg_ratio")),
            "price_to_book": fmt_ratio(db_data.get("price_to_book")),
            "profit_margin": fmt_pct(db_data.get("profit_margin")),
            "gross_margin": fmt_pct(db_data.get("gross_margin")),
            "return_on_equity": fmt_pct(db_data.get("return_on_equity")),
            "debt_to_equity": fmt_ratio(db_data.get("debt_to_equity")),
            "current_ratio": fmt_ratio(db_data.get("current_ratio")),
            "revenue_growth": fmt_pct(db_data.get("revenue_growth")),
            "earnings_growth": fmt_pct(db_data.get("earnings_growth")),
            "free_cash_flow": fmt_large_num(db_data.get("free_cash_flow")),
            "recommendation": db_data.get("recommendation"),
            "target_mean_price": db_data.get("target_mean_price"),
            "num_analyst_opinions": db_data.get("num_analyst_opinions"),
            "beta": fmt_ratio(db_data.get("beta")),
            "next_earnings_date": str(db_data.get("next_earnings_date")) if db_data.get("next_earnings_date") else None,
        }
    else:
        # Not in database, fetch live from yfinance (without storing)
        data = await fetch_fundamentals_live(symbol_upper)
        source = "api"
        is_stale = False

    if not data:
        raise NotFoundError(
            message="Fundamentals not available",
            details={"symbol": symbol_upper, "reason": "Symbol may be an ETF/index or not found"}
        )

    return SymbolFundamentalsResponse(
        symbol=symbol_upper,
        pe_ratio=data.get("pe_ratio"),
        forward_pe=data.get("forward_pe"),
        peg_ratio=float(data["peg_ratio"]) if data.get("peg_ratio") else None,
        price_to_book=float(data["price_to_book"]) if data.get("price_to_book") else None,
        profit_margin=data.get("profit_margin"),
        gross_margin=data.get("gross_margin"),
        return_on_equity=data.get("return_on_equity"),
        debt_to_equity=data.get("debt_to_equity"),
        current_ratio=data.get("current_ratio"),
        revenue_growth=data.get("revenue_growth"),
        earnings_growth=data.get("earnings_growth"),
        free_cash_flow=data.get("free_cash_flow"),
        recommendation=data.get("recommendation"),
        target_mean_price=data.get("target_mean_price"),
        num_analyst_opinions=data.get("num_analyst_opinions"),
        beta=data.get("beta"),
        next_earnings_date=data.get("next_earnings_date"),
        source=source,
        fetched_at=db_data.get("fetched_at").isoformat() if db_data and db_data.get("fetched_at") else None,
        expires_at=db_data.get("expires_at").isoformat() if db_data and db_data.get("expires_at") else None,
        is_stale=is_stale,
        refresh_task_id=refresh_task_id,
    )


@router.get(
    "",
    response_model=list[SymbolResponse],
    summary="List all symbols",
    description="Get all tracked stock symbols. Public endpoint.",
)
async def list_symbols() -> list[SymbolResponse]:
    """List all tracked symbols (public endpoint for signals page)."""
    symbols = await symbol_repo.list_symbols()
    import asyncio

    from app.services.task_tracking import get_symbol_task
    task_ids = await asyncio.gather(
        *(get_symbol_task(s.symbol) for s in symbols)
    )
    return [
        SymbolResponse(
            symbol=s.symbol,
            min_dip_pct=s.min_dip_pct,
            min_days=s.min_days,
            name=s.name,
            fetch_status=s.fetch_status,
            fetch_error=s.fetch_error,
            task_id=task_ids[index],
        )
        for index, s in enumerate(symbols)
    ]


@router.get(
    "/{symbol}",
    response_model=SymbolResponse,
    summary="Get symbol",
    description="Get a specific symbol's configuration.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def get_symbol(
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
) -> SymbolResponse:
    """Get a specific symbol's configuration."""
    config = await symbol_repo.get_symbol(symbol)
    if config is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )
    from app.services.task_tracking import get_symbol_task
    task_id = await get_symbol_task(symbol)
    return SymbolResponse(
        symbol=config.symbol,
        min_dip_pct=config.min_dip_pct,
        min_days=config.min_days,
        name=config.name,
        fetch_status=config.fetch_status,
        fetch_error=config.fetch_error,
        task_id=task_id,
    )


@router.post(
    "",
    response_model=SymbolResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create symbol",
    description="Add a new stock symbol to track.",
    responses={
        409: {"description": "Symbol already exists"},
    },
)
async def create_symbol(
    payload: SymbolCreate,
    user: TokenData = Depends(require_user),
) -> SymbolResponse:
    """
    Create a new tracked symbol.

    Symbol is queued for batch processing (runs every 10 minutes).
    Batch processing fetches data and generates AI content more efficiently.
    """
    # Check if already exists
    existing = await symbol_repo.get_symbol(payload.symbol)
    if existing is not None:
        raise ConflictError(
            message=f"Symbol '{payload.symbol}' already exists",
            details={"symbol": payload.symbol},
        )

    # Create symbol with pending status - will be picked up by batch job
    created = await symbol_repo.upsert_symbol(
        payload.symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

    # Mark as pending for batch processing (runs every 10 minutes)
    await symbol_repo.update_fetch_status(
        payload.symbol,
        fetch_status="pending",
        fetch_error=None,
    )

    return SymbolResponse(
        symbol=created.symbol,
        min_dip_pct=created.min_dip_pct,
        min_days=created.min_days,
        name=created.name,
        fetch_status='pending',  # Will be processed by batch job every 10 min
        fetch_error=None,
        task_id=None,  # No immediate task - processed in batch
    )


class ProcessSymbolsRequest(BaseModel):
    """Request to process symbols immediately."""

    symbols: list[str] | None = None  # If None, process all pending


class ProcessSymbolsResponse(BaseModel):
    """Response for immediate symbol processing."""

    processed: int
    failed: int
    message: str
    symbols: list[str]


@router.post(
    "/admin/process-now",
    response_model=ProcessSymbolsResponse,
    summary="Process symbols immediately (Admin)",
    description="Trigger immediate processing of pending symbols instead of waiting for batch job. Admin only.",
    dependencies=[Depends(require_admin)],
)
async def process_symbols_now(
    request: ProcessSymbolsRequest | None = None,
) -> ProcessSymbolsResponse:
    """
    Admin endpoint to process symbols immediately.
    
    If symbols list is provided, process those specific symbols.
    If symbols list is empty/None, process all pending symbols.
    
    This bypasses the 10-minute batch schedule for urgent processing.
    """
    from app.jobs.definitions import process_new_symbols_batch_job
    from app.repositories import symbols_orm

    # Get symbols to process
    if request and request.symbols:
        # Process specific symbols - mark them as pending first
        symbols_to_process = [s.upper() for s in request.symbols]
        for sym in symbols_to_process:
            existing = await symbols_orm.get_symbol(sym)
            if existing:
                await symbols_orm.update_fetch_status(sym, fetch_status="pending")
            else:
                # Create if doesn't exist
                await symbols_orm.upsert_symbol(sym, min_dip_pct=20.0, min_days=7)
                await symbols_orm.update_fetch_status(sym, fetch_status="pending")
    else:
        # Get all currently pending symbols
        pending = await symbols_orm.get_symbols_by_status("pending")
        symbols_to_process = [s.symbol for s in pending]

    if not symbols_to_process:
        return ProcessSymbolsResponse(
            processed=0,
            failed=0,
            message="No symbols to process",
            symbols=[],
        )

    logger.info(f"Admin triggered immediate processing of {len(symbols_to_process)} symbols: {symbols_to_process}")

    # Run the batch job immediately
    result = await process_new_symbols_batch_job()

    # Parse result to get counts
    # Result format: "Processed X/Y symbols (Z failed)"
    import re
    match = re.search(r"Processed (\d+)/(\d+) symbols \((\d+) failed\)", result)
    if match:
        processed = int(match.group(1))
        failed = int(match.group(3))
    else:
        processed = len(symbols_to_process)
        failed = 0

    return ProcessSymbolsResponse(
        processed=processed,
        failed=failed,
        message=result,
        symbols=symbols_to_process,
    )


@router.put(
    "/{symbol}",
    response_model=SymbolResponse,
    summary="Update symbol",
    description="Update a symbol's configuration.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def update_symbol(
    payload: SymbolUpdate,
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
) -> SymbolResponse:
    """
    Update a symbol's configuration.
    """
    # Check if exists
    existing = await symbol_repo.get_symbol(symbol)
    if existing is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Update symbol
    updated = await symbol_repo.upsert_symbol(
        symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

    return SymbolResponse(
        symbol=updated.symbol,
        min_dip_pct=updated.min_dip_pct,
        min_days=updated.min_days,
        name=updated.name,
        fetch_status=updated.fetch_status,
        fetch_error=updated.fetch_error,
    )


@router.delete(
    "/{symbol}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete symbol",
    description="Remove a symbol from tracking.",
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def delete_symbol(
    symbol: str = Depends(_validate_symbol_path),
    user: TokenData = Depends(require_user),
) -> None:
    """Delete a tracked symbol and its state."""
    # Check if exists
    existing = await symbol_repo.get_symbol(symbol)
    if existing is None:
        raise NotFoundError(
            message=f"Symbol '{symbol}' not found",
            details={"symbol": symbol},
        )

    # Delete symbol (cascade will delete dip_state)
    await symbol_repo.delete_symbol(symbol)


# =============================================================================
# AI Summary Management
# =============================================================================

class AISummaryResponse(BaseModel):
    """Response for AI summary regeneration."""
    symbol: str
    summary_ai: str | None
    task_id: str | None = None
    status: str | None = None


@router.post(
    "/{symbol}/ai/summary",
    response_model=AISummaryResponse,
    summary="Regenerate AI summary",
    description="Queue regeneration of the AI-generated summary for a symbol. Requires authentication.",
    dependencies=[Depends(require_user)],
)
async def regenerate_ai_summary(
    symbol: str = Depends(_validate_symbol_path),
) -> AISummaryResponse:
    """Queue regeneration of the AI summary for a symbol."""
    # Verify symbol exists
    sym = await symbol_repo.get_symbol(symbol)
    if not sym:
        raise NotFoundError(f"Symbol {symbol} not found")

    task = celery_app.send_task(
        "jobs.regenerate_symbol_summary", args=[symbol.upper()]
    )

    return AISummaryResponse(
        symbol=symbol,
        summary_ai=sym.summary_ai,
        task_id=task.id,
        status="queued",
    )


# =============================================================================
# AI AGENT ANALYSIS ENDPOINTS
# =============================================================================


class AgentVerdictResponse(BaseModel):
    """Individual agent verdict."""
    agent_id: str
    agent_name: str
    signal: str
    confidence: int
    reasoning: str
    key_factors: list[str]


class AgentAnalysisResponse(BaseModel):
    """Complete agent analysis for a stock."""
    symbol: str
    verdicts: list[AgentVerdictResponse]
    overall_signal: str
    overall_confidence: int
    summary: str
    analyzed_at: str | None = None
    expires_at: str | None = None


class AgentInfoResponse(BaseModel):
    """Information about an available agent."""
    id: str
    name: str
    philosophy: str
    focus: list[str]


@router.get(
    "/{symbol}/agents",
    response_model=AgentAnalysisResponse,
    summary="Get AI agent analysis",
    description="Get analysis from AI investor personas (Warren Buffett, Peter Lynch, etc.)",
)
async def get_agent_analysis_endpoint(
    symbol: str = Depends(_validate_symbol_path),
    force_refresh: bool = Query(False, description="Force new analysis even if cached"),
) -> AgentAnalysisResponse:
    """Get AI agent analysis for a symbol."""
    from app.services.ai_agents import get_agent_analysis, run_agent_analysis

    # Verify symbol exists
    sym = await symbol_repo.get_symbol(symbol)
    if not sym:
        raise NotFoundError(f"Symbol {symbol} not found")

    # Get existing analysis
    if not force_refresh:
        analysis = await get_agent_analysis(symbol)
        if analysis:
            return AgentAnalysisResponse(**analysis)

    # Run new analysis
    result = await run_agent_analysis(symbol)
    if not result:
        raise NotFoundError(f"Could not generate agent analysis for {symbol}")

    return AgentAnalysisResponse(
        symbol=result.symbol,
        verdicts=[
            AgentVerdictResponse(
                agent_id=v.agent_id,
                agent_name=v.agent_name,
                signal=v.signal,
                confidence=v.confidence,
                reasoning=v.reasoning,
                key_factors=v.key_factors,
            )
            for v in result.verdicts
        ],
        overall_signal=result.overall_signal,
        overall_confidence=result.overall_confidence,
        summary=result.summary,
        analyzed_at=result.analyzed_at.isoformat() if result.analyzed_at else None,
    )


@router.post(
    "/{symbol}/agents/refresh",
    response_model=AgentAnalysisResponse,
    summary="Refresh AI agent analysis",
    description="Force regenerate agent analysis for a symbol. Requires authentication.",
    dependencies=[Depends(require_user)],
)
async def refresh_agent_analysis_endpoint(
    symbol: str = Depends(_validate_symbol_path),
) -> AgentAnalysisResponse:
    """Regenerate AI agent analysis for a symbol."""
    from app.services.ai_agents import run_agent_analysis

    # Verify symbol exists
    sym = await symbol_repo.get_symbol(symbol)
    if not sym:
        raise NotFoundError(f"Symbol {symbol} not found")

    # Run new analysis
    result = await run_agent_analysis(symbol)
    if not result:
        raise NotFoundError(f"Could not generate agent analysis for {symbol}")

    logger.info(f"Refreshed agent analysis for {symbol}: {result.overall_signal} ({result.overall_confidence}%)")

    return AgentAnalysisResponse(
        symbol=result.symbol,
        verdicts=[
            AgentVerdictResponse(
                agent_id=v.agent_id,
                agent_name=v.agent_name,
                signal=v.signal,
                confidence=v.confidence,
                reasoning=v.reasoning,
                key_factors=v.key_factors,
            )
            for v in result.verdicts
        ],
        overall_signal=result.overall_signal,
        overall_confidence=result.overall_confidence,
        summary=result.summary,
        analyzed_at=result.analyzed_at.isoformat() if result.analyzed_at else None,
    )


@router.get(
    "/agents/info",
    response_model=list[AgentInfoResponse],
    summary="List available AI agents",
    description="Get information about all available AI investor persona agents.",
)
async def list_agents() -> list[AgentInfoResponse]:
    """List available AI agents and their investment philosophies."""
    from app.services.ai_agents import get_agent_info

    agents = get_agent_info()
    return [AgentInfoResponse(**a) for a in agents]
