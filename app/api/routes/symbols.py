"""Symbol CRUD routes - PostgreSQL async."""

from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, Path, Query, status
from pydantic import BaseModel
from sqlalchemy import func, or_, select

from app.api.dependencies import require_admin, require_user
from app.cache.cache import Cache
from app.celery_app import celery_app
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.database import AIPersona, Symbol, get_session
from app.repositories import symbols_orm as symbol_repo
from app.schemas.symbols import SymbolCreate, SymbolListResponse, SymbolResponse, SymbolUpdate
from app.services.stock_info import get_stock_info


logger = get_logger("api.routes.symbols")


# Cache for personas with avatars - 5 minute TTL
_avatar_cache: dict[str, float] = {}  # persona_key -> expiry timestamp
_avatar_set: set[str] = set()  # Set of persona keys that have avatars
_avatar_cache_ttl = 300  # 5 minutes


async def get_personas_with_avatars() -> set[str]:
    """Get set of persona keys that have avatars. Cached for 5 minutes."""
    import time
    
    global _avatar_set, _avatar_cache
    
    cache_key = "personas_with_avatars"
    now = time.time()
    
    if cache_key in _avatar_cache and _avatar_cache[cache_key] > now:
        return _avatar_set
    
    # Query for personas with avatars
    async with get_session() as session:
        result = await session.execute(
            select(AIPersona.key).where(AIPersona.avatar_data.isnot(None))
        )
        _avatar_set = {row[0] for row in result.fetchall()}
    
    _avatar_cache[cache_key] = now + _avatar_cache_ttl
    return _avatar_set


def get_avatar_url(persona_key: str, personas_with_avatars: set[str]) -> str | None:
    """Get avatar URL for a persona, or None if no avatar exists."""
    if persona_key in personas_with_avatars:
        return f"/api/ai-personas/{persona_key}/avatar"
    return None

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


def calculate_intrinsic_value(
    data: dict,
    current_price: float | None = None,
) -> dict:
    """Calculate intrinsic value using multiple methods.
    
    Methods (in order of preference):
    1. Analyst target price (if available with enough analysts)
    2. PEG-based fair value (if PEG ratio available)
    3. Graham Number (if P/E and Book Value available)
    
    Returns dict with: intrinsic_value, method, upside_pct, valuation_status
    """
    import math
    
    result = {
        "intrinsic_value": None,
        "intrinsic_value_method": None,
        "upside_pct": None,
        "valuation_status": None,
    }
    
    if not current_price or current_price <= 0:
        return result
    
    # Method 1: Analyst Target Price (most reliable if enough coverage)
    target_price = data.get("target_mean_price")
    num_analysts = data.get("num_analyst_opinions", 0)
    if target_price and num_analysts and num_analysts >= 5:
        result["intrinsic_value"] = round(target_price, 2)
        result["intrinsic_value_method"] = "analyst"
        upside = ((target_price / current_price) - 1) * 100
        result["upside_pct"] = round(upside, 1)
        if upside > 15:
            result["valuation_status"] = "undervalued"
        elif upside < -15:
            result["valuation_status"] = "overvalued"
        else:
            result["valuation_status"] = "fair"
        return result
    
    # Method 2: PEG-based fair value
    # Fair P/E = Growth Rate (for PEG = 1), so Fair Price = EPS * Growth Rate
    peg_ratio = data.get("peg_ratio")
    pe_ratio = data.get("pe_ratio")
    if peg_ratio and pe_ratio and peg_ratio > 0:
        # Extract growth rate from PEG: PEG = PE / Growth, so Growth = PE / PEG
        try:
            peg_val = float(peg_ratio) if isinstance(peg_ratio, str) else peg_ratio
            if peg_val > 0 and pe_ratio > 0:
                growth_rate = pe_ratio / peg_val
                # Fair P/E at PEG = 1 is the growth rate
                fair_pe = growth_rate
                # Current EPS = Current Price / P/E
                eps = current_price / pe_ratio
                fair_value = eps * fair_pe
                if fair_value > 0:
                    result["intrinsic_value"] = round(fair_value, 2)
                    result["intrinsic_value_method"] = "peg"
                    upside = ((fair_value / current_price) - 1) * 100
                    result["upside_pct"] = round(upside, 1)
                    if upside > 20:
                        result["valuation_status"] = "undervalued"
                    elif upside < -20:
                        result["valuation_status"] = "overvalued"
                    else:
                        result["valuation_status"] = "fair"
                    return result
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    # Method 3: Graham Number (simplified intrinsic value)
    # Graham Number = sqrt(22.5 * EPS * BVPS)
    price_to_book = data.get("price_to_book")
    if pe_ratio and price_to_book and pe_ratio > 0 and price_to_book > 0:
        try:
            ptb_val = float(price_to_book) if isinstance(price_to_book, str) else price_to_book
            if ptb_val > 0:
                eps = current_price / pe_ratio
                bvps = current_price / ptb_val
                # Graham Number = sqrt(22.5 * EPS * BVPS)
                if eps > 0 and bvps > 0:
                    graham = math.sqrt(22.5 * eps * bvps)
                    result["intrinsic_value"] = round(graham, 2)
                    result["intrinsic_value_method"] = "graham"
                    upside = ((graham / current_price) - 1) * 100
                    result["upside_pct"] = round(upside, 1)
                    if upside > 25:
                        result["valuation_status"] = "undervalued"
                    elif upside < -25:
                        result["valuation_status"] = "overvalued"
                    else:
                        result["valuation_status"] = "fair"
                    return result
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    return result


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
    # Intrinsic Value Estimates
    intrinsic_value: float | None = None  # Calculated fair value per share
    intrinsic_value_method: str | None = None  # Method used: 'graham', 'peg', 'dcf', 'analyst'
    upside_pct: float | None = None  # Percentage upside to intrinsic value
    valuation_status: str | None = None  # 'undervalued', 'fair', 'overvalued'
    # Domain detection
    domain: str | None = None  # bank, reit, insurer, utility, biotech, stock
    # Domain-specific metrics (Banks)
    net_interest_income: int | None = None
    net_interest_margin: float | None = None
    interest_income: int | None = None
    interest_expense: int | None = None
    # Domain-specific metrics (REITs)
    ffo: int | None = None
    ffo_per_share: float | None = None
    p_ffo: float | None = None
    # Domain-specific metrics (Insurance)
    loss_ratio: float | None = None
    expense_ratio: float | None = None
    combined_ratio: float | None = None
    # Metadata
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
    from app.repositories.dip_state_orm import get_dip_state

    symbol_upper = symbol.upper().strip()
    source = "database"

    # Get current price from dip state for intrinsic value calculation
    dip_state = await get_dip_state(symbol_upper)
    current_price = float(dip_state.current_price) if dip_state and dip_state.current_price else None

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
            # Domain detection
            "domain": db_data.get("domain"),
            # Domain-specific metrics (Banks)
            "net_interest_income": db_data.get("net_interest_income"),
            "net_interest_margin": db_data.get("net_interest_margin"),
            "interest_income": db_data.get("interest_income"),
            "interest_expense": db_data.get("interest_expense"),
            # Domain-specific metrics (REITs)
            "ffo": db_data.get("ffo"),
            "ffo_per_share": db_data.get("ffo_per_share"),
            "p_ffo": db_data.get("p_ffo"),
            # Domain-specific metrics (Insurance)
            "loss_ratio": db_data.get("loss_ratio"),
            "expense_ratio": db_data.get("expense_ratio"),
            "combined_ratio": db_data.get("combined_ratio"),
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

    # Calculate intrinsic value
    # Need raw values (not formatted strings) for calculation
    iv_data = {
        "pe_ratio": db_data.get("pe_ratio") if db_data else data.get("pe_ratio"),
        "peg_ratio": db_data.get("peg_ratio") if db_data else data.get("peg_ratio"),
        "price_to_book": db_data.get("price_to_book") if db_data else data.get("price_to_book"),
        "target_mean_price": db_data.get("target_mean_price") if db_data else data.get("target_mean_price"),
        "num_analyst_opinions": db_data.get("num_analyst_opinions") if db_data else data.get("num_analyst_opinions"),
    }
    intrinsic = calculate_intrinsic_value(iv_data, current_price)

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
        # Domain detection
        domain=data.get("domain"),
        # Domain-specific metrics (Banks)
        net_interest_income=data.get("net_interest_income"),
        net_interest_margin=data.get("net_interest_margin"),
        interest_income=data.get("interest_income"),
        interest_expense=data.get("interest_expense"),
        # Domain-specific metrics (REITs)
        ffo=data.get("ffo"),
        ffo_per_share=data.get("ffo_per_share"),
        p_ffo=data.get("p_ffo"),
        # Domain-specific metrics (Insurance)
        loss_ratio=data.get("loss_ratio"),
        expense_ratio=data.get("expense_ratio"),
        combined_ratio=data.get("combined_ratio"),
        # Intrinsic value estimates
        intrinsic_value=intrinsic.get("intrinsic_value"),
        intrinsic_value_method=intrinsic.get("intrinsic_value_method"),
        upside_pct=intrinsic.get("upside_pct"),
        valuation_status=intrinsic.get("valuation_status"),
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
    "/paged",
    response_model=SymbolListResponse,
    summary="List symbols (paginated)",
    description="Get tracked symbols with pagination and search (admin-friendly).",
)
async def list_symbols_paged(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None, min_length=1, max_length=50),
) -> SymbolListResponse:
    """List tracked symbols with pagination for admin UI."""
    from app.database.orm import Symbol as SymbolORM

    async with get_session() as session:
        stmt = select(SymbolORM).where(SymbolORM.is_active == True)
        if search:
            term = f"%{search.strip()}%"
            stmt = stmt.where(
                or_(
                    SymbolORM.symbol.ilike(term),
                    SymbolORM.name.ilike(term),
                )
            )

        total_result = await session.execute(
            select(func.count()).select_from(stmt.subquery())
        )
        total = total_result.scalar() or 0

        result = await session.execute(
            stmt.order_by(SymbolORM.symbol).offset(offset).limit(limit)
        )
        symbols = result.scalars().all()

    from app.services.task_tracking import get_symbol_task

    task_ids = await asyncio.gather(
        *(get_symbol_task(s.symbol) for s in symbols)
    )

    items = [
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

    return SymbolListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


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
    avatar_url: str | None = None


def _compute_signal_counts(verdicts: list[dict | AgentVerdictResponse]) -> tuple[int, int, int]:
    """Compute bullish/bearish/neutral counts from verdicts."""
    bullish = 0
    bearish = 0
    neutral = 0
    for v in verdicts:
        signal = v.get("signal") if isinstance(v, dict) else v.signal
        if signal in ("strong_buy", "buy"):
            bullish += 1
        elif signal in ("strong_sell", "sell"):
            bearish += 1
        else:
            neutral += 1
    return bullish, bearish, neutral


class AgentAnalysisResponse(BaseModel):
    """Complete agent analysis for a stock."""
    symbol: str
    verdicts: list[AgentVerdictResponse]
    overall_signal: str
    overall_confidence: int
    summary: str
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
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

    # Get personas that have avatars
    personas_with_avatars = await get_personas_with_avatars()

    # Get existing analysis
    if not force_refresh:
        analysis = await get_agent_analysis(symbol)
        if analysis:
            # Add avatar URLs to verdicts (only if persona has avatar)
            analysis_resp = dict(analysis)
            if "verdicts" in analysis_resp:
                for v in analysis_resp["verdicts"]:
                    v["avatar_url"] = get_avatar_url(v.get("agent_id", ""), personas_with_avatars)
                # Compute signal counts from verdicts
                bullish, bearish, neutral = _compute_signal_counts(analysis_resp["verdicts"])
                analysis_resp["bullish_count"] = bullish
                analysis_resp["bearish_count"] = bearish
                analysis_resp["neutral_count"] = neutral
            return AgentAnalysisResponse(**analysis_resp)

    # Run new analysis
    result = await run_agent_analysis(symbol)
    if not result:
        raise NotFoundError(f"Could not generate agent analysis for {symbol}")

    # Build verdict responses
    verdict_responses = [
        AgentVerdictResponse(
            agent_id=v.agent_id,
            agent_name=v.agent_name,
            signal=v.signal,
            confidence=v.confidence,
            reasoning=v.reasoning,
            key_factors=v.key_factors,
            avatar_url=get_avatar_url(v.agent_id, personas_with_avatars),
        )
        for v in result.verdicts
    ]
    bullish, bearish, neutral = _compute_signal_counts(verdict_responses)

    return AgentAnalysisResponse(
        symbol=result.symbol,
        verdicts=verdict_responses,
        overall_signal=result.overall_signal,
        overall_confidence=result.overall_confidence,
        summary=result.summary,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
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

    # Get personas that have avatars
    personas_with_avatars = await get_personas_with_avatars()

    # Run new analysis
    result = await run_agent_analysis(symbol)
    if not result:
        raise NotFoundError(f"Could not generate agent analysis for {symbol}")

    logger.info(f"Refreshed agent analysis for {symbol}: {result.overall_signal} ({result.overall_confidence}%)")

    # Build verdict responses
    verdict_responses = [
        AgentVerdictResponse(
            agent_id=v.agent_id,
            agent_name=v.agent_name,
            signal=v.signal,
            confidence=v.confidence,
            reasoning=v.reasoning,
            key_factors=v.key_factors,
            avatar_url=get_avatar_url(v.agent_id, personas_with_avatars),
        )
        for v in result.verdicts
    ]
    bullish, bearish, neutral = _compute_signal_counts(verdict_responses)

    return AgentAnalysisResponse(
        symbol=result.symbol,
        verdicts=verdict_responses,
        overall_signal=result.overall_signal,
        overall_confidence=result.overall_confidence,
        summary=result.summary,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
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


# =============================================================================
# Admin Full Refresh Endpoint
# =============================================================================


class FullRefreshRequest(BaseModel):
    """Request for full symbol refresh."""
    
    symbol: str
    refresh_fundamentals: bool = True
    refresh_prices: bool = True
    refresh_logo: bool = True
    refresh_ai_analysis: bool = True
    refresh_ai_agents: bool = True


class FullRefreshResponse(BaseModel):
    """Response for full symbol refresh."""
    
    symbol: str
    tasks_triggered: list[str]
    message: str


@router.post(
    "/admin/refresh/{symbol}",
    response_model=FullRefreshResponse,
    summary="Full symbol data refresh (Admin)",
    description="Trigger complete refresh of all data for a symbol: fundamentals, prices, logo, AI analysis, and AI agents.",
    dependencies=[Depends(require_admin)],
)
async def refresh_symbol_full(
    symbol: str = Path(..., min_length=1, max_length=10, description="Stock symbol"),
    request: FullRefreshRequest | None = None,
) -> FullRefreshResponse:
    """
    Admin endpoint to trigger full data refresh for a symbol.
    
    This is the nuclear option - refreshes ALL data sources:
    - Fundamentals (yfinance company data)
    - Price history (yfinance price data)  
    - Company logo (Logo.dev or favicon)
    - AI analysis (swipe bio, rating)
    - AI agents (Warren Buffett, Peter Lynch, etc.)
    
    Use sparingly - triggers multiple API calls and AI generations.
    """
    from app.services.logo_service import LogoTheme, get_logo
    from app.services.ai_agents import run_agent_analysis
    
    symbol_upper = symbol.upper().strip()
    tasks_triggered = []
    
    # Check symbol exists
    existing = await symbol_repo.get_symbol(symbol_upper)
    if not existing:
        raise NotFoundError(
            message=f"Symbol '{symbol_upper}' not found in database",
            details={"symbol": symbol_upper},
        )
    
    # Use request options or defaults
    refresh_fundamentals = request.refresh_fundamentals if request else True
    refresh_prices = request.refresh_prices if request else True
    refresh_logo = request.refresh_logo if request else True
    refresh_ai_analysis = request.refresh_ai_analysis if request else True
    refresh_ai_agents = request.refresh_ai_agents if request else True
    
    # 1. Refresh fundamentals via Celery task
    if refresh_fundamentals:
        task = celery_app.send_task(
            "jobs.refresh_fundamentals_symbol", args=[symbol_upper]
        )
        tasks_triggered.append(f"fundamentals:{task.id}")
        logger.info(f"Triggered fundamentals refresh for {symbol_upper}: {task.id}")
    
    # 2. Mark symbol for price refresh (will be picked up by next data_grab)
    if refresh_prices:
        await symbol_repo.update_fetch_status(symbol_upper, fetch_status="pending")
        tasks_triggered.append("prices:pending")
        logger.info(f"Marked {symbol_upper} for price refresh")
    
    # 3. Refresh logo (immediate)
    if refresh_logo:
        try:
            # Clear cached logo by fetching fresh
            from app.database.connection import get_session
            from app.database.orm import Symbol
            from sqlalchemy import update
            
            async with get_session() as session:
                await session.execute(
                    update(Symbol)
                    .where(Symbol.symbol == symbol_upper)
                    .values(logo_fetched_at=None)  # Clear cache timestamp
                )
                await session.commit()
            
            # Fetch fresh logos
            await get_logo(symbol_upper, LogoTheme.LIGHT)
            await get_logo(symbol_upper, LogoTheme.DARK)
            tasks_triggered.append("logo:refreshed")
            logger.info(f"Refreshed logos for {symbol_upper}")
        except Exception as e:
            logger.warning(f"Failed to refresh logo for {symbol_upper}: {e}")
            tasks_triggered.append(f"logo:failed ({e})")
    
    # 4. Trigger AI analysis batch for this symbol
    if refresh_ai_analysis:
        # Queue for next batch job
        from app.repositories import dip_votes_orm
        
        # Delete existing AI analysis to force regeneration
        await dip_votes_orm.delete_ai_analysis(symbol_upper)
        tasks_triggered.append("ai_analysis:queued")
        logger.info(f"Queued {symbol_upper} for AI analysis regeneration")
    
    # 5. Refresh AI agents (immediate)
    if refresh_ai_agents:
        try:
            result = await run_agent_analysis(symbol_upper)
            tasks_triggered.append(f"ai_agents:{result.overall_signal}")
            logger.info(f"Refreshed AI agents for {symbol_upper}: {result.overall_signal}")
        except Exception as e:
            logger.warning(f"Failed to refresh AI agents for {symbol_upper}: {e}")
            tasks_triggered.append(f"ai_agents:failed ({e})")
    
    return FullRefreshResponse(
        symbol=symbol_upper,
        tasks_triggered=tasks_triggered,
        message=f"Triggered {len(tasks_triggered)} refresh operations for {symbol_upper}",
    )
