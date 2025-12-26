"""Symbol CRUD routes - PostgreSQL async."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Path, Query, status
from pydantic import BaseModel

from app.api.dependencies import require_user
from app.cache.cache import Cache
from app.celery_app import celery_app
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.repositories import symbols_orm as symbol_repo
from app.schemas.symbols import SymbolCreate, SymbolResponse, SymbolUpdate
from app.services.stock_info import get_stock_info
from app.services.runtime_settings import get_cache_ttl

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
    name: Optional[str] = None
    sector: Optional[str] = None
    quote_type: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    source: Optional[str] = None  # "local", "cached", or "api"
    score: Optional[float] = None  # Relevance score (0-1)


class SymbolSearchResponse(BaseModel):
    """Response for symbol search."""
    query: str
    results: List[SymbolSearchResult]
    count: int
    suggest_fresh_search: bool = False
    search_type: str = "local"  # "local" or "api"
    next_cursor: Optional[str] = None  # Cursor for pagination


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
    cursor: Optional[str] = Query(None, description="Pagination cursor from previous response"),
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
    name: Optional[str] = None


@router.get(
    "/autocomplete/{partial}",
    response_model=List[SymbolAutocompleteResult],
    summary="Autocomplete symbols",
    description="Get autocomplete suggestions for partial symbol/name input. Optimized for speed.",
)
async def autocomplete_symbols(
    partial: str = Path(..., min_length=1, max_length=20, description="Partial input"),
    limit: int = Query(5, ge=1, le=10, description="Maximum suggestions"),
) -> List[SymbolAutocompleteResult]:
    """Get autocomplete suggestions for symbol input."""
    from app.services.symbol_search import get_symbol_suggestions
    
    results = await get_symbol_suggestions(partial, limit=limit)
    return [SymbolAutocompleteResult(**r) for r in results]


class SymbolFundamentalsResponse(BaseModel):
    """Fundamentals data for a symbol."""
    symbol: str
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    profit_margin: Optional[str] = None
    gross_margin: Optional[str] = None
    return_on_equity: Optional[str] = None
    debt_to_equity: Optional[str] = None
    current_ratio: Optional[str] = None
    revenue_growth: Optional[str] = None
    earnings_growth: Optional[str] = None
    free_cash_flow: Optional[str] = None
    recommendation: Optional[str] = None
    target_mean_price: Optional[float] = None
    num_analyst_opinions: Optional[int] = None
    beta: Optional[str] = None
    next_earnings_date: Optional[str] = None
    source: str = "database"  # "database" or "api"
    fetched_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_stale: bool = False
    refresh_task_id: Optional[str] = None


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
        get_fundamentals_with_status,
        fetch_fundamentals_live,
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
    response_model=List[SymbolResponse],
    summary="List all symbols",
    description="Get all tracked stock symbols. Public endpoint.",
)
async def list_symbols() -> List[SymbolResponse]:
    """List all tracked symbols (public endpoint for signals page)."""
    symbols = await symbol_repo.list_symbols()
    return [
        SymbolResponse(
            symbol=s.symbol,
            min_dip_pct=s.min_dip_pct,
            min_days=s.min_days,
            name=s.name,
            fetch_status=s.fetch_status,
            fetch_error=s.fetch_error,
        )
        for s in symbols
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
    return SymbolResponse(
        symbol=config.symbol,
        min_dip_pct=config.min_dip_pct,
        min_days=config.min_days,
        name=config.name,
        fetch_status=config.fetch_status,
        fetch_error=config.fetch_error,
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

    After creation, fetches initial data and generates AI summary in background.
    """
    # Check if already exists
    existing = await symbol_repo.get_symbol(payload.symbol)
    if existing is not None:
        raise ConflictError(
            message=f"Symbol '{payload.symbol}' already exists",
            details={"symbol": payload.symbol},
        )

    # Create symbol
    created = await symbol_repo.upsert_symbol(
        payload.symbol,
        payload.min_dip_pct,
        payload.min_days,
    )

    # Process symbol in background (fetch data, generate AI summary)
    task = celery_app.send_task("jobs.process_new_symbol", args=[payload.symbol.upper()])

    return SymbolResponse(
        symbol=created.symbol,
        min_dip_pct=created.min_dip_pct,
        min_days=created.min_days,
        name=created.name,
        fetch_status='fetching',  # Will be set to fetching by background task
        fetch_error=None,
        task_id=task.id,
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


async def _process_new_symbol(symbol: str) -> None:
    """Background task to process newly added symbol.
    
    Fetches Yahoo Finance data, adds to dip_state, generates AI content,
    and invalidates caches so the stock appears immediately in dashboard.
    """
    import asyncio
    from datetime import date, timedelta
    from app.repositories import symbols_orm as symbols_repo_local
    from app.repositories import dip_state_orm as dip_state_repo
    from app.services.stock_info import get_stock_info_async
    from app.services.openai_client import summarize_company, generate_bio, rate_dip
    from app.repositories import dip_votes_orm as dip_votes_repo
    from app.dipfinder.service import get_dipfinder_service
    from app.cache.cache import Cache
    
    logger.info(f"[NEW SYMBOL] Starting background processing for: {symbol}")
    
    # Wait for the main transaction to commit (race condition prevention)
    await asyncio.sleep(0.5)
    
    # Verify symbol exists before proceeding
    exists = await symbols_repo_local.symbol_exists(symbol.upper())
    if not exists:
        logger.error(f"[NEW SYMBOL] FAILED: Symbol {symbol} not found in database - aborting")
        return
    
    # Set fetch_status to 'fetching' so UI shows loading state
    await symbols_repo_local.update_fetch_status(
        symbol.upper(),
        fetch_status="fetching",
        fetch_error=None,
    )
    
    # Track what we've successfully done
    steps_completed = []
    
    try:
        # Step 1: Fetch Yahoo Finance info
        logger.info(f"[NEW SYMBOL] Step 1: Fetching Yahoo Finance data for {symbol}")
        info = await get_stock_info_async(symbol)
        if not info:
            logger.error(f"[NEW SYMBOL] FAILED: Could not fetch Yahoo data for {symbol} - aborting")
            # Mark as error
            await symbols_repo_local.update_fetch_status(
                symbol.upper(),
                fetch_status="error",
                fetch_error="Could not fetch data from Yahoo Finance",
            )
            # Still try to invalidate cache
            try:
                ranking_cache = Cache(prefix="ranking", default_ttl=3600)
                await ranking_cache.invalidate_pattern("*")
            except Exception:
                pass
            return
        
        steps_completed.append("yahoo_fetch")
        
        name = info.get("name") or info.get("short_name")
        sector = info.get("sector")
        full_summary = info.get("summary")  # longBusinessSummary from yfinance
        current_price = info.get("current_price", 0)
        ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
        dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0
        
        logger.info(f"[NEW SYMBOL] Fetched: {symbol} - name='{name}', price=${current_price}, ATH=${ath_price}, dip={dip_pct:.1f}%")
        
        # Step 2: Update symbols with name, sector (no AI yet)
        if name or sector:
            try:
                await symbols_repo_local.update_symbol_info(
                    symbol.upper(),
                    name=name,
                    sector=sector,
                )
                steps_completed.append("symbol_update")
                logger.info(f"[NEW SYMBOL] Step 2: Updated symbol table for {symbol}")
            except Exception as e:
                logger.error(f"[NEW SYMBOL] Step 2 FAILED: Could not update symbol {symbol}: {e}")
        
        # Step 3: Add to dip_state so it appears in dashboard
        try:
            await dip_state_repo.upsert_dip_state(
                symbol=symbol.upper(),
                current_price=current_price,
                ath_price=ath_price,
                dip_percentage=dip_pct,
            )
            steps_completed.append("dip_state")
            logger.info(f"[NEW SYMBOL] Step 3: Added to dip_state with dip={dip_pct:.1f}%")
        except Exception as e:
            logger.error(f"[NEW SYMBOL] Step 3 FAILED: Could not add to dip_state for {symbol}: {e}")
        
        # Step 4: Fetch price history for dipfinder
        try:
            service = get_dipfinder_service()
            prices = await service.price_provider.get_prices(
                symbol.upper(),
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
            )
            if prices is not None and not prices.empty:
                steps_completed.append("price_history")
                logger.info(f"[NEW SYMBOL] Step 4: Fetched {len(prices)} days of price history")
            else:
                logger.warning(f"[NEW SYMBOL] Step 4: No price history returned for {symbol}")
        except Exception as e:
            logger.warning(f"[NEW SYMBOL] Step 4 FAILED: Could not fetch price history for {symbol}: {e}")
        
        # Step 5: Generate AI summary (optional - continues if fails)
        ai_summary = None
        if full_summary and len(full_summary) > 100:
            try:
                ai_summary = await summarize_company(
                    symbol=symbol,
                    name=name,
                    description=full_summary,
                )
                if ai_summary:
                    await symbols_repo_local.update_symbol_info(
                        symbol.upper(),
                        summary_ai=ai_summary,
                    )
                    steps_completed.append("ai_summary")
                    logger.info(f"[NEW SYMBOL] Step 5: Generated AI summary ({len(ai_summary)} chars)")
                else:
                    logger.warning(f"[NEW SYMBOL] Step 5: No AI summary generated (OpenAI not configured?)")
            except Exception as e:
                logger.warning(f"[NEW SYMBOL] Step 5 FAILED: AI summary error for {symbol}: {e}")
        else:
            logger.info(f"[NEW SYMBOL] Step 5: Skipped AI summary (no description or too short)")
        
        # Step 6: Generate AI bio (swipe card) - optional
        bio = None
        try:
            bio = await generate_bio(
                symbol=symbol,
                name=name,
                sector=sector,
                summary=full_summary,
                dip_pct=dip_pct,
            )
            if bio:
                steps_completed.append("ai_bio")
                logger.info(f"[NEW SYMBOL] Step 6: Generated AI bio")
            else:
                logger.warning(f"[NEW SYMBOL] Step 6: No AI bio generated (OpenAI not configured?)")
        except Exception as e:
            logger.warning(f"[NEW SYMBOL] Step 6 FAILED: AI bio error for {symbol}: {e}")
        
        # Step 7: Generate AI rating - optional (includes fetching fundamentals)
        rating_data = None
        try:
            from app.services.fundamentals import get_fundamentals_for_analysis
            fundamentals = await get_fundamentals_for_analysis(symbol)
            
            rating_data = await rate_dip(
                symbol=symbol,
                current_price=current_price,
                ref_high=ath_price,
                dip_pct=dip_pct,
                days_below=0,  # New symbol, just added
                name=name,
                sector=sector,
                summary=full_summary,
                **fundamentals,  # Include all fundamental metrics
            )
            if rating_data:
                steps_completed.append("ai_rating")
                logger.info(f"[NEW SYMBOL] Step 7: Generated AI rating: {rating_data.get('rating')}")
            else:
                logger.warning(f"[NEW SYMBOL] Step 7: No AI rating generated (OpenAI not configured?)")
        except Exception as e:
            logger.warning(f"[NEW SYMBOL] Step 7 FAILED: AI rating error for {symbol}: {e}")
        
        # Step 8: Store AI analysis if any was generated
        if bio or rating_data:
            try:
                await dip_votes_repo.upsert_ai_analysis(
                    symbol=symbol,
                    swipe_bio=bio,
                    ai_rating=rating_data.get("rating") if rating_data else None,
                    ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                    is_batch=False,
                )
                steps_completed.append("ai_stored")
                logger.info(f"[NEW SYMBOL] Step 8: Stored AI analysis")
            except Exception as e:
                logger.error(f"[NEW SYMBOL] Step 8 FAILED: Could not store AI analysis for {symbol}: {e}")
        
        # Step 8.5: Run AI agent analysis (Warren Buffett, Peter Lynch, etc.)
        try:
            from app.services.ai_agents import run_agent_analysis
            agent_result = await run_agent_analysis(symbol)
            if agent_result:
                steps_completed.append("ai_agents")
                logger.info(f"[NEW SYMBOL] Step 8.5: AI agents: {agent_result.overall_signal} ({agent_result.overall_confidence}%)")
            else:
                logger.warning(f"[NEW SYMBOL] Step 8.5: No agent analysis generated")
        except Exception as e:
            logger.warning(f"[NEW SYMBOL] Step 8.5 FAILED: AI agents error for {symbol}: {e}")
        
        # Step 9: Invalidate caches so the stock appears immediately
        try:
            # Invalidate ranking cache
            ranking_cache = Cache(prefix="ranking", default_ttl=3600)
            deleted = await ranking_cache.invalidate_pattern("*")
            
            # Invalidate stock info cache
            stockinfo_cache = Cache(prefix="stockinfo", default_ttl=3600)
            await stockinfo_cache.delete(symbol.upper())
            
            # Invalidate symbols list cache
            symbols_cache = Cache(prefix="symbols", default_ttl=3600)
            await symbols_cache.invalidate_pattern("*")
            
            steps_completed.append("cache_invalidated")
            logger.info(f"[NEW SYMBOL] Step 9: Invalidated caches (ranking: {deleted} keys, stockinfo, symbols)")
        except Exception as e:
            logger.warning(f"[NEW SYMBOL] Step 9 FAILED: Could not invalidate cache: {e}")
        
        # Step 10: Mark as fetched
        from datetime import datetime as dt
        await symbols_repo_local.update_fetch_status(
            symbol.upper(),
            fetch_status="fetched",
            fetched_at=dt.utcnow(),
        )
        
        logger.info(f"[NEW SYMBOL] COMPLETED processing {symbol}. Steps completed: {', '.join(steps_completed)}")
            
    except Exception as e:
        logger.error(f"[NEW SYMBOL] FATAL ERROR processing {symbol}: {e}", exc_info=True)
        # Mark as error
        try:
            await symbols_repo_local.update_fetch_status(
                symbol.upper(),
                fetch_status="error",
                fetch_error=str(e)[:500],
            )
        except Exception:
            pass
        # Try to invalidate cache even on error
        try:
            ranking_cache = Cache(prefix="ranking", default_ttl=3600)
            await ranking_cache.invalidate_pattern("*")
        except Exception:
            pass


# =============================================================================
# AI Summary Management
# =============================================================================

class AISummaryResponse(BaseModel):
    """Response for AI summary regeneration."""
    symbol: str
    summary_ai: str | None


@router.post(
    "/{symbol}/ai/summary",
    response_model=AISummaryResponse,
    summary="Regenerate AI summary",
    description="Regenerate the AI-generated summary for a symbol. Requires authentication.",
    dependencies=[Depends(require_user)],
)
async def regenerate_ai_summary(
    symbol: str = Depends(_validate_symbol_path),
) -> AISummaryResponse:
    """Regenerate the AI summary for a symbol from its Yahoo Finance description."""
    from app.repositories import symbols_orm as symbols_repo_local
    from app.services.stock_info import get_stock_info_async
    from app.services.openai_client import summarize_company
    from app.api.routes.dips import invalidate_stock_info_cache
    
    # Verify symbol exists
    sym = await symbol_repo.get_symbol(symbol)
    if not sym:
        raise NotFoundError(f"Symbol {symbol} not found")
    
    # Fetch stock info from Yahoo
    info = await get_stock_info_async(symbol)
    if not info:
        raise NotFoundError(f"Could not fetch info for {symbol} from Yahoo Finance")
    
    description = info.get("summary")
    if not description or len(description) < 100:
        raise NotFoundError(f"No description available for {symbol}")
    
    # Generate new AI summary
    new_summary = await summarize_company(
        symbol=symbol,
        name=info.get("name"),
        description=description,
    )
    
    if new_summary:
        # Persist to database using ORM
        await symbols_repo_local.update_symbol_info(
            symbol.upper(),
            summary_ai=new_summary,
        )
        
        # Invalidate caches
        await invalidate_stock_info_cache(symbol)
        
        logger.info(f"Regenerated AI summary for {symbol}: {len(new_summary)} chars")
    
    return AISummaryResponse(symbol=symbol, summary_ai=new_summary)


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
    analyzed_at: Optional[str] = None
    expires_at: Optional[str] = None


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
