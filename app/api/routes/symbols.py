"""Symbol CRUD routes - PostgreSQL async."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, Path, status
from pydantic import BaseModel

from app.api.dependencies import require_user
from app.cache.cache import Cache
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.repositories import symbols as symbol_repo
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

    # Not in cache, validate against Yahoo Finance
    try:
        # get_stock_info is synchronous, no await needed
        info = get_stock_info(symbol_upper)
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


@router.get(
    "",
    response_model=List[SymbolResponse],
    summary="List all symbols",
    description="Get all tracked stock symbols.",
)
async def list_symbols(
    user: TokenData = Depends(require_user),
) -> List[SymbolResponse]:
    """List all tracked symbols."""
    symbols = await symbol_repo.list_symbols()
    return [
        SymbolResponse(
            symbol=s.symbol,
            min_dip_pct=s.min_dip_pct,
            min_days=s.min_days,
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
    background_tasks: BackgroundTasks,
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
    background_tasks.add_task(_process_new_symbol, payload.symbol)

    return SymbolResponse(
        symbol=created.symbol,
        min_dip_pct=created.min_dip_pct,
        min_days=created.min_days,
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
    from datetime import date, timedelta
    from app.database.connection import execute, fetch_one
    from app.services.stock_info import get_stock_info_async
    from app.services.openai_client import summarize_company, generate_bio, rate_dip
    from app.repositories import dip_votes as dip_votes_repo
    from app.dipfinder.service import get_dipfinder_service
    from app.cache.cache import Cache
    
    logger.info(f"Processing newly added symbol: {symbol}")
    
    try:
        # Step 1: Fetch Yahoo Finance info
        info = await get_stock_info_async(symbol)
        if not info:
            logger.warning(f"Could not fetch Yahoo data for {symbol}")
            return
        
        name = info.get("name") or info.get("short_name")
        sector = info.get("sector")
        full_summary = info.get("summary")  # longBusinessSummary from yfinance
        current_price = info.get("current_price", 0)
        ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
        dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0
        
        logger.info(f"Fetched data for {symbol}: price=${current_price}, ATH=${ath_price}, dip={dip_pct:.1f}%")
        
        # Step 2: Generate AI summary from the long description
        ai_summary = None
        if full_summary and len(full_summary) > 100:
            ai_summary = await summarize_company(
                symbol=symbol,
                name=name,
                description=full_summary,
            )
            if ai_summary:
                logger.info(f"Generated AI summary for {symbol}: {len(ai_summary)} chars")
        
        # Step 3: Update symbols with name, sector, and AI summary
        if name or sector or ai_summary:
            await execute(
                """UPDATE symbols SET 
                       name = COALESCE($2, name),
                       sector = COALESCE($3, sector),
                       summary_ai = COALESCE($4, summary_ai),
                       updated_at = NOW()
                   WHERE symbol = $1""",
                symbol.upper(),
                name,
                sector,
                ai_summary,
            )
            logger.info(f"Updated symbol info for {symbol}: name='{name}', sector='{sector}', summary_ai={'yes' if ai_summary else 'no'}")
        
        # Step 4: Fetch 365 days of price history for dipfinder
        try:
            service = get_dipfinder_service()
            prices = await service.price_provider.get_prices(
                symbol.upper(),
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
            )
            if prices is not None and not prices.empty:
                logger.info(f"Fetched {len(prices)} days of price history for {symbol}")
            else:
                logger.warning(f"No price history returned for {symbol}")
        except Exception as price_err:
            logger.warning(f"Failed to fetch price history for {symbol}: {price_err}")
        
        # Step 5: Add to dip_state so it appears in dashboard
        await execute(
            """INSERT INTO dip_state (symbol, current_price, ath_price, dip_percentage, first_seen, last_updated)
               VALUES ($1, $2, $3, $4, NOW(), NOW())
               ON CONFLICT (symbol) DO UPDATE SET
                   current_price = EXCLUDED.current_price,
                   ath_price = EXCLUDED.ath_price,
                   dip_percentage = EXCLUDED.dip_percentage,
                   last_updated = NOW()""",
            symbol.upper(),
            current_price,
            ath_price,
            dip_pct,
        )
        logger.info(f"Added {symbol} to dip_state with dip={dip_pct:.1f}%")
        
        # Step 6: Generate AI bio (swipe card summary)
        bio = await generate_bio(
            symbol=symbol,
            dip_pct=dip_pct,
        )
        
        # Step 7: Generate AI rating
        rating_data = await rate_dip(
            symbol=symbol,
            current_price=current_price,
            ref_high=ath_price,
            dip_pct=dip_pct,
        )
        
        # Step 8: Store AI analysis
        if bio or rating_data:
            await dip_votes_repo.upsert_ai_analysis(
                symbol=symbol,
                swipe_bio=bio,
                ai_rating=rating_data.get("rating") if rating_data else None,
                ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                is_batch=False,
            )
            logger.info(f"Generated AI content for {symbol}: bio={'yes' if bio else 'no'}, rating={rating_data.get('rating') if rating_data else 'none'}")
        else:
            logger.warning(f"No AI content generated for {symbol}")
        
        # Step 9: Invalidate caches so the stock appears immediately
        ranking_cache = Cache(prefix="ranking", default_ttl=3600)
        deleted = await ranking_cache.invalidate_pattern("*")
        logger.info(f"Completed processing {symbol}, invalidated {deleted} ranking cache keys")
            
    except Exception as e:
        logger.error(f"Error processing new symbol {symbol}: {e}")


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
    from app.database.connection import execute, fetch_one
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
        # Persist to database
        await execute(
            "UPDATE symbols SET summary_ai = $2, updated_at = NOW() WHERE symbol = $1",
            symbol.upper(),
            new_summary,
        )
        
        # Invalidate caches
        await invalidate_stock_info_cache(symbol)
        
        logger.info(f"Regenerated AI summary for {symbol}: {len(new_summary)} chars")
    
    return AISummaryResponse(symbol=symbol, summary_ai=new_summary)
