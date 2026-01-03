"""Market data API routes - sectors, industries, competitors."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from app.api.dependencies import require_admin
from app.core.logging import get_logger
from app.core.security import TokenData


logger = get_logger("api.market")

router = APIRouter(prefix="/market")


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


class SectorSummary(BaseModel):
    """Basic sector info for list endpoint."""
    key: str
    name: str
    symbol: str | None
    updated_at: str | None


class SectorDetail(BaseModel):
    """Full sector data with companies and industries."""
    key: str
    name: str
    symbol: str | None
    overview: dict | None
    top_companies: list[dict] | None
    top_etfs: dict | None
    top_mutual_funds: dict | None
    industries: list[dict]
    updated_at: str | None


class IndustryDetail(BaseModel):
    """Full industry data with companies."""
    key: str
    name: str
    symbol: str | None
    sector_key: str
    sector_name: str
    overview: dict | None
    top_companies: list[dict] | None
    top_performing_companies: list[dict] | None
    top_growth_companies: list[dict] | None
    updated_at: str | None


class CompetitorInfo(BaseModel):
    """Competitor stock info."""
    symbol: str
    name: str
    market_weight: float | None = None


class CompetitorsResponse(BaseModel):
    """Response for competitors endpoint."""
    symbol: str
    industry_key: str | None
    industry_name: str | None
    sector_key: str | None
    sector_name: str | None
    competitors: list[CompetitorInfo]
    error: str | None = None


class SyncResult(BaseModel):
    """Result of market data sync operation."""
    sectors_synced: int
    industries_synced: int
    errors: list[str]
    duration_seconds: float


# =============================================================================
# ROUTES
# =============================================================================


@router.get(
    "/sectors",
    response_model=list[SectorSummary],
    summary="List all sectors",
    description="Get a list of all market sectors with basic info.",
)
async def list_sectors():
    """Get all sectors."""
    from app.services.market_data import get_all_sectors
    
    return await get_all_sectors()


@router.get(
    "/sectors/{sector_key}",
    response_model=SectorDetail,
    summary="Get sector details",
    description="Get full sector data including top companies, ETFs, and industries.",
)
async def get_sector(sector_key: str):
    """Get sector by key."""
    from app.services.market_data import get_sector
    
    data = await get_sector(sector_key)
    if not data:
        raise HTTPException(status_code=404, detail=f"Sector not found: {sector_key}")
    
    return data


@router.get(
    "/industries/{industry_key}",
    response_model=IndustryDetail,
    summary="Get industry details",
    description="Get full industry data including top companies and performers.",
)
async def get_industry(industry_key: str):
    """Get industry by key."""
    from app.services.market_data import get_industry
    
    data = await get_industry(industry_key)
    if not data:
        raise HTTPException(status_code=404, detail=f"Industry not found: {industry_key}")
    
    return data


@router.get(
    "/competitors/{symbol}",
    response_model=CompetitorsResponse,
    summary="Get stock competitors",
    description="Find competitor stocks in the same industry as the given symbol.",
)
async def get_competitors(
    symbol: str,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """
    Get competitors for a stock.
    
    Returns companies in the same industry, ranked by market weight.
    The source stock is excluded from results by default.
    """
    from app.services.market_data import get_competitors
    
    result = await get_competitors(symbol.upper(), limit=limit)
    
    # Convert to response model
    return CompetitorsResponse(
        symbol=result.get("symbol", symbol.upper()),
        industry_key=result.get("industry_key"),
        industry_name=result.get("industry_name"),
        sector_key=result.get("sector_key"),
        sector_name=result.get("sector_name"),
        competitors=[
            CompetitorInfo(**c) for c in result.get("competitors", [])
        ],
        error=result.get("error"),
    )


@router.get(
    "/similar/{symbol}",
    response_model=list[CompetitorInfo],
    summary="Get similar stocks",
    description="Get stocks similar to the given symbol based on sector and industry.",
)
async def get_similar_stocks(
    symbol: str,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """
    Get similar stocks based on industry.
    
    Simplified version of competitors endpoint that just returns the stock list.
    """
    from app.services.market_data import get_similar_stocks
    
    stocks = await get_similar_stocks(symbol.upper(), limit=limit)
    return [CompetitorInfo(**s) for s in stocks]


# =============================================================================
# ADMIN ROUTES
# =============================================================================


@router.post(
    "/sync",
    response_model=SyncResult,
    summary="Trigger market data sync",
    description="Manually trigger a sync of all sector and industry data from yfinance.",
    dependencies=[Depends(require_admin)],
)
async def sync_market_data(background_tasks: BackgroundTasks):
    """
    Trigger full market data sync (admin only).
    
    This runs the same job as the weekly scheduled sync.
    Returns immediately and runs the sync in the background.
    """
    from app.services.market_data import sync_all_market_data
    
    # Run sync in background
    background_tasks.add_task(sync_all_market_data)
    
    return SyncResult(
        sectors_synced=0,
        industries_synced=0,
        errors=["Sync started in background"],
        duration_seconds=0,
    )


@router.post(
    "/sync/{sector_key}",
    summary="Sync single sector",
    description="Sync a single sector and its industries from yfinance.",
    dependencies=[Depends(require_admin)],
)
async def sync_single_sector(sector_key: str):
    """
    Sync a single sector (admin only).
    
    This fetches the sector and all its industries from yfinance.
    """
    from app.services.market_data import sync_sector, SECTOR_KEYS
    
    if sector_key not in SECTOR_KEYS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid sector key. Valid keys: {SECTOR_KEYS}"
        )
    
    result = await sync_sector(sector_key)
    return result
