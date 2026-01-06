"""
Market Data Service - Sector and Industry data from yfinance.

Provides weekly-updated market structure data for:
- Sector trends and top companies
- Industry analysis and competitors
- Similar stock suggestions

Usage:
    from app.services.market_data import (
        sync_all_market_data,
        get_sector,
        get_industry,
        get_competitors,
    )
    
    # Sync all sectors and industries (run weekly)
    await sync_all_market_data()
    
    # Get sector data
    sector = await get_sector("technology")
    
    # Get competitors for a stock
    competitors = await get_competitors("AAPL")
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from app.cache.cache import Cache
from app.core.data_helpers import run_in_executor
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import MarketIndustry, MarketSector


logger = get_logger("services.market_data")

# Module-level cache instance for market data
_market_cache = Cache(prefix="market", default_ttl=86400)  # 1 day default


# All sector keys from yfinance
SECTOR_KEYS = [
    "basic-materials",
    "communication-services",
    "consumer-cyclical",
    "consumer-defensive",
    "energy",
    "financial-services",
    "healthcare",
    "industrials",
    "real-estate",
    "technology",
    "utilities",
]

# Cache TTL: 1 week (data is updated weekly anyway)
CACHE_TTL = 60 * 60 * 24 * 7  # 7 days


# =============================================================================
# DATA FETCHING FROM YFINANCE
# =============================================================================


def _df_to_list(df: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    """Convert DataFrame to list of dicts for JSON storage."""
    if df is None or df.empty:
        return None
    return df.reset_index().to_dict(orient="records")


def _fetch_sector_data(sector_key: str) -> dict[str, Any] | None:
    """Fetch sector data from yfinance (blocking call)."""
    try:
        sector = yf.Sector(sector_key)
        
        return {
            "key": sector.key,
            "name": sector.name,
            "symbol": sector.symbol,
            "overview": dict(sector.overview) if sector.overview else None,
            "top_companies": _df_to_list(sector.top_companies),
            "top_etfs": dict(sector.top_etfs) if sector.top_etfs else None,
            "top_mutual_funds": dict(sector.top_mutual_funds) if sector.top_mutual_funds else None,
            "research_reports": sector.research_reports if sector.research_reports else None,
            "industries": _df_to_list(sector.industries),
        }
    except Exception as e:
        logger.error(f"Failed to fetch sector {sector_key}: {e}")
        return None


def _fetch_industry_data(industry_key: str) -> dict[str, Any] | None:
    """Fetch industry data from yfinance (blocking call)."""
    try:
        industry = yf.Industry(industry_key)
        
        return {
            "key": industry.key,
            "name": industry.name,
            "symbol": industry.symbol,
            "sector_key": industry.sector_key,
            "sector_name": industry.sector_name,
            "overview": dict(industry.overview) if industry.overview else None,
            "top_companies": _df_to_list(industry.top_companies),
            "top_performing_companies": _df_to_list(industry.top_performing_companies),
            "top_growth_companies": _df_to_list(industry.top_growth_companies),
            "research_reports": industry.research_reports if industry.research_reports else None,
        }
    except Exception as e:
        logger.error(f"Failed to fetch industry {industry_key}: {e}")
        return None


# Alias for backward compatibility
_run_in_executor = run_in_executor


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================


async def _upsert_sector(data: dict[str, Any]) -> MarketSector | None:
    """Insert or update sector in database."""
    async with get_session() as session:
        try:
            from sqlalchemy import select
            from sqlalchemy.dialects.postgresql import insert
            
            stmt = insert(MarketSector).values(
                key=data["key"],
                name=data["name"],
                symbol=data.get("symbol"),
                overview=data.get("overview"),
                top_companies=data.get("top_companies"),
                top_etfs=data.get("top_etfs"),
                top_mutual_funds=data.get("top_mutual_funds"),
                research_reports=data.get("research_reports"),
                updated_at=datetime.now(UTC),
            ).on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "name": data["name"],
                    "symbol": data.get("symbol"),
                    "overview": data.get("overview"),
                    "top_companies": data.get("top_companies"),
                    "top_etfs": data.get("top_etfs"),
                    "top_mutual_funds": data.get("top_mutual_funds"),
                    "research_reports": data.get("research_reports"),
                    "updated_at": datetime.now(UTC),
                },
            )
            await session.execute(stmt)
            await session.commit()
            
            # Fetch and return the sector
            result = await session.execute(
                select(MarketSector).where(MarketSector.key == data["key"])
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to upsert sector {data['key']}: {e}")
            await session.rollback()
            return None


async def _upsert_industry(data: dict[str, Any]) -> MarketIndustry | None:
    """Insert or update industry in database."""
    async with get_session() as session:
        try:
            from sqlalchemy import select
            from sqlalchemy.dialects.postgresql import insert
            
            stmt = insert(MarketIndustry).values(
                key=data["key"],
                name=data["name"],
                symbol=data.get("symbol"),
                sector_key=data["sector_key"],
                sector_name=data["sector_name"],
                overview=data.get("overview"),
                top_companies=data.get("top_companies"),
                top_performing_companies=data.get("top_performing_companies"),
                top_growth_companies=data.get("top_growth_companies"),
                research_reports=data.get("research_reports"),
                updated_at=datetime.now(UTC),
            ).on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "name": data["name"],
                    "symbol": data.get("symbol"),
                    "sector_key": data["sector_key"],
                    "sector_name": data["sector_name"],
                    "overview": data.get("overview"),
                    "top_companies": data.get("top_companies"),
                    "top_performing_companies": data.get("top_performing_companies"),
                    "top_growth_companies": data.get("top_growth_companies"),
                    "research_reports": data.get("research_reports"),
                    "updated_at": datetime.now(UTC),
                },
            )
            await session.execute(stmt)
            await session.commit()
            
            result = await session.execute(
                select(MarketIndustry).where(MarketIndustry.key == data["key"])
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to upsert industry {data['key']}: {e}")
            await session.rollback()
            return None


# =============================================================================
# PUBLIC API - SYNC FUNCTIONS
# =============================================================================


async def sync_sector(sector_key: str) -> dict[str, Any]:
    """
    Sync a single sector and its industries from yfinance.
    
    Returns:
        Dict with sync results: {sector, industries_synced, errors}
    """
    logger.info(f"[MARKET DATA] Syncing sector: {sector_key}")
    
    result = {
        "sector": None,
        "industries_synced": 0,
        "errors": [],
    }
    
    # Fetch sector data
    sector_data = await _run_in_executor(_fetch_sector_data, sector_key)
    if not sector_data:
        result["errors"].append(f"Failed to fetch sector: {sector_key}")
        return result
    
    # Save sector to DB
    sector = await _upsert_sector(sector_data)
    if sector:
        result["sector"] = sector_key
        # Invalidate cache
        await Cache.delete(f"market:sector:{sector_key}")
        await Cache.delete("market:sectors:all")
    else:
        result["errors"].append(f"Failed to save sector: {sector_key}")
        return result
    
    # Get industry keys from sector data
    industries_data = sector_data.get("industries", [])
    if not industries_data:
        return result
    
    # Sync each industry
    for ind in industries_data:
        industry_key = ind.get("key")
        if not industry_key:
            continue
        
        industry_data = await _run_in_executor(_fetch_industry_data, industry_key)
        if industry_data:
            industry = await _upsert_industry(industry_data)
            if industry:
                result["industries_synced"] += 1
                # Invalidate cache
                await Cache.delete(f"market:industry:{industry_key}")
            else:
                result["errors"].append(f"Failed to save industry: {industry_key}")
        else:
            result["errors"].append(f"Failed to fetch industry: {industry_key}")
    
    logger.info(
        f"[MARKET DATA] Sector {sector_key}: "
        f"{result['industries_synced']} industries synced, "
        f"{len(result['errors'])} errors"
    )
    
    return result


async def sync_all_market_data() -> dict[str, Any]:
    """
    Sync all sectors and industries from yfinance.
    
    This is the main job function, should be run weekly.
    
    Returns:
        Dict with overall sync results
    """
    logger.info("[MARKET DATA] Starting full market data sync")
    start_time = datetime.now(UTC)
    
    results = {
        "sectors_synced": 0,
        "industries_synced": 0,
        "errors": [],
        "duration_seconds": 0,
    }
    
    for sector_key in SECTOR_KEYS:
        try:
            sector_result = await sync_sector(sector_key)
            
            if sector_result["sector"]:
                results["sectors_synced"] += 1
            
            results["industries_synced"] += sector_result["industries_synced"]
            results["errors"].extend(sector_result["errors"])
            
            # Small delay between sectors to avoid rate limiting
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error syncing sector {sector_key}: {e}")
            results["errors"].append(f"Sector {sector_key}: {str(e)}")
    
    duration = (datetime.now(UTC) - start_time).total_seconds()
    results["duration_seconds"] = round(duration, 2)
    
    logger.info(
        f"[MARKET DATA] Sync complete: "
        f"{results['sectors_synced']} sectors, "
        f"{results['industries_synced']} industries, "
        f"{len(results['errors'])} errors, "
        f"{results['duration_seconds']}s"
    )
    
    return results


# =============================================================================
# PUBLIC API - READ FUNCTIONS
# =============================================================================


async def get_all_sectors() -> list[dict[str, Any]]:
    """Get all sectors with basic info."""
    # Check cache first
    cached = await _market_cache.get("sectors:all")
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(MarketSector).order_by(MarketSector.name)
        )
        sectors = result.scalars().all()
        
        data = [
            {
                "key": s.key,
                "name": s.name,
                "symbol": s.symbol,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sectors
        ]
        
        # Cache for 1 day
        await _market_cache.set("sectors:all", data, ttl=86400)
        return data


async def get_sector(sector_key: str) -> dict[str, Any] | None:
    """Get full sector data including companies and industries."""
    # Check cache first
    cached = await _market_cache.get(f"sector:{sector_key}")
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        result = await session.execute(
            select(MarketSector)
            .options(selectinload(MarketSector.industries))
            .where(MarketSector.key == sector_key)
        )
        sector = result.scalar_one_or_none()
        
        if not sector:
            return None
        
        data = {
            "key": sector.key,
            "name": sector.name,
            "symbol": sector.symbol,
            "overview": sector.overview,
            "top_companies": sector.top_companies,
            "top_etfs": sector.top_etfs,
            "top_mutual_funds": sector.top_mutual_funds,
            "industries": [
                {
                    "key": i.key,
                    "name": i.name,
                    "symbol": i.symbol,
                }
                for i in sector.industries
            ],
            "updated_at": sector.updated_at.isoformat() if sector.updated_at else None,
        }
        
        await _market_cache.set(f"sector:{sector_key}", data, ttl=CACHE_TTL)
        return data


async def get_industry(industry_key: str) -> dict[str, Any] | None:
    """Get full industry data including top companies."""
    cached = await _market_cache.get(f"industry:{industry_key}")
    if cached:
        return cached
    
    async with get_session() as session:
        from sqlalchemy import select
        
        result = await session.execute(
            select(MarketIndustry).where(MarketIndustry.key == industry_key)
        )
        industry = result.scalar_one_or_none()
        
        if not industry:
            return None
        
        data = {
            "key": industry.key,
            "name": industry.name,
            "symbol": industry.symbol,
            "sector_key": industry.sector_key,
            "sector_name": industry.sector_name,
            "overview": industry.overview,
            "top_companies": industry.top_companies,
            "top_performing_companies": industry.top_performing_companies,
            "top_growth_companies": industry.top_growth_companies,
            "updated_at": industry.updated_at.isoformat() if industry.updated_at else None,
        }
        
        await _market_cache.set(f"industry:{industry_key}", data, ttl=CACHE_TTL)
        return data


async def get_competitors(
    symbol: str,
    *,
    limit: int = 10,
    exclude_self: bool = True,
) -> dict[str, Any]:
    """
    Get competitors for a stock based on industry.
    
    Returns:
        Dict with industry info and competitor lists
    """
    from app.services.stock_info import get_stock_info_async
    
    # Get stock info to find industry
    stock_info = await get_stock_info_async(symbol)
    if not stock_info:
        return {"error": f"Stock not found: {symbol}", "competitors": []}
    
    industry_key = stock_info.get("industryKey")
    if not industry_key:
        return {"error": "No industry data available", "competitors": []}
    
    # Get industry data
    industry = await get_industry(industry_key)
    if not industry:
        return {"error": f"Industry not found: {industry_key}", "competitors": []}
    
    # Build competitor list from top companies
    competitors = []
    for company_list in [
        industry.get("top_companies", []),
        industry.get("top_performing_companies", []),
        industry.get("top_growth_companies", []),
    ]:
        if not company_list:
            continue
        for company in company_list:
            comp_symbol = company.get("symbol") or company.get("Symbol")
            if not comp_symbol:
                continue
            if exclude_self and comp_symbol.upper() == symbol.upper():
                continue
            if comp_symbol not in [c["symbol"] for c in competitors]:
                competitors.append({
                    "symbol": comp_symbol,
                    "name": company.get("name") or company.get("Name", ""),
                    "market_weight": company.get("market weight") or company.get("marketWeight"),
                })
            if len(competitors) >= limit:
                break
        if len(competitors) >= limit:
            break
    
    return {
        "symbol": symbol,
        "industry_key": industry_key,
        "industry_name": industry.get("name"),
        "sector_key": industry.get("sector_key"),
        "sector_name": industry.get("sector_name"),
        "competitors": competitors[:limit],
    }


async def get_similar_stocks(
    symbol: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Get similar stocks based on sector and industry.
    
    Combines competitors from industry with top sector companies.
    """
    competitors = await get_competitors(symbol, limit=limit)
    return competitors.get("competitors", [])
