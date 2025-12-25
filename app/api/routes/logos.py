"""Logo API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException, Depends

from fastapi.responses import Response

from app.core.logging import get_logger
from app.core.security import TokenData
from app.services.logo_service import get_logo, LogoTheme, prefetch_logos
from app.api.dependencies import require_admin

logger = get_logger("api.logos")

router = APIRouter()


@router.get(
    "/logos/{symbol}",
    summary="Get stock logo",
    description="Returns cached stock logo in WebP format. Falls back to favicon if Logo.dev unavailable.",
    responses={
        200: {
            "content": {"image/webp": {}},
            "description": "Logo image in WebP format",
        },
        404: {"description": "Logo not found"},
    },
)
async def get_stock_logo(
    symbol: str,
    theme: str = Query("light", pattern="^(light|dark)$", description="Logo theme variant"),
):
    """
    Get stock logo by symbol.
    
    Returns cached WebP logo from Logo.dev API.
    Falls back to favicon if the logo isn't available.
    """
    logo_theme = LogoTheme.LIGHT if theme == "light" else LogoTheme.DARK
    
    try:
        logo_data = await get_logo(symbol.upper(), logo_theme)
        
        if logo_data:
            return Response(
                content=logo_data,
                media_type="image/webp",
                headers={
                    "Cache-Control": "public, max-age=86400",  # Cache for 1 day
                    "X-Logo-Source": "cached",
                },
            )
        
        raise HTTPException(status_code=404, detail=f"Logo not found for {symbol}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching logo for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch logo")


@router.post(
    "/logos/prefetch",
    summary="Prefetch logos for multiple symbols",
    description="Admin endpoint to prefetch and cache logos for multiple stock symbols.",
)
async def prefetch_stock_logos(
    symbols: list[str],
    _admin: TokenData = Depends(require_admin),
):
    """
    Prefetch logos for multiple symbols (admin only).
    
    This fetches and caches both light and dark variants for each symbol.
    """
    if len(symbols) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 symbols per request",
        )
    
    results = await prefetch_logos(symbols)
    
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    return {
        "message": f"Prefetched logos for {successful} symbols, {failed} failed",
        "results": results,
    }
