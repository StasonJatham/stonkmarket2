"""
Logo service using Logo.dev API.

Fetches and caches company logos in WebP format for both light and dark themes.
Falls back to favicon if Logo.dev fails.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from enum import Enum

import httpx
from sqlalchemy import select, update

from app.core.config import settings
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import Symbol, StockSuggestion
from app.repositories import api_keys_orm as api_keys_repo

logger = get_logger("services.logo")

# Logo.dev configuration - uses settings from config
LOGO_DEV_BASE_URL = "https://img.logo.dev"
LOGO_SIZE = 128  # Retina size


class LogoTheme(str, Enum):
    """Logo theme variants."""
    LIGHT = "light"
    DARK = "dark"


async def _get_logo_dev_public_key() -> Optional[str]:
    """
    Get Logo.dev public key from env or database.
    
    Checks environment variable first, then falls back to secure database storage.
    Note: Logo.dev image CDN requires the publishable (public) key, not the secret key.
    If you get domain restriction errors, update key settings in Logo.dev dashboard.
    """
    # Check env first
    if settings.logo_dev_public_key:
        return settings.logo_dev_public_key
    
    # Check database
    try:
        key = await api_keys_repo.get_decrypted_key(api_keys_repo.LOGO_DEV_PUBLIC_KEY)
        return key
    except Exception as e:
        logger.warning(f"Failed to get Logo.dev key from database: {e}")
        return None


async def _fetch_logo_from_api(
    symbol: str,
    theme: LogoTheme,
    timeout: Optional[float] = None,
) -> Optional[bytes]:
    """
    Fetch logo from Logo.dev ticker API.
    
    Args:
        symbol: Stock ticker symbol
        theme: Light or dark theme
        timeout: Request timeout in seconds (defaults to settings.external_api_timeout)
        
    Returns:
        WebP image bytes or None if failed
    """
    if timeout is None:
        timeout = float(settings.external_api_timeout)
        
    # Get API key (must be publishable key for image CDN)
    api_key = await _get_logo_dev_public_key()
    if not api_key:
        logger.debug(f"Logo.dev API key not configured, skipping fetch for {symbol}")
        return None
    
    # Logo.dev ticker endpoint
    url = f"{LOGO_DEV_BASE_URL}/ticker/{symbol.upper()}"
    params = {
        "token": api_key,
        "format": "webp",
        "theme": theme.value,
        "retina": "true",
        "size": LOGO_SIZE,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params)
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type:
                    logger.debug(f"Fetched logo for {symbol} ({theme.value}) from Logo.dev")
                    return response.content
                    
            logger.debug(f"Logo.dev returned {response.status_code} for {symbol} ({theme.value})")
            return None
            
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching logo for {symbol} from Logo.dev")
        return None
    except Exception as e:
        logger.warning(f"Error fetching logo for {symbol}: {e}")
        return None


async def _fetch_favicon_fallback(
    website: Optional[str],
    symbol: str,
    timeout: Optional[float] = None,
) -> Optional[bytes]:
    """
    Fetch favicon as fallback using DuckDuckGo's icon service.
    Falls back to UI Avatars if favicon not available.
    
    Args:
        website: Company website URL (from yfinance)
        symbol: Stock symbol (for logging only)
        timeout: Request timeout (defaults to settings.external_api_timeout)
        
    Returns:
        Image bytes or None
    """
    from urllib.parse import urlparse
    
    if timeout is None:
        timeout = float(settings.external_api_timeout)
    
    # Extract domain from website URL
    domain = None
    if website:
        try:
            parsed = urlparse(website)
            domain = parsed.netloc or parsed.path.split('/')[0]
            # Remove www. prefix for cleaner domain
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            pass
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Try DuckDuckGo icons first (reliable, no API key needed)
        if domain:
            try:
                # DuckDuckGo icon service
                url = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
                response = await client.get(url)
                if response.status_code == 200 and len(response.content) > 100:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type or response.content[:4] in (b'\x00\x00\x01\x00', b'\x89PNG'):
                        logger.debug(f"Fetched icon for {symbol} from DuckDuckGo ({domain})")
                        return response.content
            except Exception as e:
                logger.debug(f"DuckDuckGo icon failed for {symbol}: {e}")
        
        # Final fallback: UI Avatars (always works, generates placeholder)
        try:
            # Generate a consistent color from symbol
            color = hashlib.md5(symbol.encode()).hexdigest()[:6]
            url = f"https://ui-avatars.com/api/?name={symbol}&background={color}&color=fff&size=128&format=png&bold=true"
            response = await client.get(url)
            if response.status_code == 200:
                logger.debug(f"Generated avatar for {symbol}")
                return response.content
        except Exception as e:
            logger.debug(f"UI Avatars fallback failed for {symbol}: {e}")
    
    return None


async def get_logo(
    symbol: str,
    theme: LogoTheme = LogoTheme.LIGHT,
    website: Optional[str] = None,
) -> Optional[bytes]:
    """
    Get logo for a stock symbol, using cache or fetching from API.
    
    Args:
        symbol: Stock ticker symbol
        theme: Light or dark theme
        website: Optional company website for favicon fallback
        
    Returns:
        WebP image bytes or None
    """
    symbol = symbol.upper()
    
    # Check cache first
    async with get_session() as session:
        result = await session.execute(
            select(
                Symbol.logo_light if theme == LogoTheme.LIGHT else Symbol.logo_dark,
                Symbol.logo_fetched_at,
                Symbol.logo_source,
            ).where(Symbol.symbol == symbol)
        )
        row = result.one_or_none()
    
    # Return cached logo if still valid
    if row:
        logo_data, fetched_at, logo_source = row
        if logo_data:
            if fetched_at:
                age = datetime.now(timezone.utc) - fetched_at.replace(tzinfo=timezone.utc)
                if age < timedelta(days=settings.logo_cache_days):
                    return bytes(logo_data)
    
    # Fetch fresh logo
    logo_data = await _fetch_logo_from_api(symbol, theme)
    source = "logo.dev"
    
    # Fallback to favicon if Logo.dev failed
    if not logo_data:
        # Get website from stock_suggestions table if not provided
        if not website and row:
            async with get_session() as session:
                website_result = await session.execute(
                    select(StockSuggestion.website).where(StockSuggestion.symbol == symbol)
                )
                website_row = website_result.scalar_one_or_none()
                if website_row:
                    website = website_row
        
        logo_data = await _fetch_favicon_fallback(website, symbol)
        source = "favicon" if logo_data else None
    
    # Cache the logo
    if logo_data:
        await _cache_logo(symbol, theme, logo_data, source)
    
    return logo_data


async def _cache_logo(
    symbol: str,
    theme: LogoTheme,
    data: bytes,
    source: str,
) -> None:
    """Cache logo data in the database."""
    try:
        async with get_session() as session:
            update_values = {
                "logo_fetched_at": datetime.now(timezone.utc),
                "logo_source": source,
            }
            if theme == LogoTheme.LIGHT:
                update_values["logo_light"] = data
            else:
                update_values["logo_dark"] = data
            
            await session.execute(
                update(Symbol)
                .where(Symbol.symbol == symbol.upper())
                .values(**update_values)
            )
            await session.commit()
        logger.info(f"Cached {theme.value} logo for {symbol} (source: {source}, size: {len(data)} bytes)")
    except Exception as e:
        logger.warning(f"Failed to cache logo for {symbol}: {e}")


async def prefetch_logos(symbols: list[str]) -> dict[str, bool]:
    """
    Prefetch logos for multiple symbols in parallel.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dict of symbol -> success status
    """
    results = {}
    
    async def fetch_both_themes(symbol: str) -> Tuple[str, bool]:
        try:
            # Fetch both themes in parallel
            light, dark = await asyncio.gather(
                get_logo(symbol, LogoTheme.LIGHT),
                get_logo(symbol, LogoTheme.DARK),
                return_exceptions=True,
            )
            success = bool(light) or bool(dark)
            return symbol, success
        except Exception as e:
            logger.warning(f"Failed to prefetch logos for {symbol}: {e}")
            return symbol, False
    
    # Process in batches to avoid rate limiting
    batch_size = 5
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[fetch_both_themes(s) for s in batch],
            return_exceptions=True,
        )
        for result in batch_results:
            if isinstance(result, tuple):
                results[result[0]] = result[1]
        
        # Small delay between batches to avoid rate limiting
        if i + batch_size < len(symbols):
            await asyncio.sleep(0.5)
    
    return results


async def get_logo_url(symbol: str, theme: LogoTheme = LogoTheme.LIGHT) -> str:
    """
    Get the URL for a logo (for frontend use).
    
    Returns the API endpoint URL that will serve the cached logo.
    """
    return f"/api/logos/{symbol}?theme={theme.value}"
