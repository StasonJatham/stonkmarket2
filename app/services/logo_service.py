"""
Logo service using Logo.dev API.

Fetches and caches company logos in WebP format for both light and dark themes.
Falls back to favicon if Logo.dev fails.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple
from enum import Enum

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.database.connection import fetch_one, execute
from app.repositories import api_keys as api_keys_repo

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
    timeout: float = 10.0,
) -> Optional[bytes]:
    """
    Fetch logo from Logo.dev ticker API.
    
    Args:
        symbol: Stock ticker symbol
        theme: Light or dark theme
        timeout: Request timeout in seconds
        
    Returns:
        WebP image bytes or None if failed
    """
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
    timeout: float = 5.0,
) -> Optional[bytes]:
    """
    Fetch favicon as fallback using Google's favicon service.
    
    Args:
        website: Company website URL (from yfinance)
        symbol: Stock symbol (for logging only)
        timeout: Request timeout
        
    Returns:
        Image bytes or None
    """
    # Extract domain from website URL - no hardcoded fallback mappings
    domain = None
    if website:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(website)
            domain = parsed.netloc or parsed.path.split('/')[0]
        except Exception:
            pass
    
    # No domain means no website available from yfinance - can't fetch favicon
    if not domain:
        logger.debug(f"No website available for {symbol}, skipping favicon fallback")
        return None
    
    # Google favicon service
    url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code == 200:
                logger.debug(f"Fetched favicon for {symbol} from {domain}")
                return response.content
    except Exception as e:
        logger.debug(f"Favicon fallback failed for {symbol}: {e}")
    
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
    cache_column = "logo_light" if theme == LogoTheme.LIGHT else "logo_dark"
    
    row = await fetch_one(
        f"""
        SELECT {cache_column}, logo_fetched_at, logo_source
        FROM symbols 
        WHERE symbol = $1
        """,
        symbol,
    )
    
    # Return cached logo if still valid
    if row and row[cache_column]:
        fetched_at = row.get("logo_fetched_at")
        if fetched_at:
            age = datetime.utcnow() - fetched_at.replace(tzinfo=None)
            if age < timedelta(days=settings.logo_cache_days):
                return bytes(row[cache_column])
    
    # Fetch fresh logo
    logo_data = await _fetch_logo_from_api(symbol, theme)
    source = "logo.dev"
    
    # Fallback to favicon if Logo.dev failed
    if not logo_data:
        # Get website from symbols table if not provided
        if not website and row:
            website_row = await fetch_one(
                "SELECT website FROM stock_suggestions WHERE symbol = $1",
                symbol,
            )
            if website_row:
                website = website_row.get("website")
        
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
    cache_column = "logo_light" if theme == LogoTheme.LIGHT else "logo_dark"
    
    try:
        await execute(
            f"""
            UPDATE symbols 
            SET {cache_column} = $2,
                logo_fetched_at = NOW(),
                logo_source = $3
            WHERE symbol = $1
            """,
            symbol.upper(),
            data,
            source,
        )
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
