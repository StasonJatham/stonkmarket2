"""
Data version tracking for cache invalidation.

When backend data changes (prices refreshed, dips recalculated, etc.),
the version is incremented. Frontend can check this version and invalidate
stale caches automatically.
"""

from __future__ import annotations

import time
from typing import Final

from app.core.logging import get_logger

from .client import get_valkey_client
from .http_cache import set_data_version


logger = get_logger("data_version")

# Redis key for storing the global data version
DATA_VERSION_KEY: Final[str] = "stonkmarket:data_version"


async def get_data_version() -> int:
    """Get current data version timestamp (Unix milliseconds)."""
    client = await get_valkey_client()
    version = await client.get(DATA_VERSION_KEY)
    if version is None:
        # Initialize with current timestamp
        version = int(time.time() * 1000)
        await client.set(DATA_VERSION_KEY, str(version))
    else:
        version = int(version)
    
    # Update the sync version for response headers
    set_data_version(str(version))
    return version


async def bump_data_version() -> int:
    """
    Bump data version to current timestamp.
    
    Call this after:
    - Price data refresh
    - Dip recalculation
    - Symbol additions/removals
    - Any bulk data update
    
    Returns the new version.
    """
    version = int(time.time() * 1000)
    client = await get_valkey_client()
    await client.set(DATA_VERSION_KEY, str(version))
    
    # Update the sync version for response headers
    set_data_version(str(version))
    
    logger.info(f"Data version bumped to {version}")
    return version


async def init_data_version() -> None:
    """Initialize data version on app startup."""
    await get_data_version()


async def get_data_version_header() -> dict[str, str]:
    """Get data version as a header dict for responses."""
    version = await get_data_version()
    return {"X-Data-Version": str(version)}
