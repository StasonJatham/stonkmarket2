"""Runtime settings service module.

Provides access to runtime settings stored in the database.
Settings are loaded on startup and cached in memory.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.core.logging import get_logger
from app.database.connection import fetch_one, fetch_all, execute

logger = get_logger("runtime_settings")

# Settings that affect specific caches
CACHE_AFFECTING_SETTINGS = {
    "benchmarks": ["ranking", "chart"],  # Changing benchmarks invalidates these caches
    "signal_threshold_strong_buy": ["ranking"],
    "signal_threshold_buy": ["ranking"],
    "signal_threshold_hold": ["ranking"],
}

# Default settings (used if database is empty or unavailable)
DEFAULT_SETTINGS: Dict[str, Any] = {
    "signal_threshold_strong_buy": 80.0,
    "signal_threshold_buy": 60.0,
    "signal_threshold_hold": 40.0,
    "ai_enrichment_enabled": True,
    "ai_batch_size": 0,  # 0 = process all stocks
    "ai_model": "gpt-5-mini",
    "suggestion_cleanup_days": 30,
    "auto_approve_votes": 10,
    "benchmarks": [
        {
            "id": "SP500",
            "symbol": "^GSPC",
            "name": "S&P 500",
            "description": "US Large Cap Index",
        },
        {
            "id": "MSCI_WORLD",
            "symbol": "URTH",
            "name": "MSCI World",
            "description": "Global Developed Markets",
        },
    ],
}

# In-memory cache of settings
_settings_cache: Dict[str, Any] = {}
_cache_initialized: bool = False


def _parse_jsonb_value(value: Any) -> Any:
    """Parse a JSONB value that may be returned as a string by asyncpg."""
    if value is None:
        return None
    # If it's already the right type, return as-is
    if isinstance(value, (dict, list, bool, int, float)):
        return value
    # If it's a string, try to parse as JSON
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


async def _load_settings_from_db() -> Dict[str, Any]:
    """Load all settings from the database."""
    settings = DEFAULT_SETTINGS.copy()
    try:
        rows = await fetch_all("SELECT key, value FROM runtime_settings")
        for row in rows:
            key = row["key"]
            # Parse JSONB value (asyncpg may return as string)
            value = _parse_jsonb_value(row["value"])
            settings[key] = value
        logger.debug(f"Loaded {len(rows)} settings from database")
    except Exception as e:
        logger.warning(f"Failed to load settings from database, using defaults: {e}")
    return settings


async def _save_setting_to_db(key: str, value: Any) -> None:
    """Save a single setting to the database."""
    try:
        # Convert Python value to JSON for storage
        json_value = json.dumps(value)
        await execute(
            """
            INSERT INTO runtime_settings (key, value, updated_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = $2::jsonb,
                updated_at = NOW()
            """,
            key,
            json_value,
        )
        logger.debug(f"Saved setting {key} to database")
    except Exception as e:
        logger.error(f"Failed to save setting {key} to database: {e}")


async def init_runtime_settings() -> None:
    """Initialize runtime settings from database on startup."""
    global _settings_cache, _cache_initialized
    _settings_cache = await _load_settings_from_db()
    _cache_initialized = True
    logger.info(f"Initialized {len(_settings_cache)} runtime settings")


def get_runtime_setting(key: str, default: Any = None) -> Any:
    """Get a runtime setting by key.
    
    Args:
        key: The setting key to retrieve
        default: Default value if key not found
        
    Returns:
        The setting value or default
    """
    if not _cache_initialized:
        # Return from defaults if cache not yet initialized
        return DEFAULT_SETTINGS.get(key, default)
    return _settings_cache.get(key, default)


def set_runtime_setting(key: str, value: Any) -> None:
    """Set a runtime setting (memory only - call save_runtime_settings to persist).
    
    Args:
        key: The setting key to set
        value: The value to set
    """
    _settings_cache[key] = value


async def save_runtime_setting(key: str, value: Any) -> None:
    """Set and persist a runtime setting.
    
    Args:
        key: The setting key to set
        value: The value to set
    """
    _settings_cache[key] = value
    await _save_setting_to_db(key, value)


async def update_runtime_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update multiple runtime settings and persist to database.
    
    Also invalidates affected caches when settings change.
    
    Args:
        updates: Dictionary of settings to update
        
    Returns:
        The updated settings dictionary
    """
    from app.cache.cache import Cache
    
    # Collect caches to invalidate
    caches_to_invalidate: set[str] = set()
    
    for key, value in updates.items():
        _settings_cache[key] = value
        await _save_setting_to_db(key, value)
        
        # Check if this setting affects any caches
        if key in CACHE_AFFECTING_SETTINGS:
            caches_to_invalidate.update(CACHE_AFFECTING_SETTINGS[key])
    
    # Invalidate affected caches
    for cache_prefix in caches_to_invalidate:
        try:
            cache = Cache(prefix=cache_prefix)
            deleted = await cache.invalidate_pattern("*")
            logger.info(f"Invalidated {deleted} keys in cache '{cache_prefix}' due to settings change")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache '{cache_prefix}': {e}")
    
    logger.info(f"Updated {len(updates)} runtime settings")
    return _settings_cache.copy()


def get_all_runtime_settings() -> Dict[str, Any]:
    """Get all runtime settings.
    
    Returns:
        Copy of all runtime settings
    """
    if not _cache_initialized:
        return DEFAULT_SETTINGS.copy()
    return _settings_cache.copy()


async def check_openai_configured() -> bool:
    """Check if OpenAI API key is configured (env or database)."""
    import os
    from app.repositories import api_keys as api_keys_repo
    
    # Check environment variable first
    if os.environ.get("OPENAI_API_KEY"):
        return True
    
    # Check database
    try:
        key = await api_keys_repo.get_key(api_keys_repo.OPENAI_API_KEY)
        return key is not None and key.get("encrypted_key")
    except Exception:
        return False
