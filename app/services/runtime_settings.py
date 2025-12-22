"""Runtime settings service module.

Provides access to runtime settings stored in memory.
In a production environment, these would be persisted to the database.
"""

from typing import Any, Dict

# Runtime settings store - imported and used by admin_settings route
_runtime_settings: Dict[str, Any] = {
    "signal_threshold_strong_buy": 80.0,
    "signal_threshold_buy": 60.0,
    "signal_threshold_hold": 40.0,
    "ai_enrichment_enabled": False,
    "ai_batch_size": 5,
    "ai_model": "gpt-4o-mini",
    "suggestion_cleanup_days": 30,
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


def get_runtime_setting(key: str, default: Any = None) -> Any:
    """Get a runtime setting by key.
    
    Args:
        key: The setting key to retrieve
        default: Default value if key not found
        
    Returns:
        The setting value or default
    """
    return _runtime_settings.get(key, default)


def set_runtime_setting(key: str, value: Any) -> None:
    """Set a runtime setting.
    
    Args:
        key: The setting key to set
        value: The value to set
    """
    _runtime_settings[key] = value


def update_runtime_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update multiple runtime settings.
    
    Args:
        updates: Dictionary of settings to update
        
    Returns:
        The updated settings dictionary
    """
    _runtime_settings.update(updates)
    return _runtime_settings.copy()


def get_all_runtime_settings() -> Dict[str, Any]:
    """Get all runtime settings.
    
    Returns:
        Copy of all runtime settings
    """
    return _runtime_settings.copy()
