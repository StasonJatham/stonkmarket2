"""API module with routers and dependencies."""

from .app import create_api_app
from .dependencies import (
    get_db,
    require_user,
    require_admin,
    get_current_user,
    get_client_ip,
)

__all__ = [
    "create_api_app",
    "get_db",
    "require_user",
    "require_admin",
    "get_current_user",
    "get_client_ip",
]
