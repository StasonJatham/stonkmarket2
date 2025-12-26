"""API module with routers and dependencies."""

from .app import create_api_app
from .dependencies import (
    get_client_ip,
    get_current_user,
    require_admin,
    require_user,
)


__all__ = [
    "create_api_app",
    "get_client_ip",
    "get_current_user",
    "require_admin",
    "require_user",
]
