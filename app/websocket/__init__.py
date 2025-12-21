"""WebSocket module for real-time updates."""

from .manager import ConnectionManager, get_connection_manager
from .events import WSEventType, WSEvent

__all__ = [
    "ConnectionManager",
    "get_connection_manager",
    "WSEventType",
    "WSEvent",
]
