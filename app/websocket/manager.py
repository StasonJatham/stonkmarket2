"""WebSocket connection manager with authentication."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from weakref import WeakSet

from fastapi import WebSocket, WebSocketDisconnect

from app.core.logging import get_logger
from app.core.security import decode_access_token, TokenData
from .events import WSEvent, WSEventType

logger = get_logger("websocket.manager")


@dataclass
class AuthenticatedConnection:
    """A WebSocket connection with associated user info."""
    
    websocket: WebSocket
    user: TokenData
    connected_at: datetime = field(default_factory=datetime.utcnow)
    subscriptions: set[str] = field(default_factory=set)
    
    async def send_event(self, event: WSEvent) -> bool:
        """Send an event to this connection."""
        try:
            await self.websocket.send_json(event.model_dump(mode="json"))
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {self.user.username}: {e}")
            return False


class ConnectionManager:
    """Manages authenticated WebSocket connections."""
    
    def __init__(self):
        self._connections: dict[str, AuthenticatedConnection] = {}
        self._lock = asyncio.Lock()
        self._admin_connections: WeakSet[AuthenticatedConnection] = WeakSet()
    
    async def authenticate(self, websocket: WebSocket, token: str) -> Optional[TokenData]:
        """Authenticate a WebSocket connection using JWT token."""
        try:
            user = decode_access_token(token)
            if user:
                logger.debug(f"WebSocket authenticated: {user.username}")
                return user
        except Exception as e:
            logger.warning(f"WebSocket auth failed: {e}")
        return None
    
    async def connect(
        self,
        websocket: WebSocket,
        user: TokenData,
        subscriptions: Optional[set[str]] = None,
    ) -> AuthenticatedConnection:
        """Accept and register a new connection."""
        await websocket.accept()
        
        conn = AuthenticatedConnection(
            websocket=websocket,
            user=user,
            subscriptions=subscriptions or {"all"},
        )
        
        async with self._lock:
            # Use connection ID (id of websocket object)
            conn_id = str(id(websocket))
            self._connections[conn_id] = conn
            
            # Track admin connections separately for priority broadcasting
            if user.is_admin:
                self._admin_connections.add(conn)
        
        logger.info(f"WebSocket connected: {user.username} (admin={user.is_admin})")
        
        # Send connected event
        await conn.send_event(WSEvent(
            type=WSEventType.CONNECTED,
            message=f"Connected as {user.username}",
            data={"username": user.username, "is_admin": user.is_admin},
        ))
        
        return conn
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a connection."""
        conn_id = str(id(websocket))
        
        async with self._lock:
            conn = self._connections.pop(conn_id, None)
            if conn:
                self._admin_connections.discard(conn)
                logger.info(f"WebSocket disconnected: {conn.user.username}")
    
    async def broadcast(
        self,
        event: WSEvent,
        subscription: Optional[str] = None,
        admin_only: bool = False,
    ) -> int:
        """Broadcast an event to all matching connections.
        
        Args:
            event: The event to broadcast
            subscription: Only send to connections subscribed to this channel
            admin_only: Only send to admin connections
            
        Returns:
            Number of connections that received the event
        """
        async with self._lock:
            connections = list(self._connections.values())
        
        if admin_only:
            connections = [c for c in connections if c.user.is_admin]
        
        if subscription:
            connections = [
                c for c in connections
                if subscription in c.subscriptions or "all" in c.subscriptions
            ]
        
        # Send to all matching connections concurrently
        results = await asyncio.gather(
            *[conn.send_event(event) for conn in connections],
            return_exceptions=True
        )
        
        sent_count = sum(1 for r in results if r is True)
        logger.debug(f"Broadcast {event.type}: sent to {sent_count}/{len(connections)}")
        
        return sent_count
    
    async def send_to_user(self, username: str, event: WSEvent) -> bool:
        """Send an event to a specific user."""
        async with self._lock:
            for conn in self._connections.values():
                if conn.user.username == username:
                    return await conn.send_event(event)
        return False
    
    async def broadcast_to_admins(self, event: WSEvent) -> int:
        """Convenience method to broadcast only to admins."""
        return await self.broadcast(event, admin_only=True)
    
    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self._connections)
    
    @property  
    def admin_connection_count(self) -> int:
        """Number of active admin connections."""
        return len(self._admin_connections)


# Global connection manager instance
_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
