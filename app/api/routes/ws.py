"""WebSocket routes for real-time updates."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from app.core.logging import get_logger
from app.websocket import get_connection_manager, WSEvent, WSEventType

logger = get_logger("api.routes.ws")

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time updates.
    
    Connect with JWT token as query param: ws://host/api/ws?token=<jwt>
    
    Events you'll receive:
    - fetch_started/progress/complete: Data fetch progress
    - cronjob_started/progress/complete: Cronjob execution updates
    - suggestion_new/approved/rejected: Stock suggestion updates (admin only)
    """
    manager = get_connection_manager()
    
    # Authenticate
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return
    
    user = await manager.authenticate(websocket, token)
    if not user:
        await websocket.close(code=4003, reason="Invalid or expired token")
        return
    
    # Connect and register
    conn = await manager.connect(websocket, user)
    
    try:
        while True:
            # Wait for messages (ping/pong, subscription changes)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0  # 1 minute timeout for ping
                )
                
                msg_type = data.get("type", "")
                
                if msg_type == "ping":
                    await conn.send_event(WSEvent(type=WSEventType.PONG))
                
                elif msg_type == "subscribe":
                    # Subscribe to specific channels
                    channels = data.get("channels", [])
                    if isinstance(channels, list):
                        conn.subscriptions.update(channels)
                        logger.debug(f"{user.username} subscribed to: {channels}")
                
                elif msg_type == "unsubscribe":
                    channels = data.get("channels", [])
                    if isinstance(channels, list):
                        conn.subscriptions.difference_update(channels)
                        logger.debug(f"{user.username} unsubscribed from: {channels}")
                        
            except asyncio.TimeoutError:
                # Send ping to check if client is still alive
                try:
                    await conn.send_event(WSEvent(type=WSEventType.PING))
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        logger.debug(f"Client disconnected: {user.username}")
    except Exception as e:
        logger.error(f"WebSocket error for {user.username}: {e}")
        await conn.send_event(WSEvent(
            type=WSEventType.ERROR,
            message=str(e)
        ))
    finally:
        await manager.disconnect(websocket)


@router.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics (admin endpoint)."""
    manager = get_connection_manager()
    return {
        "total_connections": manager.connection_count,
        "admin_connections": manager.admin_connection_count,
    }
