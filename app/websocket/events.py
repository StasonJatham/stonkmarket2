"""WebSocket event types and models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class WSEventType(str, Enum):
    """WebSocket event types."""

    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

    # Data fetch events (yfinance)
    FETCH_STARTED = "fetch_started"
    FETCH_PROGRESS = "fetch_progress"
    FETCH_SYMBOL_START = "fetch_symbol_start"
    FETCH_SYMBOL_COMPLETE = "fetch_symbol_complete"
    FETCH_SYMBOL_ERROR = "fetch_symbol_error"
    FETCH_COMPLETE = "fetch_complete"

    # Cronjob events
    CRONJOB_STARTED = "cronjob_started"
    CRONJOB_PROGRESS = "cronjob_progress"
    CRONJOB_COMPLETE = "cronjob_complete"
    CRONJOB_ERROR = "cronjob_error"

    # Suggestion events
    SUGGESTION_NEW = "suggestion_new"
    SUGGESTION_APPROVED = "suggestion_approved"
    SUGGESTION_REJECTED = "suggestion_rejected"
    SUGGESTION_VOTE = "suggestion_vote"


class WSEvent(BaseModel):
    """WebSocket event payload."""

    type: WSEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Optional[dict[str, Any]] = None
    message: Optional[str] = None

    model_config = {"use_enum_values": True}


class FetchProgressData(BaseModel):
    """Data for fetch progress events."""

    job_id: str
    total_symbols: int
    completed: int
    current_symbol: Optional[str] = None
    success_count: int = 0
    error_count: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)


class CronjobProgressData(BaseModel):
    """Data for cronjob progress events."""

    job_name: str
    status: str  # running, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    result: Optional[str] = None
    error: Optional[str] = None
