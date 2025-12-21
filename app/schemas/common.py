"""Common schemas and error responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response schema (RFC 7807 inspired)."""

    error: str = Field(..., description="Error code", examples=["NOT_FOUND"])
    message: str = Field(..., description="Human-readable error message")
    status: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )

    model_config = {"json_schema_extra": {"example": {"error": "NOT_FOUND", "message": "Symbol not found", "status": 404}}}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall health status", examples=["healthy", "degraded", "unhealthy"])
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, bool] = Field(default_factory=dict, description="Individual service health checks")


class PaginationParams(BaseModel):
    """Pagination parameters."""

    limit: int = Field(default=50, ge=1, le=500, description="Number of items to return")
    offset: int = Field(default=0, ge=0, description="Number of items to skip")


class PaginatedResponse(BaseModel):
    """Base paginated response."""

    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")


class MessageResponse(BaseModel):
    """Simple message response."""

    message: str = Field(..., description="Response message")
