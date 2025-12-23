"""Web Vitals metrics collection endpoint."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["Metrics"])


class WebVitalMetric(BaseModel):
    """Individual Web Vital metric."""

    name: Literal["LCP", "INP", "CLS", "FCP", "TTFB"]
    value: float
    rating: Literal["good", "needs-improvement", "poor"]
    delta: float
    id: str
    navigationType: str = "navigate"


class WebVitalsPayload(BaseModel):
    """Payload from frontend Web Vitals reporter."""

    metrics: list[WebVitalMetric] = Field(default_factory=list)
    url: str
    userAgent: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# In-memory storage for development (use Redis/DB in production)
_vitals_buffer: list[dict] = []
_MAX_BUFFER_SIZE = 1000


@router.post(
    "/vitals",
    summary="Collect Web Vitals metrics",
    description="Receives Core Web Vitals metrics from the frontend for aggregation.",
)
async def collect_vitals(payload: WebVitalsPayload) -> dict:
    """Collect Web Vitals metrics from frontend."""
    global _vitals_buffer

    for metric in payload.metrics:
        entry = {
            "name": metric.name,
            "value": metric.value,
            "rating": metric.rating,
            "url": payload.url,
            "timestamp": payload.timestamp,
        }
        _vitals_buffer.append(entry)

        # Log poor metrics for alerting
        if metric.rating == "poor":
            logger.warning(
                f"Poor Web Vital: {metric.name}={metric.value:.2f} on {payload.url}"
            )

    # Trim buffer to prevent memory issues
    if len(_vitals_buffer) > _MAX_BUFFER_SIZE:
        _vitals_buffer = _vitals_buffer[-_MAX_BUFFER_SIZE:]

    return {"status": "ok", "collected": len(payload.metrics)}


@router.get(
    "/vitals/summary",
    summary="Get Web Vitals summary",
    description="Returns aggregated Web Vitals statistics.",
)
async def get_vitals_summary() -> dict:
    """Get aggregated Web Vitals summary."""
    if not _vitals_buffer:
        return {"status": "no_data", "metrics": {}}

    # Aggregate by metric name
    aggregated: dict[str, dict] = {}

    for entry in _vitals_buffer:
        name = entry["name"]
        if name not in aggregated:
            aggregated[name] = {
                "values": [],
                "ratings": {"good": 0, "needs-improvement": 0, "poor": 0},
            }

        aggregated[name]["values"].append(entry["value"])
        aggregated[name]["ratings"][entry["rating"]] += 1

    # Calculate statistics
    result = {}
    for name, data in aggregated.items():
        values = data["values"]
        ratings = data["ratings"]
        total = len(values)

        result[name] = {
            "count": total,
            "p50": sorted(values)[total // 2] if values else 0,
            "p75": sorted(values)[int(total * 0.75)] if values else 0,
            "p95": sorted(values)[int(total * 0.95)] if values else 0,
            "avg": sum(values) / total if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "good_pct": (ratings["good"] / total * 100) if total else 0,
            "poor_pct": (ratings["poor"] / total * 100) if total else 0,
        }

    return {
        "status": "ok",
        "sample_size": len(_vitals_buffer),
        "metrics": result,
    }


@router.delete(
    "/vitals",
    summary="Clear Web Vitals buffer",
    description="Clears the in-memory Web Vitals buffer.",
)
async def clear_vitals() -> dict:
    """Clear the Web Vitals buffer."""
    global _vitals_buffer
    count = len(_vitals_buffer)
    _vitals_buffer = []
    return {"status": "ok", "cleared": count}
