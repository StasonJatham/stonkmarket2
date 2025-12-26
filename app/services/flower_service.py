"""Flower API client."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger("services.flower")


def _parse_basic_auth() -> Optional[Tuple[str, str]]:
    if not settings.flower_basic_auth:
        return None

    if ":" not in settings.flower_basic_auth:
        logger.warning("FLOWER_BASIC_AUTH is missing ':' separator")
        return None

    username, password = settings.flower_basic_auth.split(":", 1)
    if not username or not password:
        logger.warning("FLOWER_BASIC_AUTH missing username or password")
        return None

    return username, password


async def fetch_flower(path: str, params: Optional[dict[str, Any]] = None) -> Any:
    """Fetch JSON from Flower API."""
    base_url = settings.flower_api_url.rstrip("/")
    url = f"{base_url}/{path.lstrip('/')}"
    auth = _parse_basic_auth()

    try:
        async with httpx.AsyncClient(timeout=settings.external_api_timeout) as client:
            response = await client.get(url, params=params, auth=auth)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        logger.warning(f"Flower request failed: {exc}")
        raise ExternalServiceError(message="Flower API unavailable") from exc
    except httpx.HTTPStatusError as exc:
        logger.warning(f"Flower API error: {exc.response.status_code}")
        raise ExternalServiceError(
            message="Flower API returned an error",
            details={"status_code": exc.response.status_code},
        ) from exc
