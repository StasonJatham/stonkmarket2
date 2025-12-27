"""Flower API client."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger


logger = get_logger("services.flower")


def _parse_basic_auth() -> tuple[str, str] | None:
    """Parse FLOWER_BASIC_AUTH into (username, password) tuple."""
    auth_str = settings.flower_basic_auth
    if not auth_str:
        logger.debug("FLOWER_BASIC_AUTH not configured")
        return None

    auth_str = auth_str.strip()
    if not auth_str or ":" not in auth_str:
        logger.warning("FLOWER_BASIC_AUTH is missing ':' separator or empty")
        return None

    username, password = auth_str.split(":", 1)
    if not username or not password:
        logger.warning("FLOWER_BASIC_AUTH missing username or password")
        return None

    return username, password


async def fetch_flower(path: str, params: dict[str, Any] | None = None) -> Any:
    """Fetch JSON from Flower API."""
    base_url = settings.flower_api_url.rstrip("/")
    url = f"{base_url}/{path.lstrip('/')}"
    auth = _parse_basic_auth()
    
    if auth is None:
        logger.debug("No Flower auth configured, Flower may reject request")

    try:
        async with httpx.AsyncClient(timeout=settings.external_api_timeout) as client:
            response = await client.get(url, params=params, auth=auth)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        logger.warning(f"Flower request failed: {exc}")
        raise ExternalServiceError(message="Flower API unavailable") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            logger.warning("Flower API 401 Unauthorized - check FLOWER_BASIC_AUTH env var")
        else:
            logger.warning(f"Flower API error: {exc.response.status_code}")
        raise ExternalServiceError(
            message="Flower API returned an error",
            details={"status_code": exc.response.status_code},
        ) from exc
