"""Admin settings API routes - PostgreSQL async."""

from __future__ import annotations

import os
from typing import Dict, Any

from fastapi import APIRouter, Depends

from app.api.dependencies import require_admin
from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import TokenData
from app.database.connection import fetch_one, fetch_all
from app.schemas.admin_settings import (
    AppSettingsResponse,
    RuntimeSettingsResponse,
    RuntimeSettingsUpdate,
    CronJobSummary,
    SystemStatusResponse,
)
from app.services.runtime_settings import (
    get_all_runtime_settings,
    update_runtime_settings as update_settings_async,
    check_openai_configured,
)

logger = get_logger("api.admin_settings")

router = APIRouter()


@router.get(
    "/app",
    response_model=AppSettingsResponse,
    summary="Get application settings",
    description="Get read-only application settings from configuration.",
)
async def get_app_settings(
    user: TokenData = Depends(require_admin),
) -> AppSettingsResponse:
    """Get application settings (read-only from config)."""
    return AppSettingsResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
        default_min_dip_pct=settings.default_min_dip_pct,
        default_min_days=settings.default_min_days,
        history_days=settings.history_days,
        chart_days=settings.chart_days,
        vote_cooldown_days=settings.vote_cooldown_days,
        auto_approve_enabled=settings.auto_approve_enabled,
        auto_approve_votes=settings.auto_approve_votes,
        auto_approve_unique_voters=settings.auto_approve_unique_voters,
        auto_approve_min_age_hours=settings.auto_approve_min_age_hours,
        rate_limit_enabled=settings.rate_limit_enabled,
        rate_limit_auth=settings.rate_limit_auth,
        rate_limit_api_anonymous=settings.rate_limit_api_anonymous,
        rate_limit_api_authenticated=settings.rate_limit_api_authenticated,
        scheduler_enabled=settings.scheduler_enabled,
        scheduler_timezone=settings.scheduler_timezone,
        external_api_timeout=settings.external_api_timeout,
        external_api_retries=settings.external_api_retries,
    )


@router.get(
    "/runtime",
    response_model=RuntimeSettingsResponse,
    summary="Get runtime settings",
    description="Get runtime settings that can be modified.",
)
async def get_runtime_settings(
    user: TokenData = Depends(require_admin),
) -> RuntimeSettingsResponse:
    """Get runtime settings."""
    return RuntimeSettingsResponse(**get_all_runtime_settings())


@router.patch(
    "/runtime",
    response_model=RuntimeSettingsResponse,
    summary="Update runtime settings",
    description="Update runtime settings.",
)
async def update_runtime_settings(
    updates: RuntimeSettingsUpdate,
    user: TokenData = Depends(require_admin),
) -> RuntimeSettingsResponse:
    """Update runtime settings."""
    update_dict = updates.model_dump(exclude_none=True)
    
    # If trying to enable AI, check if OpenAI is configured
    if update_dict.get("ai_enrichment_enabled") is True:
        if not await check_openai_configured():
            from app.core.exceptions import ValidationError
            raise ValidationError(
                "Cannot enable AI enrichment: OpenAI API key is not configured. "
                "Set OPENAI_API_KEY environment variable or add via API Keys settings."
            )
    
    updated = await update_settings_async(update_dict)
    logger.info(f"Runtime settings updated by {user.sub}: {update_dict}")
    return RuntimeSettingsResponse(**updated)


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get system status",
    description="Get complete system status including settings and service health.",
)
async def get_system_status(
    user: TokenData = Depends(require_admin),
) -> SystemStatusResponse:
    """Get complete system status."""
    # Get cronjobs
    cronjobs = []
    try:
        rows = await fetch_all(
            """
            SELECT name, cron, description, last_run, last_status
            FROM cronjobs
            ORDER BY name
            """
        )
        for row in rows:
            cronjobs.append(
                CronJobSummary(
                    name=row["name"],
                    cron=row["cron"],
                    description=row["description"],
                    last_run=row["last_run"],
                    last_status=row["last_status"],
                )
            )
    except Exception as e:
        logger.warning(f"Failed to fetch cronjobs: {e}")

    # Check OpenAI configuration
    openai_configured = await check_openai_configured()

    # Get symbol count
    total_symbols = 0
    try:
        row = await fetch_one("SELECT COUNT(*) as count FROM symbols")
        if row:
            total_symbols = row["count"]
    except Exception as e:
        logger.warning(f"Failed to count symbols: {e}")

    # Get pending suggestions count
    pending_suggestions = 0
    try:
        row = await fetch_one(
            "SELECT COUNT(*) as count FROM stock_suggestions WHERE status = 'pending'"
        )
        if row:
            pending_suggestions = row["count"]
    except Exception as e:
        logger.warning(f"Failed to count suggestions: {e}")

    return SystemStatusResponse(
        app_settings=await get_app_settings(user),
        runtime_settings=RuntimeSettingsResponse(**get_all_runtime_settings()),
        cronjobs=cronjobs,
        openai_configured=openai_configured,
        total_symbols=total_symbols,
        pending_suggestions=pending_suggestions,
    )


@router.get(
    "/openai-status",
    summary="Check OpenAI configuration status",
    description="Check if OpenAI API key is configured.",
)
async def get_openai_status(
    user: TokenData = Depends(require_admin),
) -> dict:
    """Check if OpenAI is configured."""
    return {"configured": await check_openai_configured()}


@router.get(
    "/suspicious-votes",
    summary="Get suspicious vote log",
    description="View recent votes flagged as potentially fraudulent.",
)
async def get_suspicious_votes(
    limit: int = 100,
    user: TokenData = Depends(require_admin),
) -> dict:
    """Get suspicious vote log for admin review."""
    from app.core.client_identity import get_suspicious_log
    
    entries = await get_suspicious_log(limit)
    return {
        "entries": entries,
        "count": len(entries),
    }
