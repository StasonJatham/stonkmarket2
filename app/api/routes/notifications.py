"""Notification management routes.

Provides endpoints for managing notification channels, rules, and viewing history.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import TokenData
from app.repositories import auth_user_orm as auth_repo
from app.repositories import notifications_orm as notifications_repo
from app.schemas.notifications import (
    ChannelCreate,
    ChannelResponse,
    ChannelTestResponse,
    ChannelUpdate,
    ComparisonOperator,
    NotificationHistoryResponse,
    NotificationLogEntry,
    NotificationSummary,
    RuleCreate,
    RuleResponse,
    RuleTestResponse,
    RuleUpdate,
    TriggerType,
    TRIGGER_TYPE_INFO,
)


router = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _get_user_id(user: TokenData) -> int:
    """Get the database user ID from token data."""
    db_user = await auth_repo.get_user(user.sub)
    if not db_user:
        raise NotFoundError("User not found")
    return db_user.id


# =============================================================================
# CHANNEL ENDPOINTS
# =============================================================================


@router.get(
    "/channels",
    response_model=list[ChannelResponse],
    summary="List notification channels",
    description="Get all notification channels for the current user.",
)
async def list_channels(
    user: TokenData = Depends(require_user),
    active_only: bool = Query(False, description="Only return active channels"),
) -> list[ChannelResponse]:
    """List all channels for the authenticated user."""
    user_id = await _get_user_id(user)
    channels = await notifications_repo.list_channels(
        user_id=user_id,
        active_only=active_only,
        include_rules=True,
    )
    
    return [
        ChannelResponse(
            id=ch["id"],
            name=ch["name"],
            channel_type=ch["channel_type"],
            is_verified=ch["is_verified"],
            last_verified_at=ch.get("last_verified_at"),
            is_active=ch["is_active"],
            error_count=ch["error_count"],
            last_error=ch.get("last_error"),
            last_used_at=ch.get("last_used_at"),
            rule_count=ch.get("rule_count", 0),
            created_at=ch["created_at"],
            updated_at=ch["updated_at"],
        )
        for ch in channels
    ]


@router.post(
    "/channels",
    response_model=ChannelResponse,
    summary="Create notification channel",
    description="Create a new notification channel (Discord, Telegram, etc.).",
)
async def create_channel(
    payload: ChannelCreate,
    user: TokenData = Depends(require_user),
) -> ChannelResponse:
    """Create a new notification channel."""
    from app.services.notifications.sender import validate_apprise_url
    
    user_id = await _get_user_id(user)
    
    # Validate the Apprise URL
    is_valid, detected_type, error = validate_apprise_url(payload.apprise_url)
    if not is_valid:
        raise ValidationError(f"Invalid Apprise URL: {error}")
    
    # Use detected type if not provided
    channel_type = payload.channel_type or detected_type or "other"
    
    channel = await notifications_repo.create_channel(
        user_id=user_id,
        name=payload.name,
        channel_type=channel_type,
        apprise_url=payload.apprise_url,
        is_active=True,
    )
    
    return ChannelResponse(
        id=channel["id"],
        name=channel["name"],
        channel_type=channel["channel_type"],
        is_verified=channel["is_verified"],
        last_verified_at=channel.get("last_verified_at"),
        is_active=channel["is_active"],
        error_count=channel["error_count"],
        last_error=channel.get("last_error"),
        last_used_at=channel.get("last_used_at"),
        rule_count=0,
        created_at=channel["created_at"],
        updated_at=channel["updated_at"],
    )


@router.get(
    "/channels/{channel_id}",
    response_model=ChannelResponse,
    summary="Get channel details",
    description="Get details of a specific notification channel.",
)
async def get_channel(
    channel_id: int,
    user: TokenData = Depends(require_user),
) -> ChannelResponse:
    """Get a specific channel."""
    user_id = await _get_user_id(user)
    channel = await notifications_repo.get_channel(channel_id, user_id)
    
    if not channel:
        raise NotFoundError(f"Channel {channel_id} not found")
    
    return ChannelResponse(
        id=channel["id"],
        name=channel["name"],
        channel_type=channel["channel_type"],
        is_verified=channel["is_verified"],
        last_verified_at=channel.get("last_verified_at"),
        is_active=channel["is_active"],
        error_count=channel["error_count"],
        last_error=channel.get("last_error"),
        last_used_at=channel.get("last_used_at"),
        rule_count=0,
        created_at=channel["created_at"],
        updated_at=channel["updated_at"],
    )


@router.patch(
    "/channels/{channel_id}",
    response_model=ChannelResponse,
    summary="Update channel",
    description="Update a notification channel's settings.",
)
async def update_channel(
    channel_id: int,
    payload: ChannelUpdate,
    user: TokenData = Depends(require_user),
) -> ChannelResponse:
    """Update a channel."""
    from app.services.notifications.sender import validate_apprise_url
    
    user_id = await _get_user_id(user)
    
    # Validate new URL if provided
    if payload.apprise_url:
        is_valid, _, error = validate_apprise_url(payload.apprise_url)
        if not is_valid:
            raise ValidationError(f"Invalid Apprise URL: {error}")
    
    channel = await notifications_repo.update_channel(
        channel_id=channel_id,
        user_id=user_id,
        name=payload.name,
        channel_type=payload.channel_type,
        apprise_url=payload.apprise_url,
        is_active=payload.is_active,
    )
    
    if not channel:
        raise NotFoundError(f"Channel {channel_id} not found")
    
    return ChannelResponse(
        id=channel["id"],
        name=channel["name"],
        channel_type=channel["channel_type"],
        is_verified=channel["is_verified"],
        last_verified_at=channel.get("last_verified_at"),
        is_active=channel["is_active"],
        error_count=channel["error_count"],
        last_error=channel.get("last_error"),
        last_used_at=channel.get("last_used_at"),
        rule_count=0,
        created_at=channel["created_at"],
        updated_at=channel["updated_at"],
    )


@router.delete(
    "/channels/{channel_id}",
    summary="Delete channel",
    description="Delete a notification channel and all associated rules.",
)
async def delete_channel(
    channel_id: int,
    user: TokenData = Depends(require_user),
) -> dict:
    """Delete a channel."""
    user_id = await _get_user_id(user)
    deleted = await notifications_repo.delete_channel(channel_id, user_id)
    
    if not deleted:
        raise NotFoundError(f"Channel {channel_id} not found")
    
    return {"deleted": True, "channel_id": channel_id}


@router.post(
    "/channels/{channel_id}/test",
    response_model=ChannelTestResponse,
    summary="Test channel",
    description="Send a test notification to verify channel configuration.",
)
async def test_channel(
    channel_id: int,
    user: TokenData = Depends(require_user),
) -> ChannelTestResponse:
    """Send a test notification to a channel."""
    from app.services.notifications.sender import test_channel as do_test
    
    user_id = await _get_user_id(user)
    result = await do_test(channel_id, user_id)
    
    return ChannelTestResponse(
        success=result["success"],
        message=result["message"],
        verified_at=result.get("verified_at"),
    )


# =============================================================================
# RULE ENDPOINTS
# =============================================================================


@router.get(
    "/rules",
    response_model=list[RuleResponse],
    summary="List notification rules",
    description="Get all notification rules for the current user.",
)
async def list_rules(
    user: TokenData = Depends(require_user),
    active_only: bool = Query(False, description="Only return active rules"),
    channel_id: int | None = Query(None, description="Filter by channel"),
    trigger_type: str | None = Query(None, description="Filter by trigger type"),
) -> list[RuleResponse]:
    """List all rules for the authenticated user."""
    user_id = await _get_user_id(user)
    rules = await notifications_repo.list_rules(
        user_id=user_id,
        active_only=active_only,
        channel_id=channel_id,
        trigger_type=trigger_type,
    )
    
    return [
        RuleResponse(
            id=r["id"],
            channel_id=r["channel_id"],
            name=r["name"],
            trigger_type=r["trigger_type"],
            comparison_operator=r["comparison_operator"],
            target_value=r.get("target_value"),
            target_symbol=r.get("target_symbol"),
            target_portfolio_id=r.get("target_portfolio_id"),
            smart_payload=r.get("smart_payload"),
            cooldown_minutes=r["cooldown_minutes"],
            last_triggered_at=r.get("last_triggered_at"),
            trigger_count=r["trigger_count"],
            is_active=r["is_active"],
            priority=r["priority"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rules
    ]


@router.post(
    "/rules",
    response_model=RuleResponse,
    summary="Create notification rule",
    description="Create a new notification rule with trigger conditions.",
)
async def create_rule(
    payload: RuleCreate,
    user: TokenData = Depends(require_user),
) -> RuleResponse:
    """Create a new notification rule."""
    from decimal import Decimal
    
    user_id = await _get_user_id(user)
    
    # Validate channel exists
    channel = await notifications_repo.get_channel(payload.channel_id, user_id)
    if not channel:
        raise NotFoundError(f"Channel {payload.channel_id} not found")
    
    # Validate trigger type
    try:
        TriggerType(payload.trigger_type)
    except ValueError:
        raise ValidationError(f"Invalid trigger type: {payload.trigger_type}")
    
    # Validate comparison operator
    try:
        ComparisonOperator(payload.comparison_operator)
    except ValueError:
        raise ValidationError(f"Invalid comparison operator: {payload.comparison_operator}")
    
    # Get trigger info for validation
    trigger_info = TRIGGER_TYPE_INFO.get(payload.trigger_type, {})
    
    # Validate symbol or portfolio is provided as needed
    if trigger_info.get("requires_symbol") and not payload.target_symbol:
        raise ValidationError(f"Trigger type {payload.trigger_type} requires a symbol")
    if trigger_info.get("requires_portfolio") and not payload.target_portfolio_id:
        raise ValidationError(f"Trigger type {payload.trigger_type} requires a portfolio")
    
    rule = await notifications_repo.create_rule(
        user_id=user_id,
        channel_id=payload.channel_id,
        name=payload.name,
        trigger_type=payload.trigger_type,
        comparison_operator=payload.comparison_operator,
        target_value=Decimal(str(payload.target_value)) if payload.target_value is not None else None,
        target_symbol=payload.target_symbol,
        target_portfolio_id=payload.target_portfolio_id,
        smart_payload=payload.smart_payload,
        cooldown_minutes=payload.cooldown_minutes,
        priority=payload.priority,
        is_active=True,
    )
    
    return RuleResponse(
        id=rule["id"],
        channel_id=rule["channel_id"],
        name=rule["name"],
        trigger_type=rule["trigger_type"],
        comparison_operator=rule["comparison_operator"],
        target_value=rule.get("target_value"),
        target_symbol=rule.get("target_symbol"),
        target_portfolio_id=rule.get("target_portfolio_id"),
        smart_payload=rule.get("smart_payload"),
        cooldown_minutes=rule["cooldown_minutes"],
        last_triggered_at=rule.get("last_triggered_at"),
        trigger_count=rule["trigger_count"],
        is_active=rule["is_active"],
        priority=rule["priority"],
        created_at=rule["created_at"],
        updated_at=rule["updated_at"],
    )


@router.get(
    "/rules/{rule_id}",
    response_model=RuleResponse,
    summary="Get rule details",
    description="Get details of a specific notification rule.",
)
async def get_rule(
    rule_id: int,
    user: TokenData = Depends(require_user),
) -> RuleResponse:
    """Get a specific rule."""
    user_id = await _get_user_id(user)
    rule = await notifications_repo.get_rule(rule_id, user_id)
    
    if not rule:
        raise NotFoundError(f"Rule {rule_id} not found")
    
    return RuleResponse(
        id=rule["id"],
        channel_id=rule["channel_id"],
        name=rule["name"],
        trigger_type=rule["trigger_type"],
        comparison_operator=rule["comparison_operator"],
        target_value=rule.get("target_value"),
        target_symbol=rule.get("target_symbol"),
        target_portfolio_id=rule.get("target_portfolio_id"),
        smart_payload=rule.get("smart_payload"),
        cooldown_minutes=rule["cooldown_minutes"],
        last_triggered_at=rule.get("last_triggered_at"),
        trigger_count=rule["trigger_count"],
        is_active=rule["is_active"],
        priority=rule["priority"],
        created_at=rule["created_at"],
        updated_at=rule["updated_at"],
    )


@router.patch(
    "/rules/{rule_id}",
    response_model=RuleResponse,
    summary="Update rule",
    description="Update a notification rule's settings.",
)
async def update_rule(
    rule_id: int,
    payload: RuleUpdate,
    user: TokenData = Depends(require_user),
) -> RuleResponse:
    """Update a rule."""
    from decimal import Decimal
    
    user_id = await _get_user_id(user)
    
    # Validate channel if changed
    if payload.channel_id is not None:
        channel = await notifications_repo.get_channel(payload.channel_id, user_id)
        if not channel:
            raise NotFoundError(f"Channel {payload.channel_id} not found")
    
    # Validate trigger type if changed
    if payload.trigger_type is not None:
        try:
            TriggerType(payload.trigger_type)
        except ValueError:
            raise ValidationError(f"Invalid trigger type: {payload.trigger_type}")
    
    # Validate comparison operator if changed
    if payload.comparison_operator is not None:
        try:
            ComparisonOperator(payload.comparison_operator)
        except ValueError:
            raise ValidationError(f"Invalid comparison operator: {payload.comparison_operator}")
    
    # Build update kwargs
    update_kwargs = {}
    if payload.name is not None:
        update_kwargs["name"] = payload.name
    if payload.channel_id is not None:
        update_kwargs["channel_id"] = payload.channel_id
    if payload.trigger_type is not None:
        update_kwargs["trigger_type"] = payload.trigger_type
    if payload.comparison_operator is not None:
        update_kwargs["comparison_operator"] = payload.comparison_operator
    if payload.target_value is not None:
        update_kwargs["target_value"] = Decimal(str(payload.target_value))
    if payload.target_symbol is not None:
        update_kwargs["target_symbol"] = payload.target_symbol
    if payload.target_portfolio_id is not None:
        update_kwargs["target_portfolio_id"] = payload.target_portfolio_id
    if payload.smart_payload is not None:
        update_kwargs["smart_payload"] = payload.smart_payload
    if payload.cooldown_minutes is not None:
        update_kwargs["cooldown_minutes"] = payload.cooldown_minutes
    if payload.priority is not None:
        update_kwargs["priority"] = payload.priority
    if payload.is_active is not None:
        update_kwargs["is_active"] = payload.is_active
    
    rule = await notifications_repo.update_rule(
        rule_id=rule_id,
        user_id=user_id,
        **update_kwargs,
    )
    
    if not rule:
        raise NotFoundError(f"Rule {rule_id} not found")
    
    return RuleResponse(
        id=rule["id"],
        channel_id=rule["channel_id"],
        name=rule["name"],
        trigger_type=rule["trigger_type"],
        comparison_operator=rule["comparison_operator"],
        target_value=rule.get("target_value"),
        target_symbol=rule.get("target_symbol"),
        target_portfolio_id=rule.get("target_portfolio_id"),
        smart_payload=rule.get("smart_payload"),
        cooldown_minutes=rule["cooldown_minutes"],
        last_triggered_at=rule.get("last_triggered_at"),
        trigger_count=rule["trigger_count"],
        is_active=rule["is_active"],
        priority=rule["priority"],
        created_at=rule["created_at"],
        updated_at=rule["updated_at"],
    )


@router.delete(
    "/rules/{rule_id}",
    summary="Delete rule",
    description="Delete a notification rule.",
)
async def delete_rule(
    rule_id: int,
    user: TokenData = Depends(require_user),
) -> dict:
    """Delete a rule."""
    user_id = await _get_user_id(user)
    deleted = await notifications_repo.delete_rule(rule_id, user_id)
    
    if not deleted:
        raise NotFoundError(f"Rule {rule_id} not found")
    
    return {"deleted": True, "rule_id": rule_id}


@router.post(
    "/rules/{rule_id}/test",
    response_model=RuleTestResponse,
    summary="Test rule",
    description="Evaluate a rule against current data without sending a notification.",
)
async def test_rule(
    rule_id: int,
    user: TokenData = Depends(require_user),
) -> RuleTestResponse:
    """Test a rule without sending."""
    from app.services.notifications.checker import check_single_rule
    
    user_id = await _get_user_id(user)
    result = await check_single_rule(rule_id, user_id, ignore_cooldown=True)
    
    if "error" in result:
        raise NotFoundError(result["error"])
    
    return RuleTestResponse(
        triggered=result["triggered"],
        would_send=result["would_send"],
        actual_value=result.get("actual_value"),
        threshold_value=result.get("threshold_value"),
        description=result.get("description"),
        skip_reasons=result.get("skip_reasons", []),
        data_preview=result.get("data_preview"),
    )


@router.post(
    "/rules/{rule_id}/clear-cooldown",
    summary="Clear rule cooldown",
    description="Clear the cooldown for a rule to allow immediate triggering.",
)
async def clear_rule_cooldown(
    rule_id: int,
    user: TokenData = Depends(require_user),
) -> dict:
    """Clear cooldown for a rule."""
    from app.services.notifications.cooldown import clear_cooldown
    
    user_id = await _get_user_id(user)
    
    # Verify rule exists and belongs to user
    rule = await notifications_repo.get_rule(rule_id, user_id)
    if not rule:
        raise NotFoundError(f"Rule {rule_id} not found")
    
    cleared = await clear_cooldown(user_id, rule_id)
    return {"cleared": cleared, "rule_id": rule_id}


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================


@router.get(
    "/history",
    response_model=NotificationHistoryResponse,
    summary="Get notification history",
    description="Get history of sent notifications with pagination.",
)
async def get_history(
    user: TokenData = Depends(require_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="Filter by status"),
    trigger_type: str | None = Query(None, description="Filter by trigger type"),
) -> NotificationHistoryResponse:
    """Get notification history."""
    user_id = await _get_user_id(user)
    
    logs, total = await notifications_repo.list_logs(
        user_id=user_id,
        limit=limit,
        offset=offset,
        status=status,
        trigger_type=trigger_type,
    )
    
    return NotificationHistoryResponse(
        entries=[
            NotificationLogEntry(
                id=log["id"],
                rule_id=log.get("rule_id"),
                channel_id=log.get("channel_id"),
                trigger_type=log["trigger_type"],
                trigger_symbol=log.get("trigger_symbol"),
                trigger_value=log.get("trigger_value"),
                threshold_value=log.get("threshold_value"),
                title=log["title"],
                body=log["body"],
                status=log["status"],
                error_message=log.get("error_message"),
                triggered_at=log["triggered_at"],
                sent_at=log.get("sent_at"),
            )
            for log in logs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/summary",
    response_model=NotificationSummary,
    summary="Get notification summary",
    description="Get summary statistics for notifications.",
)
async def get_summary(
    user: TokenData = Depends(require_user),
) -> NotificationSummary:
    """Get notification summary stats."""
    from app.services.notifications.safety import get_rate_limit_status
    
    user_id = await _get_user_id(user)
    
    stats = await notifications_repo.get_notification_stats(user_id)
    rate_status = await get_rate_limit_status(user_id)
    
    return NotificationSummary(
        total_notifications=stats["total_notifications"],
        sent_last_24h=stats["last_24h"]["sent"],
        failed_last_24h=stats["last_24h"]["failed"],
        skipped_last_24h=stats["last_24h"]["skipped"],
        active_channels=stats["active_channels"],
        active_rules=stats["active_rules"],
        rate_limit_remaining=rate_status["remaining"],
        rate_limit_reset_in=rate_status["reset_in"],
    )


# =============================================================================
# TRIGGER TYPE ENDPOINTS
# =============================================================================


class TriggerTypeInfo(BaseModel):
    """Information about a trigger type."""
    
    value: str
    category: str
    default_operator: str
    default_value: float | None
    value_unit: str | None
    requires_symbol: bool
    requires_portfolio: bool


@router.get(
    "/trigger-types",
    response_model=list[TriggerTypeInfo],
    summary="List available trigger types",
    description="Get all available trigger types with their metadata.",
)
async def list_trigger_types() -> list[TriggerTypeInfo]:
    """List all available trigger types."""
    result = []
    for tt in TriggerType:
        info = TRIGGER_TYPE_INFO.get(tt)
        if info:
            result.append(TriggerTypeInfo(
                value=tt.value,
                category=info.category,
                default_operator=info.default_operator or "GT",
                default_value=info.default_value,
                value_unit=info.value_unit,
                requires_symbol=info.requires_symbol,
                requires_portfolio=info.requires_portfolio,
            ))
        else:
            # Fallback for unmapped trigger types
            result.append(TriggerTypeInfo(
                value=tt.value,
                category="other",
                default_operator="GT",
                default_value=None,
                value_unit=None,
                requires_symbol=False,
                requires_portfolio=False,
            ))
    return result
