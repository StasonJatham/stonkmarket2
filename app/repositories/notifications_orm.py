"""Notifications repository using SQLAlchemy ORM.

CRUD operations for notification channels, rules, and logs.

Usage:
    from app.repositories.notifications_orm import (
        create_channel, get_channel, list_channels,
        create_rule, get_rule, list_rules,
        create_log_entry, list_logs,
    )
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select, func, update
from sqlalchemy.orm import selectinload

from app.core.encryption import encrypt_api_key, decrypt_api_key
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import NotificationChannel, NotificationRule, NotificationLog


logger = get_logger("repositories.notifications_orm")


# =============================================================================
# CHANNEL OPERATIONS
# =============================================================================


async def create_channel(
    user_id: int,
    name: str,
    channel_type: str,
    apprise_url: str,
    is_active: bool = True,
) -> dict[str, Any]:
    """Create a new notification channel.
    
    Args:
        user_id: Owner of the channel
        name: Display name for the channel
        channel_type: Type (discord, telegram, email, etc.)
        apprise_url: Apprise-compatible URL (will be encrypted)
        is_active: Whether the channel is enabled
        
    Returns:
        Created channel as dict
    """
    encrypted_url = encrypt_api_key(apprise_url)
    
    async with get_session() as session:
        channel = NotificationChannel(
            user_id=user_id,
            name=name,
            channel_type=channel_type,
            encrypted_url=encrypted_url,
            is_active=is_active,
        )
        session.add(channel)
        await session.commit()
        await session.refresh(channel)
        
        return _channel_to_dict(channel)


async def get_channel(channel_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    """Get a channel by ID, optionally filtered by user.
    
    Args:
        channel_id: The channel ID
        user_id: Optional user ID for access control
        
    Returns:
        Channel dict or None if not found
    """
    async with get_session() as session:
        stmt = select(NotificationChannel).where(NotificationChannel.id == channel_id)
        if user_id is not None:
            stmt = stmt.where(NotificationChannel.user_id == user_id)
            
        result = await session.execute(stmt)
        channel = result.scalar_one_or_none()
        
        if channel:
            return _channel_to_dict(channel)
        return None


async def list_channels(
    user_id: int,
    active_only: bool = False,
    include_rules: bool = False,
) -> list[dict[str, Any]]:
    """List all channels for a user.
    
    Args:
        user_id: The user ID
        active_only: If True, only return active channels
        include_rules: If True, include associated rules count
        
    Returns:
        List of channel dicts
    """
    async with get_session() as session:
        stmt = select(NotificationChannel).where(NotificationChannel.user_id == user_id)
        
        if active_only:
            stmt = stmt.where(NotificationChannel.is_active == True)  # noqa: E712
        
        if include_rules:
            stmt = stmt.options(selectinload(NotificationChannel.rules))
            
        stmt = stmt.order_by(NotificationChannel.name)
        
        result = await session.execute(stmt)
        channels = result.scalars().all()
        
        return [
            _channel_to_dict(ch, include_rule_count=include_rules)
            for ch in channels
        ]


async def update_channel(
    channel_id: int,
    user_id: int,
    *,
    name: str | None = None,
    channel_type: str | None = None,
    apprise_url: str | None = None,
    is_active: bool | None = None,
    is_verified: bool | None = None,
    error_count: int | None = None,
    last_error: str | None = None,
    last_used_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Update a channel.
    
    Args:
        channel_id: The channel to update
        user_id: Owner (for access control)
        **kwargs: Fields to update
        
    Returns:
        Updated channel dict or None if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(NotificationChannel).where(
                NotificationChannel.id == channel_id,
                NotificationChannel.user_id == user_id,
            )
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            return None
        
        if name is not None:
            channel.name = name
        if channel_type is not None:
            channel.channel_type = channel_type
        if apprise_url is not None:
            channel.encrypted_url = encrypt_api_key(apprise_url)
        if is_active is not None:
            channel.is_active = is_active
        if is_verified is not None:
            channel.is_verified = is_verified
            if is_verified:
                channel.last_verified_at = datetime.now(UTC)
        if error_count is not None:
            channel.error_count = error_count
        if last_error is not None:
            channel.last_error = last_error
        if last_used_at is not None:
            channel.last_used_at = last_used_at
            
        channel.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(channel)
        
        return _channel_to_dict(channel)


async def delete_channel(channel_id: int, user_id: int) -> bool:
    """Delete a channel (cascades to rules and logs).
    
    Args:
        channel_id: The channel to delete
        user_id: Owner (for access control)
        
    Returns:
        True if deleted, False if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(NotificationChannel).where(
                NotificationChannel.id == channel_id,
                NotificationChannel.user_id == user_id,
            )
        )
        channel = result.scalar_one_or_none()
        
        if not channel:
            return False
            
        await session.delete(channel)
        await session.commit()
        return True


async def get_channel_decrypted_url(channel_id: int, user_id: int) -> str | None:
    """Get the decrypted Apprise URL for a channel.
    
    Args:
        channel_id: The channel ID
        user_id: Owner (for access control)
        
    Returns:
        Decrypted URL or None if not found
    """
    channel = await get_channel(channel_id, user_id)
    if channel and channel.get("encrypted_url"):
        return decrypt_api_key(channel["encrypted_url"])
    return None


async def increment_channel_error(channel_id: int, error_message: str) -> None:
    """Increment error count and record error message.
    
    Args:
        channel_id: The channel ID
        error_message: The error that occurred
    """
    async with get_session() as session:
        await session.execute(
            update(NotificationChannel)
            .where(NotificationChannel.id == channel_id)
            .values(
                error_count=NotificationChannel.error_count + 1,
                last_error=error_message,
                updated_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def reset_channel_errors(channel_id: int) -> None:
    """Reset error count after successful send.
    
    Args:
        channel_id: The channel ID
    """
    async with get_session() as session:
        await session.execute(
            update(NotificationChannel)
            .where(NotificationChannel.id == channel_id)
            .values(
                error_count=0,
                last_error=None,
                last_used_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        await session.commit()


def _channel_to_dict(channel: NotificationChannel, include_rule_count: bool = False) -> dict[str, Any]:
    """Convert channel ORM object to dict."""
    result = {
        "id": channel.id,
        "user_id": channel.user_id,
        "name": channel.name,
        "channel_type": channel.channel_type,
        "encrypted_url": channel.encrypted_url,  # Never expose, but keep for internal use
        "is_verified": channel.is_verified,
        "last_verified_at": channel.last_verified_at,
        "is_active": channel.is_active,
        "error_count": channel.error_count,
        "last_error": channel.last_error,
        "last_used_at": channel.last_used_at,
        "created_at": channel.created_at,
        "updated_at": channel.updated_at,
    }
    
    if include_rule_count:
        result["rule_count"] = len(channel.rules) if channel.rules else 0
        
    return result


# =============================================================================
# RULE OPERATIONS
# =============================================================================


async def create_rule(
    user_id: int,
    channel_id: int,
    name: str,
    trigger_type: str,
    comparison_operator: str = "GT",
    target_value: Decimal | None = None,
    target_symbol: str | None = None,
    target_portfolio_id: int | None = None,
    smart_payload: dict[str, Any] | None = None,
    cooldown_minutes: int = 60,
    priority: str = "normal",
    is_active: bool = True,
) -> dict[str, Any]:
    """Create a new notification rule.
    
    Args:
        user_id: Owner of the rule
        channel_id: Channel to send notifications to
        name: Display name for the rule
        trigger_type: Type of trigger (from TriggerType enum)
        comparison_operator: Comparison operator (GT, LT, etc.)
        target_value: Threshold value for the condition
        target_symbol: Symbol to monitor (for symbol-specific triggers)
        target_portfolio_id: Portfolio to monitor (for portfolio triggers)
        smart_payload: Additional trigger-specific configuration
        cooldown_minutes: Minutes between repeated triggers
        priority: Alert priority (low, normal, high, critical)
        is_active: Whether the rule is enabled
        
    Returns:
        Created rule as dict
    """
    async with get_session() as session:
        rule = NotificationRule(
            user_id=user_id,
            channel_id=channel_id,
            name=name,
            trigger_type=trigger_type,
            comparison_operator=comparison_operator,
            target_value=target_value,
            target_symbol=target_symbol.upper() if target_symbol else None,
            target_portfolio_id=target_portfolio_id,
            smart_payload=smart_payload or {},
            cooldown_minutes=cooldown_minutes,
            priority=priority,
            is_active=is_active,
        )
        session.add(rule)
        await session.commit()
        await session.refresh(rule)
        
        return _rule_to_dict(rule)


async def get_rule(rule_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    """Get a rule by ID, optionally filtered by user.
    
    Args:
        rule_id: The rule ID
        user_id: Optional user ID for access control
        
    Returns:
        Rule dict or None if not found
    """
    async with get_session() as session:
        stmt = select(NotificationRule).where(NotificationRule.id == rule_id)
        if user_id is not None:
            stmt = stmt.where(NotificationRule.user_id == user_id)
            
        result = await session.execute(stmt)
        rule = result.scalar_one_or_none()
        
        if rule:
            return _rule_to_dict(rule)
        return None


async def list_rules(
    user_id: int,
    active_only: bool = False,
    channel_id: int | None = None,
    trigger_type: str | None = None,
    target_symbol: str | None = None,
) -> list[dict[str, Any]]:
    """List rules for a user with optional filters.
    
    Args:
        user_id: The user ID
        active_only: If True, only return active rules
        channel_id: Filter by channel
        trigger_type: Filter by trigger type
        target_symbol: Filter by symbol
        
    Returns:
        List of rule dicts
    """
    async with get_session() as session:
        stmt = select(NotificationRule).where(NotificationRule.user_id == user_id)
        
        if active_only:
            stmt = stmt.where(NotificationRule.is_active == True)  # noqa: E712
        if channel_id is not None:
            stmt = stmt.where(NotificationRule.channel_id == channel_id)
        if trigger_type is not None:
            stmt = stmt.where(NotificationRule.trigger_type == trigger_type)
        if target_symbol is not None:
            stmt = stmt.where(NotificationRule.target_symbol == target_symbol.upper())
            
        stmt = stmt.order_by(NotificationRule.name)
        
        result = await session.execute(stmt)
        rules = result.scalars().all()
        
        return [_rule_to_dict(r) for r in rules]


async def list_active_rules_for_checker() -> list[dict[str, Any]]:
    """List all active rules for the notification checker.
    
    Returns rules with their channel info for batch processing.
    
    Returns:
        List of active rules with channel info
    """
    async with get_session() as session:
        stmt = (
            select(NotificationRule)
            .options(selectinload(NotificationRule.channel))
            .where(
                NotificationRule.is_active == True,  # noqa: E712
            )
            .order_by(NotificationRule.user_id, NotificationRule.trigger_type)
        )
        
        result = await session.execute(stmt)
        rules = result.scalars().all()
        
        return [
            {
                **_rule_to_dict(r),
                "channel": _channel_to_dict(r.channel) if r.channel else None,
            }
            for r in rules
            if r.channel and r.channel.is_active  # Skip rules with inactive channels
        ]


async def update_rule(
    rule_id: int,
    user_id: int,
    *,
    name: str | None = None,
    channel_id: int | None = None,
    trigger_type: str | None = None,
    comparison_operator: str | None = None,
    target_value: Decimal | None = ...,  # Use ... as sentinel for "not provided"
    target_symbol: str | None = ...,
    target_portfolio_id: int | None = ...,
    smart_payload: dict[str, Any] | None = ...,
    cooldown_minutes: int | None = None,
    priority: str | None = None,
    is_active: bool | None = None,
) -> dict[str, Any] | None:
    """Update a rule.
    
    Args:
        rule_id: The rule to update
        user_id: Owner (for access control)
        **kwargs: Fields to update (... means not provided vs None which clears)
        
    Returns:
        Updated rule dict or None if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(NotificationRule).where(
                NotificationRule.id == rule_id,
                NotificationRule.user_id == user_id,
            )
        )
        rule = result.scalar_one_or_none()
        
        if not rule:
            return None
        
        if name is not None:
            rule.name = name
        if channel_id is not None:
            rule.channel_id = channel_id
        if trigger_type is not None:
            rule.trigger_type = trigger_type
        if comparison_operator is not None:
            rule.comparison_operator = comparison_operator
        if target_value is not ...:
            rule.target_value = target_value
        if target_symbol is not ...:
            rule.target_symbol = target_symbol.upper() if target_symbol else None
        if target_portfolio_id is not ...:
            rule.target_portfolio_id = target_portfolio_id
        if smart_payload is not ...:
            rule.smart_payload = smart_payload or {}
        if cooldown_minutes is not None:
            rule.cooldown_minutes = cooldown_minutes
        if priority is not None:
            rule.priority = priority
        if is_active is not None:
            rule.is_active = is_active
            
        rule.updated_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(rule)
        
        return _rule_to_dict(rule)


async def delete_rule(rule_id: int, user_id: int) -> bool:
    """Delete a rule (cascades to logs).
    
    Args:
        rule_id: The rule to delete
        user_id: Owner (for access control)
        
    Returns:
        True if deleted, False if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(NotificationRule).where(
                NotificationRule.id == rule_id,
                NotificationRule.user_id == user_id,
            )
        )
        rule = result.scalar_one_or_none()
        
        if not rule:
            return False
            
        await session.delete(rule)
        await session.commit()
        return True


async def update_rule_triggered(rule_id: int) -> None:
    """Update last_triggered_at and increment trigger_count.
    
    Args:
        rule_id: The rule that was triggered
    """
    async with get_session() as session:
        await session.execute(
            update(NotificationRule)
            .where(NotificationRule.id == rule_id)
            .values(
                last_triggered_at=datetime.now(UTC),
                trigger_count=NotificationRule.trigger_count + 1,
                updated_at=datetime.now(UTC),
            )
        )
        await session.commit()


def _rule_to_dict(rule: NotificationRule) -> dict[str, Any]:
    """Convert rule ORM object to dict."""
    return {
        "id": rule.id,
        "user_id": rule.user_id,
        "channel_id": rule.channel_id,
        "name": rule.name,
        "trigger_type": rule.trigger_type,
        "comparison_operator": rule.comparison_operator,
        "target_value": float(rule.target_value) if rule.target_value else None,
        "target_symbol": rule.target_symbol,
        "target_portfolio_id": rule.target_portfolio_id,
        "smart_payload": rule.smart_payload,
        "cooldown_minutes": rule.cooldown_minutes,
        "last_triggered_at": rule.last_triggered_at,
        "trigger_count": rule.trigger_count,
        "is_active": rule.is_active,
        "priority": rule.priority,
        "created_at": rule.created_at,
        "updated_at": rule.updated_at,
    }


# =============================================================================
# LOG OPERATIONS
# =============================================================================


async def create_log_entry(
    user_id: int,
    trigger_type: str,
    title: str,
    body: str,
    status: str = "pending",
    rule_id: int | None = None,
    channel_id: int | None = None,
    trigger_symbol: str | None = None,
    trigger_value: Decimal | None = None,
    threshold_value: Decimal | None = None,
    error_message: str | None = None,
    content_hash: str | None = None,
) -> dict[str, Any]:
    """Create a notification log entry.
    
    Args:
        user_id: The user who owns this notification
        trigger_type: What triggered the notification
        title: Notification title
        body: Notification body
        status: pending, sent, failed, skipped
        rule_id: The rule that triggered (optional)
        channel_id: The channel used (optional)
        trigger_symbol: Symbol involved (optional)
        trigger_value: Actual value that triggered (optional)
        threshold_value: Threshold that was exceeded (optional)
        error_message: Error if failed (optional)
        content_hash: SHA-256 hash for deduplication (optional)
        
    Returns:
        Created log entry as dict
    """
    async with get_session() as session:
        log = NotificationLog(
            user_id=user_id,
            rule_id=rule_id,
            channel_id=channel_id,
            trigger_type=trigger_type,
            trigger_symbol=trigger_symbol.upper() if trigger_symbol else None,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            title=title,
            body=body,
            status=status,
            error_message=error_message,
            content_hash=content_hash,
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
        
        return _log_to_dict(log)


async def update_log_status(
    log_id: int,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update a log entry status.
    
    Args:
        log_id: The log entry to update
        status: New status (sent, failed, skipped)
        error_message: Error message if failed
    """
    async with get_session() as session:
        values: dict[str, Any] = {"status": status}
        if status == "sent":
            values["sent_at"] = datetime.now(UTC)
        if error_message:
            values["error_message"] = error_message
            
        await session.execute(
            update(NotificationLog)
            .where(NotificationLog.id == log_id)
            .values(**values)
        )
        await session.commit()


async def list_logs(
    user_id: int,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    trigger_type: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """List notification logs for a user with pagination.
    
    Args:
        user_id: The user ID
        limit: Max entries to return
        offset: Offset for pagination
        status: Filter by status
        trigger_type: Filter by trigger type
        
    Returns:
        Tuple of (log entries, total count)
    """
    async with get_session() as session:
        # Build base query
        base = select(NotificationLog).where(NotificationLog.user_id == user_id)
        
        if status:
            base = base.where(NotificationLog.status == status)
        if trigger_type:
            base = base.where(NotificationLog.trigger_type == trigger_type)
        
        # Get total count
        count_result = await session.execute(
            select(func.count()).select_from(base.subquery())
        )
        total = count_result.scalar_one()
        
        # Get paginated results
        stmt = (
            base
            .order_by(NotificationLog.triggered_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await session.execute(stmt)
        logs = result.scalars().all()
        
        return [_log_to_dict(log) for log in logs], total


async def check_recent_hash(content_hash: str, user_id: int, hours: int = 24) -> bool:
    """Check if a notification with this hash was sent recently.
    
    Used for deduplication to avoid sending identical notifications.
    
    Args:
        content_hash: The content hash to check
        user_id: The user ID
        hours: How far back to look
        
    Returns:
        True if a matching hash was found recently
    """
    async with get_session() as session:
        from datetime import timedelta
        
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        result = await session.execute(
            select(func.count())
            .where(
                NotificationLog.user_id == user_id,
                NotificationLog.content_hash == content_hash,
                NotificationLog.status == "sent",
                NotificationLog.triggered_at >= cutoff,
            )
        )
        count = result.scalar_one()
        
        return count > 0


async def get_notification_stats(user_id: int) -> dict[str, Any]:
    """Get notification statistics for a user.
    
    Args:
        user_id: The user ID
        
    Returns:
        Stats including counts by status, recent activity, etc.
    """
    async with get_session() as session:
        from datetime import timedelta
        
        now = datetime.now(UTC)
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        # Count by status (last 24h)
        status_result = await session.execute(
            select(
                NotificationLog.status,
                func.count().label("count"),
            )
            .where(
                NotificationLog.user_id == user_id,
                NotificationLog.triggered_at >= last_24h,
            )
            .group_by(NotificationLog.status)
        )
        status_counts = {row.status: row.count for row in status_result}
        
        # Total counts
        total_result = await session.execute(
            select(func.count())
            .where(NotificationLog.user_id == user_id)
        )
        total = total_result.scalar_one()
        
        # Channels count
        channels_result = await session.execute(
            select(func.count())
            .where(
                NotificationChannel.user_id == user_id,
                NotificationChannel.is_active == True,  # noqa: E712
            )
        )
        active_channels = channels_result.scalar_one()
        
        # Rules count
        rules_result = await session.execute(
            select(func.count())
            .where(
                NotificationRule.user_id == user_id,
                NotificationRule.is_active == True,  # noqa: E712
            )
        )
        active_rules = rules_result.scalar_one()
        
        return {
            "total_notifications": total,
            "last_24h": {
                "sent": status_counts.get("sent", 0),
                "failed": status_counts.get("failed", 0),
                "skipped": status_counts.get("skipped", 0),
                "pending": status_counts.get("pending", 0),
            },
            "active_channels": active_channels,
            "active_rules": active_rules,
        }


def _log_to_dict(log: NotificationLog) -> dict[str, Any]:
    """Convert log ORM object to dict."""
    return {
        "id": log.id,
        "rule_id": log.rule_id,
        "channel_id": log.channel_id,
        "user_id": log.user_id,
        "trigger_type": log.trigger_type,
        "trigger_symbol": log.trigger_symbol,
        "trigger_value": float(log.trigger_value) if log.trigger_value else None,
        "threshold_value": float(log.threshold_value) if log.threshold_value else None,
        "title": log.title,
        "body": log.body,
        "status": log.status,
        "error_message": log.error_message,
        "triggered_at": log.triggered_at,
        "sent_at": log.sent_at,
        "content_hash": log.content_hash,
    }
