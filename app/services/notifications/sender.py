"""Notification sender using Apprise.

Handles the actual delivery of notifications to various channels.
Supports Discord, Telegram, Email, Slack, Pushover, ntfy, webhooks, etc.

Apprise URL formats:
    discord://webhook_id/webhook_token
    tgram://bot_token/chat_id
    mailto://user:pass@gmail.com
    slack://TokenA/TokenB/TokenC/#channel
    pover://user@token
    ntfy://topic
    json://hostname/path
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.core.encryption import decrypt_api_key
from app.core.logging import get_logger
from app.repositories import notifications_orm as repo


logger = get_logger("notifications.sender")


async def send_via_apprise(
    apprise_url: str,
    title: str,
    body: str,
    priority: str = "normal",
) -> tuple[bool, str | None]:
    """Send a notification via Apprise.
    
    Args:
        apprise_url: Decrypted Apprise URL
        title: Notification title
        body: Notification body
        priority: Priority level (low, normal, high, critical)
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        import apprise
        
        # Create Apprise instance
        apobj = apprise.Apprise()
        
        # Add the URL
        if not apobj.add(apprise_url):
            return False, "Invalid Apprise URL format"
        
        # Map priority to Apprise notification type
        notify_type = apprise.NotifyType.INFO
        if priority == "critical":
            notify_type = apprise.NotifyType.FAILURE
        elif priority == "high":
            notify_type = apprise.NotifyType.WARNING
        elif priority == "low":
            notify_type = apprise.NotifyType.SUCCESS
        
        # Send the notification
        result = apobj.notify(
            title=title,
            body=body,
            notify_type=notify_type,
        )
        
        if result:
            logger.info("Notification sent successfully")
            return True, None
        else:
            return False, "Apprise notify() returned False"
            
    except ImportError:
        logger.error("Apprise package not installed")
        return False, "Apprise package not installed"
    except Exception as e:
        logger.exception("Failed to send notification via Apprise")
        return False, str(e)


async def send_notification(
    channel_id: int,
    user_id: int,
    title: str,
    body: str,
    priority: str = "normal",
    rule_id: int | None = None,
    trigger_type: str | None = None,
    trigger_symbol: str | None = None,
    trigger_value: float | None = None,
    threshold_value: float | None = None,
) -> tuple[bool, int | None, str | None]:
    """Send a notification and log the result.
    
    Args:
        channel_id: The channel to send to
        user_id: The user sending the notification
        title: Notification title
        body: Notification body
        priority: Priority level
        rule_id: The rule that triggered (optional)
        trigger_type: Type of trigger
        trigger_symbol: Symbol involved
        trigger_value: Value that triggered
        threshold_value: Threshold that was exceeded
        
    Returns:
        Tuple of (success, log_id, error_message)
    """
    from decimal import Decimal
    from app.services.notifications.safety import compute_content_hash
    
    # Get channel and decrypt URL
    channel = await repo.get_channel(channel_id, user_id)
    if not channel:
        logger.error("Channel not found", extra={"channel_id": channel_id})
        return False, None, "Channel not found"
    
    if not channel.get("is_active"):
        return False, None, "Channel is disabled"
    
    # Create log entry as pending
    content_hash = compute_content_hash(title, body)
    log_entry = await repo.create_log_entry(
        user_id=user_id,
        rule_id=rule_id,
        channel_id=channel_id,
        trigger_type=trigger_type or "MANUAL",
        trigger_symbol=trigger_symbol,
        trigger_value=Decimal(str(trigger_value)) if trigger_value else None,
        threshold_value=Decimal(str(threshold_value)) if threshold_value else None,
        title=title,
        body=body,
        status="pending",
        content_hash=content_hash,
    )
    log_id = log_entry["id"]
    
    # Decrypt the Apprise URL
    try:
        apprise_url = decrypt_api_key(channel["encrypted_url"])
    except Exception as e:
        await repo.update_log_status(log_id, "failed", f"Decryption failed: {e}")
        return False, log_id, "Failed to decrypt channel URL"
    
    # Send via Apprise
    success, error = await send_via_apprise(apprise_url, title, body, priority)
    
    if success:
        await repo.update_log_status(log_id, "sent")
        await repo.reset_channel_errors(channel_id)
        
        logger.info(
            "Notification sent",
            extra={
                "channel_id": channel_id,
                "rule_id": rule_id,
                "trigger_type": trigger_type,
            }
        )
    else:
        await repo.update_log_status(log_id, "failed", error)
        await repo.increment_channel_error(channel_id, error or "Unknown error")
        
        logger.warning(
            "Notification failed",
            extra={
                "channel_id": channel_id,
                "error": error,
            }
        )
    
    return success, log_id, error


async def test_channel(channel_id: int, user_id: int) -> dict[str, Any]:
    """Send a test notification to verify channel configuration.
    
    Args:
        channel_id: The channel to test
        user_id: The user (for access control)
        
    Returns:
        Dict with success, message, and optional details
    """
    channel = await repo.get_channel(channel_id, user_id)
    if not channel:
        return {
            "success": False,
            "message": "Channel not found",
        }
    
    # Decrypt URL
    try:
        apprise_url = decrypt_api_key(channel["encrypted_url"])
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to decrypt channel URL: {e}",
        }
    
    # Send test message
    title = "Stonkmarket Test Notification"
    body = (
        f"This is a test notification sent at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}.\n\n"
        f"Your channel '{channel['name']}' is configured correctly!"
    )
    
    success, error = await send_via_apprise(apprise_url, title, body)
    
    if success:
        # Update verification status
        await repo.update_channel(
            channel_id,
            user_id,
            is_verified=True,
            error_count=0,
            last_error=None,
            last_used_at=datetime.now(UTC),
        )
        
        return {
            "success": True,
            "message": "Test notification sent successfully!",
            "verified_at": datetime.now(UTC).isoformat(),
        }
    else:
        await repo.increment_channel_error(channel_id, error or "Test failed")
        
        return {
            "success": False,
            "message": f"Failed to send: {error}",
            "error": error,
        }


def validate_apprise_url(url: str) -> tuple[bool, str | None, str | None]:
    """Validate an Apprise URL format without sending.
    
    Args:
        url: The Apprise URL to validate
        
    Returns:
        Tuple of (is_valid, channel_type, error_message)
    """
    try:
        import apprise
        
        apobj = apprise.Apprise()
        if not apobj.add(url):
            return False, None, "Invalid URL format"
        
        # Get the service type
        # Apprise URLs start with the service type
        service_type = url.split("://")[0].lower() if "://" in url else None
        
        # Map common Apprise schemes to our channel types
        type_map = {
            "discord": "discord",
            "tgram": "telegram",
            "mailto": "email",
            "slack": "slack",
            "pover": "pushover",
            "ntfy": "ntfy",
            "json": "webhook",
            "xml": "webhook",
            "form": "webhook",
        }
        
        channel_type = type_map.get(service_type, "other")
        
        return True, channel_type, None
        
    except ImportError:
        return False, None, "Apprise package not installed"
    except Exception as e:
        return False, None, str(e)
