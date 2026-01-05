"""Main notification checker engine.

Runs periodically to evaluate all active rules and send notifications.
Implements batch optimization, safety checks, and logging.

Usage:
    # In Celery job
    from app.services.notifications import check_all_rules
    await check_all_rules()
    
    # For testing a single rule
    from app.services.notifications import check_single_rule
    result = await check_single_rule(rule_id, user_id)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.core.logging import get_logger
from app.repositories import notifications_orm as repo
from app.services.notifications.cooldown import check_cooldown, set_cooldown
from app.services.notifications.data_fetcher import (
    batch_fetch_symbol_data,
    batch_fetch_portfolio_data,
    batch_fetch_watchlist_data,
)
from app.services.notifications.message_builder import build_notification_message
from app.services.notifications.safety import (
    check_duplicate,
    check_rate_limit,
    check_staleness,
    increment_rate_limit,
)
from app.services.notifications.sender import send_notification
from app.services.notifications.triggers import evaluate_trigger


logger = get_logger("notifications.checker")


async def check_all_rules() -> dict[str, Any]:
    """Check all active notification rules and send triggered notifications.
    
    This is the main entry point called by the Celery periodic task.
    Implements batch optimization to minimize database queries.
    
    Returns:
        Dict with statistics about the run
    """
    start_time = datetime.now(UTC)
    stats = {
        "started_at": start_time.isoformat(),
        "rules_checked": 0,
        "rules_triggered": 0,
        "notifications_sent": 0,
        "notifications_skipped": 0,
        "notifications_failed": 0,
        "errors": [],
    }
    
    try:
        # Get all active rules with their channels
        rules = await repo.list_active_rules_for_checker()
        stats["rules_checked"] = len(rules)
        
        if not rules:
            logger.info("No active rules to check")
            return stats
        
        # Group rules by data type for batch fetching
        symbol_rules, portfolio_rules, watchlist_rules, symbols, portfolio_ids, watchlist_ids = _group_rules(rules)
        
        # Batch fetch all needed data
        symbol_data = await batch_fetch_symbol_data(symbols) if symbols else {}
        portfolio_data = await batch_fetch_portfolio_data(portfolio_ids) if portfolio_ids else {}
        watchlist_data = await batch_fetch_watchlist_data(watchlist_ids) if watchlist_ids else {}
        
        # Process each rule
        for rule in rules:
            try:
                result = await _process_rule(rule, symbol_data, portfolio_data, watchlist_data, stats)
                if result.get("triggered"):
                    stats["rules_triggered"] += 1
            except Exception as e:
                logger.exception(f"Error processing rule {rule['id']}")
                stats["errors"].append({
                    "rule_id": rule["id"],
                    "error": str(e),
                })
        
        logger.info(
            "Notification check completed",
            extra={
                "rules_checked": stats["rules_checked"],
                "rules_triggered": stats["rules_triggered"],
                "notifications_sent": stats["notifications_sent"],
                "duration_ms": (datetime.now(UTC) - start_time).total_seconds() * 1000,
            }
        )
        
    except Exception as e:
        logger.exception("Fatal error in notification checker")
        stats["errors"].append({"fatal": str(e)})
    
    stats["completed_at"] = datetime.now(UTC).isoformat()
    return stats


def _group_rules(
    rules: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict], list[str], list[int], list[int]]:
    """Group rules by what data they need.
    
    Returns:
        Tuple of (symbol_rules, portfolio_rules, watchlist_rules, unique_symbols, unique_portfolio_ids, unique_watchlist_ids)
    """
    symbol_rules = []
    portfolio_rules = []
    watchlist_rules = []
    symbols = set()
    portfolio_ids = set()
    watchlist_ids = set()
    
    for rule in rules:
        if rule.get("target_symbol"):
            symbol_rules.append(rule)
            symbols.add(rule["target_symbol"])
        elif rule.get("target_portfolio_id"):
            portfolio_rules.append(rule)
            portfolio_ids.add(rule["target_portfolio_id"])
        elif rule.get("target_watchlist_id"):
            watchlist_rules.append(rule)
            watchlist_ids.add(rule["target_watchlist_id"])
    
    return symbol_rules, portfolio_rules, watchlist_rules, list(symbols), list(portfolio_ids), list(watchlist_ids)


async def _process_rule(
    rule: dict[str, Any],
    symbol_data: dict[str, dict[str, Any]],
    portfolio_data: dict[int, dict[str, Any]],
    watchlist_data: dict[int, dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Process a single rule and send notification if triggered.
    
    Args:
        rule: The rule to process
        symbol_data: Pre-fetched symbol data
        portfolio_data: Pre-fetched portfolio data
        watchlist_data: Pre-fetched watchlist data
        stats: Stats dict to update
        
    Returns:
        Result dict with triggered, sent, skipped flags
    """
    result = {
        "rule_id": rule["id"],
        "triggered": False,
        "sent": False,
        "skipped": False,
        "skip_reason": None,
    }
    
    user_id = rule["user_id"]
    rule_id = rule["id"]
    channel = rule.get("channel")
    
    if not channel:
        result["skip_reason"] = "No channel configured"
        stats["notifications_skipped"] += 1
        return result
    
    # Check cooldown
    if await check_cooldown(user_id, rule_id):
        result["skip_reason"] = "In cooldown"
        stats["notifications_skipped"] += 1
        return result
    
    # Check rate limit
    is_limited, count = await check_rate_limit(user_id)
    if is_limited:
        result["skip_reason"] = f"Rate limited ({count} notifications this hour)"
        stats["notifications_skipped"] += 1
        return result
    
    # Get the relevant data
    if rule.get("target_symbol"):
        data = symbol_data.get(rule["target_symbol"], {})
    elif rule.get("target_portfolio_id"):
        data = portfolio_data.get(rule["target_portfolio_id"], {})
    elif rule.get("target_watchlist_id"):
        data = watchlist_data.get(rule["target_watchlist_id"], {})
    else:
        data = {}
    
    if not data:
        result["skip_reason"] = "No data available"
        stats["notifications_skipped"] += 1
        return result
    
    # Check data staleness
    updated_at = (
        data.get("dip_updated_at") or
        data.get("fundamentals_updated_at") or
        data.get("quant_updated_at") or
        data.get("portfolio_updated_at") or
        data.get("watchlist_updated_at")
    )
    is_stale, age_hours = check_staleness(updated_at)
    if is_stale:
        result["skip_reason"] = f"Data too stale ({age_hours:.1f}h old)"
        stats["notifications_skipped"] += 1
        return result
    
    # Evaluate trigger
    triggered, actual_value, description = evaluate_trigger(
        trigger_type=rule["trigger_type"],
        data=data,
        comparison_operator=rule["comparison_operator"],
        target_value=rule.get("target_value"),
        smart_payload=rule.get("smart_payload"),
    )
    
    if not triggered:
        return result
    
    result["triggered"] = True
    
    # Build message
    title, body = build_notification_message(
        trigger_type=rule["trigger_type"],
        actual_value=actual_value,
        threshold_value=rule.get("target_value"),
        symbol=rule.get("target_symbol"),
        portfolio_name=data.get("portfolio_name"),
        rule_name=rule.get("name"),
    )
    
    # Check for duplicate
    is_dup, content_hash = await check_duplicate(title, body, user_id)
    if is_dup:
        result["skip_reason"] = "Duplicate notification"
        result["skipped"] = True
        stats["notifications_skipped"] += 1
        return result
    
    # Send notification
    success, log_id, error = await send_notification(
        channel_id=channel["id"],
        user_id=user_id,
        title=title,
        body=body,
        priority=rule.get("priority", "normal"),
        rule_id=rule_id,
        trigger_type=rule["trigger_type"],
        trigger_symbol=rule.get("target_symbol"),
        trigger_value=float(actual_value) if actual_value is not None else None,
        threshold_value=float(rule.get("target_value")) if rule.get("target_value") else None,
    )
    
    if success:
        result["sent"] = True
        stats["notifications_sent"] += 1
        
        # Set cooldown
        await set_cooldown(user_id, rule_id, rule.get("cooldown_minutes", 60))
        
        # Increment rate limit
        await increment_rate_limit(user_id)
        
        # Update rule stats
        await repo.update_rule_triggered(rule_id)
        
        logger.info(
            "Notification sent",
            extra={
                "rule_id": rule_id,
                "trigger_type": rule["trigger_type"],
                "symbol": rule.get("target_symbol"),
            }
        )
    else:
        stats["notifications_failed"] += 1
        result["error"] = error
        logger.warning(
            "Notification failed",
            extra={
                "rule_id": rule_id,
                "error": error,
            }
        )
    
    return result


async def check_single_rule(
    rule_id: int,
    user_id: int,
    ignore_cooldown: bool = False,
) -> dict[str, Any]:
    """Check a single rule (for testing or manual trigger).
    
    Args:
        rule_id: The rule to check
        user_id: The user ID (for access control)
        ignore_cooldown: If True, bypass cooldown check
        
    Returns:
        Result dict with details
    """
    result = {
        "rule_id": rule_id,
        "triggered": False,
        "would_send": False,
        "actual_value": None,
        "threshold_value": None,
        "description": None,
        "skip_reasons": [],
    }
    
    # Get the rule
    rule = await repo.get_rule(rule_id, user_id)
    if not rule:
        result["error"] = "Rule not found"
        return result
    
    result["rule_name"] = rule["name"]
    result["trigger_type"] = rule["trigger_type"]
    result["threshold_value"] = rule.get("target_value")
    
    # Get channel
    channel = await repo.get_channel(rule["channel_id"], user_id)
    if not channel:
        result["skip_reasons"].append("Channel not found")
        return result
    
    if not channel.get("is_active"):
        result["skip_reasons"].append("Channel is disabled")
    
    # Check cooldown
    if not ignore_cooldown and await check_cooldown(user_id, rule_id):
        result["skip_reasons"].append("In cooldown")
    
    # Check rate limit
    is_limited, count = await check_rate_limit(user_id)
    if is_limited:
        result["skip_reasons"].append(f"Rate limited ({count}/{50})")
    
    # Get data
    if rule.get("target_symbol"):
        data = await batch_fetch_symbol_data([rule["target_symbol"]])
        data = data.get(rule["target_symbol"], {})
    elif rule.get("target_portfolio_id"):
        data = await batch_fetch_portfolio_data([rule["target_portfolio_id"]])
        data = data.get(rule["target_portfolio_id"], {})
    else:
        data = {}
    
    if not data:
        result["skip_reasons"].append("No data available")
        return result
    
    result["data_preview"] = {
        k: v for k, v in data.items()
        if k in ["current_price", "dip_percent", "quant_score", "portfolio_total_value"]
    }
    
    # Check staleness
    updated_at = data.get("dip_updated_at") or data.get("portfolio_updated_at")
    is_stale, age_hours = check_staleness(updated_at)
    if is_stale:
        result["skip_reasons"].append(f"Data stale ({age_hours:.1f}h old)")
    
    # Evaluate trigger
    triggered, actual_value, description = evaluate_trigger(
        trigger_type=rule["trigger_type"],
        data=data,
        comparison_operator=rule["comparison_operator"],
        target_value=rule.get("target_value"),
        smart_payload=rule.get("smart_payload"),
    )
    
    result["triggered"] = triggered
    result["actual_value"] = actual_value
    result["description"] = description
    
    # Would it send?
    result["would_send"] = triggered and len(result["skip_reasons"]) == 0
    
    return result
