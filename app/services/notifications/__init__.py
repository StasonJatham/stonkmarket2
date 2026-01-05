"""
Notification Service Package - Context-aware notification system with Apprise.

This package provides a complete notification system with:
- Multiple channel types (Discord, Telegram, Email, Slack, etc.)
- Trigger-based rules with thresholds and comparisons
- Cooldown management to prevent spam
- Batch-optimized data fetching
- Content deduplication

Usage:
    from app.services.notifications import (
        # Core operations
        check_all_rules,
        send_notification,
        test_channel,
        
        # Trigger evaluation
        evaluate_trigger,
        get_trigger_data,
        
        # Cooldown management
        check_cooldown,
        set_cooldown,
        
        # Safety checks
        check_rate_limit,
        check_staleness,
    )
"""

from app.services.notifications.sender import (
    send_notification,
    send_via_apprise,
    test_channel,
)
from app.services.notifications.cooldown import (
    check_cooldown,
    set_cooldown,
    clear_cooldown,
    get_remaining_cooldown,
)
from app.services.notifications.safety import (
    check_rate_limit,
    increment_rate_limit,
    check_staleness,
    check_duplicate,
)
from app.services.notifications.triggers import (
    evaluate_trigger,
    compare_values,
)
from app.services.notifications.data_fetcher import (
    get_trigger_data,
    batch_fetch_symbol_data,
    batch_fetch_portfolio_data,
)
from app.services.notifications.checker import (
    check_all_rules,
    check_single_rule,
)
from app.services.notifications.message_builder import (
    build_notification_message,
    format_trigger_value,
)

__all__ = [
    # Sender
    "send_notification",
    "send_via_apprise",
    "test_channel",
    # Cooldown
    "check_cooldown",
    "set_cooldown",
    "clear_cooldown",
    "get_remaining_cooldown",
    # Safety
    "check_rate_limit",
    "increment_rate_limit",
    "check_staleness",
    "check_duplicate",
    # Triggers
    "evaluate_trigger",
    "compare_values",
    # Data fetching
    "get_trigger_data",
    "batch_fetch_symbol_data",
    "batch_fetch_portfolio_data",
    # Checker
    "check_all_rules",
    "check_single_rule",
    # Message building
    "build_notification_message",
    "format_trigger_value",
]
