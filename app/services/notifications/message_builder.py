"""Message building for notifications.

Constructs human-readable notification messages from trigger data.
Supports different formats for different channel types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.schemas.notifications import TriggerType, TRIGGER_TYPE_INFO


def build_notification_message(
    trigger_type: str,
    actual_value: Any,
    threshold_value: Any,
    symbol: str | None = None,
    portfolio_name: str | None = None,
    rule_name: str | None = None,
    additional_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build notification title and body.
    
    Args:
        trigger_type: The type of trigger
        actual_value: The value that triggered
        threshold_value: The threshold that was exceeded
        symbol: Stock symbol (if applicable)
        portfolio_name: Portfolio name (if applicable)
        rule_name: The rule name for reference
        additional_context: Extra data for rich messages
        
    Returns:
        Tuple of (title, body)
    """
    trigger_info = TRIGGER_TYPE_INFO.get(trigger_type, {})
    category = trigger_info.get("category", "other")
    value_unit = trigger_info.get("value_unit", "")
    
    # Build title
    title = _build_title(trigger_type, symbol, portfolio_name)
    
    # Build body
    body_parts = []
    
    # Main message
    main_message = _get_trigger_message(trigger_type, actual_value, threshold_value, value_unit)
    body_parts.append(main_message)
    
    # Add context
    if symbol:
        body_parts.append(f"Symbol: {symbol}")
    
    if portfolio_name:
        body_parts.append(f"Portfolio: {portfolio_name}")
    
    if rule_name:
        body_parts.append(f"Rule: {rule_name}")
    
    # Add formatted values
    if actual_value is not None:
        formatted = format_trigger_value(actual_value, value_unit)
        body_parts.append(f"Current: {formatted}")
    
    if threshold_value is not None:
        formatted = format_trigger_value(threshold_value, value_unit)
        body_parts.append(f"Threshold: {formatted}")
    
    # Add timestamp
    body_parts.append(f"Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    
    body = "\n".join(body_parts)
    
    return title, body


def _build_title(
    trigger_type: str,
    symbol: str | None,
    portfolio_name: str | None,
) -> str:
    """Build notification title based on trigger type."""
    
    title_map = {
        # Price triggers
        TriggerType.PRICE_DROPS_BELOW.value: f"Price Alert: {symbol} Below Target",
        TriggerType.PRICE_RISES_ABOVE.value: f"Price Alert: {symbol} Above Target",
        TriggerType.PRICE_CHANGE_PERCENT.value: f"Price Movement: {symbol}",
        
        # Dip triggers
        TriggerType.DIP_EXCEEDS_PERCENT.value: f"Dip Alert: {symbol}",
        TriggerType.DIP_RECOVERY.value: f"Recovery Alert: {symbol}",
        TriggerType.DIP_52W_THRESHOLD.value: f"52-Week Dip: {symbol}",
        TriggerType.DIPFINDER_ALERT.value: f"Dipfinder: {symbol}",
        
        # Quant triggers
        TriggerType.QUANT_SCORE_ABOVE.value: f"Quant Score High: {symbol}",
        TriggerType.QUANT_SCORE_BELOW.value: f"Quant Score Low: {symbol}",
        TriggerType.QUANT_RECOMMENDATION.value: f"Quant Recommendation: {symbol}",
        TriggerType.VALUE_SCORE_THRESHOLD.value: f"Value Score: {symbol}",
        TriggerType.GROWTH_SCORE_THRESHOLD.value: f"Growth Score: {symbol}",
        TriggerType.MOMENTUM_SCORE_THRESHOLD.value: f"Momentum Score: {symbol}",
        
        # Fundamental triggers
        TriggerType.PE_RATIO_BELOW.value: f"P/E Ratio Alert: {symbol}",
        TriggerType.PE_RATIO_ABOVE.value: f"P/E Ratio Alert: {symbol}",
        TriggerType.DIVIDEND_YIELD_ABOVE.value: f"Dividend Alert: {symbol}",
        TriggerType.MARKET_CAP_BELOW.value: f"Market Cap Alert: {symbol}",
        TriggerType.MARKET_CAP_ABOVE.value: f"Market Cap Alert: {symbol}",
        
        # Strategy triggers
        TriggerType.STRATEGY_SIGNAL_BUY.value: f"Buy Signal: {symbol}",
        TriggerType.STRATEGY_SIGNAL_SELL.value: f"Sell Signal: {symbol}",
        TriggerType.STRATEGY_CONFIDENCE.value: f"Strategy Alert: {symbol}",
        
        # AI triggers
        TriggerType.AI_RATING_STRONG_BUY.value: f"AI Strong Buy: {symbol}",
        TriggerType.AI_RATING_SELL.value: f"AI Sell Alert: {symbol}",
        TriggerType.AI_OPPORTUNITY_DETECTED.value: f"AI Opportunity: {symbol}",
        TriggerType.AI_RISK_ALERT.value: f"AI Risk Alert: {symbol}",
        
        # Portfolio triggers
        TriggerType.PORTFOLIO_TOTAL_VALUE.value: f"Portfolio Value: {portfolio_name}",
        TriggerType.PORTFOLIO_DAILY_CHANGE.value: f"Portfolio Change: {portfolio_name}",
        TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS.value: f"Portfolio Drawdown: {portfolio_name}",
        TriggerType.PORTFOLIO_POSITION_SIZE.value: f"Position Size Alert: {portfolio_name}",
        TriggerType.PORTFOLIO_GAIN_ABOVE.value: f"Position Gain: {portfolio_name}",
        TriggerType.PORTFOLIO_LOSS_BELOW.value: f"Position Loss: {portfolio_name}",
        
        # Calendar triggers
        TriggerType.EARNINGS_UPCOMING.value: f"Earnings Soon: {symbol}",
        TriggerType.EX_DIVIDEND_UPCOMING.value: f"Ex-Dividend Soon: {symbol}",
        TriggerType.STOCK_SPLIT_UPCOMING.value: f"Stock Split Soon: {symbol}",
    }
    
    return title_map.get(trigger_type, f"Stonkmarket Alert: {symbol or portfolio_name}")


def _get_trigger_message(
    trigger_type: str,
    actual_value: Any,
    threshold_value: Any,
    value_unit: str | None,
) -> str:
    """Get the main message for a trigger."""
    
    formatted_actual = format_trigger_value(actual_value, value_unit)
    formatted_threshold = format_trigger_value(threshold_value, value_unit)
    
    messages = {
        # Price triggers
        TriggerType.PRICE_DROPS_BELOW.value: f"Price dropped to {formatted_actual} (threshold: {formatted_threshold})",
        TriggerType.PRICE_RISES_ABOVE.value: f"Price rose to {formatted_actual} (threshold: {formatted_threshold})",
        TriggerType.PRICE_CHANGE_PERCENT.value: f"Price changed by {formatted_actual}",
        
        # Dip triggers
        TriggerType.DIP_EXCEEDS_PERCENT.value: f"Stock is down {formatted_actual} from peak",
        TriggerType.DIP_RECOVERY.value: f"Stock has recovered {formatted_actual} from dip",
        TriggerType.DIP_52W_THRESHOLD.value: f"Stock is {formatted_actual} from 52-week high",
        TriggerType.DIPFINDER_ALERT.value: "Dipfinder detected a buying opportunity",
        
        # Quant triggers
        TriggerType.QUANT_SCORE_ABOVE.value: f"Quant score reached {formatted_actual}",
        TriggerType.QUANT_SCORE_BELOW.value: f"Quant score dropped to {formatted_actual}",
        TriggerType.QUANT_RECOMMENDATION.value: f"New quant recommendation: {actual_value}",
        
        # Portfolio triggers
        TriggerType.PORTFOLIO_TOTAL_VALUE.value: f"Portfolio value is {formatted_actual}",
        TriggerType.PORTFOLIO_DAILY_CHANGE.value: f"Portfolio changed {formatted_actual} today",
        TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS.value: f"Portfolio drawdown is {formatted_actual}",
    }
    
    return messages.get(trigger_type, f"Condition met: {formatted_actual}")


def format_trigger_value(value: Any, unit: str | None = None) -> str:
    """Format a trigger value for display.
    
    Args:
        value: The value to format
        unit: Optional unit (%, $, days, etc.)
        
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    # Convert to float for formatting
    if isinstance(value, Decimal):
        value = float(value)
    
    unit = unit or ""
    
    if unit == "%":
        return f"{value:+.2f}%"
    elif unit == "$":
        if abs(value) >= 1e9:
            return f"${value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.2f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.2f}K"
        else:
            return f"${value:,.2f}"
    elif unit == "days":
        return f"{int(value)} days"
    elif unit == "score":
        return f"{value:.0f}/100"
    elif isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"
    else:
        return str(value)


def format_discord_message(title: str, body: str, priority: str = "normal") -> dict[str, Any]:
    """Format message for Discord embed.
    
    Args:
        title: Notification title
        body: Notification body
        priority: Priority level
        
    Returns:
        Discord embed dict
    """
    color_map = {
        "low": 0x808080,      # Gray
        "normal": 0x3498db,   # Blue
        "high": 0xf39c12,     # Orange
        "critical": 0xe74c3c, # Red
    }
    
    return {
        "embeds": [{
            "title": title,
            "description": body,
            "color": color_map.get(priority, 0x3498db),
            "timestamp": datetime.now(UTC).isoformat(),
            "footer": {
                "text": "Stonkmarket Notifications"
            }
        }]
    }


def format_telegram_message(title: str, body: str) -> str:
    """Format message for Telegram (Markdown).
    
    Args:
        title: Notification title
        body: Notification body
        
    Returns:
        Markdown-formatted message
    """
    return f"*{title}*\n\n{body}"


def format_email_message(title: str, body: str, priority: str = "normal") -> tuple[str, str]:
    """Format message for email (subject + HTML body).
    
    Args:
        title: Notification title
        body: Notification body
        priority: Priority level
        
    Returns:
        Tuple of (subject, html_body)
    """
    subject = f"[Stonkmarket] {title}"
    
    priority_colors = {
        "low": "#808080",
        "normal": "#3498db",
        "high": "#f39c12",
        "critical": "#e74c3c",
    }
    color = priority_colors.get(priority, "#3498db")
    
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
            <h2 style="margin: 0;">{title}</h2>
        </div>
        <div style="background: #f5f5f5; padding: 20px; border-radius: 0 0 5px 5px;">
            <pre style="white-space: pre-wrap; font-family: inherit;">{body}</pre>
        </div>
        <p style="color: #888; font-size: 12px; text-align: center; margin-top: 20px;">
            Sent by Stonkmarket Notifications
        </p>
    </body>
    </html>
    """
    
    return subject, html_body
