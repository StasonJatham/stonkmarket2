"""Trigger evaluation for notification rules.

Maps trigger types to data sources and evaluates conditions.

Each trigger type corresponds to a specific metric that can be compared
against a threshold value using standard comparison operators.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.core.logging import get_logger
from app.schemas.notifications import (
    TriggerType,
    ComparisonOperator,
    TRIGGER_TYPE_INFO,
)


logger = get_logger("notifications.triggers")


def compare_values(
    actual: Decimal | float | int | None,
    threshold: Decimal | float | int | None,
    operator: str,
) -> bool:
    """Compare actual value against threshold using operator.
    
    Args:
        actual: The actual value from data
        threshold: The threshold value from rule
        operator: Comparison operator (GT, LT, GTE, LTE, EQ, NEQ, CHANGE)
        
    Returns:
        True if condition is met, False otherwise
    """
    if actual is None or threshold is None:
        return False
    
    # Convert to Decimal for precision
    actual_d = Decimal(str(actual))
    threshold_d = Decimal(str(threshold))
    
    op = ComparisonOperator(operator.upper())
    
    if op == ComparisonOperator.GT:
        return actual_d > threshold_d
    elif op == ComparisonOperator.LT:
        return actual_d < threshold_d
    elif op == ComparisonOperator.GTE:
        return actual_d >= threshold_d
    elif op == ComparisonOperator.LTE:
        return actual_d <= threshold_d
    elif op == ComparisonOperator.EQ:
        return actual_d == threshold_d
    elif op == ComparisonOperator.NEQ:
        return actual_d != threshold_d
    elif op == ComparisonOperator.CHANGE:
        # CHANGE: absolute value exceeds threshold (for detecting any significant change)
        return abs(actual_d) >= abs(threshold_d)
    
    return False


def get_trigger_info(trigger_type: str) -> dict[str, Any]:
    """Get metadata about a trigger type.
    
    Args:
        trigger_type: The trigger type string
        
    Returns:
        Dict with category, default_operator, default_value, value_unit, etc.
    """
    return TRIGGER_TYPE_INFO.get(trigger_type, {
        "category": "other",
        "default_operator": "GT",
        "default_value": None,
        "value_unit": None,
        "requires_symbol": False,
        "requires_portfolio": False,
    })


def evaluate_trigger(
    trigger_type: str,
    data: dict[str, Any],
    comparison_operator: str,
    target_value: Decimal | float | int | None,
    smart_payload: dict[str, Any] | None = None,
) -> tuple[bool, Any, str]:
    """Evaluate if a trigger condition is met.
    
    Args:
        trigger_type: The type of trigger (from TriggerType enum)
        data: The data dict with relevant metrics
        comparison_operator: How to compare (GT, LT, etc.)
        target_value: The threshold value
        smart_payload: Additional configuration for complex triggers
        
    Returns:
        Tuple of (triggered, actual_value, description)
    """
    try:
        tt = TriggerType(trigger_type)
    except ValueError:
        logger.warning(f"Unknown trigger type: {trigger_type}")
        return False, None, f"Unknown trigger type: {trigger_type}"
    
    # Extract the relevant value based on trigger type
    actual_value, description = _extract_trigger_value(tt, data, smart_payload)
    
    if actual_value is None:
        return False, None, "Data not available"
    
    # Compare
    triggered = compare_values(actual_value, target_value, comparison_operator)
    
    return triggered, actual_value, description


def _extract_trigger_value(
    trigger_type: TriggerType,
    data: dict[str, Any],
    smart_payload: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Extract the value to compare from data based on trigger type.
    
    Args:
        trigger_type: The trigger type
        data: The data dict
        smart_payload: Additional configuration
        
    Returns:
        Tuple of (value, description)
    """
    smart_payload = smart_payload or {}
    
    # ==========================================================================
    # PRICE & DIP TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.PRICE_DROPS_BELOW:
        price = data.get("current_price")
        return price, f"Price: ${price:.2f}" if price else "Price data unavailable"
    
    if trigger_type == TriggerType.PRICE_RISES_ABOVE:
        price = data.get("current_price")
        return price, f"Price: ${price:.2f}" if price else "Price data unavailable"
    
    if trigger_type == TriggerType.DIP_EXCEEDS_PERCENT:
        dip = data.get("dip_percent")
        return dip, f"Dip: {dip:.1f}%" if dip else "Dip data unavailable"
    
    if trigger_type == TriggerType.DIP_DURATION_EXCEEDS:
        days = data.get("dip_duration_days")
        return days, f"Dip Duration: {days} days" if days else "Dip duration unavailable"
    
    if trigger_type == TriggerType.TAIL_EVENT_DETECTED:
        is_tail = data.get("is_tail_event")
        return 1 if is_tail else 0, "Tail Event Detected" if is_tail else "No tail event"
    
    if trigger_type == TriggerType.NEW_ATH_REACHED:
        is_ath = data.get("is_ath")
        return 1 if is_ath else 0, "New All-Time High" if is_ath else "Not at ATH"
    
    # ==========================================================================
    # SIGNAL TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.DIPFINDER_ALERT:
        is_dip = data.get("dipfinder_active")
        return 1 if is_dip else 0, "DipFinder Alert" if is_dip else "No DipFinder alert"
    
    if trigger_type == TriggerType.DIPFINDER_SCORE_ABOVE:
        score = data.get("dipfinder_confidence")
        if score is not None:
            score = score * 100  # Convert to percentage
        return score, f"DipFinder Score: {score:.0f}" if score else "DipFinder score unavailable"
    
    if trigger_type == TriggerType.STRATEGY_SIGNAL_BUY:
        has_buy = data.get("strategy_signal") == "buy"
        return 1 if has_buy else 0, data.get("strategy_name", "Strategy") + " Buy Signal"
    
    if trigger_type == TriggerType.STRATEGY_SIGNAL_SELL:
        has_sell = data.get("strategy_signal") == "sell"
        return 1 if has_sell else 0, data.get("strategy_name", "Strategy") + " Sell Signal"
    
    if trigger_type == TriggerType.ENTRY_SIGNAL_TRIGGERED:
        has_entry = data.get("entry_signal_triggered")
        return 1 if has_entry else 0, "Entry Signal Triggered" if has_entry else "No entry signal"
    
    if trigger_type == TriggerType.WIN_RATE_ABOVE:
        rate = data.get("strategy_win_rate")
        return rate, f"Win Rate: {rate:.1f}%" if rate else "Win rate unavailable"
    
    # ==========================================================================
    # FUNDAMENTAL TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.PE_RATIO_BELOW:
        pe = data.get("pe_ratio")
        return pe, f"P/E Ratio: {pe:.1f}" if pe else "P/E unavailable"
    
    if trigger_type == TriggerType.PE_RATIO_ABOVE:
        pe = data.get("pe_ratio")
        return pe, f"P/E Ratio: {pe:.1f}" if pe else "P/E unavailable"
    
    if trigger_type == TriggerType.ANALYST_UPGRADE:
        # recommendation_mean: 1.0=strong buy, 2.0=buy, 3.0=hold
        # Lower is more bullish - trigger when below threshold (e.g., < 2.0 = strong buy)
        rating = data.get("recommendation_mean")
        if rating is not None:
            is_bullish = rating <= 2.0  # Strong buy or buy
            return rating, f"Analyst Rating: {rating:.1f} (bullish)" if is_bullish else f"Analyst Rating: {rating:.1f}"
        return None, "Analyst rating unavailable"
    
    if trigger_type == TriggerType.ANALYST_DOWNGRADE:
        # recommendation_mean: 4.0=sell, 5.0=strong sell
        # Higher is more bearish - trigger when above threshold (e.g., > 3.5 = sell/strong sell)
        rating = data.get("recommendation_mean")
        if rating is not None:
            is_bearish = rating >= 3.5  # Sell or strong sell
            return rating, f"Analyst Rating: {rating:.1f} (bearish)" if is_bearish else f"Analyst Rating: {rating:.1f}"
        return None, "Analyst rating unavailable"
    
    if trigger_type == TriggerType.PRICE_BELOW_TARGET:
        price = data.get("current_price")
        target = data.get("analyst_price_target")
        if price and target and target > 0:
            discount_pct = ((target - price) / target) * 100
            return discount_pct, f"Price ${price:.2f} is {discount_pct:.1f}% below target ${target:.2f}"
        return None, "Price/target data unavailable"
    
    if trigger_type == TriggerType.EARNINGS_APPROACHING:
        days = data.get("earnings_days_until")
        return days, f"Earnings in {days} days" if days is not None else "No upcoming earnings"
    
    if trigger_type == TriggerType.QUALITY_SCORE_ABOVE:
        score = data.get("value_score")  # Using value_score as quality proxy
        return score, f"Quality Score: {score:.0f}" if score else "Quality score unavailable"
    
    if trigger_type == TriggerType.MOMENTUM_SCORE_ABOVE:
        score = data.get("momentum_score")
        return score, f"Momentum Score: {score:.0f}" if score else "Momentum score unavailable"
    
    # ==========================================================================
    # AI ANALYSIS TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.AI_RATING_STRONG_BUY:
        rating = data.get("ai_rating")
        is_strong_buy = rating == "strong_buy"
        return 1 if is_strong_buy else 0, f"AI Rating: {rating}" if rating else "AI rating unavailable"
    
    if trigger_type == TriggerType.AI_RATING_CHANGE:
        current = data.get("ai_rating")
        previous = data.get("ai_rating_prev")
        has_change = current and previous and current != previous
        return 1 if has_change else 0, f"AI Rating changed: {previous} → {current}" if has_change else "No rating change"
    
    if trigger_type == TriggerType.AI_CONFIDENCE_HIGH:
        confidence = data.get("ai_confidence")
        return confidence, f"AI Confidence: {confidence:.1f}/10" if confidence else "AI confidence unavailable"
    
    if trigger_type == TriggerType.AI_CONSENSUS_BUY:
        consensus_count = data.get("ai_consensus_buy_count", 0)
        return consensus_count, f"AI Consensus: {consensus_count} personas agree BUY"
    
    # ==========================================================================
    # PORTFOLIO TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.PORTFOLIO_VALUE_ABOVE:
        value = data.get("portfolio_total_value")
        return value, f"Portfolio Value: ${value:,.0f}" if value else "Value unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_VALUE_BELOW:
        value = data.get("portfolio_total_value")
        return value, f"Portfolio Value: ${value:,.0f}" if value else "Value unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS:
        dd = data.get("portfolio_drawdown")
        return dd, f"Drawdown: {dd:.1f}%" if dd else "Drawdown unavailable"
    
    if trigger_type == TriggerType.POSITION_WEIGHT_EXCEEDS:
        # Find max position weight
        max_weight = data.get("max_position_weight_percent")
        symbol = data.get("max_position_symbol", "Position")
        return max_weight, f"{symbol}: {max_weight:.1f}% of portfolio" if max_weight else "Position weight unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_GAIN_EXCEEDS:
        gain = data.get("portfolio_daily_change_percent")
        if gain is not None and gain > 0:
            return gain, f"Portfolio Gain: +{gain:.2f}%"
        return 0, "No portfolio gain"
    
    if trigger_type == TriggerType.PORTFOLIO_LOSS_EXCEEDS:
        change = data.get("portfolio_daily_change_percent")
        if change is not None and change < 0:
            loss = abs(change)
            return loss, f"Portfolio Loss: -{loss:.2f}%"
        return 0, "No portfolio loss"
    
    # ==========================================================================
    # WATCHLIST TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.WATCHLIST_STOCK_DIPS:
        # Would check watchlist stocks for dips
        dipping_count = data.get("watchlist_dipping_count", 0)
        return dipping_count, f"{dipping_count} watchlist stocks dipping" if dipping_count else "No watchlist dips"
    
    if trigger_type == TriggerType.WATCHLIST_OPPORTUNITY:
        has_opp = data.get("watchlist_has_opportunity")
        return 1 if has_opp else 0, "Watchlist opportunity found" if has_opp else "No watchlist opportunity"
    
    # Default case
    logger.warning(f"No extractor for trigger type: {trigger_type}")
    return None, f"Unsupported trigger type: {trigger_type}"


def get_required_data_fields(trigger_type: str) -> list[str]:
    """Get list of data fields required for a trigger type.
    
    Useful for optimizing data fetching.
    
    Args:
        trigger_type: The trigger type string
        
    Returns:
        List of field names needed from data
    """
    field_map = {
        # Price & Dip
        TriggerType.PRICE_DROPS_BELOW.value: ["current_price"],
        TriggerType.PRICE_RISES_ABOVE.value: ["current_price"],
        TriggerType.DIP_EXCEEDS_PERCENT.value: ["dip_percent"],
        TriggerType.DIP_DURATION_EXCEEDS.value: ["dip_duration_days"],
        TriggerType.TAIL_EVENT_DETECTED.value: ["is_tail_event"],
        TriggerType.NEW_ATH_REACHED.value: ["is_ath"],
        
        # Signals
        TriggerType.DIPFINDER_ALERT.value: ["dipfinder_active"],
        TriggerType.DIPFINDER_SCORE_ABOVE.value: ["dipfinder_confidence"],
        TriggerType.STRATEGY_SIGNAL_BUY.value: ["strategy_signal", "strategy_name"],
        TriggerType.STRATEGY_SIGNAL_SELL.value: ["strategy_signal", "strategy_name"],
        TriggerType.ENTRY_SIGNAL_TRIGGERED.value: ["entry_signal_triggered"],
        TriggerType.WIN_RATE_ABOVE.value: ["strategy_win_rate"],
        
        # Fundamentals
        TriggerType.PE_RATIO_BELOW.value: ["pe_ratio"],
        TriggerType.PE_RATIO_ABOVE.value: ["pe_ratio"],
        TriggerType.ANALYST_UPGRADE.value: ["recommendation_mean"],
        TriggerType.ANALYST_DOWNGRADE.value: ["recommendation_mean"],
        TriggerType.PRICE_BELOW_TARGET.value: ["current_price", "analyst_price_target"],
        TriggerType.EARNINGS_APPROACHING.value: ["earnings_days_until"],
        TriggerType.QUALITY_SCORE_ABOVE.value: ["value_score"],
        TriggerType.MOMENTUM_SCORE_ABOVE.value: ["momentum_score"],
        
        # AI Analysis
        TriggerType.AI_RATING_STRONG_BUY.value: ["ai_rating"],
        TriggerType.AI_RATING_CHANGE.value: ["ai_rating", "ai_rating_prev"],
        TriggerType.AI_CONFIDENCE_HIGH.value: ["ai_confidence"],
        TriggerType.AI_CONSENSUS_BUY.value: ["ai_consensus_buy_count"],
        
        # Portfolio
        TriggerType.PORTFOLIO_VALUE_ABOVE.value: ["portfolio_total_value"],
        TriggerType.PORTFOLIO_VALUE_BELOW.value: ["portfolio_total_value"],
        TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS.value: ["portfolio_drawdown"],
        TriggerType.POSITION_WEIGHT_EXCEEDS.value: ["max_position_weight_percent", "max_position_symbol"],
        TriggerType.PORTFOLIO_GAIN_EXCEEDS.value: ["portfolio_daily_change_percent"],
        TriggerType.PORTFOLIO_LOSS_EXCEEDS.value: ["portfolio_daily_change_percent"],
        
        # Watchlist
        TriggerType.WATCHLIST_STOCK_DIPS.value: ["watchlist_dipping_count"],
        TriggerType.WATCHLIST_OPPORTUNITY.value: ["watchlist_has_opportunity"],
    }
    
    return field_map.get(trigger_type, [])
