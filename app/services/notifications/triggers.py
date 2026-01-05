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
    # PRICE TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.PRICE_DROPS_BELOW:
        price = data.get("current_price")
        return price, f"Price: ${price:.2f}" if price else "Price data unavailable"
    
    if trigger_type == TriggerType.PRICE_RISES_ABOVE:
        price = data.get("current_price")
        return price, f"Price: ${price:.2f}" if price else "Price data unavailable"
    
    if trigger_type == TriggerType.PRICE_CHANGE_PERCENT:
        change = data.get("price_change_percent")
        return change, f"Change: {change:+.2f}%" if change else "Change data unavailable"
    
    # ==========================================================================
    # DIP TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.DIP_EXCEEDS_PERCENT:
        dip = data.get("dip_percent")
        return dip, f"Dip: {dip:.1f}%" if dip else "Dip data unavailable"
    
    if trigger_type == TriggerType.DIP_RECOVERY:
        recovery = data.get("recovery_percent")
        return recovery, f"Recovery: {recovery:.1f}%" if recovery else "Recovery data unavailable"
    
    if trigger_type == TriggerType.DIP_52W_THRESHOLD:
        dip_52w = data.get("dip_from_52w_high")
        return dip_52w, f"52W Dip: {dip_52w:.1f}%" if dip_52w else "52W data unavailable"
    
    if trigger_type == TriggerType.DIPFINDER_ALERT:
        is_dip = data.get("dipfinder_active")
        return 1 if is_dip else 0, "Dipfinder Alert" if is_dip else "No dipfinder alert"
    
    # ==========================================================================
    # QUANT TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.QUANT_SCORE_ABOVE:
        score = data.get("quant_score")
        return score, f"Quant Score: {score}" if score else "Quant score unavailable"
    
    if trigger_type == TriggerType.QUANT_SCORE_BELOW:
        score = data.get("quant_score")
        return score, f"Quant Score: {score}" if score else "Quant score unavailable"
    
    if trigger_type == TriggerType.QUANT_RECOMMENDATION:
        rec = data.get("quant_recommendation")
        # Map recommendation to numeric for comparison
        rec_map = {
            "strong_buy": 5,
            "buy": 4,
            "hold": 3,
            "sell": 2,
            "strong_sell": 1,
        }
        return rec_map.get(rec, 0), f"Recommendation: {rec}" if rec else "Recommendation unavailable"
    
    if trigger_type == TriggerType.VALUE_SCORE_THRESHOLD:
        score = data.get("value_score")
        return score, f"Value Score: {score}" if score else "Value score unavailable"
    
    if trigger_type == TriggerType.GROWTH_SCORE_THRESHOLD:
        score = data.get("growth_score")
        return score, f"Growth Score: {score}" if score else "Growth score unavailable"
    
    if trigger_type == TriggerType.MOMENTUM_SCORE_THRESHOLD:
        score = data.get("momentum_score")
        return score, f"Momentum Score: {score}" if score else "Momentum score unavailable"
    
    # ==========================================================================
    # FUNDAMENTAL TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.PE_RATIO_BELOW:
        pe = data.get("pe_ratio")
        return pe, f"P/E Ratio: {pe:.1f}" if pe else "P/E unavailable"
    
    if trigger_type == TriggerType.PE_RATIO_ABOVE:
        pe = data.get("pe_ratio")
        return pe, f"P/E Ratio: {pe:.1f}" if pe else "P/E unavailable"
    
    if trigger_type == TriggerType.DIVIDEND_YIELD_ABOVE:
        div = data.get("dividend_yield")
        return div, f"Dividend Yield: {div:.2f}%" if div else "Dividend yield unavailable"
    
    if trigger_type == TriggerType.MARKET_CAP_BELOW:
        cap = data.get("market_cap")
        cap_b = cap / 1e9 if cap else None
        return cap, f"Market Cap: ${cap_b:.1f}B" if cap_b else "Market cap unavailable"
    
    if trigger_type == TriggerType.MARKET_CAP_ABOVE:
        cap = data.get("market_cap")
        cap_b = cap / 1e9 if cap else None
        return cap, f"Market Cap: ${cap_b:.1f}B" if cap_b else "Market cap unavailable"
    
    # ==========================================================================
    # STRATEGY SIGNAL TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.STRATEGY_SIGNAL_BUY:
        has_buy = data.get("strategy_signal") == "buy"
        return 1 if has_buy else 0, data.get("strategy_name", "Strategy") + " Buy Signal"
    
    if trigger_type == TriggerType.STRATEGY_SIGNAL_SELL:
        has_sell = data.get("strategy_signal") == "sell"
        return 1 if has_sell else 0, data.get("strategy_name", "Strategy") + " Sell Signal"
    
    if trigger_type == TriggerType.STRATEGY_CONFIDENCE:
        conf = data.get("strategy_confidence")
        return conf, f"Strategy Confidence: {conf:.0%}" if conf else "Confidence unavailable"
    
    # ==========================================================================
    # AI ANALYSIS TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.AI_RATING_STRONG_BUY:
        rating = data.get("ai_rating")
        is_strong_buy = rating == "strong_buy"
        return 1 if is_strong_buy else 0, f"AI Rating: {rating}" if rating else "AI rating unavailable"
    
    if trigger_type == TriggerType.AI_RATING_SELL:
        rating = data.get("ai_rating")
        is_sell = rating in ("sell", "strong_sell")
        return 1 if is_sell else 0, f"AI Rating: {rating}" if rating else "AI rating unavailable"
    
    if trigger_type == TriggerType.AI_OPPORTUNITY_DETECTED:
        has_opp = data.get("ai_opportunity")
        return 1 if has_opp else 0, data.get("ai_opportunity_type", "Opportunity") if has_opp else "No opportunity"
    
    if trigger_type == TriggerType.AI_RISK_ALERT:
        has_risk = data.get("ai_risk_alert")
        return 1 if has_risk else 0, data.get("ai_risk_type", "Risk Alert") if has_risk else "No risk alert"
    
    # ==========================================================================
    # PORTFOLIO TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.PORTFOLIO_TOTAL_VALUE:
        value = data.get("portfolio_total_value")
        return value, f"Portfolio Value: ${value:,.0f}" if value else "Value unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_DAILY_CHANGE:
        change = data.get("portfolio_daily_change_percent")
        return change, f"Daily Change: {change:+.2f}%" if change else "Change unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS:
        dd = data.get("portfolio_drawdown")
        return dd, f"Drawdown: {dd:.1f}%" if dd else "Drawdown unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_POSITION_SIZE:
        size = data.get("position_size_percent")
        symbol = data.get("position_symbol", "Position")
        return size, f"{symbol}: {size:.1f}%" if size else "Position size unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_GAIN_ABOVE:
        gain = data.get("position_gain_percent")
        symbol = data.get("position_symbol", "Position")
        return gain, f"{symbol} Gain: {gain:+.1f}%" if gain else "Gain unavailable"
    
    if trigger_type == TriggerType.PORTFOLIO_LOSS_BELOW:
        loss = data.get("position_gain_percent")  # Negative for loss
        symbol = data.get("position_symbol", "Position")
        return loss, f"{symbol} Loss: {loss:.1f}%" if loss else "Loss unavailable"
    
    # ==========================================================================
    # CALENDAR TRIGGERS
    # ==========================================================================
    
    if trigger_type == TriggerType.EARNINGS_UPCOMING:
        days = data.get("earnings_days_until")
        return days, f"Earnings in {days} days" if days else "No upcoming earnings"
    
    if trigger_type == TriggerType.EX_DIVIDEND_UPCOMING:
        days = data.get("ex_dividend_days_until")
        return days, f"Ex-dividend in {days} days" if days else "No upcoming ex-dividend"
    
    if trigger_type == TriggerType.STOCK_SPLIT_UPCOMING:
        days = data.get("split_days_until")
        return days, f"Stock split in {days} days" if days else "No upcoming split"
    
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
        TriggerType.TAIL_EVENT_DETECTED.value: ["tail_event"],
        TriggerType.NEW_ATH_REACHED.value: ["is_ath"],
        
        # Signals
        TriggerType.DIPFINDER_ALERT.value: ["dipfinder_active"],
        TriggerType.DIPFINDER_SCORE_ABOVE.value: ["dipfinder_score"],
        TriggerType.STRATEGY_SIGNAL_BUY.value: ["strategy_signal", "strategy_name"],
        TriggerType.STRATEGY_SIGNAL_SELL.value: ["strategy_signal", "strategy_name"],
        TriggerType.ENTRY_SIGNAL_TRIGGERED.value: ["entry_signal"],
        TriggerType.WIN_RATE_ABOVE.value: ["win_rate"],
        
        # Fundamentals
        TriggerType.PE_RATIO_BELOW.value: ["pe_ratio"],
        TriggerType.PE_RATIO_ABOVE.value: ["pe_ratio"],
        TriggerType.ANALYST_UPGRADE.value: ["analyst_rating_change"],
        TriggerType.ANALYST_DOWNGRADE.value: ["analyst_rating_change"],
        TriggerType.PRICE_BELOW_TARGET.value: ["current_price", "price_target"],
        TriggerType.EARNINGS_APPROACHING.value: ["earnings_days_until"],
        TriggerType.QUALITY_SCORE_ABOVE.value: ["quality_score"],
        TriggerType.MOMENTUM_SCORE_ABOVE.value: ["momentum_score"],
        
        # AI Analysis
        TriggerType.AI_RATING_STRONG_BUY.value: ["ai_rating"],
        TriggerType.AI_RATING_CHANGE.value: ["ai_rating", "ai_rating_prev"],
        TriggerType.AI_CONFIDENCE_HIGH.value: ["ai_confidence"],
        TriggerType.AI_CONSENSUS_BUY.value: ["ai_consensus"],
        
        # Portfolio
        TriggerType.PORTFOLIO_VALUE_ABOVE.value: ["portfolio_total_value"],
        TriggerType.PORTFOLIO_VALUE_BELOW.value: ["portfolio_total_value"],
        TriggerType.PORTFOLIO_DRAWDOWN_EXCEEDS.value: ["portfolio_drawdown"],
        TriggerType.POSITION_WEIGHT_EXCEEDS.value: ["position_weight_percent", "position_symbol"],
        TriggerType.PORTFOLIO_GAIN_EXCEEDS.value: ["portfolio_gain_percent"],
        TriggerType.PORTFOLIO_LOSS_EXCEEDS.value: ["portfolio_loss_percent"],
        
        # Watchlist
        TriggerType.WATCHLIST_STOCK_DIPS.value: ["watchlist_dip_percent"],
        TriggerType.WATCHLIST_OPPORTUNITY.value: ["watchlist_opportunity"],
    }
    
    return field_map.get(trigger_type, [])
