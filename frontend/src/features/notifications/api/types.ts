/**
 * TypeScript types for the Notification System.
 * 
 * These types mirror the backend Pydantic schemas from app/schemas/notifications.py
 */

// =============================================================================
// ENUMS
// =============================================================================

export type TriggerType =
  // Price & Dip
  | 'PRICE_DROPS_BELOW'
  | 'PRICE_RISES_ABOVE'
  | 'DIP_EXCEEDS_PERCENT'
  | 'DIP_DURATION_EXCEEDS'
  | 'TAIL_EVENT_DETECTED'
  | 'NEW_ATH_REACHED'
  // Signals
  | 'DIPFINDER_ALERT'
  | 'DIPFINDER_SCORE_ABOVE'
  | 'STRATEGY_SIGNAL_BUY'
  | 'STRATEGY_SIGNAL_SELL'
  | 'ENTRY_SIGNAL_TRIGGERED'
  | 'WIN_RATE_ABOVE'
  // Fundamentals
  | 'PE_RATIO_BELOW'
  | 'PE_RATIO_ABOVE'
  | 'ANALYST_UPGRADE'
  | 'ANALYST_DOWNGRADE'
  | 'PRICE_BELOW_TARGET'
  | 'EARNINGS_APPROACHING'
  | 'QUALITY_SCORE_ABOVE'
  | 'MOMENTUM_SCORE_ABOVE'
  // AI Analysis
  | 'AI_RATING_STRONG_BUY'
  | 'AI_RATING_CHANGE'
  | 'AI_CONFIDENCE_HIGH'
  | 'AI_CONSENSUS_BUY'
  // Portfolio
  | 'PORTFOLIO_VALUE_ABOVE'
  | 'PORTFOLIO_VALUE_BELOW'
  | 'PORTFOLIO_DRAWDOWN_EXCEEDS'
  | 'POSITION_WEIGHT_EXCEEDS'
  | 'PORTFOLIO_GAIN_EXCEEDS'
  | 'PORTFOLIO_LOSS_EXCEEDS'
  // Watchlist
  | 'WATCHLIST_STOCK_DIPS'
  | 'WATCHLIST_OPPORTUNITY';

export type ComparisonOperator = 'GT' | 'LT' | 'GTE' | 'LTE' | 'EQ' | 'NEQ' | 'CHANGE';

export type ChannelType = 
  | 'discord' 
  | 'telegram' 
  | 'email' 
  | 'slack' 
  | 'pushover' 
  | 'ntfy' 
  | 'webhook';

export type RulePriority = 'low' | 'normal' | 'high' | 'critical';

export type NotificationStatus = 'pending' | 'sent' | 'failed' | 'skipped';

// =============================================================================
// TRIGGER TYPE INFO
// =============================================================================

export interface TriggerTypeInfo {
  type: TriggerType;
  name: string;
  description: string;
  category: string;
  requires_symbol: boolean;
  requires_portfolio: boolean;
  default_value: number | null;
  default_operator: ComparisonOperator;
  value_unit: string | null;
  is_boolean: boolean;
}

// =============================================================================
// CHANNEL TYPES
// =============================================================================

export interface NotificationChannel {
  id: number;
  name: string;
  channel_type: ChannelType;
  is_verified: boolean;
  is_active: boolean;
  error_count: number;
  last_error: string | null;
  last_used_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChannelCreateRequest {
  name: string;
  channel_type: ChannelType;
  apprise_url: string;
}

export interface ChannelUpdateRequest {
  name?: string;
  apprise_url?: string;
  is_active?: boolean;
}

export interface ChannelTestResponse {
  success: boolean;
  message: string;
}

// =============================================================================
// RULE TYPES
// =============================================================================

export interface NotificationRule {
  id: number;
  name: string;
  trigger_type: TriggerType;
  target_symbol: string | null;
  target_portfolio_id: number | null;
  comparison_operator: ComparisonOperator;
  target_value: number | null;
  smart_payload: Record<string, unknown>;
  cooldown_minutes: number;
  priority: RulePriority;
  is_active: boolean;
  last_triggered_at: string | null;
  trigger_count: number;
  channel: NotificationChannel;
  created_at: string;
  updated_at: string;
}

export interface RuleCreateRequest {
  name: string;
  channel_id: number;
  trigger_type: TriggerType;
  target_symbol?: string | null;
  target_portfolio_id?: number | null;
  comparison_operator?: ComparisonOperator;
  target_value?: number | null;
  smart_payload?: Record<string, unknown>;
  cooldown_minutes?: number;
  priority?: RulePriority;
}

export interface RuleUpdateRequest {
  name?: string;
  channel_id?: number;
  comparison_operator?: ComparisonOperator;
  target_value?: number | null;
  smart_payload?: Record<string, unknown>;
  cooldown_minutes?: number;
  priority?: RulePriority;
  is_active?: boolean;
}

export interface RuleTestResponse {
  would_trigger: boolean;
  current_value: number | null;
  threshold_value: number | null;
  message: string;
}

// =============================================================================
// NOTIFICATION LOG TYPES
// =============================================================================

export interface NotificationLogEntry {
  id: number;
  rule_id: number | null;
  rule_name: string | null;
  channel_name: string | null;
  trigger_type: TriggerType;
  trigger_symbol: string | null;
  trigger_value: number | null;
  threshold_value: number | null;
  title: string;
  body: string;
  status: NotificationStatus;
  error_message: string | null;
  triggered_at: string;
  sent_at: string | null;
}

export interface NotificationHistoryResponse {
  notifications: NotificationLogEntry[];
  total: number;
  page: number;
  page_size: number;
}

// =============================================================================
// SUMMARY TYPES
// =============================================================================

export interface NotificationSummary {
  total_channels: number;
  active_channels: number;
  total_rules: number;
  active_rules: number;
  notifications_today: number;
  notifications_this_week: number;
}
