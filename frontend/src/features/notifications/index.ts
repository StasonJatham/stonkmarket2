/**
 * Notifications feature - clean exports.
 */

// Types
export type {
  TriggerType,
  ComparisonOperator,
  ChannelType,
  RulePriority,
  NotificationStatus,
  TriggerTypeInfo,
  NotificationChannel,
  ChannelCreateRequest,
  ChannelUpdateRequest,
  ChannelTestResponse,
  NotificationRule,
  RuleCreateRequest,
  RuleUpdateRequest,
  RuleTestResponse,
  NotificationLogEntry,
  NotificationHistoryResponse,
  NotificationSummary,
} from './api/types';

// Queries and Mutations
export {
  // Channel hooks
  useNotificationChannels,
  useCreateChannel,
  useUpdateChannel,
  useDeleteChannel,
  useTestChannel,
  // Rule hooks
  useNotificationRules,
  useCreateRule,
  useUpdateRule,
  useDeleteRule,
  useTestRule,
  useClearRuleCooldown,
  // History and summary
  useNotificationHistory,
  useNotificationSummary,
  // Trigger types
  useTriggerTypes,
  // Utils
  invalidateNotificationQueries,
} from './api/queries';
