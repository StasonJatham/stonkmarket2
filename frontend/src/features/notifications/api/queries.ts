/**
 * React Query hooks for Notification System.
 * 
 * Provides queries and mutations for managing notification channels,
 * rules, and viewing notification history.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet, apiPost, apiPatch, apiDelete } from '@/lib/api-client';
import type {
  NotificationChannel,
  NotificationRule,
  NotificationSummary,
  NotificationHistoryResponse,
  TriggerTypeInfo,
  ChannelCreateRequest,
  ChannelUpdateRequest,
  ChannelTestResponse,
  RuleCreateRequest,
  RuleUpdateRequest,
  RuleTestResponse,
} from './types';

// =============================================================================
// CHANNELS - Queries
// =============================================================================

async function fetchChannels(): Promise<NotificationChannel[]> {
  return apiGet<NotificationChannel[]>('/notifications/channels');
}

export function useNotificationChannels() {
  return useQuery({
    queryKey: queryKeys.notifications.channels(),
    queryFn: fetchChannels,
    staleTime: 5 * 60 * 1000,
  });
}

// =============================================================================
// CHANNELS - Mutations
// =============================================================================

async function createChannel(data: ChannelCreateRequest): Promise<NotificationChannel> {
  return apiPost<NotificationChannel>('/notifications/channels', data);
}

export function useCreateChannel() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: createChannel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.channels() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.summary() });
    },
  });
}

interface UpdateChannelParams {
  channelId: number;
  data: ChannelUpdateRequest;
}

async function updateChannel({ channelId, data }: UpdateChannelParams): Promise<NotificationChannel> {
  return apiPatch<NotificationChannel>(`/notifications/channels/${channelId}`, data);
}

export function useUpdateChannel() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updateChannel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.channels() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.summary() });
    },
  });
}

async function deleteChannel(channelId: number): Promise<void> {
  return apiDelete<void>(`/notifications/channels/${channelId}`);
}

export function useDeleteChannel() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: deleteChannel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.channels() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.rules() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.summary() });
    },
  });
}

async function testChannel(channelId: number): Promise<ChannelTestResponse> {
  return apiPost<ChannelTestResponse>(`/notifications/channels/${channelId}/test`, {});
}

export function useTestChannel() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: testChannel,
    onSuccess: () => {
      // Refresh channels to get updated verification status
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.channels() });
    },
  });
}

// =============================================================================
// RULES - Queries
// =============================================================================

async function fetchRules(): Promise<NotificationRule[]> {
  return apiGet<NotificationRule[]>('/notifications/rules');
}

export function useNotificationRules() {
  return useQuery({
    queryKey: queryKeys.notifications.rules(),
    queryFn: fetchRules,
    staleTime: 5 * 60 * 1000,
  });
}

// =============================================================================
// RULES - Mutations
// =============================================================================

async function createRule(data: RuleCreateRequest): Promise<NotificationRule> {
  return apiPost<NotificationRule>('/notifications/rules', data);
}

export function useCreateRule() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: createRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.rules() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.summary() });
    },
  });
}

interface UpdateRuleParams {
  ruleId: number;
  data: RuleUpdateRequest;
}

async function updateRule({ ruleId, data }: UpdateRuleParams): Promise<NotificationRule> {
  return apiPatch<NotificationRule>(`/notifications/rules/${ruleId}`, data);
}

export function useUpdateRule() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updateRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.rules() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.summary() });
    },
  });
}

async function deleteRule(ruleId: number): Promise<void> {
  return apiDelete<void>(`/notifications/rules/${ruleId}`);
}

export function useDeleteRule() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: deleteRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.rules() });
      queryClient.invalidateQueries({ queryKey: queryKeys.notifications.summary() });
    },
  });
}

async function testRule(ruleId: number): Promise<RuleTestResponse> {
  return apiPost<RuleTestResponse>(`/notifications/rules/${ruleId}/test`, {});
}

export function useTestRule() {
  return useMutation({
    mutationFn: testRule,
  });
}

async function clearRuleCooldown(ruleId: number): Promise<{ message: string }> {
  return apiPost<{ message: string }>(`/notifications/rules/${ruleId}/clear-cooldown`, {});
}

export function useClearRuleCooldown() {
  return useMutation({
    mutationFn: clearRuleCooldown,
  });
}

// =============================================================================
// HISTORY - Queries
// =============================================================================

interface HistoryParams {
  page?: number;
  pageSize?: number;
}

async function fetchHistory({ page = 1, pageSize = 20 }: HistoryParams = {}): Promise<NotificationHistoryResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  return apiGet<NotificationHistoryResponse>(`/notifications/history?${params}`);
}

export function useNotificationHistory(params: HistoryParams = {}) {
  return useQuery({
    queryKey: queryKeys.notifications.history(params.page, params.pageSize),
    queryFn: () => fetchHistory(params),
    staleTime: 2 * 60 * 1000,
  });
}

// =============================================================================
// SUMMARY - Queries
// =============================================================================

async function fetchSummary(): Promise<NotificationSummary> {
  return apiGet<NotificationSummary>('/notifications/summary');
}

export function useNotificationSummary() {
  return useQuery({
    queryKey: queryKeys.notifications.summary(),
    queryFn: fetchSummary,
    staleTime: 2 * 60 * 1000,
  });
}

// =============================================================================
// TRIGGER TYPES - Queries
// =============================================================================

async function fetchTriggerTypes(): Promise<TriggerTypeInfo[]> {
  return apiGet<TriggerTypeInfo[]>('/notifications/trigger-types');
}

export function useTriggerTypes() {
  return useQuery({
    queryKey: queryKeys.notifications.triggerTypes(),
    queryFn: fetchTriggerTypes,
    staleTime: 60 * 60 * 1000, // Cache for 1 hour (static data)
  });
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Invalidate all notification-related queries.
 * Useful after bulk operations or when syncing state.
 */
export function invalidateNotificationQueries(queryClient: ReturnType<typeof useQueryClient>) {
  queryClient.invalidateQueries({ queryKey: queryKeys.notifications.all });
}
