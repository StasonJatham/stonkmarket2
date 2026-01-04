/**
 * React Query hooks for Admin features.
 * 
 * Includes API key management, job scheduling, and suggestions management.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet, apiPost, apiPatch, apiDelete } from '@/lib/api-client';

// ============================================================================
// API Key Management
// ============================================================================

interface ApiKeyInfo {
  id: number;
  key_name: string;
  key_hint: string;
  created_at: string;
  updated_at: string;
  created_by: string;
}

interface ApiKeysResponse {
  keys: ApiKeyInfo[];
}

async function fetchApiKeys(): Promise<ApiKeysResponse> {
  return apiGet<ApiKeysResponse>('/api-keys');
}

export function useApiKeys() {
  return useQuery({
    queryKey: queryKeys.admin.apiKeys(),
    queryFn: fetchApiKeys,
    staleTime: 30 * 1000,
    select: (data) => data.keys,
  });
}

interface CreateApiKeyParams {
  key_name: string;
  key_value: string;
  mfa_code?: string;
}

async function createApiKey(params: CreateApiKeyParams): Promise<void> {
  return apiPost<void>('/api-keys', params);
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: createApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.admin.apiKeys() });
    },
  });
}

interface RevealApiKeyParams {
  key_name: string;
  mfa_code?: string;
}

interface RevealApiKeyResponse {
  key_value: string;
}

async function revealApiKey(params: RevealApiKeyParams): Promise<RevealApiKeyResponse> {
  return apiPost<RevealApiKeyResponse>(`/api-keys/${params.key_name}/reveal`, {
    mfa_code: params.mfa_code,
  });
}

export function useRevealApiKey() {
  return useMutation({
    mutationFn: revealApiKey,
  });
}

interface DeleteApiKeyParams {
  key_name: string;
  mfa_code?: string;
}

async function deleteApiKey(params: DeleteApiKeyParams): Promise<void> {
  const searchParams = params.mfa_code 
    ? `?mfa_code=${encodeURIComponent(params.mfa_code)}` 
    : '';
  return apiDelete<void>(`/api-keys/${params.key_name}${searchParams}`);
}

export function useDeleteApiKey() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: deleteApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.admin.apiKeys() });
    },
  });
}

// ============================================================================
// Job Scheduling
// ============================================================================

interface CronJob {
  job_name: string;
  description: string | null;
  schedule: string | null;
  is_paused: boolean;
  last_run: string | null;
  next_run: string | null;
  last_status: string | null;
  last_duration_ms: number | null;
}

interface CronJobsResponse {
  jobs: CronJob[];
}

async function fetchCronJobs(): Promise<CronJobsResponse> {
  return apiGet<CronJobsResponse>('/admin/cronjobs');
}

export function useCronJobs() {
  return useQuery({
    queryKey: queryKeys.admin.jobs(),
    queryFn: fetchCronJobs,
    staleTime: 10 * 1000,
    refetchInterval: 30 * 1000, // Auto-refresh every 30s
    select: (data) => data.jobs,
  });
}

interface UpdateCronJobParams {
  job_name: string;
  schedule?: string;
  is_paused?: boolean;
}

async function updateCronJob({ job_name, ...data }: UpdateCronJobParams): Promise<CronJob> {
  return apiPatch<CronJob>(`/admin/cronjobs/${job_name}`, data);
}

export function useUpdateCronJob() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updateCronJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.admin.jobs() });
    },
  });
}

interface RunCronJobParams {
  job_name: string;
}

async function runCronJob({ job_name }: RunCronJobParams): Promise<void> {
  return apiPost<void>(`/admin/cronjobs/${job_name}/run`, {});
}

export function useRunCronJob() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: runCronJob,
    onSuccess: () => {
      // Delay refetch to allow job to start
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: queryKeys.admin.jobs() });
      }, 1000);
    },
  });
}

// ============================================================================
// Batch Jobs
// ============================================================================

interface BatchJobRequest {
  job_type: string;
  symbols?: string[];
  max_workers?: number;
  force?: boolean;
}

interface BatchJobResponse {
  job_id: string;
  job_type: string;
  status: string;
  total_symbols: number;
  queued_at: string;
}

async function runBatchJob(params: BatchJobRequest): Promise<BatchJobResponse> {
  return apiPost<BatchJobResponse>('/admin/batch-jobs', params);
}

export function useRunBatchJob() {
  return useMutation({
    mutationFn: runBatchJob,
  });
}

// ============================================================================
// Suggestions Management
// ============================================================================

interface Suggestion {
  symbol: string;
  name: string | null;
  suggested_by: string | null;
  vote_count: number;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
}

async function fetchSuggestions(): Promise<Suggestion[]> {
  return apiGet<Suggestion[]>('/admin/suggestions');
}

export function useSuggestions() {
  return useQuery({
    queryKey: queryKeys.suggestions.list(),
    queryFn: fetchSuggestions,
    staleTime: 30 * 1000,
  });
}

interface ApproveSuggestionParams {
  symbol: string;
}

async function approveSuggestion({ symbol }: ApproveSuggestionParams): Promise<void> {
  return apiPost<void>(`/admin/suggestions/${symbol}/approve`, {});
}

export function useApproveSuggestion() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: approveSuggestion,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.suggestions.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.admin.symbols() });
    },
  });
}

interface RejectSuggestionParams {
  symbol: string;
}

async function rejectSuggestion({ symbol }: RejectSuggestionParams): Promise<void> {
  return apiPost<void>(`/admin/suggestions/${symbol}/reject`, {});
}

export function useRejectSuggestion() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: rejectSuggestion,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.suggestions.all });
    },
  });
}

// ============================================================================
// Symbol Management
// ============================================================================

interface Symbol {
  symbol: string;
  name: string | null;
  exchange: string | null;
  currency: string | null;
  is_active: boolean;
  created_at: string;
}

async function fetchSymbols(): Promise<Symbol[]> {
  return apiGet<Symbol[]>('/admin/symbols');
}

export function useSymbols() {
  return useQuery({
    queryKey: queryKeys.admin.symbols(),
    queryFn: fetchSymbols,
    staleTime: 60 * 1000,
  });
}

interface AddSymbolParams {
  symbol: string;
  name?: string;
  exchange?: string;
  currency?: string;
}

async function addSymbol(params: AddSymbolParams): Promise<Symbol> {
  return apiPost<Symbol>('/admin/symbols', params);
}

export function useAddSymbol() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: addSymbol,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.admin.symbols() });
    },
  });
}

interface UpdateSymbolParams {
  symbol: string;
  is_active?: boolean;
}

async function updateSymbol({ symbol, ...data }: UpdateSymbolParams): Promise<Symbol> {
  return apiPatch<Symbol>(`/admin/symbols/${symbol}`, data);
}

export function useUpdateSymbol() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updateSymbol,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.admin.symbols() });
    },
  });
}

// ============================================================================
// MFA Status
// ============================================================================

interface MFAStatus {
  enabled: boolean;
}

async function fetchMFAStatus(): Promise<MFAStatus> {
  return apiGet<MFAStatus>('/auth/mfa/status');
}

export function useMFAStatus() {
  return useQuery({
    queryKey: ['auth', 'mfaStatus'],
    queryFn: fetchMFAStatus,
    staleTime: 60 * 1000,
  });
}

// ============================================================================
// MFA Session Check
// ============================================================================

interface MFASession {
  has_session: boolean;
}

async function fetchMFASession(): Promise<MFASession> {
  return apiGet<MFASession>('/auth/mfa/session');
}

export function useMFASession() {
  return useQuery({
    queryKey: ['auth', 'mfaSession'],
    queryFn: fetchMFASession,
    staleTime: 30 * 1000,
  });
}

// ============================================================================
// System Status
// ============================================================================

interface SystemStatus {
  logo_dev_configured: boolean;
  database_connected: boolean;
}

async function fetchSystemStatus(): Promise<SystemStatus> {
  return apiGet<SystemStatus>('/system/status');
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ['system', 'status'],
    queryFn: fetchSystemStatus,
    staleTime: 60 * 1000,
  });
}
