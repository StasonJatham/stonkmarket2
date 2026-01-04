/**
 * React Query mutations for Portfolio feature.
 * 
 * Replaces manual async/await + setState patterns in Portfolio.tsx.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiPost, apiPatch, apiDelete } from '@/lib/api-client';
import type {
  Portfolio,
  Holding,
  PortfolioAnalyticsResponse,
  PortfolioAllocationRecommendation,
} from '@/services/api';

// ============================================================================
// Portfolio CRUD
// ============================================================================

interface CreatePortfolioParams {
  name: string;
  description?: string;
  base_currency?: string;
}

async function createPortfolio(params: CreatePortfolioParams): Promise<Portfolio> {
  return apiPost<Portfolio>('/portfolios', params);
}

export function useCreatePortfolio() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: createPortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.all });
    },
  });
}

interface UpdatePortfolioParams {
  portfolioId: number;
  data: Partial<{
    name: string;
    description: string;
    base_currency: string;
    is_active: boolean;
  }>;
}

async function updatePortfolio({ portfolioId, data }: UpdatePortfolioParams): Promise<Portfolio> {
  return apiPatch<Portfolio>(`/portfolios/${portfolioId}`, data);
}

export function useUpdatePortfolio() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: updatePortfolio,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.detail(variables.portfolioId) });
    },
  });
}

async function deletePortfolio(portfolioId: number): Promise<void> {
  return apiDelete<void>(`/portfolios/${portfolioId}`);
}

export function useDeletePortfolio() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: deletePortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.all });
    },
  });
}

// ============================================================================
// Holding CRUD
// ============================================================================

interface UpsertHoldingParams {
  portfolioId: number;
  symbol: string;
  quantity: number;
  avg_cost?: number;
  target_weight?: number;
}

async function upsertHolding({ portfolioId, ...data }: UpsertHoldingParams): Promise<Holding> {
  return apiPost<Holding>(`/portfolios/${portfolioId}/holdings`, data);
}

export function useUpsertHolding() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: upsertHolding,
    onSuccess: (_, variables) => {
      // Invalidate all portfolio-related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.detail(variables.portfolioId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.riskAnalytics(variables.portfolioId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.analytics(variables.portfolioId) });
    },
  });
}

interface DeleteHoldingParams {
  portfolioId: number;
  symbol: string;
}

async function deleteHolding({ portfolioId, symbol }: DeleteHoldingParams): Promise<void> {
  return apiDelete<void>(`/portfolios/${portfolioId}/holdings/${symbol}`);
}

export function useDeleteHolding() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: deleteHolding,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.detail(variables.portfolioId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.riskAnalytics(variables.portfolioId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.analytics(variables.portfolioId) });
    },
  });
}

// ============================================================================
// Bulk Import
// ============================================================================

interface BulkImportPosition {
  symbol: string;
  quantity: number;
  avg_cost?: number;
  currency?: string;
}

interface BulkImportParams {
  portfolioId: number;
  positions: BulkImportPosition[];
  skip_duplicates?: boolean;
}

interface BulkImportResult {
  imported: number;
  skipped: number;
  errors: string[];
}

async function bulkImportHoldings({ portfolioId, ...data }: BulkImportParams): Promise<BulkImportResult> {
  return apiPost<BulkImportResult>(`/portfolios/${portfolioId}/bulk-import`, data);
}

export function useBulkImportHoldings() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: bulkImportHoldings,
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.detail(variables.portfolioId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.riskAnalytics(variables.portfolioId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.analytics(variables.portfolioId) });
    },
  });
}

// ============================================================================
// Run Analytics
// ============================================================================

interface RunAnalyticsParams {
  portfolioId: number;
  tools?: string[];
  force_refresh?: boolean;
}

async function runAnalytics({ portfolioId, ...data }: RunAnalyticsParams): Promise<PortfolioAnalyticsResponse> {
  return apiPost<PortfolioAnalyticsResponse>(`/portfolios/${portfolioId}/analytics`, data);
}

export function useRunAnalytics() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: runAnalytics,
    onSuccess: (data, variables) => {
      // Update analytics cache with new results
      queryClient.setQueryData(
        queryKeys.portfolios.analytics(variables.portfolioId),
        data
      );
    },
  });
}

// ============================================================================
// Get Allocation Recommendation (mutation because it's a POST)
// ============================================================================

interface GetAllocationParams {
  portfolioId: number;
  inflow_eur: number;
  method?: string;
}

async function getAllocation({ portfolioId, ...params }: GetAllocationParams): Promise<PortfolioAllocationRecommendation> {
  const searchParams = new URLSearchParams({
    inflow_eur: String(params.inflow_eur),
  });
  if (params.method && params.method !== 'auto') {
    searchParams.append('method', params.method);
  }
  return apiPost<PortfolioAllocationRecommendation>(
    `/portfolios/${portfolioId}/allocate?${searchParams.toString()}`,
    {}
  );
}

export function useGetAllocation() {
  return useMutation({
    mutationFn: getAllocation,
  });
}
