/**
 * React Query hooks for Portfolio feature.
 * 
 * Replaces manual useState/useEffect patterns in Portfolio.tsx.
 */

import { useQuery, useQueries } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet, buildUrl } from '@/lib/api-client';
import type {
  Portfolio,
  PortfolioDetail,
  PortfolioRiskAnalyticsResponse,
  PortfolioAnalyticsResponse,
  PortfolioAnalyticsJob,
  PortfolioAllocationRecommendation,
  StockInfo,
  HoldingSparklineData,
  BatchSparklineResponse,
} from '@/services/api';

// ============================================================================
// Portfolio List
// ============================================================================

async function fetchPortfolios(): Promise<Portfolio[]> {
  return apiGet<Portfolio[]>('/portfolios');
}

export function usePortfolios() {
  return useQuery({
    queryKey: queryKeys.portfolios.all,
    queryFn: fetchPortfolios,
    staleTime: 5 * 60 * 1000,
  });
}

// ============================================================================
// Portfolio Detail
// ============================================================================

async function fetchPortfolioDetail(portfolioId: number): Promise<PortfolioDetail> {
  return apiGet<PortfolioDetail>(`/portfolios/${portfolioId}`);
}

export function usePortfolioDetail(portfolioId: number | null) {
  return useQuery({
    queryKey: queryKeys.portfolios.detail(portfolioId ?? 0),
    queryFn: () => fetchPortfolioDetail(portfolioId!),
    enabled: portfolioId !== null && portfolioId > 0,
    staleTime: 2 * 60 * 1000,
  });
}

// ============================================================================
// Risk Analytics
// ============================================================================

async function fetchRiskAnalytics(portfolioId: number): Promise<PortfolioRiskAnalyticsResponse> {
  return apiGet<PortfolioRiskAnalyticsResponse>(`/portfolios/${portfolioId}/analytics`);
}

export function usePortfolioRiskAnalytics(portfolioId: number | null, hasHoldings: boolean) {
  return useQuery({
    queryKey: queryKeys.portfolios.riskAnalytics(portfolioId ?? 0),
    queryFn: () => fetchRiskAnalytics(portfolioId!),
    enabled: portfolioId !== null && portfolioId > 0 && hasHoldings,
    staleTime: 5 * 60 * 1000,
  });
}

// ============================================================================
// Portfolio Analytics Job (polling)
// ============================================================================

async function fetchAnalyticsJob(portfolioId: number, jobId: string): Promise<PortfolioAnalyticsJob> {
  return apiGet<PortfolioAnalyticsJob>(`/portfolios/${portfolioId}/analytics/jobs/${jobId}`);
}

export function usePortfolioAnalyticsJob(
  portfolioId: number | null,
  jobId: string | null,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: queryKeys.portfolios.analyticsJob(portfolioId ?? 0, jobId ?? ''),
    queryFn: () => fetchAnalyticsJob(portfolioId!, jobId!),
    enabled: enabled && portfolioId !== null && jobId !== null,
    refetchInterval: (query) => {
      const data = query.state.data;
      // Stop polling when job is complete or failed
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
  });
}

// ============================================================================
// Allocation Recommendation
// ============================================================================

async function fetchAllocation(
  portfolioId: number,
  inflowEur: number,
  method: string
): Promise<PortfolioAllocationRecommendation> {
  const params = new URLSearchParams({
    inflow_eur: String(inflowEur),
  });
  if (method && method !== 'auto') {
    params.append('method', method);
  }
  return apiGet<PortfolioAllocationRecommendation>(
    `/portfolios/${portfolioId}/allocate?${params.toString()}`
  );
}

export function useAllocationRecommendation(
  portfolioId: number | null,
  inflowEur: number,
  method: string,
  enabled: boolean = false
) {
  return useQuery({
    queryKey: queryKeys.portfolios.allocation(portfolioId ?? 0, inflowEur, method),
    queryFn: () => fetchAllocation(portfolioId!, inflowEur, method),
    enabled: enabled && portfolioId !== null && portfolioId > 0 && inflowEur > 0,
    staleTime: 2 * 60 * 1000,
  });
}

// ============================================================================
// Stock Info for Holdings
// ============================================================================

async function fetchStockInfo(symbol: string): Promise<StockInfo | null> {
  try {
    return await apiGet<StockInfo>(buildUrl(`/stocks/${symbol}`));
  } catch {
    return null;
  }
}

export function useHoldingsStockInfo(symbols: string[]) {
  return useQueries({
    queries: symbols.map((symbol) => ({
      queryKey: queryKeys.stocks.info(symbol),
      queryFn: () => fetchStockInfo(symbol),
      staleTime: 10 * 60 * 1000,
    })),
    combine: (results) => {
      const map: Record<string, StockInfo | null> = {};
      results.forEach((result, index) => {
        map[symbols[index]] = result.data ?? null;
      });
      return {
        data: map,
        isLoading: results.some((r) => r.isLoading),
        isError: results.some((r) => r.isError),
      };
    },
  });
}

// ============================================================================
// Holdings Sparklines
// ============================================================================

async function fetchSparklines(
  portfolioId: number,
  symbols: string[],
  days: number
): Promise<Record<string, HoldingSparklineData>> {
  if (symbols.length === 0) return {};
  const response = await apiGet<BatchSparklineResponse>(`/portfolios/${portfolioId}/sparklines`, {
    method: 'POST',
    body: JSON.stringify({ symbols, days }),
  });
  return response.sparklines;
}

export function useHoldingsSparklines(
  portfolioId: number | null,
  symbols: string[],
  days: number = 180
) {
  return useQuery({
    queryKey: queryKeys.portfolios.sparklines(portfolioId ?? 0, symbols, days),
    queryFn: () => fetchSparklines(portfolioId!, symbols, days),
    enabled: portfolioId !== null && portfolioId > 0 && symbols.length > 0,
    staleTime: 10 * 60 * 1000,
  });
}

// ============================================================================
// Combined Portfolio Data Hook
// ============================================================================

export interface PortfolioPageData {
  // Portfolios
  portfolios: Portfolio[];
  isLoadingPortfolios: boolean;
  portfoliosError: Error | null;
  
  // Selected portfolio detail
  detail: PortfolioDetail | null;
  isLoadingDetail: boolean;
  detailError: Error | null;
  
  // Risk analytics
  riskAnalytics: PortfolioRiskAnalyticsResponse | null;
  isLoadingRiskAnalytics: boolean;
  riskAnalyticsError: Error | null;
  
  // Stock info for holdings
  stockInfoMap: Record<string, StockInfo | null>;
  isLoadingStockInfo: boolean;
  
  // Sparklines
  sparklineData: Record<string, HoldingSparklineData>;
  isLoadingSpark: boolean;
}

export function usePortfolioPageData(selectedId: number | null): PortfolioPageData {
  // Fetch all portfolios
  const portfoliosQuery = usePortfolios();
  
  // Fetch detail for selected portfolio
  const detailQuery = usePortfolioDetail(selectedId);
  
  // Derive holding symbols
  const holdingSymbols = detailQuery.data?.holdings?.map(h => h.symbol) ?? [];
  const hasHoldings = holdingSymbols.length > 0;
  
  // Fetch risk analytics (only if has holdings)
  const riskQuery = usePortfolioRiskAnalytics(selectedId, hasHoldings);
  
  // Fetch stock info for each holding
  const stockInfoQuery = useHoldingsStockInfo(holdingSymbols);
  
  // Fetch sparklines
  const sparklinesQuery = useHoldingsSparklines(selectedId, holdingSymbols, 180);
  
  return {
    portfolios: portfoliosQuery.data ?? [],
    isLoadingPortfolios: portfoliosQuery.isLoading,
    portfoliosError: portfoliosQuery.error,
    
    detail: detailQuery.data ?? null,
    isLoadingDetail: detailQuery.isLoading,
    detailError: detailQuery.error,
    
    riskAnalytics: riskQuery.data ?? null,
    isLoadingRiskAnalytics: riskQuery.isLoading,
    riskAnalyticsError: riskQuery.error,
    
    stockInfoMap: stockInfoQuery.data,
    isLoadingStockInfo: stockInfoQuery.isLoading,
    
    sparklineData: sparklinesQuery.data ?? {},
    isLoadingSpark: sparklinesQuery.isLoading,
  };
}
