/**
 * React Query hooks for market data.
 * 
 * These replace the manual useEffect + useState patterns throughout the app.
 * Each hook handles loading, error, and caching automatically.
 */

import { useQuery, useQueries } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet, buildUrl } from '@/lib/api-client';
import {
  StockInfoSchema,
  ChartDataSchema,
  RankingArraySchema,
  SignalTriggersResponseSchema,
  CurrentSignalsSchema,
  AgentAnalysisSchema,
  DipAnalysisSchema,
  SymbolFundamentalsSchema,
  BenchmarkSchema,
  DipCardSchema,
  safeParse,
  type StockInfo,
  type ChartDataPoint,
  type RankingResponse,
  type SignalTriggersResponse,
  type CurrentSignals,
  type AgentAnalysis,
  type DipAnalysis,
  type SymbolFundamentals,
  type Benchmark,
  type DipCard,
} from './schemas';

// ============================================================================
// Stock Info
// ============================================================================

async function fetchStockInfo(symbol: string): Promise<StockInfo> {
  const data = await apiGet<unknown>(`/stocks/${symbol}/info`);
  return safeParse(StockInfoSchema, data, `StockInfo:${symbol}`);
}

export function useStockInfo(symbol: string | undefined) {
  return useQuery({
    queryKey: queryKeys.stocks.info(symbol ?? ''),
    queryFn: () => fetchStockInfo(symbol!),
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000, // 10 minutes - info changes less frequently
  });
}

// ============================================================================
// Stock Chart
// ============================================================================

async function fetchStockChart(symbol: string, days: number): Promise<ChartDataPoint[]> {
  const data = await apiGet<unknown>(buildUrl(`/dips/${symbol}/chart`, { days }));
  return safeParse(ChartDataSchema, data, `Chart:${symbol}:${days}`);
}

export function useStockChart(symbol: string | undefined, days: number = 180) {
  return useQuery({
    queryKey: queryKeys.stocks.chart(symbol ?? '', days),
    queryFn: () => fetchStockChart(symbol!, days),
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// ============================================================================
// Stock Ranking (Dip Ranking)
// ============================================================================

async function fetchRanking(showAll: boolean): Promise<RankingResponse> {
  const data = await apiGet<unknown>(buildUrl('/dips/ranking', { show_all: showAll }));
  // API returns a raw array, parse it and transform to RankingResponse structure
  const ranking = safeParse(RankingArraySchema, data, 'Ranking');
  
  // Transform to RankingResponse structure with UI calculations
  const enrichedRanking = ranking.map((stock, index) => ({
    ...stock,
    dip_score: 100 - (index * (100 / Math.max(ranking.length, 1))),
    recovery_potential: stock.depth * 100,
  }));
  
  return {
    ranking: enrichedRanking,
    last_updated: ranking.length > 0 && ranking[0].updated_at ? ranking[0].updated_at : null,
  };
}

export function useRanking(showAll: boolean = false) {
  return useQuery({
    queryKey: queryKeys.stocks.ranking(showAll),
    queryFn: () => fetchRanking(showAll),
    staleTime: 5 * 60 * 1000,
  });
}

// ============================================================================
// Signal Triggers
// ============================================================================

async function fetchSignalTriggers(symbol: string, days: number): Promise<SignalTriggersResponse> {
  const data = await apiGet<unknown>(buildUrl(`/signals/${symbol}/triggers`, { days }));
  return safeParse(SignalTriggersResponseSchema, data, `Signals:${symbol}`);
}

export function useSignalTriggers(symbol: string | undefined, days: number = 365) {
  return useQuery({
    queryKey: queryKeys.stocks.signals(symbol ?? ''),
    queryFn: () => fetchSignalTriggers(symbol!, Math.min(730, days)),
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000,
  });
}

// ============================================================================
// Current Signals
// ============================================================================

async function fetchCurrentSignals(symbol: string): Promise<CurrentSignals> {
  const data = await apiGet<unknown>(`/signals/${symbol}/current`);
  return safeParse(CurrentSignalsSchema, data, `CurrentSignals:${symbol}`);
}

export function useCurrentSignals(symbol: string | undefined) {
  return useQuery({
    queryKey: ['stocks', 'currentSignals', symbol],
    queryFn: () => fetchCurrentSignals(symbol!),
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000,
  });
}

// ============================================================================
// Agent Analysis (AI Personas)
// ============================================================================

async function fetchAgentAnalysis(symbol: string): Promise<AgentAnalysis | null> {
  const data = await apiGet<unknown>(`/symbols/${symbol}/agents`);
  if (!data) return null;
  return safeParse(AgentAnalysisSchema, data, `AgentAnalysis:${symbol}`);
}

export function useAgentAnalysis(symbol: string | undefined) {
  return useQuery({
    queryKey: ['stocks', 'agentAnalysis', symbol],
    queryFn: () => fetchAgentAnalysis(symbol!),
    enabled: !!symbol,
    staleTime: 30 * 60 * 1000, // 30 minutes - AI analysis is expensive
  });
}

// ============================================================================
// Dip Card (Swipe Card / Stock Detail dip info)
// ============================================================================

async function fetchDipCard(symbol: string, refreshAi = false): Promise<DipCard> {
  const url = buildUrl(`/swipe/cards/${symbol}`, { refresh_ai: refreshAi });
  const data = await apiGet<unknown>(url);
  return safeParse(DipCardSchema, data, `DipCard:${symbol}`);
}

export function useDipCard(symbol: string | undefined, refreshAi = false) {
  return useQuery({
    queryKey: ['stocks', 'dipCard', symbol, refreshAi],
    queryFn: () => fetchDipCard(symbol!, refreshAi),
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// ============================================================================
// Dip Analysis
// ============================================================================

async function fetchDipAnalysis(symbol: string): Promise<DipAnalysis> {
  const data = await apiGet<unknown>(`/dips/${symbol}/analysis`);
  return safeParse(DipAnalysisSchema, data, `DipAnalysis:${symbol}`);
}

export function useDipAnalysis(symbol: string | undefined) {
  return useQuery({
    queryKey: ['stocks', 'dipAnalysis', symbol],
    queryFn: () => fetchDipAnalysis(symbol!),
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000,
  });
}

// ============================================================================
// Fundamentals
// ============================================================================

async function fetchFundamentals(symbol: string): Promise<SymbolFundamentals> {
  const data = await apiGet<unknown>(`/stocks/${symbol}/fundamentals`);
  return safeParse(SymbolFundamentalsSchema, data, `Fundamentals:${symbol}`);
}

export function useFundamentals(symbol: string | undefined) {
  return useQuery({
    queryKey: queryKeys.stocks.fundamentals(symbol ?? ''),
    queryFn: () => fetchFundamentals(symbol!),
    enabled: !!symbol,
    staleTime: 60 * 60 * 1000, // 1 hour - fundamentals change rarely
  });
}

// ============================================================================
// Benchmarks
// ============================================================================

async function fetchBenchmarks(): Promise<Benchmark[]> {
  const data = await apiGet<unknown[]>('/benchmarks');
  return data.map(b => safeParse(BenchmarkSchema, b, 'Benchmark'));
}

export function useBenchmarks() {
  return useQuery({
    queryKey: queryKeys.benchmarks.list(),
    queryFn: fetchBenchmarks,
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

async function fetchBenchmarkChart(symbol: string, days: number): Promise<ChartDataPoint[]> {
  const data = await apiGet<unknown>(buildUrl(`/benchmarks/${symbol}/chart`, { days }));
  return safeParse(ChartDataSchema, data, `BenchmarkChart:${symbol}`);
}

export function useBenchmarkChart(symbol: string | undefined, days: number = 365) {
  return useQuery({
    queryKey: queryKeys.benchmarks.chart(symbol ?? '', days),
    queryFn: () => fetchBenchmarkChart(symbol!, days),
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000,
  });
}

// ============================================================================
// Combined Stock Detail Query (replaces useEffect chains)
// ============================================================================

/**
 * Fetches all data needed for stock detail view in parallel.
 * Replaces 4+ separate useEffect chains with a single hook.
 */
export function useStockDetail(symbol: string | undefined, chartDays: number = 365) {
  const results = useQueries({
    queries: [
      {
        queryKey: queryKeys.stocks.info(symbol ?? ''),
        queryFn: () => fetchStockInfo(symbol!),
        enabled: !!symbol,
        staleTime: 10 * 60 * 1000,
      },
      {
        queryKey: queryKeys.stocks.chart(symbol ?? '', chartDays),
        queryFn: () => fetchStockChart(symbol!, chartDays),
        enabled: !!symbol,
        staleTime: 5 * 60 * 1000,
      },
      {
        queryKey: queryKeys.stocks.signals(symbol ?? ''),
        queryFn: () => fetchSignalTriggers(symbol!, Math.min(730, chartDays + 30)),
        enabled: !!symbol,
        staleTime: 10 * 60 * 1000,
      },
      {
        queryKey: queryKeys.stocks.fundamentals(symbol ?? ''),
        queryFn: () => fetchFundamentals(symbol!),
        enabled: !!symbol,
        staleTime: 60 * 60 * 1000,
      },
      {
        queryKey: ['stocks', 'dipCard', symbol, false],
        queryFn: () => fetchDipCard(symbol!, false),
        enabled: !!symbol,
        staleTime: 5 * 60 * 1000,
      },
      {
        queryKey: ['stocks', 'agentAnalysis', symbol],
        queryFn: () => fetchAgentAnalysis(symbol!),
        enabled: !!symbol,
        staleTime: 30 * 60 * 1000,
      },
    ],
  });

  const [infoQuery, chartQuery, signalsQuery, fundamentalsQuery, dipCardQuery, agentQuery] = results;

  return {
    info: infoQuery.data,
    chartData: chartQuery.data ?? [],
    signals: signalsQuery.data,
    fundamentals: fundamentalsQuery.data,
    dipCard: dipCardQuery.data,
    agentAnalysis: agentQuery.data,
    isLoading: results.some(r => r.isLoading),
    isError: results.some(r => r.isError),
    error: results.find(r => r.error)?.error,
    // Individual loading states for progressive rendering
    isLoadingInfo: infoQuery.isLoading,
    isLoadingChart: chartQuery.isLoading,
    isLoadingSignals: signalsQuery.isLoading,
    isLoadingFundamentals: fundamentalsQuery.isLoading,
    isLoadingDipCard: dipCardQuery.isLoading,
    isLoadingAgentAnalysis: agentQuery.isLoading,
    // Refetch functions for manual refresh
    refetchChart: chartQuery.refetch,
  };
}

// ============================================================================
// Prefetch utilities
// ============================================================================

import { queryClient } from '@/lib/query';

/**
 * Prefetch stock data for hover/preload scenarios
 */
export function prefetchStock(symbol: string) {
  queryClient.prefetchQuery({
    queryKey: queryKeys.stocks.info(symbol),
    queryFn: () => fetchStockInfo(symbol),
    staleTime: 10 * 60 * 1000,
  });
  queryClient.prefetchQuery({
    queryKey: queryKeys.stocks.chart(symbol, 180),
    queryFn: () => fetchStockChart(symbol, 180),
    staleTime: 5 * 60 * 1000,
  });
}

/**
 * Prefetch multiple stocks (e.g., for ranking page)
 */
export function prefetchStocks(symbols: string[]) {
  symbols.slice(0, 20).forEach(symbol => {
    prefetchStock(symbol);
  });
}
