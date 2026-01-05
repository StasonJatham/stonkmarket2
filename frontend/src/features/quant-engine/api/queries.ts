/**
 * React Query hooks for quant engine.
 * 
 * This REPLACES QuantContext.tsx - no more manual context management!
 */

import { useQuery } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet, buildUrl } from '@/lib/api-client';
import { quantToDipStock } from '@/services/api';
import type { QuantEngineResponse, DipStock } from '@/services/api';

// ============================================================================
// Quant Recommendations
// ============================================================================

async function fetchQuantRecommendations(inflow: number, limit: number): Promise<QuantEngineResponse> {
  // Use api.ts types directly - the full API response has more fields than the Zod schema
  return apiGet<QuantEngineResponse>(buildUrl('/quant/recommendations', { 
    inflow_eur: inflow, 
    limit 
  }));
}

/**
 * Primary hook for quant engine recommendations.
 * Replaces QuantContext completely.
 */
export function useQuantRecommendations(inflow: number = 1000, limit: number = 40) {
  return useQuery({
    queryKey: queryKeys.quant.recommendations(inflow, limit),
    queryFn: () => fetchQuantRecommendations(inflow, limit),
    staleTime: 5 * 60 * 1000, // 5 minutes
    // Transform to include derived data
    select: (data) => ({
      ...data,
      // Convert to DipStock format for backward compatibility
      stocks: data.recommendations.map(quantToDipStock),
    }),
  });
}

// Re-export quantToDipStock for components that need direct access
export { quantToDipStock };
export type { DipStock };

// ============================================================================
// Portfolio Stats (derived from recommendations)
// ============================================================================

export interface PortfolioStats {
  expectedReturn: number;
  expectedRisk: number;
  totalTrades: number;
  transactionCostEur: number;
}

/**
 * Convenience hook that just returns portfolio stats.
 */
export function usePortfolioStats(inflow: number = 1000, limit: number = 40) {
  const query = useQuantRecommendations(inflow, limit);
  
  return {
    ...query,
    data: query.data ? {
      expectedReturn: query.data.expected_portfolio_return,
      expectedRisk: query.data.expected_portfolio_risk,
      totalTrades: query.data.total_trades,
      transactionCostEur: query.data.total_transaction_cost_eur,
    } : undefined,
  };
}

// ============================================================================
// Landing Page Data (combines recommendations + charts + hero analysis)
// ============================================================================

import { 
  useStockChart, 
  useBatchCharts,
  useAgentAnalysis 
} from '@/features/market-data/api/queries';
import { getSignalTriggers, getDipEntry } from '@/services/api';

const LANDING_SIGNAL_BOARD_COUNT = 8;
const LANDING_MINI_CHART_DAYS = 45;
const LANDING_HERO_CHART_DAYS = 365;
const LANDING_SIGNAL_LOOKBACK_DAYS = 365;

/**
 * Hook for signal triggers with full summary data.
 * Uses the quant engine endpoint which includes benchmark comparison metrics.
 */
function useHeroSignalTriggers(symbol: string | undefined, lookbackDays: number = 365) {
  return useQuery({
    queryKey: ['quant', 'signalTriggers', symbol, lookbackDays],
    queryFn: () => getSignalTriggers(symbol!, lookbackDays),
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Hook for dip entry analysis data.
 */
function useHeroDipEntry(symbol: string | undefined) {
  return useQuery({
    queryKey: ['quant', 'dipEntry', symbol],
    queryFn: () => getDipEntry(symbol!),
    enabled: !!symbol,
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}

/**
 * Combined hook for Landing page.
 * Replaces the massive useEffect chain with declarative data fetching.
 */
export function useLandingData(inflow: number = 1000, limit: number = 25) {
  // Primary data: quant recommendations
  const recsQuery = useQuantRecommendations(inflow, limit);
  
  // Derive hero symbol (first BUY recommendation)
  const recommendations = recsQuery.data?.recommendations ?? [];
  const heroRec = recommendations.find(r => r.action === 'BUY') ?? recommendations[0] ?? null;
  const heroSymbol = heroRec?.ticker ?? '';
  
  // Signal board symbols (top N recommendations)
  const signalBoardSymbols = recommendations.slice(0, LANDING_SIGNAL_BOARD_COUNT).map(r => r.ticker);
  
  // Batch charts for signal board
  const batchChartsQuery = useBatchCharts(signalBoardSymbols, LANDING_MINI_CHART_DAYS);
  
  // Hero chart (larger timeframe for featured stock)
  const heroChartQuery = useStockChart(heroSymbol || undefined, LANDING_HERO_CHART_DAYS);
  
  // Hero signals - use the full quant endpoint with benchmark metrics
  const heroSignalsQuery = useHeroSignalTriggers(heroSymbol || undefined, LANDING_SIGNAL_LOOKBACK_DAYS);
  
  // Hero agent analysis
  const heroAgentQuery = useAgentAnalysis(heroSymbol || undefined);
  
  // Hero dip entry analysis (for fallback when no strategy beats B&H)
  const heroDipEntryQuery = useHeroDipEntry(heroSymbol || undefined);
  
  // Derive as_of timestamp
  const asOfDate = recsQuery.data?.as_of_date;
  const lastUpdatedAt = asOfDate ? Date.parse(asOfDate) : null;
  
  return {
    // Recommendations data
    recommendations,
    marketMessage: recsQuery.data?.market_message ?? null,
    portfolioStats: recsQuery.data ? {
      expectedReturn: recsQuery.data.expected_portfolio_return,
      expectedRisk: recsQuery.data.expected_portfolio_risk,
      totalTrades: recsQuery.data.total_trades,
    } : { expectedReturn: 0, expectedRisk: 0, totalTrades: 0 },
    lastUpdatedAt,
    
    // Chart data
    chartDataMap: batchChartsQuery.data ?? {},
    heroChart: heroChartQuery.data ?? [],
    heroSymbol,
    heroRec,
    
    // Hero analysis
    heroAgentAnalysis: heroAgentQuery.data ?? null,
    heroAgentPending: heroAgentQuery.data?.agent_pending ?? false,
    heroSignals: heroSignalsQuery.data?.triggers ?? [],
    heroSignalSummary: heroSignalsQuery.data ? {
      edgeVsBuyHoldPct: heroSignalsQuery.data.edge_vs_buy_hold_pct,
      buyHoldReturnPct: heroSignalsQuery.data.buy_hold_return_pct,
      signalReturnPct: heroSignalsQuery.data.signal_return_pct,
      nTrades: heroSignalsQuery.data.n_trades,
      beatsBuyHold: heroSignalsQuery.data.beats_buy_hold,
      signalName: heroSignalsQuery.data.signal_name,
    } : null,
    heroDipEntry: heroDipEntryQuery.data ?? null,
    
    // Loading states
    isLoading: recsQuery.isLoading,
    isLoadingCharts: batchChartsQuery.isLoading,
    isLoadingHero: heroChartQuery.isLoading || heroSignalsQuery.isLoading,
    isLoadingAgents: heroAgentQuery.isLoading,
    isLoadingDipEntry: heroDipEntryQuery.isLoading,
    
    // Error state
    isError: recsQuery.isError,
    error: recsQuery.error,
  };
}
