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
