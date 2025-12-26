/* eslint-disable react-refresh/only-export-components -- Context module exports hook alongside provider */
import { createContext, useContext, useState, useEffect, useCallback, useMemo, type ReactNode } from 'react';
import { getQuantRecommendations, preloadStockLogos, quantToDipStock } from '@/services/api';
import type { QuantRecommendation, QuantEngineResponse, QuantAuditBlock, DipStock } from '@/services/api';

interface QuantContextType {
  recommendations: QuantRecommendation[];
  /** Recommendations converted to DipStock for backward compatible components */
  stocks: DipStock[];
  audit: QuantAuditBlock | null;
  asOfDate: string | null;
  portfolioStats: {
    expectedReturn: number;
    expectedRisk: number;
    totalTrades: number;
    transactionCostEur: number;
  } | null;
  isLoading: boolean;
  error: string | null;
  refreshRecommendations: (skipCache?: boolean) => Promise<void>;
  inflow: number;
  setInflow: (amount: number) => void;
}

const QuantContext = createContext<QuantContextType | null>(null);

export function QuantProvider({ children }: { children: ReactNode }) {
  const [recommendations, setRecommendations] = useState<QuantRecommendation[]>([]);
  const [audit, setAudit] = useState<QuantAuditBlock | null>(null);
  const [asOfDate, setAsOfDate] = useState<string | null>(null);
  const [portfolioStats, setPortfolioStats] = useState<QuantContextType['portfolioStats']>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [inflow, setInflow] = useState(1000);

  // Convert recommendations to DipStock for backward compatibility
  const stocks = useMemo(() => {
    return recommendations.map(quantToDipStock);
  }, [recommendations]);

  const refreshRecommendations = useCallback(async (_skipCache = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const data: QuantEngineResponse = await getQuantRecommendations(inflow, 40);
      setRecommendations(data.recommendations);
      setAudit(data.audit);
      setAsOfDate(data.as_of_date);
      setPortfolioStats({
        expectedReturn: data.expected_portfolio_return,
        expectedRisk: data.expected_portfolio_risk,
        totalTrades: data.total_trades,
        transactionCostEur: data.total_transaction_cost_eur,
      });
      
      // Preload logos for all tickers
      const tickers = data.recommendations.map(r => r.ticker);
      preloadStockLogos(tickers);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load quant recommendations');
    } finally {
      setIsLoading(false);
    }
  }, [inflow]);

  // Load on mount and when inflow changes
  useEffect(() => {
    refreshRecommendations();
  }, [refreshRecommendations]);

  return (
    <QuantContext.Provider value={{
      recommendations,
      stocks,
      audit,
      asOfDate,
      portfolioStats,
      isLoading,
      error,
      refreshRecommendations,
      inflow,
      setInflow,
    }}>
      {children}
    </QuantContext.Provider>
  );
}

export function useQuant() {
  const context = useContext(QuantContext);
  if (!context) {
    throw new Error('useQuant must be used within a QuantProvider');
  }
  return context;
}
