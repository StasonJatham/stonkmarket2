/* eslint-disable react-refresh/only-export-components -- Context module exports hook alongside provider */
import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import { getRanking } from '@/services/api';
import type { DipStock } from '@/services/api';

interface DipContextType {
  stocks: DipStock[];
  tickerStocks: DipStock[]; // Always ALL stocks for ticker (max 40)
  isLoading: boolean;
  isLoadingTicker: boolean;
  lastUpdated: string | null;
  error: string | null;
  refreshRanking: (skipCache?: boolean, showAll?: boolean) => Promise<void>;
  showAllStocks: boolean;
  setShowAllStocks: (show: boolean) => void;
}

const DipContext = createContext<DipContextType | null>(null);

export function DipProvider({ children }: { children: ReactNode }) {
  const [stocks, setStocks] = useState<DipStock[]>([]);
  const [tickerStocks, setTickerStocks] = useState<DipStock[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingTicker, setIsLoadingTicker] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAllStocks, setShowAllStocks] = useState(false);

  // Load ticker stocks once on mount (always ALL stocks)
  useEffect(() => {
    const loadTickerStocks = async () => {
      setIsLoadingTicker(true);
      try {
        const data = await getRanking(false, true); // Always showAll=true for ticker
        setTickerStocks(data.ranking.slice(0, 40));
      } catch (err) {
        console.error('Failed to load ticker stocks:', err);
      } finally {
        setIsLoadingTicker(false);
      }
    };
    loadTickerStocks();
  }, []);

  const refreshRanking = useCallback(async (skipCache = false, showAll?: boolean) => {
    const useShowAll = showAll ?? showAllStocks;
    setIsLoading(true);
    setError(null);
    try {
      const data = await getRanking(skipCache, useShowAll);
      setStocks(data.ranking);
      setLastUpdated(data.last_updated);
      // Also update ticker stocks if we got all stocks
      if (useShowAll) {
        setTickerStocks(data.ranking.slice(0, 40));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load ranking');
    } finally {
      setIsLoading(false);
    }
  }, [showAllStocks]);

  // Load on mount and when showAllStocks changes
  useEffect(() => {
    refreshRanking(false, showAllStocks);
  }, [showAllStocks]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <DipContext.Provider value={{
      stocks,
      tickerStocks,
      isLoading,
      isLoadingTicker,
      lastUpdated,
      error,
      refreshRanking,
      showAllStocks,
      setShowAllStocks,
    }}>
      {children}
    </DipContext.Provider>
  );
}

export function useDips() {
  const context = useContext(DipContext);
  if (!context) {
    throw new Error('useDips must be used within a DipProvider');
  }
  return context;
}
