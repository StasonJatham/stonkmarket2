/* eslint-disable react-refresh/only-export-components */
/**
 * React Query Client Configuration
 * 
 * Single source of truth for all server state caching.
 * Replaces the custom apiCache layer with React Query's built-in cache.
 * 
 * Cache Strategy for Daily Price Updates:
 * - staleTime: 5 minutes (data considered fresh)
 * - gcTime: 30 minutes (keep in memory for quick access)
 * - No aggressive refetching (prices only update daily)
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { type ReactNode } from 'react';

// Create a stable query client instance
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Data is considered fresh for 2 minutes
      // (short enough to catch manual refreshes, long enough to prevent spam)
      staleTime: 2 * 60 * 1000,
      // Keep unused data in cache for 10 minutes
      gcTime: 10 * 60 * 1000,
      // Refetch on window focus to catch daily updates when user returns
      refetchOnWindowFocus: true,
      // Refetch on reconnect to get latest data after network issues
      refetchOnReconnect: true,
      // Retry failed requests 2 times with exponential backoff
      retry: 2,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    },
    mutations: {
      // Retry mutations once on network errors
      retry: 1,
    },
  },
});

interface QueryProviderProps {
  children: ReactNode;
}

export function QueryProvider({ children }: QueryProviderProps) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {import.meta.env.DEV && (
        <ReactQueryDevtools initialIsOpen={false} buttonPosition="bottom-left" />
      )}
    </QueryClientProvider>
  );
}

/**
 * Query key factory for consistent cache keys across the app.
 * Using a factory pattern ensures type-safe, consistent keys.
 */
export const queryKeys = {
  // Market data
  stocks: {
    all: ['stocks'] as const,
    ranking: (showAll: boolean) => ['stocks', 'ranking', { showAll }] as const,
    detail: (symbol: string) => ['stocks', 'detail', symbol] as const,
    chart: (symbol: string, days: number) => ['stocks', 'chart', symbol, { days }] as const,
    info: (symbol: string) => ['stocks', 'info', symbol] as const,
    signals: (symbol: string) => ['stocks', 'signals', symbol] as const,
    fundamentals: (symbol: string) => ['stocks', 'fundamentals', symbol] as const,
  },
  
  // Quant engine
  quant: {
    all: ['quant'] as const,
    recommendations: (inflow: number, limit: number) => 
      ['quant', 'recommendations', { inflow, limit }] as const,
  },
  
  // Dip cards (swipe feature)
  dips: {
    all: ['dips'] as const,
    cards: (limit: number) => ['dips', 'cards', { limit }] as const,
    card: (symbol: string) => ['dips', 'card', symbol] as const,
  },
  
  // Portfolios
  portfolios: {
    all: ['portfolios'] as const,
    list: () => ['portfolios', 'list'] as const,
    detail: (id: number) => ['portfolios', 'detail', id] as const,
    analytics: (id: number) => ['portfolios', 'analytics', id] as const,
    risk: (id: number) => ['portfolios', 'risk', id] as const,
    riskAnalytics: (id: number) => ['portfolios', 'riskAnalytics', id] as const,
    allocation: (id: number, inflow?: number, method?: string) => 
      ['portfolios', 'allocation', id, { inflow, method }] as const,
    analyticsJob: (id: number, jobId: string) => ['portfolios', 'analyticsJob', id, jobId] as const,
    sparklines: (id: number, symbols: string[], days: number) => 
      ['portfolios', 'sparklines', id, { symbols: symbols.join(','), days }] as const,
  },
  
  // Suggestions
  suggestions: {
    all: ['suggestions'] as const,
    list: () => ['suggestions', 'list'] as const,
    top: () => ['suggestions', 'top'] as const,
    settings: () => ['suggestions', 'settings'] as const,
    metrics: () => ['suggestions', 'metrics'] as const,
    history: () => ['suggestions', 'history'] as const,
  },
  
  // Swipe voting
  swipe: {
    all: ['swipe'] as const,
    cards: () => ['swipe', 'cards'] as const,
    history: () => ['swipe', 'history'] as const,
  },
  
  // Admin
  admin: {
    symbols: () => ['admin', 'symbols'] as const,
    jobs: () => ['admin', 'jobs'] as const,
    apiKeys: () => ['admin', 'apiKeys'] as const,
    settings: () => ['admin', 'settings'] as const,
    benchmarks: () => ['admin', 'benchmarks'] as const,
  },
  
  // Notifications
  notifications: {
    all: ['notifications'] as const,
    channels: () => ['notifications', 'channels'] as const,
    rules: () => ['notifications', 'rules'] as const,
    history: (page?: number, pageSize?: number) => 
      ['notifications', 'history', { page, pageSize }] as const,
    summary: () => ['notifications', 'summary'] as const,
    triggerTypes: () => ['notifications', 'triggerTypes'] as const,
  },
  
  // Benchmarks
  benchmarks: {
    all: ['benchmarks'] as const,
    list: () => ['benchmarks', 'list'] as const,
    chart: (symbol: string, days: number) => ['benchmarks', 'chart', symbol, { days }] as const,
  },
} as const;

/**
 * Utility to invalidate related queries after mutations.
 * Example: After adding a holding, invalidate portfolio detail and analytics.
 */
export function invalidatePortfolioQueries(portfolioId: number) {
  queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.detail(portfolioId) });
  queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.analytics(portfolioId) });
  queryClient.invalidateQueries({ queryKey: queryKeys.portfolios.risk(portfolioId) });
}

export function invalidateStockQueries(symbol: string) {
  queryClient.invalidateQueries({ queryKey: queryKeys.stocks.detail(symbol) });
  queryClient.invalidateQueries({ queryKey: queryKeys.stocks.chart(symbol, 0) }); // All chart periods
  queryClient.invalidateQueries({ queryKey: queryKeys.stocks.info(symbol) });
}
