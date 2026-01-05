/**
 * React Query hooks for Watchlist feature.
 */

import { useQuery } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet } from '@/lib/api-client';
import type {
  Watchlist,
  WatchlistDetail,
  WatchlistDippingStock,
  WatchlistOpportunity,
} from './types';

// ============================================================================
// Watchlist List
// ============================================================================

interface WatchlistListResponse {
  watchlists: Watchlist[];
  total_count: number;
}

async function fetchWatchlists(): Promise<Watchlist[]> {
  const response = await apiGet<WatchlistListResponse>('/watchlists');
  return response.watchlists;
}

export function useWatchlists() {
  return useQuery({
    queryKey: queryKeys.watchlists.all,
    queryFn: fetchWatchlists,
    staleTime: 2 * 60 * 1000,
  });
}

// ============================================================================
// Default Watchlist
// ============================================================================

async function fetchDefaultWatchlist(): Promise<Watchlist | null> {
  try {
    return await apiGet<Watchlist>('/watchlists/default');
  } catch {
    return null;
  }
}

export function useDefaultWatchlist() {
  return useQuery({
    queryKey: queryKeys.watchlists.default(),
    queryFn: fetchDefaultWatchlist,
    staleTime: 2 * 60 * 1000,
  });
}

// ============================================================================
// Watchlist Detail with Items
// ============================================================================

async function fetchWatchlistDetail(watchlistId: number): Promise<WatchlistDetail> {
  return apiGet<WatchlistDetail>(`/watchlists/${watchlistId}`);
}

export function useWatchlistDetail(watchlistId: number | null) {
  return useQuery({
    queryKey: queryKeys.watchlists.detail(watchlistId ?? 0),
    queryFn: () => fetchWatchlistDetail(watchlistId!),
    enabled: watchlistId !== null && watchlistId > 0,
    staleTime: 1 * 60 * 1000, // Shorter stale time for real-time price updates
  });
}

// ============================================================================
// Watchlist Items (with enriched market data)
// ============================================================================

async function fetchWatchlistItems(watchlistId: number): Promise<WatchlistDetail['items']> {
  return apiGet<WatchlistDetail['items']>(`/watchlists/${watchlistId}/items`);
}

export function useWatchlistItems(watchlistId: number | null) {
  return useQuery({
    queryKey: queryKeys.watchlists.items(watchlistId ?? 0),
    queryFn: () => fetchWatchlistItems(watchlistId!),
    enabled: watchlistId !== null && watchlistId > 0,
    staleTime: 1 * 60 * 1000,
  });
}

// ============================================================================
// Dipping Stocks across all Watchlists
// ============================================================================

interface WatchlistDippingResponse {
  stocks: WatchlistDippingStock[];
  total_count: number;
}

async function fetchDippingStocks(minDipPct?: number): Promise<WatchlistDippingStock[]> {
  const params = minDipPct !== undefined ? `?min_dip_pct=${minDipPct}` : '';
  const response = await apiGet<WatchlistDippingResponse>(`/watchlists/dipping${params}`);
  return response.stocks;
}

export function useDippingStocks(minDipPct?: number) {
  return useQuery({
    queryKey: queryKeys.watchlists.dipping(minDipPct),
    queryFn: () => fetchDippingStocks(minDipPct),
    staleTime: 1 * 60 * 1000,
  });
}

// ============================================================================
// Opportunities (stocks at or below target price)
// ============================================================================

interface WatchlistOpportunitiesResponse {
  opportunities: WatchlistOpportunity[];
  total_count: number;
}

async function fetchOpportunities(): Promise<WatchlistOpportunity[]> {
  const response = await apiGet<WatchlistOpportunitiesResponse>('/watchlists/opportunities');
  return response.opportunities;
}

export function useOpportunities() {
  return useQuery({
    queryKey: queryKeys.watchlists.opportunities(),
    queryFn: fetchOpportunities,
    staleTime: 1 * 60 * 1000,
  });
}
