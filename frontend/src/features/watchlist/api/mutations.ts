/**
 * React Query mutation hooks for Watchlist feature.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiPost, apiPatch, apiDelete } from '@/lib/api-client';
import type {
  Watchlist,
  WatchlistDetail,
  WatchlistItem,
  WatchlistCreateRequest,
  WatchlistUpdateRequest,
  WatchlistItemAddRequest,
  WatchlistItemUpdateRequest,
  WatchlistBulkAddRequest,
} from './types';

// ============================================================================
// Create Watchlist
// ============================================================================

export function useCreateWatchlist() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data: WatchlistCreateRequest) => 
      apiPost<Watchlist>('/watchlists', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
    },
  });
}

// ============================================================================
// Update Watchlist
// ============================================================================

export function useUpdateWatchlist() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: WatchlistUpdateRequest }) => 
      apiPatch<Watchlist>(`/watchlists/${id}`, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.detail(id) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.default() });
    },
  });
}

// ============================================================================
// Delete Watchlist
// ============================================================================

export function useDeleteWatchlist() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: number) => 
      apiDelete(`/watchlists/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.default() });
    },
  });
}

// ============================================================================
// Add Item to Watchlist
// ============================================================================

export function useAddWatchlistItem() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ watchlistId, data }: { watchlistId: number; data: WatchlistItemAddRequest }) => 
      apiPost<WatchlistItem>(`/watchlists/${watchlistId}/items`, data),
    onSuccess: (_, { watchlistId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.detail(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.items(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.dipping() });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.opportunities() });
    },
  });
}

// ============================================================================
// Bulk Add Items to Watchlist
// ============================================================================

export function useBulkAddWatchlistItems() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ watchlistId, data }: { watchlistId: number; data: WatchlistBulkAddRequest }) => 
      apiPost<WatchlistDetail>(`/watchlists/${watchlistId}/items/bulk`, data),
    onSuccess: (_, { watchlistId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.detail(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.items(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.dipping() });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.opportunities() });
    },
  });
}

// ============================================================================
// Update Watchlist Item
// ============================================================================

export function useUpdateWatchlistItem() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ watchlistId, itemId, data }: { 
      watchlistId: number; 
      itemId: number; 
      data: WatchlistItemUpdateRequest 
    }) => 
      apiPatch<WatchlistItem>(`/watchlists/${watchlistId}/items/${itemId}`, data),
    onSuccess: (_, { watchlistId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.detail(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.items(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.dipping() });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.opportunities() });
    },
  });
}

// ============================================================================
// Delete Watchlist Item
// ============================================================================

export function useDeleteWatchlistItem() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ watchlistId, itemId }: { watchlistId: number; itemId: number }) => 
      apiDelete(`/watchlists/${watchlistId}/items/${itemId}`),
    onSuccess: (_, { watchlistId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.detail(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.items(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.dipping() });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.opportunities() });
    },
  });
}

// ============================================================================
// Delete Item by Symbol (convenience)
// ============================================================================

export function useDeleteWatchlistItemBySymbol() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ watchlistId, symbol }: { watchlistId: number; symbol: string }) => 
      apiDelete(`/watchlists/${watchlistId}/items/symbol/${symbol}`),
    onSuccess: (_, { watchlistId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.detail(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.items(watchlistId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.dipping() });
      queryClient.invalidateQueries({ queryKey: queryKeys.watchlists.opportunities() });
    },
  });
}
