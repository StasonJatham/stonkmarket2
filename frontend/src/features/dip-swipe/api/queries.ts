/**
 * React Query hooks for DipSwipe feature.
 * 
 * Replaces manual useState/useEffect patterns in DipSwipe.tsx.
 */

import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query';
import { apiGet, apiPost, buildUrl } from '@/lib/api-client';
import type {
  DipCard,
  TopSuggestion,
  ChartDataPoint,
  VoteType,
} from '@/services/api';

// ============================================================================
// Dip Cards
// ============================================================================

interface DipCardsResponse {
  cards: DipCard[];
  count: number;
}

async function fetchDipCards(excludeVoted: boolean = true): Promise<DipCardsResponse> {
  return apiGet<DipCardsResponse>(buildUrl('/dips/cards', { 
    exclude_voted: excludeVoted,
    include_chart: true,
  }));
}

export function useDipCards(excludeVoted: boolean = true) {
  return useQuery({
    queryKey: queryKeys.dips.cards(excludeVoted ? 1 : 0),
    queryFn: () => fetchDipCards(excludeVoted),
    staleTime: 5 * 60 * 1000,
    select: (data) => data.cards,
  });
}

// ============================================================================
// Top Suggestions
// ============================================================================

async function fetchTopSuggestions(limit: number = 50, excludeVoted: boolean = true): Promise<TopSuggestion[]> {
  return apiGet<TopSuggestion[]>(buildUrl('/suggestions/top', { 
    limit,
    exclude_voted: excludeVoted,
  }));
}

export function useTopSuggestions(limit: number = 50, excludeVoted: boolean = true) {
  return useQuery({
    queryKey: queryKeys.suggestions.top(),
    queryFn: () => fetchTopSuggestions(limit, excludeVoted),
    staleTime: 5 * 60 * 1000,
  });
}

// ============================================================================
// Suggestion Settings
// ============================================================================

interface SuggestionSettings {
  auto_approve_votes: number;
  max_dip_queue: number;
}

async function fetchSuggestionSettings(): Promise<SuggestionSettings> {
  return apiGet<SuggestionSettings>('/suggestions/settings');
}

export function useSuggestionSettings() {
  return useQuery({
    queryKey: queryKeys.suggestions.settings(),
    queryFn: fetchSuggestionSettings,
    staleTime: 60 * 60 * 1000, // 1 hour - settings rarely change
  });
}

// ============================================================================
// Chart Data (batch prefetch for current + next card)
// ============================================================================

async function fetchChartData(symbol: string, days: number = 90): Promise<ChartDataPoint[]> {
  return apiGet<ChartDataPoint[]>(buildUrl(`/stocks/${symbol}/chart`, { days }));
}

export function useSwipeCharts(symbols: string[], days: number = 90) {
  return useQueries({
    queries: symbols.filter(Boolean).map((symbol) => ({
      queryKey: queryKeys.stocks.chart(symbol, days),
      queryFn: () => fetchChartData(symbol, days),
      staleTime: 10 * 60 * 1000,
      enabled: !!symbol,
    })),
    combine: (results) => {
      const map: Record<string, ChartDataPoint[]> = {};
      results.forEach((result, index) => {
        const symbol = symbols[index];
        if (symbol && result.data) {
          map[symbol] = result.data;
        }
      });
      return {
        data: map,
        isLoading: results.some((r) => r.isLoading),
      };
    },
  });
}

// ============================================================================
// Vote Mutations
// ============================================================================

interface VoteDipParams {
  symbol: string;
  vote: VoteType;
}

async function voteDip({ symbol, vote }: VoteDipParams): Promise<void> {
  return apiPost<void>(`/dips/${symbol}/vote`, { vote });
}

export function useVoteDip() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: voteDip,
    onSuccess: () => {
      // Invalidate dip cards to refetch fresh data
      queryClient.invalidateQueries({ queryKey: queryKeys.dips.all });
    },
  });
}

interface VoteSuggestionParams {
  symbol: string;
}

async function voteForSuggestion({ symbol }: VoteSuggestionParams): Promise<void> {
  return apiPost<void>(`/suggestions/${symbol}/vote`, {});
}

export function useVoteSuggestion() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: voteForSuggestion,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.suggestions.all });
    },
  });
}

// ============================================================================
// Combined Hook for Swipe Page
// ============================================================================

export type SwipeMode = 'dips' | 'suggestions';

export function useSwipeData(mode: SwipeMode) {
  const dipsQuery = useDipCards(true);
  const suggestionsQuery = useTopSuggestions(50, true);
  const settingsQuery = useSuggestionSettings();
  
  // Derive current data based on mode
  const cards = mode === 'dips' ? dipsQuery.data ?? [] : [];
  const suggestions = mode === 'suggestions' ? suggestionsQuery.data ?? [] : [];
  
  return {
    cards,
    suggestions,
    autoApproveVotes: settingsQuery.data?.auto_approve_votes ?? 10,
    isLoading: mode === 'dips' ? dipsQuery.isLoading : suggestionsQuery.isLoading,
    error: mode === 'dips' ? dipsQuery.error : suggestionsQuery.error,
    refetch: mode === 'dips' ? dipsQuery.refetch : suggestionsQuery.refetch,
  };
}
