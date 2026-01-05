/**
 * Watchlist API Types
 */

export interface Watchlist {
  id: number;
  name: string;
  description: string | null;
  is_default: boolean;
  item_count: number;
  created_at: string;
  updated_at: string;
}

export interface WatchlistItem {
  id: number;
  watchlist_id: number;
  symbol: string;
  notes: string | null;
  target_price: number | null;
  alert_on_dip: boolean;
  created_at: string | null;
  updated_at: string | null;
  // Enriched with current market data
  current_price: number | null;
  dip_percent: number | null;
  days_below: number | null;
  is_tail_event: boolean | null;
  ath_price: number | null;
}

export interface WatchlistDetail extends Watchlist {
  items: WatchlistItem[];
}

export interface WatchlistDippingStock {
  symbol: string;
  watchlist_id: number;
  notes: string | null;
  target_price: number | null;
  current_price: number | null;
  dip_percent: number | null;
  days_below: number | null;
  is_tail_event: boolean;
}

export interface WatchlistOpportunity {
  symbol: string;
  watchlist_id: number;
  notes: string | null;
  target_price: number;
  current_price: number;
  discount_percent: number;
}

// Request types
export interface WatchlistCreateRequest {
  name: string;
  description?: string;
  is_default?: boolean;
}

export interface WatchlistUpdateRequest {
  name?: string;
  description?: string | null;
  is_default?: boolean;
}

export interface WatchlistItemAddRequest {
  symbol: string;
  notes?: string;
  target_price?: number;
  alert_on_dip?: boolean;
}

export interface WatchlistItemUpdateRequest {
  notes?: string | null;
  target_price?: number | null;
  alert_on_dip?: boolean;
}

export interface WatchlistBulkAddRequest {
  symbols: string[];
  alert_on_dip?: boolean;
}
