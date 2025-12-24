import { getAuthHeaders } from './auth';
import { apiCache, CACHE_TTL } from '@/lib/cache';
import { getDeviceFingerprint, getDeviceFingerprintSync, recordLocalVote, hasVotedLocally, getLocalVotes } from '@/lib/fingerprint';

const API_BASE = '/api';

// ETag storage for conditional requests
const etagStore = new Map<string, string>();
// Data store for 304 responses - keeps last response even when cache TTL=0
const etagDataStore = new Map<string, unknown>();

interface FetchOptions extends RequestInit {
  useEtag?: boolean;
}

async function fetchAPI<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const { useEtag = false, ...fetchOptions } = options;
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...getAuthHeaders(),
    ...fetchOptions.headers,
  };
  
  // Add If-None-Match header if we have a stored ETag for this endpoint
  if (useEtag) {
    const storedEtag = etagStore.get(endpoint);
    if (storedEtag) {
      (headers as Record<string, string>)['If-None-Match'] = storedEtag;
    }
  }
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...fetchOptions,
    headers,
  });

  // Handle 304 Not Modified - return cached data
  if (response.status === 304) {
    // First try ETag data store (always populated on 200 with ETag)
    const etagData = etagDataStore.get(endpoint);
    if (etagData !== undefined) {
      return etagData as T;
    }
    // Fallback to regular cache
    const cached = apiCache.get<T>(`api:${endpoint}`);
    if (cached) {
      return cached.data;
    }
    // No cached data - clear the stale ETag and refetch without If-None-Match
    etagStore.delete(endpoint);
    return fetchAPI<T>(endpoint, { ...options, useEtag: false });
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.message || error.detail || `HTTP ${response.status}`);
  }

  const data = await response.json();

  // Store ETag and data from response if present
  const etag = response.headers.get('ETag');
  if (etag) {
    etagStore.set(endpoint, etag);
    etagDataStore.set(endpoint, data);
  }

  return data;
}

/**
 * Invalidate ETag cache for specific endpoints or patterns.
 * Call this after mutations to ensure fresh data on next fetch.
 */
function invalidateEtagCache(pattern?: string | RegExp): void {
  if (!pattern) {
    etagStore.clear();
    etagDataStore.clear();
    return;
  }
  
  const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
  for (const key of etagStore.keys()) {
    if (regex.test(key)) {
      etagStore.delete(key);
      etagDataStore.delete(key);
    }
  }
}

// Types
export interface DipStock {
  symbol: string;
  name: string | null;
  depth: number;
  last_price: number;
  previous_close: number | null;
  change_percent: number | null;
  days_since_dip: number | null;
  high_52w: number | null;
  low_52w: number | null;
  market_cap: number | null;
  sector: string | null;
  pe_ratio: number | null;
  volume: number | null;
  symbol_type?: 'stock' | 'index';
  // Added for UI calculations
  dip_score?: number;
  recovery_potential?: number;
}

export interface RankingResponse {
  ranking: DipStock[];
  last_updated: string | null;
}

export interface StockInfo {
  symbol: string;
  name: string | null;
  sector: string | null;
  industry: string | null;
  market_cap: number | null;
  pe_ratio: number | null;
  forward_pe: number | null;
  dividend_yield: number | null;
  beta: number | null;
  avg_volume: number | null;
  summary: string | null;
  summary_ai: string | null;  // AI-generated short summary (~300 chars)
  website: string | null;
  recommendation: string | null;
}

export interface ChartDataPoint {
  date: string;
  close: number;
  threshold: number | null;
  ref_high: number | null;
  drawdown: number | null;
  since_dip: number | null;
  ref_high_date: string | null;
  dip_start_date: string | null;
}

export interface CronJob {
  name: string;
  cron: string;
  description: string | null;
}

export interface CronLogEntry {
  name: string;
  status: string;
  message: string | null;
  created_at: string;
}

export interface CronLogsResponse {
  logs: CronLogEntry[];
  total: number;
}

// API functions with caching
export async function getRanking(skipCache = false, showAll = false): Promise<RankingResponse> {
  const endpoint = `/dips/ranking?show_all=${showAll}`;
  const cacheKey = `ranking:${showAll}`;
  
  const fetcher = async () => {
    const ranking = await fetchAPI<DipStock[]>(endpoint, { useEtag: true });
    return {
      ranking: ranking.map((stock, index) => ({
        ...stock,
        dip_score: 100 - (index * (100 / Math.max(ranking.length, 1))),
        recovery_potential: stock.depth * 100,
      })),
      last_updated: new Date().toISOString(),
    };
  };
  
  if (skipCache) {
    apiCache.invalidate(cacheKey);
  }
  
  return apiCache.fetch(cacheKey, fetcher, { 
    ttl: CACHE_TTL.RANKING,
    staleWhileRevalidate: true,
  });
}

export async function getStockChart(symbol: string, days: number = 180): Promise<ChartDataPoint[]> {
  const endpoint = `/dips/${symbol}/chart?days=${days}`;
  const cacheKey = `chart:${symbol}:${days}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<ChartDataPoint[]>(endpoint, { useEtag: true }),
    { ttl: CACHE_TTL.CHART, staleWhileRevalidate: true }
  );
}

export async function getStockInfo(symbol: string): Promise<StockInfo> {
  const cacheKey = `info:${symbol}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<StockInfo>(`/dips/${symbol}/info`),
    { ttl: CACHE_TTL.STOCK_INFO, staleWhileRevalidate: true }
  );
}

// =============================================================================
// PREFETCH UTILITIES
// =============================================================================

/**
 * Prefetch chart data for a stock in background (non-blocking).
 * Call this on hover or when you anticipate user interaction.
 * Defaults to 90 days (3 months) - fetch more data on demand.
 */
export function prefetchStockChart(symbol: string, days: number = 90): void {
  const endpoint = `/dips/${symbol}/chart?days=${days}`;
  const cacheKey = `chart:${symbol}:${days}`;
  
  apiCache.prefetch(
    cacheKey,
    () => fetchAPI<ChartDataPoint[]>(endpoint, { useEtag: true }),
    { ttl: CACHE_TTL.CHART }
  );
}

/**
 * Prefetch stock info in background.
 */
export function prefetchStockInfo(symbol: string): void {
  const cacheKey = `info:${symbol}`;
  
  apiCache.prefetch(
    cacheKey,
    () => fetchAPI<StockInfo>(`/dips/${symbol}/info`),
    { ttl: CACHE_TTL.STOCK_INFO }
  );
}

/**
 * Prefetch data for multiple stocks.
 * Called after ranking loads to pre-warm cache for all active dips.
 * Defaults to 90 days (3 months) - more data is fetched on demand.
 */
export function prefetchTopStocks(symbols: string[], chartDays: number = 90): void {
  // Stagger prefetches to avoid hammering the API
  symbols.forEach((symbol, index) => {
    setTimeout(() => {
      prefetchStockChart(symbol, chartDays);
      prefetchStockInfo(symbol);
    }, index * 100); // 100ms between each prefetch
  });
}

/**
 * Preload stock logo images for faster display.
 * Uses link preload for browser-native prefetching.
 */
export function preloadStockLogos(symbols: string[], theme: 'light' | 'dark' = 'light'): void {
  if (typeof document === 'undefined') return;
  
  // Create a fragment to batch DOM operations
  const fragment = document.createDocumentFragment();
  const existingPreloads = new Set(
    Array.from(document.querySelectorAll('link[rel="preload"][as="image"]'))
      .map(link => link.getAttribute('href'))
  );
  
  symbols.forEach(symbol => {
    const logoUrl = `/api/logos/${symbol.toUpperCase()}?theme=${theme}`;
    if (existingPreloads.has(logoUrl)) return;
    
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'image';
    link.href = logoUrl;
    fragment.appendChild(link);
  });
  
  document.head.appendChild(fragment);
}

export interface PublicBenchmark {
  id: string;
  symbol: string;
  name: string;
  description?: string | null;
}

export async function getAvailableBenchmarks(): Promise<PublicBenchmark[]> {
  const cacheKey = 'benchmarks';
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<PublicBenchmark[]>('/dips/benchmarks'),
    { ttl: CACHE_TTL.BENCHMARK, staleWhileRevalidate: true }
  );
}

export async function getCronJobs(): Promise<CronJob[]> {
  return apiCache.fetch(
    'cronjobs',
    () => fetchAPI<CronJob[]>('/cronjobs'),
    { ttl: CACHE_TTL.CRON_JOBS }
  );
}

export async function getCronLogs(
  limit: number = 50,
  offset: number = 0,
  search?: string,
  status?: string
): Promise<CronLogsResponse> {
  const params = new URLSearchParams();
  params.append('limit', limit.toString());
  params.append('offset', offset.toString());
  if (search) params.append('search', search);
  if (status) params.append('status', status);
  return fetchAPI<CronLogsResponse>(`/cronjobs/logs/all?${params.toString()}`);
}

export async function updateCronJob(name: string, cron: string): Promise<CronJob> {
  return fetchAPI<CronJob>(`/cronjobs/${name}`, {
    method: 'PUT',
    body: JSON.stringify({ cron }),
  });
}

export async function runCronJobNow(name: string): Promise<CronLogEntry> {
  return fetchAPI<CronLogEntry>(`/cronjobs/${name}/run`, {
    method: 'POST',
  });
}

export async function refreshData(): Promise<DipStock[]> {
  return fetchAPI<DipStock[]>('/dips/refresh', {
    method: 'POST',
  });
}

// Benchmark types and functions
// BenchmarkType is now dynamic - can be any benchmark ID from the API or null
export type BenchmarkType = string | null;

export interface BenchmarkDataPoint {
  date: string;
  close: number;
  normalizedValue: number; // Normalized to percentage change from start
}

export interface ComparisonChartData {
  date: string;
  displayDate: string;
  stockValue: number;
  stockNormalized: number;
  benchmarkValue?: number;
  benchmarkNormalized?: number;
}

// Legacy benchmark symbol mapping (fallback for known benchmarks)
const LEGACY_BENCHMARK_SYMBOLS: Record<string, string> = {
  SP500: '^GSPC',      // S&P 500 Index
  MSCI_WORLD: 'URTH',  // iShares MSCI World ETF (proxy for MSCI World)
};

const LEGACY_BENCHMARK_NAMES: Record<string, string> = {
  SP500: 'S&P 500',
  MSCI_WORLD: 'MSCI World',
};

export function getBenchmarkName(benchmark: BenchmarkType): string {
  if (!benchmark) return '';
  return LEGACY_BENCHMARK_NAMES[benchmark] || benchmark;
}

// Cache for benchmark config lookup
let benchmarkConfigCache: PublicBenchmark[] | null = null;

// Clear benchmark config cache - called when settings are updated
export function clearBenchmarkConfigCache(): void {
  benchmarkConfigCache = null;
}

async function getBenchmarkConfig(): Promise<PublicBenchmark[]> {
  if (benchmarkConfigCache) return benchmarkConfigCache;
  try {
    benchmarkConfigCache = await getAvailableBenchmarks();
    return benchmarkConfigCache;
  } catch {
    return [];
  }
}

export async function getBenchmarkChart(
  benchmark: string,
  days: number = 365
): Promise<ChartDataPoint[]> {
  // First try to get symbol from API config
  const configs = await getBenchmarkConfig();
  const config = configs.find(b => b.id === benchmark);
  
  // Use symbol from config, or fallback to legacy mapping, or use benchmark as symbol
  const symbol = config?.symbol || LEGACY_BENCHMARK_SYMBOLS[benchmark] || benchmark;
  
  if (!symbol) {
    console.error(`Unknown benchmark: ${benchmark}`);
    return [];
  }
  // Use /chart endpoint which exists, not /history
  // URL encode the symbol in case it contains special chars like ^
  return fetchAPI<ChartDataPoint[]>(`/dips/${encodeURIComponent(symbol)}/chart?days=${days}`);
}

// Normalize chart data to percentage change from first value
export function normalizeChartData(data: ChartDataPoint[]): ChartDataPoint[] {
  if (data.length === 0) return data;
  const firstValue = data[0].close;
  return data.map(point => ({
    ...point,
    close: ((point.close - firstValue) / firstValue) * 100, // Percentage change
  }));
}

// Merge stock and benchmark data for comparison charts
export function mergeChartData(
  stockData: ChartDataPoint[],
  benchmarkData: ChartDataPoint[],
  normalizeForComparison: boolean = true
): ComparisonChartData[] {
  if (stockData.length === 0) return [];
  
  // Create a map of benchmark data by date
  const benchmarkMap = new Map<string, ChartDataPoint>();
  benchmarkData.forEach(point => {
    benchmarkMap.set(point.date.split('T')[0], point);
  });
  
  const stockFirstValue = stockData[0].close;
  const benchmarkFirstValue = benchmarkData[0]?.close || 1;
  
  return stockData.map(point => {
    const dateKey = point.date.split('T')[0];
    const benchmarkPoint = benchmarkMap.get(dateKey);
    
    const stockNormalized = normalizeForComparison
      ? ((point.close - stockFirstValue) / stockFirstValue) * 100
      : point.close;
    
    const benchmarkNormalized = benchmarkPoint && normalizeForComparison
      ? ((benchmarkPoint.close - benchmarkFirstValue) / benchmarkFirstValue) * 100
      : benchmarkPoint?.close;
    
    return {
      date: point.date,
      displayDate: new Date(point.date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      stockValue: point.close,
      stockNormalized,
      benchmarkValue: benchmarkPoint?.close,
      benchmarkNormalized,
    };
  });
}

// Aggregate portfolio performance (average of all dips)
export interface AggregatedPerformance {
  date: string;
  displayDate: string;
  portfolioAvg: number;
  portfolioMin: number;
  portfolioMax: number;
  benchmarkValue?: number;
}

export function aggregatePortfolioPerformance(
  stocksData: Map<string, ChartDataPoint[]>,
  benchmarkData: ChartDataPoint[]
): AggregatedPerformance[] {
  if (stocksData.size === 0) return [];
  
  // Get all unique dates
  const allDates = new Set<string>();
  stocksData.forEach(data => {
    data.forEach(point => allDates.add(point.date.split('T')[0]));
  });
  
  // Create benchmark map
  const benchmarkMap = new Map<string, number>();
  const benchmarkFirstValue = benchmarkData[0]?.close || 1;
  benchmarkData.forEach(point => {
    const normalized = ((point.close - benchmarkFirstValue) / benchmarkFirstValue) * 100;
    benchmarkMap.set(point.date.split('T')[0], normalized);
  });
  
  // Create stock maps with normalized values
  const stockMaps: Map<string, number>[] = [];
  stocksData.forEach(data => {
    const map = new Map<string, number>();
    const firstValue = data[0]?.close || 1;
    data.forEach(point => {
      const normalized = ((point.close - firstValue) / firstValue) * 100;
      map.set(point.date.split('T')[0], normalized);
    });
    stockMaps.push(map);
  });
  
  // Sort dates
  const sortedDates = Array.from(allDates).sort();
  
  // Aggregate
  return sortedDates.map(dateKey => {
    const values: number[] = [];
    stockMaps.forEach(map => {
      const value = map.get(dateKey);
      if (value !== undefined) values.push(value);
    });
    
    const avg = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const min = values.length > 0 ? Math.min(...values) : 0;
    const max = values.length > 0 ? Math.max(...values) : 0;
    
    return {
      date: dateKey,
      displayDate: new Date(dateKey).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      portfolioAvg: avg,
      portfolioMin: min,
      portfolioMax: max,
      benchmarkValue: benchmarkMap.get(dateKey),
    };
  });
}

// =============================================================================
// SWIPE VOTING TYPES & API
// =============================================================================

export interface VoteCounts {
  buy: number;
  sell: number;
  buy_weighted: number;
  sell_weighted: number;
  net_score: number;
}

export interface DipCard {
  symbol: string;
  name: string | null;
  sector: string | null;
  industry: string | null;
  website: string | null;
  current_price: number;
  ref_high: number;
  dip_pct: number;
  days_below: number;
  min_dip_pct: number | null;
  summary_ai: string | null;
  swipe_bio: string | null;
  ai_rating: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell' | null;
  ai_reasoning: string | null;
  ai_confidence: number | null;
  vote_counts: VoteCounts;
  ipo_year: number | null;
}

export interface DipCardList {
  cards: DipCard[];
  total: number;
}

export interface DipStats {
  symbol: string;
  vote_counts: VoteCounts;
  total_votes: number;
  weighted_total: number;
  buy_pct: number;
  sell_pct: number;
  sentiment: 'very_bullish' | 'bullish' | 'neutral' | 'bearish' | 'very_bearish';
}

export type VoteType = 'buy' | 'sell';

export async function getDipCards(includeAi: boolean = false, excludeVoted: boolean = false): Promise<DipCardList> {
  const cacheKey = `swipe:cards:${includeAi}:${excludeVoted}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<DipCardList>(`/swipe/cards?include_ai=${includeAi}&exclude_voted=${excludeVoted}`),
    { ttl: CACHE_TTL.RANKING }
  );
}

export async function getDipCard(symbol: string, refreshAi: boolean = false): Promise<DipCard> {
  return fetchAPI<DipCard>(`/swipe/cards/${symbol}?refresh_ai=${refreshAi}`);
}

export async function voteDip(symbol: string, voteType: VoteType): Promise<{ symbol: string; vote_type: string; message: string }> {
  return fetchAPI(`/swipe/cards/${symbol}/vote`, {
    method: 'PUT',
    body: JSON.stringify({ vote_type: voteType }),
  });
}

export async function getDipStats(symbol: string): Promise<DipStats> {
  return fetchAPI<DipStats>(`/swipe/cards/${symbol}/stats`);
}

export async function refreshAiAnalysis(symbol: string): Promise<DipCard> {
  return fetchAPI<DipCard>(`/swipe/cards/${symbol}/refresh-ai`, {
    method: 'POST',
  });
}

export type AiFieldType = 'rating' | 'bio' | 'summary';
export type SwipeAiFieldType = 'rating' | 'bio';

/**
 * Regenerate AI field for a dip card.
 * Use for swipe-specific fields: rating and bio.
 * For summary, use regenerateSymbolAiSummary instead.
 */
export async function refreshAiField(symbol: string, field: AiFieldType): Promise<DipCard> {
  // Route summary to the proper symbols endpoint
  if (field === 'summary') {
    await regenerateSymbolAiSummary(symbol);
    // Fetch the updated card to return consistent type
    return getDipCard(symbol);
  }
  
  const result = await fetchAPI<DipCard>(`/swipe/cards/${symbol}/refresh-ai/${field}`, {
    method: 'POST',
  });
  
  return result;
}

/**
 * Regenerate AI summary for a symbol.
 * This is a symbol property, not swipe-specific.
 */
export async function regenerateSymbolAiSummary(symbol: string): Promise<{ symbol: string; summary_ai: string | null }> {
  const result = await fetchAPI<{ symbol: string; summary_ai: string | null }>(`/symbols/${symbol}/ai/summary`, {
    method: 'POST',
  });
  
  // Invalidate relevant caches
  apiCache.invalidate(`info:${symbol}`);
  apiCache.invalidate(`swipe:cards:true:false`);
  apiCache.invalidate(`swipe:cards:true:true`);
  apiCache.invalidate(`swipe:cards:false:false`);
  apiCache.invalidate(`swipe:cards:false:true`);
  
  return result;
}

// =============================================================================
// DIPFINDER TYPES & API
// =============================================================================

export interface QualityFactors {
  score: number;
  pe_ratio: number | null;
  forward_pe: number | null;
  pb_ratio: number | null;
  ps_ratio: number | null;
  profit_margin: number | null;
  roe: number | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  factors_available: number;
}

export interface StabilityFactors {
  score: number;
  beta: number | null;
  volatility_30d: number | null;
  volatility_90d: number | null;
  avg_volume: number | null;
  market_cap: number | null;
  factors_available: number;
}

export interface DipSignal {
  ticker: string;
  window: number;
  benchmark: string;
  as_of_date: string;
  dip_stock: number;
  peak_stock: number;
  current_price: number;
  dip_pctl: number;
  dip_vs_typical: number;
  persist_days: number;
  dip_mkt: number;
  excess_dip: number;
  dip_class: 'MARKET_DIP' | 'STOCK_SPECIFIC' | 'MIXED';
  quality_score: number;
  stability_score: number;
  dip_score: number;
  final_score: number;
  alert_level: 'NONE' | 'GOOD' | 'STRONG';
  should_alert: boolean;
  reason: string;
  quality_factors?: QualityFactors;
  stability_factors?: StabilityFactors;
}

export interface DipSignalListResponse {
  signals: DipSignal[];
  count: number;
  benchmark: string;
  window: number;
  as_of_date: string;
}

export async function getDipFinderSignals(
  tickers: string[],
  options: { window?: number; benchmark?: string; includeFactors?: boolean } = {}
): Promise<DipSignalListResponse> {
  const params = new URLSearchParams();
  params.append('tickers', tickers.join(','));
  if (options.window) params.append('window', options.window.toString());
  if (options.benchmark) params.append('benchmark', options.benchmark);
  if (options.includeFactors) params.append('include_factors', 'true');
  
  return fetchAPI<DipSignalListResponse>(`/dipfinder/signals?${params.toString()}`);
}

export async function getDipFinderSignal(
  ticker: string,
  options: { window?: number; benchmark?: string; forceRefresh?: boolean } = {}
): Promise<DipSignal> {
  const params = new URLSearchParams();
  if (options.window) params.append('window', options.window.toString());
  if (options.benchmark) params.append('benchmark', options.benchmark);
  if (options.forceRefresh) params.append('force_refresh', 'true');
  
  return fetchAPI<DipSignal>(`/dipfinder/signals/${ticker}?${params.toString()}`);
}

// =============================================================================
// SYMBOLS CRUD
// =============================================================================

export interface Symbol {
  symbol: string;
  min_dip_pct: number;
  min_days: number;
  name?: string | null;
  fetch_status?: 'pending' | 'fetching' | 'fetched' | 'error' | null;
  fetch_error?: string | null;
}

export async function getSymbols(skipCache = false): Promise<Symbol[]> {
  if (skipCache) {
    return fetchAPI<Symbol[]>('/symbols');
  }
  return apiCache.fetch(
    'symbols-list',
    () => fetchAPI<Symbol[]>('/symbols'),
    { ttl: CACHE_TTL.SYMBOLS }
  );
}

export async function getSymbol(symbol: string): Promise<Symbol> {
  return fetchAPI<Symbol>(`/symbols/${symbol}`);
}

export async function createSymbol(data: { symbol: string; min_dip_pct?: number; min_days?: number }): Promise<Symbol> {
  const result = await fetchAPI<Symbol>('/symbols', {
    method: 'POST',
    body: JSON.stringify(data),
  });
  // Invalidate symbols and ranking cache on add
  apiCache.invalidate(/^symbols/);
  apiCache.invalidate(/^ranking/);
  invalidateEtagCache(/\/symbols|\/dips\/ranking/);
  return result;
}

export async function updateSymbol(symbol: string, data: { min_dip_pct?: number; min_days?: number }): Promise<Symbol> {
  const result = await fetchAPI<Symbol>(`/symbols/${symbol}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
  // Invalidate symbols and ranking cache on update
  apiCache.invalidate(/^symbols/);
  apiCache.invalidate(/^ranking/);
  invalidateEtagCache(/\/symbols|\/dips\/ranking/);
  return result;
}

export async function deleteSymbol(symbol: string): Promise<void> {
  await fetchAPI(`/symbols/${symbol}`, {
    method: 'DELETE',
  });
  // Invalidate symbols and ranking cache on delete
  apiCache.invalidate(/^symbols/);
  apiCache.invalidate(/^ranking/);
  invalidateEtagCache(/\/symbols|\/dips\/ranking/);
}

// =============================================================================
// SYMBOL VALIDATION (Yahoo Finance)
// =============================================================================

export interface SymbolValidationResponse {
  valid: boolean;
  symbol: string;
  name?: string;
  sector?: string;
  summary?: string;
  error?: string;
}

export async function validateSymbol(symbol: string): Promise<SymbolValidationResponse> {
  try {
    const result = await fetchAPI<SymbolValidationResponse>(`/symbols/validate/${symbol.toUpperCase()}`);
    return result;
  } catch {
    return { valid: false, symbol: symbol.toUpperCase(), error: 'Failed to validate symbol' };
  }
}

// =============================================================================
// MFA TYPES & API
// =============================================================================

export interface MFAStatus {
  enabled: boolean;
  has_backup_codes: boolean;
  backup_codes_remaining: number | null;
}

export interface MFASetupResponse {
  secret: string;
  provisioning_uri: string;
  qr_code_base64: string;
}

export interface MFAVerifyResponse {
  enabled: boolean;
  backup_codes: string[];
}

export async function getMFAStatus(): Promise<MFAStatus> {
  return fetchAPI<MFAStatus>('/auth/mfa/status');
}

export async function setupMFA(): Promise<MFASetupResponse> {
  return fetchAPI<MFASetupResponse>('/auth/mfa/setup', { method: 'POST' });
}

export async function verifyMFA(code: string): Promise<MFAVerifyResponse> {
  return fetchAPI<MFAVerifyResponse>('/auth/mfa/verify', {
    method: 'POST',
    body: JSON.stringify({ code }),
  });
}

export async function disableMFA(code: string): Promise<void> {
  await fetchAPI('/auth/mfa/disable', {
    method: 'POST',
    body: JSON.stringify({ code }),
  });
}

export async function regenerateBackupCodes(code: string): Promise<{ backup_codes: string[] }> {
  return fetchAPI<{ backup_codes: string[] }>('/auth/mfa/backup-codes/regenerate', {
    method: 'POST',
    body: JSON.stringify({ code }),
  });
}

// =============================================================================
// API KEYS TYPES & API (OpenAI, etc.)
// =============================================================================

export interface ApiKeyInfo {
  id: number;
  key_name: string;
  key_hint: string;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface ApiKeyList {
  keys: ApiKeyInfo[];
}

export async function listApiKeys(): Promise<ApiKeyList> {
  return fetchAPI<ApiKeyList>('/admin/api-keys');
}

export async function createApiKey(keyName: string, apiKey: string, mfaCode: string): Promise<ApiKeyInfo> {
  return fetchAPI<ApiKeyInfo>('/admin/api-keys', {
    method: 'POST',
    body: JSON.stringify({ key_name: keyName, api_key: apiKey, mfa_code: mfaCode }),
  });
}

export async function revealApiKey(keyName: string, mfaCode: string): Promise<{ key_name: string; api_key: string }> {
  return fetchAPI<{ key_name: string; api_key: string }>(`/admin/api-keys/${keyName}/reveal`, {
    method: 'POST',
    body: JSON.stringify({ mfa_code: mfaCode }),
  });
}

export async function deleteApiKey(keyName: string, mfaCode: string): Promise<void> {
  await fetchAPI(`/admin/api-keys/${keyName}`, {
    method: 'DELETE',
    body: JSON.stringify({ mfa_code: mfaCode }),
  });
}

export async function checkApiKey(keyName: string): Promise<{ key_name: string; exists: boolean; key_hint: string | null }> {
  return fetchAPI<{ key_name: string; exists: boolean; key_hint: string | null }>(`/admin/api-keys/check/${keyName}`);
}

export async function checkMfaSession(): Promise<{ has_session: boolean }> {
  return fetchAPI<{ has_session: boolean }>('/admin/api-keys/mfa-session');
}

// =============================================================================
// USER API KEYS (External access keys for the stonkmarket API)
// =============================================================================

export interface UserApiKey {
  id: number;
  key_prefix: string;
  name: string;
  description: string | null;
  vote_weight: number;
  rate_limit_bypass: boolean;
  is_active: boolean;
  usage_count: number;
  last_used_at: string | null;
  created_at: string;
  expires_at: string | null;
}

export interface UserApiKeyStats {
  total_keys: number;
  active_keys: number;
  inactive_keys: number;
  expired_keys: number;
  total_usage: number;
  last_used: string | null;
}

export interface CreateUserKeyRequest {
  name: string;
  description?: string;
  vote_weight?: number;
  rate_limit_bypass?: boolean;
  expires_days?: number;
}

export interface CreateUserKeyResponse {
  key: string;
  id: number;
  key_prefix: string;
  name: string;
  vote_weight: number;
  rate_limit_bypass: boolean;
  expires_at: string | null;
  warning: string;
}

export async function listUserApiKeys(activeOnly: boolean = true): Promise<UserApiKey[]> {
  return fetchAPI<UserApiKey[]>(`/admin/user-keys?active_only=${activeOnly}`);
}

export async function createUserApiKey(request: CreateUserKeyRequest): Promise<CreateUserKeyResponse> {
  return fetchAPI<CreateUserKeyResponse>('/admin/user-keys', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getUserApiKeyStats(): Promise<UserApiKeyStats> {
  return fetchAPI<UserApiKeyStats>('/admin/user-keys/stats');
}

export async function getUserApiKey(keyId: number): Promise<UserApiKey> {
  return fetchAPI<UserApiKey>(`/admin/user-keys/${keyId}`);
}

export async function updateUserApiKey(
  keyId: number, 
  updates: Partial<{ name: string; description: string; vote_weight: number; rate_limit_bypass: boolean }>
): Promise<UserApiKey> {
  return fetchAPI<UserApiKey>(`/admin/user-keys/${keyId}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
}

export async function deactivateUserApiKey(keyId: number): Promise<void> {
  await fetchAPI(`/admin/user-keys/${keyId}/deactivate`, {
    method: 'POST',
  });
}

export async function reactivateUserApiKey(keyId: number): Promise<void> {
  await fetchAPI(`/admin/user-keys/${keyId}/reactivate`, {
    method: 'POST',
  });
}

export async function deleteUserApiKey(keyId: number): Promise<void> {
  await fetchAPI(`/admin/user-keys/${keyId}`, {
    method: 'DELETE',
  });
}

// =============================================================================
// ADMIN SETTINGS
// =============================================================================

export interface AppSettings {
  app_name: string;
  app_version: string;
  environment: string;
  debug: boolean;
  default_min_dip_pct: number;
  default_min_days: number;
  history_days: number;
  chart_days: number;
  vote_cooldown_days: number;
  auto_approve_enabled: boolean;
  auto_approve_votes: number;
  auto_approve_unique_voters: number;
  auto_approve_min_age_hours: number;
  rate_limit_enabled: boolean;
  rate_limit_auth: string;
  rate_limit_api_anonymous: string;
  rate_limit_api_authenticated: string;
  scheduler_enabled: boolean;
  scheduler_timezone: string;
  external_api_timeout: number;
  external_api_retries: number;
}

export interface BenchmarkConfig {
  id: string;
  symbol: string;
  name: string;
  description?: string | null;
}

export interface RuntimeSettings {
  signal_threshold_strong_buy: number;
  signal_threshold_buy: number;
  signal_threshold_hold: number;
  ai_enrichment_enabled: boolean;
  ai_batch_size: number;
  ai_model: string;
  suggestion_cleanup_days: number;
  auto_approve_votes: number;
  // Cache TTL settings (in seconds)
  cache_ttl_symbols: number;
  cache_ttl_suggestions: number;
  cache_ttl_ai_content: number;
  cache_ttl_ranking: number;
  cache_ttl_charts: number;
  benchmarks: BenchmarkConfig[];
}

export interface CronJobSummary {
  name: string;
  cron: string;
  description: string | null;
  last_run: string | null;
  last_status: string | null;
}

export interface SystemStatus {
  app_settings: AppSettings;
  runtime_settings: RuntimeSettings;
  cronjobs: CronJobSummary[];
  openai_configured: boolean;
  logo_dev_configured: boolean;
  total_symbols: number;
  pending_suggestions: number;
}

export async function getAppSettings(): Promise<AppSettings> {
  return fetchAPI<AppSettings>('/admin/settings/app');
}

export async function getRuntimeSettings(): Promise<RuntimeSettings> {
  return apiCache.fetch(
    'runtime-settings',
    () => fetchAPI<RuntimeSettings>('/admin/settings/runtime'),
    { ttl: CACHE_TTL.SETTINGS }
  );
}

export async function updateRuntimeSettings(updates: Partial<RuntimeSettings>): Promise<RuntimeSettings> {
  const result = await fetchAPI<RuntimeSettings>('/admin/settings/runtime', {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });
  // Invalidate settings caches on update
  apiCache.invalidate(/^(runtime-settings|suggestion-settings|benchmarks)/);
  invalidateEtagCache(/\/admin\/settings|\/dips\/ranking|\/dips\/benchmarks/);
  // Also clear the module-level benchmark config cache
  clearBenchmarkConfigCache();
  return result;
}

export async function getSystemStatus(): Promise<SystemStatus> {
  return fetchAPI<SystemStatus>('/admin/settings/status');
}

export async function checkOpenAIStatus(): Promise<{ configured: boolean }> {
  return fetchAPI<{ configured: boolean }>('/admin/settings/openai-status');
}

// =============================================================================
// USER SETTINGS / CREDENTIALS
// =============================================================================

export interface UserCredentialsUpdate {
  current_password: string;
  new_password: string;
  new_username?: string;
}

export async function updateCredentials(data: UserCredentialsUpdate): Promise<{ username: string; is_admin: boolean }> {
  return fetchAPI<{ username: string; is_admin: boolean }>('/auth/credentials', {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

// =============================================================================
// SUGGESTIONS TYPES & API
// =============================================================================

export type SuggestionStatus = 'pending' | 'approved' | 'rejected';
export type FetchStatus = 'pending' | 'fetching' | 'fetched' | 'rate_limited' | 'error' | 'invalid';

export interface Suggestion {
  id: number;
  symbol: string;
  status: SuggestionStatus;
  vote_count: number;
  name: string | null;
  sector: string | null;
  summary: string | null;
  website: string | null;
  ipo_year: number | null;
  current_price: number | null;
  ath_price: number | null;
  fetch_status: FetchStatus | null;
  fetch_error: string | null;
  fetched_at: string | null;
  created_at: string;
  reviewed_at?: string | null;
  approved_by?: number | null;
}

export interface SuggestionListResponse {
  items: Suggestion[];
  total: number;
  page: number;
  page_size: number;
}

export interface TopSuggestion {
  symbol: string;
  name: string | null;
  vote_count: number;
  sector: string | null;
  summary: string | null;
  ipo_year: number | null;
  website: string | null;
  fetch_status: FetchStatus | null;
}

// Public endpoints (no auth required)
export async function suggestStock(symbol: string): Promise<{ message: string; symbol: string; vote_count: number; status: string }> {
  // Get client fingerprint to send to server
  const clientFingerprint = await getDeviceFingerprint();
  
  const result = await fetchAPI<{ message: string; symbol: string; vote_count: number; status: string }>(
    `/suggestions?symbol=${encodeURIComponent(symbol.toUpperCase())}`,
    { 
      method: 'POST',
      headers: {
        'X-Client-Fingerprint': clientFingerprint,
      },
    }
  );
  
  // Record local vote (suggesting also counts as voting)
  recordLocalVote(symbol);
  
  return result;
}

export async function voteForSuggestion(symbol: string): Promise<{ message: string; symbol: string; auto_approved?: boolean }> {
  // Check local vote record first (client-side protection)
  if (hasVotedLocally(symbol)) {
    throw new Error('You have already voted for this stock');
  }
  
  // Get client fingerprint to send to server
  const clientFingerprint = await getDeviceFingerprint();
  
  const result = await fetchAPI<{ message: string; symbol: string; auto_approved?: boolean }>(
    `/suggestions/${encodeURIComponent(symbol.toUpperCase())}/vote`, 
    {
      method: 'PUT',
      headers: {
        'X-Client-Fingerprint': clientFingerprint,
      },
    }
  );
  
  // Record vote locally for client-side duplicate prevention
  recordLocalVote(symbol);
  
  return result;
}

// Re-export fingerprint utilities for components
export { hasVotedLocally, getLocalVotes };

export async function getTopSuggestions(limit: number = 10, excludeVoted: boolean = false): Promise<TopSuggestion[]> {
  // Include client fingerprint sync ID for exclude_voted to work correctly
  const clientFingerprint = getDeviceFingerprintSync();
  const headers: Record<string, string> = {};
  if (clientFingerprint) {
    headers['X-Client-Fingerprint'] = clientFingerprint;
  }
  
  const response = await fetch(`${API_BASE}/suggestions/top?limit=${limit}&exclude_voted=${excludeVoted}`, {
    headers: {
      ...getAuthHeaders(),
      ...headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return response.json();
}

export async function getSuggestionSettings(): Promise<{ auto_approve_votes: number }> {
  return apiCache.fetch(
    'suggestion-settings',
    () => fetchAPI<{ auto_approve_votes: number }>('/suggestions/settings'),
    { ttl: CACHE_TTL.SUGGESTIONS }
  );
}

export async function getPendingSuggestions(page: number = 1, pageSize: number = 20): Promise<SuggestionListResponse> {
  return fetchAPI<SuggestionListResponse>(`/suggestions/pending?page=${page}&page_size=${pageSize}`);
}

// Admin endpoints
export async function getAllSuggestions(
  status?: SuggestionStatus,
  page: number = 1,
  pageSize: number = 20
): Promise<SuggestionListResponse> {
  const params = new URLSearchParams();
  if (status) params.append('status', status);
  params.append('page', page.toString());
  params.append('page_size', pageSize.toString());
  return fetchAPI<SuggestionListResponse>(`/suggestions?${params.toString()}`);
}

export async function approveSuggestion(suggestionId: number): Promise<{ message: string; symbol: string }> {
  const result = await fetchAPI<{ message: string; symbol: string }>(`/suggestions/${suggestionId}/approve`, {
    method: 'POST',
  });
  // Invalidate symbols and ranking cache so the new stock appears immediately
  apiCache.invalidate(/^symbols/);
  apiCache.invalidate(/^ranking/);
  invalidateEtagCache(/\/symbols|\/dips\/ranking/);
  return result;
}

export async function rejectSuggestion(suggestionId: number, reason?: string): Promise<{ message: string }> {
  const params = reason ? `?reason=${encodeURIComponent(reason)}` : '';
  return fetchAPI<{ message: string }>(`/suggestions/${suggestionId}/reject${params}`, {
    method: 'POST',
  });
}

export async function updateSuggestion(
  suggestionId: number, 
  newSymbol: string
): Promise<{ message: string; old_symbol: string; new_symbol: string }> {
  return fetchAPI<{ message: string; old_symbol: string; new_symbol: string }>(
    `/suggestions/${suggestionId}?new_symbol=${encodeURIComponent(newSymbol.toUpperCase())}`,
    { method: 'PATCH' }
  );
}

export async function refreshSuggestionData(
  suggestionId: number
): Promise<{ message: string; symbol: string }> {
  return fetchAPI<{ message: string; symbol: string }>(
    `/suggestions/${suggestionId}/refresh`,
    { method: 'POST' }
  );
}

export async function retrySuggestionFetch(
  suggestionId: number
): Promise<{ message: string; symbol: string; fetch_status: FetchStatus; fetch_error: string | null }> {
  return fetchAPI<{ message: string; symbol: string; fetch_status: FetchStatus; fetch_error: string | null }>(
    `/suggestions/${suggestionId}/retry`,
    { method: 'POST' }
  );
}

export async function backfillSuggestions(
  limit: number = 10
): Promise<{ message: string; processed: number; total: number; errors: Array<{ symbol: string; error: string }> | null }> {
  return fetchAPI<{ message: string; processed: number; total: number; errors: Array<{ symbol: string; error: string }> | null }>(
    `/suggestions/backfill?limit=${limit}`,
    { method: 'POST' }
  );
}
