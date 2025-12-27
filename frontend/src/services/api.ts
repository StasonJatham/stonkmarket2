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
  peg_ratio: number | null;
  dividend_yield: number | null;
  beta: number | null;
  avg_volume: number | null;
  summary: string | null;
  summary_ai: string | null;  // AI-generated short summary (~300 chars)
  website: string | null;
  recommendation: string | null;
  // Extended fundamentals
  profit_margin: number | null;
  gross_margin: number | null;
  return_on_equity: number | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  revenue_growth: number | null;
  free_cash_flow: number | null;
  target_mean_price: number | null;
  num_analyst_opinions: number | null;
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
  next_run?: string | null;
  next_runs?: string[] | null;
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

export interface TaskStatus {
  task_id: string;
  status: string;
  result?: string | null;
  error?: string | null;
  traceback?: string | null;
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

export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  return fetchAPI<TaskStatus>(`/cronjobs/tasks/${taskId}`);
}

export interface CeleryWorkerInfo {
  status?: string;
  active?: unknown[];
  processed?: number;
  total?: Record<string, number>;
  loadavg?: string | number[];
  pool?: Record<string, unknown>;
  uptime?: number;
}

export type CeleryWorkersResponse = Record<string, CeleryWorkerInfo>;

export interface CeleryQueueInfo {
  name?: string;
  messages?: number;
  consumers?: number;
  state?: string;
  [key: string]: unknown;
}

export type CeleryBrokerInfo = Record<string, unknown>;

export async function getCeleryWorkers(): Promise<CeleryWorkersResponse> {
  return fetchAPI<CeleryWorkersResponse>('/celery/workers');
}

function normalizeCeleryQueues(payload: unknown): CeleryQueueInfo[] {
  if (Array.isArray(payload)) {
    return payload as CeleryQueueInfo[];
  }

  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>;
    const fromKey = record.queues ?? record.data ?? record.items;
    if (Array.isArray(fromKey)) {
      return fromKey as CeleryQueueInfo[];
    }

    return Object.entries(record).map(([name, info]) => ({
      name,
      ...(typeof info === 'object' && info !== null ? (info as CeleryQueueInfo) : {}),
    }));
  }

  return [];
}

export async function getCeleryQueues(): Promise<CeleryQueueInfo[]> {
  const payload = await fetchAPI<unknown>('/celery/queues');
  return normalizeCeleryQueues(payload);
}

export async function getCeleryBroker(): Promise<CeleryBrokerInfo> {
  return fetchAPI<CeleryBrokerInfo>('/celery/broker');
}

// Celery task info from Flower
export interface CeleryTaskInfo {
  uuid: string;
  name: string;
  state: string;
  received?: number;
  started?: number;
  succeeded?: number;
  failed?: number;
  runtime?: number;
  args?: string;
  kwargs?: string;
  worker?: string;
}

export type CeleryTasksResponse = Record<string, CeleryTaskInfo>;

export async function getCeleryTasks(limit = 50): Promise<CeleryTaskInfo[]> {
  const payload = await fetchAPI<CeleryTasksResponse>(`/celery/tasks?limit=${limit}`);
  // Convert the dict response to an array sorted by received time
  return Object.entries(payload)
    .map(([uuid, info]) => ({ ...info, uuid }))
    .sort((a, b) => (b.received || 0) - (a.received || 0));
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
  ai_pending?: boolean | null;
  ai_task_id?: string | null;
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
  const result = await fetchAPI<{ symbol: string; vote_type: string; message: string }>(`/swipe/cards/${symbol}/vote`, {
    method: 'PUT',
    body: JSON.stringify({ vote_type: voteType }),
  });

  // Ensure swipe lists reflect vote exclusions immediately
  apiCache.invalidate(/^swipe:cards:/);
  
  return result;
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

export interface AISummaryResponse {
  symbol: string;
  summary_ai: string | null;
  task_id?: string | null;
  status?: string | null;
}

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
export async function regenerateSymbolAiSummary(symbol: string): Promise<AISummaryResponse> {
  const result = await fetchAPI<AISummaryResponse>(`/symbols/${symbol}/ai/summary`, {
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
// QUANT ENGINE RECOMMENDATIONS
// =============================================================================

export interface QuantRecommendation {
  ticker: string;
  name: string | null;
  action: 'BUY' | 'SELL' | 'HOLD';
  notional_eur: number;
  delta_weight: number;
  target_weight: number;
  mu_hat: number;
  uncertainty: number;
  risk_contribution: number;
  dip_score: number | null;
  dip_bucket: string | null;
  marginal_utility: number;
  // Legacy compatibility
  legacy_dip_pct: number | null;
  legacy_days_in_dip: number | null;
  legacy_domain_score: number | null;
}

export interface QuantAuditBlock {
  timestamp: string;
  config_hash: number;
  mu_hat_summary: { mean: number; std: number; min: number; max: number };
  risk_model_summary: Record<string, unknown>;
  optimizer_status: string;
  constraint_binding: string[];
  turnover_realized: number;
  regime_state: string;
  dip_stats: Record<string, unknown> | null;
  error_message: string | null;
}

export interface QuantEngineResponse {
  recommendations: QuantRecommendation[];
  as_of_date: string;
  portfolio_value_eur: number;
  inflow_eur: number;
  total_trades: number;
  total_transaction_cost_eur: number;
  expected_portfolio_return: number;
  expected_portfolio_risk: number;
  audit: QuantAuditBlock;
}

export async function getQuantRecommendations(
  inflow_eur: number = 1000,
  limit: number = 40
): Promise<QuantEngineResponse> {
  const params = new URLSearchParams();
  params.append('inflow_eur', inflow_eur.toString());
  params.append('limit', limit.toString());
  
  return fetchAPI<QuantEngineResponse>(`/recommendations?${params.toString()}`);
}

/**
 * Convert QuantRecommendation to DipStock for backward compatibility.
 * This adapter allows quant-ranked stocks to be displayed in components
 * that still expect the legacy DipStock interface.
 */
export function quantToDipStock(rec: QuantRecommendation): DipStock {
  return {
    symbol: rec.ticker,
    name: rec.name,
    // Use legacy dip percentage as depth, or fallback to 0
    depth: rec.legacy_dip_pct !== null ? rec.legacy_dip_pct : 0,
    // We don't have price in quant data, use 0 as placeholder
    // The StockDetailsPanel will load full info separately
    last_price: 0,
    previous_close: null,
    change_percent: null,
    days_since_dip: rec.legacy_days_in_dip,
    high_52w: null,
    low_52w: null,
    market_cap: null,
    sector: null,
    pe_ratio: null,
    volume: null,
    // Map quant metrics to dip metrics for sorting compatibility
    // marginal_utility is the optimizer's ranking signal
    dip_score: rec.marginal_utility,
    recovery_potential: rec.mu_hat,
  };
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
  task_id?: string | null;
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
// SYMBOL SEARCH & AUTOCOMPLETE (yfinance)
// =============================================================================

export interface SymbolSearchResult {
  symbol: string;
  name: string | null;
  sector?: string | null;
  quote_type?: string | null;
  market_cap?: number | null;
  source?: string;
}

export interface SymbolSearchResponse {
  query: string;
  results: SymbolSearchResult[];
  count: number;
}

export async function searchSymbols(query: string, limit = 10, fresh = true): Promise<SymbolSearchResult[]> {
  if (query.length < 2) return [];
  try {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    if (fresh) params.append('fresh', 'true');
    const response = await fetchAPI<SymbolSearchResponse>(`/symbols/search/${encodeURIComponent(query)}?${params.toString()}`);
    return response.results;
  } catch {
    return [];
  }
}

export interface AutocompleteResult {
  symbol: string;
  name: string | null;
}

export async function autocompleteSymbols(partial: string, limit = 5): Promise<AutocompleteResult[]> {
  if (partial.length < 1) return [];
  try {
    const results = await fetchAPI<AutocompleteResult[]>(`/symbols/autocomplete/${encodeURIComponent(partial)}?limit=${limit}`);
    return results;
  } catch {
    return [];
  }
}

// =============================================================================
// AI AGENT ANALYSIS
// =============================================================================

export interface AgentVerdict {
  agent_id: string;
  agent_name: string;
  signal: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  reasoning: string;
  key_factors: string[];
}

export interface AgentAnalysis {
  symbol: string;
  analyzed_at: string;
  verdicts: AgentVerdict[];
  overall_signal: 'bullish' | 'bearish' | 'neutral';
  overall_confidence: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
}

export interface AgentInfo {
  id: string;
  name: string;
  description: string;
  style: string;
}

export interface AgentInfoResponse {
  agents: AgentInfo[];
  count: number;
}

export async function getAgentAnalysis(
  symbol: string,
  options: { forceRefresh?: boolean } = {}
): Promise<AgentAnalysis | null> {
  const params = new URLSearchParams();
  if (options.forceRefresh) params.append('force_refresh', 'true');
  const queryString = params.toString();
  
  return fetchAPI<AgentAnalysis | null>(`/symbols/${symbol}/agents${queryString ? `?${queryString}` : ''}`);
}

export async function refreshAgentAnalysis(symbol: string): Promise<AgentAnalysis> {
  return fetchAPI<AgentAnalysis>(`/symbols/${symbol}/agents/refresh`, {
    method: 'POST',
  });
}

export async function getAgentsInfo(): Promise<AgentInfoResponse> {
  return apiCache.fetch(
    'agents-info',
    () => fetchAPI<AgentInfoResponse>('/symbols/agents/info'),
    { ttl: CACHE_TTL.STOCK_INFO }  // 5 minutes - agent list rarely changes
  );
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
  task_id?: string | null;
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

// Stored search result (from DB, no yfinance call)
export interface StoredSearchResult {
  symbol: string;
  name: string | null;
  sector: string | null;
  source: 'tracked' | 'suggestion';
  vote_count: number | null;
}

// Fast cached search - searches stored suggestions and tracked symbols
export async function searchStoredSuggestions(query: string, limit: number = 10): Promise<StoredSearchResult[]> {
  if (query.length < 1) return [];
  try {
    return await fetchAPI<StoredSearchResult[]>(
      `/suggestions/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
  } catch {
    return [];
  }
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

export async function approveSuggestion(suggestionId: number): Promise<{ message: string; symbol: string; task_id?: string | null }> {
  const result = await fetchAPI<{ message: string; symbol: string; task_id?: string | null }>(`/suggestions/${suggestionId}/approve`, {
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
): Promise<{ message: string; symbol: string; task_id?: string | null }> {
  return fetchAPI<{ message: string; symbol: string; task_id?: string | null }>(
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

// =============================================================================
// BATCH JOBS API
// =============================================================================

export type BatchJobStatus = 'pending' | 'validating' | 'in_progress' | 'finalizing' | 'completed' | 'failed' | 'expired' | 'cancelled';

export interface BatchJob {
  id: number;
  batch_id: string;
  job_type: string;
  status: BatchJobStatus;
  total_requests: number;
  completed_requests: number;
  failed_requests: number;
  estimated_cost_usd: number | null;
  actual_cost_usd: number | null;
  created_at: string | null;
  completed_at: string | null;
}

export interface BatchJobListResponse {
  jobs: BatchJob[];
  total: number;
  active_count: number;
}

export async function getBatchJobs(limit: number = 20, includeCompleted: boolean = true): Promise<BatchJobListResponse> {
  return fetchAPI<BatchJobListResponse>(`/admin/settings/batch-jobs?limit=${limit}&include_completed=${includeCompleted}`);
}

export async function cancelBatchJob(batchId: string): Promise<{ success: boolean; batch_id: string; status: string }> {
  return fetchAPI<{ success: boolean; batch_id: string; status: string }>(
    `/admin/settings/batch-jobs/${batchId}/cancel`,
    { method: 'POST' }
  );
}

export async function deleteBatchJob(jobId: number): Promise<{ success: boolean; job_id: number; batch_id: string }> {
  return fetchAPI<{ success: boolean; job_id: number; batch_id: string }>(
    `/admin/settings/batch-jobs/${jobId}`,
    { method: 'DELETE' }
  );
}

// =============================================================================
// PORTFOLIO API
// =============================================================================

export interface Portfolio {
  id: number;
  user_id: number;
  name: string;
  description?: string | null;
  base_currency: string;
  cash_balance?: number | null;
  is_active: boolean;
  created_at: string;
  updated_at?: string | null;
}

export interface Holding {
  id: number;
  portfolio_id: number;
  symbol: string;
  quantity: number;
  avg_cost?: number | null;
  target_weight?: number | null;
  created_at: string;
  updated_at?: string | null;
}

export interface Transaction {
  id: number;
  portfolio_id: number;
  symbol: string;
  side: string;
  quantity?: number | null;
  price?: number | null;
  fees?: number | null;
  trade_date: string;
  notes?: string | null;
  created_at: string;
}

export interface PortfolioDetail extends Portfolio {
  holdings: Holding[];
  transactions: Transaction[];
}

export interface PortfolioAnalyticsResult {
  tool: string;
  status: string;
  data: Record<string, unknown>;
  warnings: string[];
  source?: string | null;
  generated_at?: string | null;
}

export interface PortfolioAnalyticsResponse {
  portfolio_id: number;
  as_of_date: string;
  results: PortfolioAnalyticsResult[];
  job_id?: string | null;
  job_status?: string | null;
  scheduled_tools: string[];
}

export interface PortfolioAnalyticsJob {
  job_id: string;
  portfolio_id: number;
  status: string;
  tools: string[];
  results_count: number;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export async function getPortfolios(): Promise<Portfolio[]> {
  return fetchAPI<Portfolio[]>('/portfolios');
}

export async function createPortfolio(payload: {
  name: string;
  description?: string;
  base_currency?: string;
  cash_balance?: number;
}): Promise<Portfolio> {
  const result = await fetchAPI<Portfolio>('/portfolios', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  invalidateEtagCache(/\/portfolios/);
  return result;
}

export async function updatePortfolio(
  portfolioId: number,
  payload: Partial<{
    name: string;
    description: string;
    base_currency: string;
    cash_balance: number;
    is_active: boolean;
  }>
): Promise<Portfolio> {
  const result = await fetchAPI<Portfolio>(`/portfolios/${portfolioId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  });
  invalidateEtagCache(/\/portfolios/);
  return result;
}

export async function deletePortfolio(portfolioId: number): Promise<void> {
  await fetchAPI<void>(`/portfolios/${portfolioId}`, { method: 'DELETE' });
  invalidateEtagCache(/\/portfolios/);
}

export async function getPortfolioDetail(portfolioId: number): Promise<PortfolioDetail> {
  return fetchAPI<PortfolioDetail>(`/portfolios/${portfolioId}`);
}

export async function upsertHolding(
  portfolioId: number,
  payload: { symbol: string; quantity: number; avg_cost?: number; target_weight?: number }
): Promise<Holding> {
  const result = await fetchAPI<Holding>(`/portfolios/${portfolioId}/holdings`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  invalidateEtagCache(new RegExp(`/portfolios/${portfolioId}`));
  return result;
}

export async function deleteHolding(portfolioId: number, symbol: string): Promise<void> {
  await fetchAPI<void>(`/portfolios/${portfolioId}/holdings/${symbol}`, { method: 'DELETE' });
  invalidateEtagCache(new RegExp(`/portfolios/${portfolioId}`));
}

export async function getTransactions(portfolioId: number, limit: number = 200): Promise<Transaction[]> {
  return fetchAPI<Transaction[]>(`/portfolios/${portfolioId}/transactions?limit=${limit}`);
}

export async function addTransaction(
  portfolioId: number,
  payload: {
    symbol: string;
    side: string;
    quantity?: number;
    price?: number;
    fees?: number;
    trade_date: string;
    notes?: string;
  }
): Promise<Transaction> {
  const result = await fetchAPI<Transaction>(`/portfolios/${portfolioId}/transactions`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  invalidateEtagCache(new RegExp(`/portfolios/${portfolioId}`));
  return result;
}

export async function deleteTransaction(portfolioId: number, transactionId: number): Promise<void> {
  await fetchAPI<void>(`/portfolios/${portfolioId}/transactions/${transactionId}`, { method: 'DELETE' });
  invalidateEtagCache(new RegExp(`/portfolios/${portfolioId}`));
}

export async function runPortfolioAnalytics(
  portfolioId: number,
  payload: {
    tools?: string[];
    window?: string;
    start_date?: string;
    end_date?: string;
    benchmark?: string;
    params?: Record<string, unknown>;
    force_refresh?: boolean;
  }
): Promise<PortfolioAnalyticsResponse> {
  return fetchAPI<PortfolioAnalyticsResponse>(`/portfolios/${portfolioId}/analytics`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getPortfolioAnalyticsJob(
  portfolioId: number,
  jobId: string
): Promise<PortfolioAnalyticsJob> {
  return fetchAPI<PortfolioAnalyticsJob>(
    `/portfolios/${portfolioId}/analytics/jobs/${jobId}`
  );
}
