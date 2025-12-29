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

/** Mini chart data for card backgrounds */
export interface MiniChartData {
  symbol: string;
  points: { date: string; close: number }[];
}

/** Get batch mini chart data for multiple symbols (for card backgrounds) */
export async function getBatchCharts(symbols: string[], days: number = 60): Promise<Record<string, ChartDataPoint[]>> {
  const cacheKey = `batch-charts:${symbols.sort().join(',')}:${days}`;
  
  const fetcher = async (): Promise<Record<string, ChartDataPoint[]>> => {
    const response = await fetchAPI<{ charts: MiniChartData[] }>('/dips/batch/charts', {
      method: 'POST',
      body: JSON.stringify({ symbols, days }),
    });
    
    // Use Record instead of Map for proper JSON serialization in cache
    const result: Record<string, ChartDataPoint[]> = {};
    for (const chart of response.charts) {
      // Convert mini chart to ChartDataPoint format
      result[chart.symbol] = chart.points.map(p => ({
        date: p.date,
        close: p.close,
        ref_high: 0,
        threshold: 0,
        drawdown: 0,
        since_dip: null,
        dip_start_date: null,
        ref_high_date: null,
      }));
    }
    return result;
  };
  
  return apiCache.fetch(cacheKey, fetcher, { ttl: CACHE_TTL.CHART, staleWhileRevalidate: true });
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
// FUNDAMENTALS (DOMAIN-SPECIFIC METRICS)
// =============================================================================

export type DomainType = 'bank' | 'reit' | 'insurer' | 'utility' | 'biotech' | 'stock';

export interface SymbolFundamentals {
  symbol: string;
  // Standard metrics
  pe_ratio: number | null;
  forward_pe: number | null;
  peg_ratio: number | null;
  price_to_book: number | null;
  profit_margin: string | null;
  gross_margin: string | null;
  return_on_equity: string | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  revenue_growth: string | null;
  earnings_growth: string | null;
  free_cash_flow: number | null;
  recommendation: string | null;
  target_mean_price: number | null;
  num_analyst_opinions: number | null;
  beta: string | null;
  next_earnings_date: string | null;
  
  // Intrinsic value estimates
  intrinsic_value: number | null;
  intrinsic_value_method: 'analyst' | 'peg' | 'graham' | 'dcf' | null;
  upside_pct: number | null;
  valuation_status: 'undervalued' | 'fair' | 'overvalued' | null;
  
  // Domain classification
  domain: DomainType;
  
  // Bank-specific metrics
  net_interest_income: number | null;  // NII in dollars
  net_interest_margin: number | null;  // NIM as decimal (0.0179 = 1.79%)
  interest_income: number | null;
  interest_expense: number | null;
  
  // REIT-specific metrics
  ffo: number | null;                  // Funds From Operations
  ffo_per_share: number | null;
  p_ffo: number | null;                // Price/FFO ratio
  
  // Insurance-specific metrics
  loss_ratio: number | null;
  expense_ratio: number | null;
  combined_ratio: number | null;       // Loss ratio + expense ratio (< 100% = profitable)
  
  // Data freshness
  source: 'live' | 'database' | 'cache';
  fetched_at: string | null;
  expires_at: string | null;
  is_stale: boolean;
}

export async function getSymbolFundamentals(symbol: string): Promise<SymbolFundamentals | null> {
  const cacheKey = `fundamentals:${symbol}`;
  
  try {
    return await apiCache.fetch(
      cacheKey,
      () => fetchAPI<SymbolFundamentals>(`/symbols/fundamentals/${symbol}`),
      { ttl: CACHE_TTL.STOCK_INFO, staleWhileRevalidate: true }
    );
  } catch {
    return null;
  }
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

// Signal trigger for chart markers
export interface SignalTrigger {
  date: string;
  signal_name: string;
  price: number;
  win_rate: number;
  avg_return_pct: number;
  holding_days: number;
  drawdown_pct: number;  // The threshold that triggered (for drawdown signals)
  signal_type: 'entry' | 'exit';  // "entry" for buy signals, "exit" for sell signals
}

export interface SignalTriggersResponse {
  symbol: string;
  signal_name: string | null;
  triggers: SignalTrigger[];
  // Benchmark comparison
  buy_hold_return_pct: number;  // Return from buying and holding
  signal_return_pct: number;  // Aggregate return from signal trading
  edge_vs_buy_hold_pct: number;  // Signal return - buy hold (the "alpha")
  n_trades: number;  // Number of signals in the period
  beats_buy_hold: boolean;  // Whether signal strategy outperformed buy-and-hold
  actual_win_rate: number;  // True win rate from visible trades (not backtest estimate)
}

/** Get historical signal triggers for chart markers with benchmark comparison */
export async function getSignalTriggers(symbol: string, lookbackDays: number = 365): Promise<SignalTriggersResponse> {
  const cacheKey = `signal-triggers:${symbol}:${lookbackDays}`;
  
  return apiCache.fetch(
    cacheKey,
    async () => {
      const response = await fetchAPI<SignalTriggersResponse>(
        `/recommendations/${symbol}/signal-triggers?lookback_days=${lookbackDays}`
      );
      return response;  // Return full response with benchmark data
    },
    { ttl: CACHE_TTL.RECOMMENDATIONS, staleWhileRevalidate: true }
  );
}

// =============================================================================
// TRADE ENGINE API (Buy + Sell Signals, Full Trade Optimization)
// =============================================================================

export interface TradeCycle {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  return_pct: number;
  holding_days: number;
  entry_signal: string;
  exit_signal: string;
}

export interface FullTradeResponse {
  symbol: string;
  entry_signal_name: string;
  entry_threshold: number;
  entry_description: string;
  exit_strategy_name: string;
  exit_threshold: number;
  exit_description: string;
  n_complete_trades: number;
  win_rate: number;
  avg_return_pct: number;
  total_return_pct: number;
  max_return_pct: number;
  max_drawdown_pct: number;
  avg_holding_days: number;
  sharpe_ratio: number;
  // Benchmark comparison
  buy_hold_return_pct: number;
  spy_return_pct: number;
  edge_vs_buy_hold_pct: number;
  edge_vs_spy_pct: number;
  beats_both_benchmarks: boolean;
  // Exit timing analysis
  exit_predictability: number;  // How consistent is exit timing?
  upside_captured_pct: number;  // What % of potential gain was captured?
  trades: TradeCycle[];
  current_buy_signal: boolean;
  current_sell_signal: boolean;
  days_since_last_signal: number;
}

export interface CombinedSignalResult {
  name: string;
  component_signals: string[];
  logic: string;
  win_rate: number;
  avg_return_pct: number;
  n_signals: number;
  improvement_vs_best_single: number;
}

export interface DipAnalysis {
  symbol: string;
  current_drawdown_pct: number;
  typical_dip_pct: number;
  max_historical_dip_pct: number;
  dip_zscore: number;  // How many std devs from typical
  is_unusually_deep: boolean;
  deviation_from_typical: number;
  // Technical analysis
  technical_score: number;  // -1 (bearish) to +1 (bullish)
  trend_broken: boolean;  // Below SMA 200?
  volume_confirmation: boolean;  // High volume = capitulation
  momentum_divergence: boolean;  // Price down but RSI up
  // Classification
  dip_type: 'overreaction' | 'normal_volatility' | 'fundamental_decline' | 'insufficient_data';
  confidence: number;
  action: 'STRONG_BUY' | 'BUY' | 'WAIT' | 'AVOID';
  reasoning: string;
  // Probability estimates
  recovery_probability: number;
  expected_return_if_buy: number;
  expected_loss_if_knife: number;
}

export interface CurrentSignals {
  symbol: string;
  buy_signals: Array<{ name: string; value: number; threshold: number; description?: string }>;
  sell_signals: Array<{ name: string; value: number; threshold: number; description?: string }>;
  overall_action: 'STRONG_BUY' | 'BUY' | 'WEAK_BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  reasoning: string;
}

/** Get optimized full trade strategy (entry + exit) for a stock */
export async function getFullTradeStrategy(symbol: string): Promise<FullTradeResponse> {
  const cacheKey = `full-trade:${symbol}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<FullTradeResponse>(`/recommendations/${symbol}/full-trade`),
    { ttl: CACHE_TTL.RECOMMENDATIONS, staleWhileRevalidate: true }
  );
}

/** Get signal combinations (multiple indicators together) */
export async function getSignalCombinations(symbol: string): Promise<CombinedSignalResult[]> {
  const cacheKey = `signal-combos:${symbol}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<CombinedSignalResult[]>(`/recommendations/${symbol}/signal-combinations`),
    { ttl: CACHE_TTL.RECOMMENDATIONS, staleWhileRevalidate: true }
  );
}

/** Analyze if dip is overreaction or real decline */
export async function getDipAnalysis(symbol: string): Promise<DipAnalysis> {
  const cacheKey = `dip-analysis:${symbol}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<DipAnalysis>(`/recommendations/${symbol}/dip-analysis`),
    { ttl: CACHE_TTL.RECOMMENDATIONS, staleWhileRevalidate: true }
  );
}

/** Get current real-time buy and sell signals */
export async function getCurrentSignals(symbol: string): Promise<CurrentSignals> {
  const cacheKey = `current-signals:${symbol}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<CurrentSignals>(`/recommendations/${symbol}/current-signals`),
    { ttl: 60_000, staleWhileRevalidate: true }  // Short TTL for real-time signals
  );
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

export async function getTaskStatuses(taskIds: string[]): Promise<TaskStatus[]> {
  if (taskIds.length === 0) return [];
  const response = await fetchAPI<{ tasks: TaskStatus[] }>(`/cronjobs/tasks/batch`, {
    method: 'POST',
    body: JSON.stringify({ task_ids: taskIds }),
  });
  return response.tasks;
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

export interface CelerySnapshotResponse {
  workers: CeleryWorkersResponse;
  queues: CeleryQueueInfo[];
  broker: CeleryBrokerInfo | null;
  tasks: CeleryTaskInfo[];
}

export async function getCelerySnapshot(limit = 100): Promise<CelerySnapshotResponse> {
  const payload = await fetchAPI<{
    workers: CeleryWorkersResponse;
    queues: unknown;
    broker: CeleryBrokerInfo | null;
    tasks: CeleryTaskInfo[];
  }>(`/celery/snapshot?limit=${limit}`);

  return {
    workers: payload.workers || {},
    queues: normalizeCeleryQueues(payload.queues),
    broker: payload.broker ?? null,
    tasks: Array.isArray(payload.tasks) ? payload.tasks : [],
  };
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

export interface DipCardsQuery {
  includeAi?: boolean;
  excludeVoted?: boolean;
  limit?: number;
  offset?: number;
  search?: string;
  skipCache?: boolean;
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

export async function getDipCardsPaged(query: DipCardsQuery = {}): Promise<DipCardList> {
  const params = new URLSearchParams();
  if (query.includeAi) params.append('include_ai', 'true');
  if (query.excludeVoted) params.append('exclude_voted', 'true');
  if (query.limit) params.append('limit', String(query.limit));
  if (query.offset) params.append('offset', String(query.offset));
  if (query.search) params.append('search', query.search);

  const endpoint = `/swipe/cards?${params.toString()}`;
  const cacheKey = `swipe:cards:paged:${params.toString()}`;

  if (query.skipCache) {
    return fetchAPI<DipCardList>(endpoint);
  }

  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<DipCardList>(endpoint),
    { ttl: CACHE_TTL.RANKING, staleWhileRevalidate: true }
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

// Evidence block for APUS + DOUS scoring transparency
export interface EvidenceBlock {
  // Statistical validation
  p_outperf: number;      // P(edge > 0) from stationary bootstrap
  ci_low: number;         // 95% CI lower bound for edge
  ci_high: number;        // 95% CI upper bound for edge
  dsr: number;            // Deflated Sharpe Ratio
  psr: number;            // Probabilistic Sharpe Ratio
  
  // Edge metrics
  median_edge: number;    // Median edge over benchmarks
  mean_edge: number;      // Mean edge over benchmarks
  edge_vs_stock: number;  // Edge vs buy-and-hold stock
  edge_vs_spy: number;    // Edge vs SPY benchmark
  worst_regime_edge: number;  // Worst edge across regimes
  cvar_5: number;         // Conditional VaR at 5%
  
  // Sharpe metrics
  observed_sharpe: number;
  sr_max: number;         // Expected max Sharpe under null
  n_effective: number;    // Effective number of strategies
  
  // Regime edges
  edge_bull: number;
  edge_bear: number;
  edge_high_vol: number;
  
  // Stability
  sharpe_degradation: number;
  n_trades: number;
  
  // Fundamental metrics
  fund_mom: number;       // Fundamental momentum score
  val_z: number;          // Valuation z-score
  event_risk: boolean;    // Event risk flag (earnings/dividend)
  
  // Dip metrics
  p_recovery: number;     // P(recovery within H days)
  expected_value: number; // Expected value of dip entry
  sector_relative: number; // Sector relative drawdown
}

export interface QuantRecommendation {
  ticker: string;
  name: string | null;
  action: 'BUY' | 'SELL' | 'HOLD';
  notional_eur: number;
  delta_weight: number;
  target_weight: number;
  // Price data
  last_price: number | null;
  change_percent: number | null;
  market_cap: number | null;
  // Sector info
  sector: string | null;
  sector_etf: string | null;
  // Signal metrics
  mu_hat: number;
  uncertainty: number;
  risk_contribution: number;
  // Top technical signal
  top_signal_name: string | null;
  top_signal_is_buy: boolean;
  top_signal_strength: number;
  top_signal_description: string | null;
  // Opportunity score
  opportunity_score: number | null;
  opportunity_rating: 'strong_buy' | 'buy' | 'hold' | 'avoid' | null;
  // Recovery and unusual dip metrics (quant engine backtest data)
  expected_recovery_days: number | null;  // Expected time to recover based on optimal holding period
  typical_dip_pct: number | null;  // Stock's typical dip size as percentage
  dip_vs_typical: number | null;  // Current dip / typical dip ratio (>1.5 = unusual)
  is_unusual_dip: boolean;  // True if current dip is significantly larger than typical
  win_rate: number | null;  // Historical win rate for similar signals
  // Dip metrics
  dip_score: number | null;
  dip_bucket: string | null;
  marginal_utility: number;
  // AI analysis
  ai_summary: string | null;
  ai_rating: string | null;
  // AI Persona Analysis
  ai_persona_signal: string | null;
  ai_persona_confidence: number | null;
  ai_persona_buy_count: number;
  ai_persona_summary: string | null;
  // Strategy Signal (optimized timing strategy)
  strategy_beats_bh: boolean;
  strategy_signal: string | null;
  strategy_win_rate: number | null;
  strategy_vs_bh_pct: number | null;
  // Dip Entry Optimizer
  dip_entry_optimal_pct: number | null;
  dip_entry_price: number | null;
  dip_entry_is_buy_now: boolean;
  dip_entry_strength: number | null;
  // Best Chance Score
  best_chance_score: number;
  best_chance_reason: string | null;
  // APUS + DOUS Dual-Mode Scoring
  quant_mode: 'CERTIFIED_BUY' | 'DIP_ENTRY' | 'HOLD' | 'DOWNTREND' | null;  // Scoring mode
  quant_score_a: number | null;   // Mode A (APUS) score 0-100
  quant_score_b: number | null;   // Mode B (DOUS) score 0-100
  quant_gate_pass: boolean;       // Whether Mode A gate criteria passed
  quant_evidence: EvidenceBlock | null;  // Full evidence block for transparency
  // Domain-specific analysis (sector-aware analysis)
  domain_context: string | null;
  domain_adjustment: number | null;
  domain_adjustment_reason: string | null;
  domain_risk_level: string | null;
  domain_risk_factors: string[] | null;
  domain_recovery_days: number | null;
  domain_warnings: string[] | null;
  volatility_regime: string | null;
  volatility_percentile: number | null;
  vs_sector_performance: number | null;
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
  market_message: string | null;
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
    // legacy_dip_pct is a percentage (16.7), convert to fraction (0.167) for DipStock.depth
    depth: rec.legacy_dip_pct !== null ? rec.legacy_dip_pct / 100 : 0,
    // Use price from backend, fallback to 0
    last_price: rec.last_price ?? 0,
    previous_close: null,
    change_percent: rec.change_percent,
    days_since_dip: rec.legacy_days_in_dip,
    high_52w: null,
    low_52w: null,
    market_cap: rec.market_cap,
    sector: null,
    pe_ratio: null,
    volume: null,
    // Use best_chance_score for sorting (higher = better opportunity)
    dip_score: rec.best_chance_score ?? rec.marginal_utility ?? rec.dip_score ?? 0,
    recovery_potential: rec.mu_hat,
  };
}

// =============================================================================
// STOCK CARD V2 DATA ADAPTER
// =============================================================================

/**
 * StockCardData - Enhanced card data for StockCardV2 component.
 * Extends DipStock with additional metrics for information-dense display.
 */
export interface StockCardData {
  // Core identity
  symbol: string;
  name: string | null;
  sector: string | null;
  
  // Price & performance
  last_price: number;
  change_percent: number | null;
  high_52w: number | null;
  low_52w: number | null;
  market_cap: number | null;
  
  // Dip metrics
  depth: number;
  days_since_dip: number | null;
  dip_bucket: string | null;
  
  // Recovery and unusual dip metrics (quant engine backtest data)
  expected_recovery_days?: number | null;  // Expected time to recover based on optimal holding period
  typical_dip_pct?: number | null;  // Stock's typical dip size as percentage
  dip_vs_typical?: number | null;  // Current dip / typical dip ratio (>1.5 = unusual)
  is_unusual_dip?: boolean;  // True if current dip is significantly larger than typical
  win_rate?: number | null;  // Historical win rate for similar signals
  
  // Technical signals
  top_signal?: {
    name: string;
    is_buy: boolean;
    strength: number;
    description?: string;
  } | null;
  
  // Sector comparison
  sector_delta?: number | null;
  sector_etf?: string | null;
  
  // Opportunity rating
  opportunity_score?: number | null;
  opportunity_rating?: 'strong_buy' | 'buy' | 'hold' | 'avoid' | null;
  
  // AI analysis
  ai_rating?: string | null;
  ai_summary?: string | null;
  domain_analysis?: string | null;
  
  // Domain-specific analysis (sector-aware)
  domain_context?: string | null;
  domain_adjustment?: number | null;
  domain_adjustment_reason?: string | null;
  domain_risk_level?: string | null;
  domain_risk_factors?: string[] | null;
  domain_recovery_days?: number | null;
  domain_warnings?: string[] | null;
  volatility_regime?: string | null;
  volatility_percentile?: number | null;
  vs_sector_performance?: number | null;
  
  // Signal metrics
  mu_hat?: number;
  uncertainty?: number;
  marginal_utility?: number;
  
  // APUS + DOUS Dual-Mode Scoring
  quant_mode?: 'CERTIFIED_BUY' | 'DIP_ENTRY' | 'HOLD' | 'DOWNTREND' | null;
  quant_score_a?: number | null;  // Mode A (APUS) score 0-100
  quant_score_b?: number | null;  // Mode B (DOUS) score 0-100
  quant_gate_pass?: boolean;
  quant_evidence?: EvidenceBlock | null;
}

/**
 * Sector ETF mapping for benchmarking.
 */
const SECTOR_ETF_MAP: Record<string, string> = {
  'Technology': 'XLK',
  'Information Technology': 'XLK',
  'Healthcare': 'XLV',
  'Health Care': 'XLV',
  'Financials': 'XLF',
  'Financial Services': 'XLF',
  'Consumer Discretionary': 'XLY',
  'Consumer Cyclical': 'XLY',
  'Consumer Staples': 'XLP',
  'Consumer Defensive': 'XLP',
  'Energy': 'XLE',
  'Industrials': 'XLI',
  'Materials': 'XLB',
  'Basic Materials': 'XLB',
  'Real Estate': 'XLRE',
  'Utilities': 'XLU',
  'Communication Services': 'XLC',
  'Communication': 'XLC',
};

/**
 * Convert QuantRecommendation to StockCardData for StockCardV2.
 * Now uses pre-computed fields from backend.
 */
export function quantToStockCardData(
  rec: QuantRecommendation,
  stockInfo?: StockInfo | null
): StockCardData {
  // Use sector from backend first, fallback to stockInfo
  const sector = rec.sector || stockInfo?.sector || null;
  const sectorEtf = rec.sector_etf || (sector ? SECTOR_ETF_MAP[sector] : null) || null;
  
  // Use pre-computed opportunity score from backend, or fallback calculation
  let opportunityScore = rec.opportunity_score ?? 50;
  let opportunityRating = rec.opportunity_rating ?? 'hold';
  
  // Only calculate locally if backend didn't provide
  if (rec.opportunity_score === null || rec.opportunity_score === undefined) {
    opportunityScore = 50;
    
    const dipPct = rec.legacy_dip_pct || 0;
    if (dipPct > 5) opportunityScore += Math.min(dipPct / 2, 25);
    
    if (rec.legacy_domain_score && rec.legacy_domain_score > 50) {
      opportunityScore += (rec.legacy_domain_score - 50) * 0.4;
    }
    
    if (rec.ai_rating) {
      const ratingBonus: Record<string, number> = {
        'strong_buy': 15, 'buy': 10, 'hold': 0, 'sell': -10, 'strong_sell': -15,
      };
      opportunityScore += ratingBonus[rec.ai_rating] || 0;
    }
    
    if (rec.mu_hat > 0) {
      opportunityScore += Math.min(rec.mu_hat * 100, 10);
    }
    
    opportunityScore = Math.max(0, Math.min(100, opportunityScore));
    
    if (opportunityScore >= 75) opportunityRating = 'strong_buy';
    else if (opportunityScore >= 60) opportunityRating = 'buy';
    else if (opportunityScore >= 40) opportunityRating = 'hold';
    else opportunityRating = 'avoid';
  }
  
  // Build top signal from backend data
  const topSignal = rec.top_signal_name ? {
    name: rec.top_signal_name,
    is_buy: rec.top_signal_is_buy,
    strength: rec.top_signal_strength,
    description: rec.top_signal_description || undefined,
  } : null;
  
  return {
    symbol: rec.ticker,
    name: rec.name,
    sector,
    last_price: rec.last_price ?? 0,
    change_percent: rec.change_percent,
    high_52w: null,
    low_52w: null,
    market_cap: rec.market_cap,
    depth: rec.legacy_dip_pct !== null ? rec.legacy_dip_pct / 100 : 0,
    days_since_dip: rec.legacy_days_in_dip,
    dip_bucket: rec.dip_bucket,
    // Recovery and unusual dip metrics from quant engine
    expected_recovery_days: rec.expected_recovery_days,
    typical_dip_pct: rec.typical_dip_pct,
    dip_vs_typical: rec.dip_vs_typical,
    is_unusual_dip: rec.is_unusual_dip,
    win_rate: rec.win_rate,
    top_signal: topSignal,
    sector_delta: rec.vs_sector_performance ?? null,
    sector_etf: sectorEtf,
    opportunity_score: opportunityScore,
    opportunity_rating: opportunityRating as 'strong_buy' | 'buy' | 'hold' | 'avoid',
    ai_rating: rec.ai_rating,
    ai_summary: rec.ai_summary,
    domain_analysis: rec.domain_context || rec.ai_summary || null,
    // Domain-specific analysis
    domain_context: rec.domain_context,
    domain_adjustment: rec.domain_adjustment,
    domain_adjustment_reason: rec.domain_adjustment_reason,
    domain_risk_level: rec.domain_risk_level,
    domain_risk_factors: rec.domain_risk_factors,
    domain_recovery_days: rec.domain_recovery_days,
    domain_warnings: rec.domain_warnings,
    volatility_regime: rec.volatility_regime,
    volatility_percentile: rec.volatility_percentile,
    vs_sector_performance: rec.vs_sector_performance,
    mu_hat: rec.mu_hat,
    uncertainty: rec.uncertainty,
    marginal_utility: rec.marginal_utility,
    // APUS + DOUS Dual-Mode Scoring
    quant_mode: rec.quant_mode,
    quant_score_a: rec.quant_score_a,
    quant_score_b: rec.quant_score_b,
    quant_gate_pass: rec.quant_gate_pass,
    quant_evidence: rec.quant_evidence,
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

export interface SymbolListResponse {
  items: Symbol[];
  total: number;
  limit: number;
  offset: number;
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

export async function getSymbolsPaged(
  limit: number = 50,
  offset: number = 0,
  search?: string,
  skipCache: boolean = false
): Promise<SymbolListResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (search) {
    params.append('search', search);
  }
  const endpoint = `/symbols/paged?${params.toString()}`;
  const cacheKey = `symbols-paged:${limit}:${offset}:${search ?? ''}`;

  if (skipCache) {
    return fetchAPI<SymbolListResponse>(endpoint);
  }

  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<SymbolListResponse>(endpoint),
    { ttl: CACHE_TTL.SYMBOLS, staleWhileRevalidate: true }
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
  signal: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell' | 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  reasoning: string;
  key_factors: string[];
  avatar_url?: string;
}

export interface AgentAnalysis {
  symbol: string;
  analyzed_at: string;
  verdicts: AgentVerdict[];
  overall_signal: 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell' | 'bullish' | 'bearish' | 'neutral';
  overall_confidence: number;
  summary?: string;
  expires_at?: string;
  bullish_count?: number;
  bearish_count?: number;
  neutral_count?: number;
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

export interface SectorETFConfig {
  sector: string;
  symbol: string;
  name: string;
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
  sector_etfs: SectorETFConfig[];
  // Trading/Backtest Configuration
  trading_initial_capital: number;
  trading_flat_cost_per_trade: number;
  trading_slippage_bps: number;
  trading_stop_loss_pct: number;
  trading_take_profit_pct: number;
  trading_max_holding_days: number;
  trading_min_trades_required: number;
  trading_walk_forward_folds: number;
  trading_train_ratio: number;
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
  const cacheKey = `suggestions:${status ?? 'all'}:${page}:${pageSize}`;
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<SuggestionListResponse>(`/suggestions?${params.toString()}`),
    { ttl: CACHE_TTL.SUGGESTIONS, staleWhileRevalidate: true }
  );
}

export async function approveSuggestion(suggestionId: number): Promise<{ message: string; symbol: string; task_id?: string | null }> {
  const result = await fetchAPI<{ message: string; symbol: string; task_id?: string | null }>(`/suggestions/${suggestionId}/approve`, {
    method: 'POST',
  });
  // Invalidate symbols and ranking cache so the new stock appears immediately
  apiCache.invalidate(/^symbols/);
  apiCache.invalidate(/^ranking/);
  apiCache.invalidate(/^suggestions:/);
  invalidateEtagCache(/\/symbols|\/dips\/ranking/);
  return result;
}

export async function rejectSuggestion(suggestionId: number, reason?: string): Promise<{ message: string }> {
  const params = reason ? `?reason=${encodeURIComponent(reason)}` : '';
  const result = await fetchAPI<{ message: string }>(`/suggestions/${suggestionId}/reject${params}`, {
    method: 'POST',
  });
  apiCache.invalidate(/^suggestions:/);
  return result;
}

export async function updateSuggestion(
  suggestionId: number, 
  newSymbol: string
): Promise<{ message: string; old_symbol: string; new_symbol: string }> {
  const result = await fetchAPI<{ message: string; old_symbol: string; new_symbol: string }>(
    `/suggestions/${suggestionId}?new_symbol=${encodeURIComponent(newSymbol.toUpperCase())}`,
    { method: 'PATCH' }
  );
  apiCache.invalidate(/^suggestions:/);
  return result;
}

export async function refreshSuggestionData(
  suggestionId: number
): Promise<{ message: string; symbol: string; task_id?: string | null }> {
  const result = await fetchAPI<{ message: string; symbol: string; task_id?: string | null }>(
    `/suggestions/${suggestionId}/refresh`,
    { method: 'POST' }
  );
  apiCache.invalidate(/^suggestions:/);
  return result;
}

export async function retrySuggestionFetch(
  suggestionId: number
): Promise<{ message: string; symbol: string; fetch_status: FetchStatus; fetch_error: string | null }> {
  const result = await fetchAPI<{ message: string; symbol: string; fetch_status: FetchStatus; fetch_error: string | null }>(
    `/suggestions/${suggestionId}/retry`,
    { method: 'POST' }
  );
  apiCache.invalidate(/^suggestions:/);
  return result;
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
  limit: number;
  offset: number;
}

export async function getBatchJobs(
  limit: number = 20,
  offset: number = 0,
  includeCompleted: boolean = true
): Promise<BatchJobListResponse> {
  return fetchAPI<BatchJobListResponse>(
    `/admin/settings/batch-jobs?limit=${limit}&offset=${offset}&include_completed=${includeCompleted}`
  );
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

// =============================================================================
// BULK IMPORT API
// =============================================================================

export type ExtractionConfidence = 'high' | 'medium' | 'low';

export interface ExtractedPosition {
  symbol: string | null;
  name: string | null;
  isin: string | null;
  quantity: number | null;
  avg_cost: number | null;
  current_price: number | null;
  total_value: number | null;
  currency: string;
  exchange: string | null;
  confidence: ExtractionConfidence;
  raw_text: string | null;
  notes: string | null;
  skip: boolean;
}

export interface ImageExtractionResponse {
  success: boolean;
  positions: ExtractedPosition[];
  image_quality: string | null;
  detected_broker: string | null;
  currency_hint: string | null;
  processing_time_ms: number | null;
  error_message: string | null;
  warnings: string[];
}

export interface BulkImportPosition {
  symbol: string;
  quantity: number;
  avg_cost?: number;
  currency?: string;
}

export interface BulkImportRequest {
  positions: BulkImportPosition[];
  skip_duplicates?: boolean;
}

export type ImportResultStatus = 'created' | 'updated' | 'skipped' | 'failed';

export interface ImportPositionResult {
  symbol: string;
  status: ImportResultStatus;
  message: string | null;
  holding_id: number | null;
}

export interface BulkImportResponse {
  success: boolean;
  total: number;
  created: number;
  updated: number;
  skipped: number;
  failed: number;
  results: ImportPositionResult[];
}

export async function extractPositionsFromImage(
  portfolioId: number,
  file: File
): Promise<ImageExtractionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE}/portfolios/${portfolioId}/import/extract-image`,
    {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.message || error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function bulkImportPositions(
  portfolioId: number,
  payload: BulkImportRequest
): Promise<BulkImportResponse> {
  const result = await fetchAPI<BulkImportResponse>(
    `/portfolios/${portfolioId}/import/bulk`,
    {
      method: 'POST',
      body: JSON.stringify(payload),
    }
  );
  invalidateEtagCache(new RegExp(`/portfolios/${portfolioId}`));
  return result;
}

// =============================================================================
// AI PERSONAS API
// =============================================================================

export interface AIPersona {
  id: number;
  key: string;
  name: string;
  description: string | null;
  philosophy: string | null;
  has_avatar: boolean;
  avatar_url: string | null;
  is_active: boolean;
  display_order: number;
}

export interface AIPersonaUpdate {
  name?: string;
  description?: string;
  philosophy?: string;
  is_active?: boolean;
  display_order?: number;
}

/**
 * Get all AI personas
 */
export async function getAIPersonas(activeOnly = true): Promise<AIPersona[]> {
  return fetchAPI<AIPersona[]>(`/ai-personas?active_only=${activeOnly}`);
}

/**
 * Update an AI persona
 */
export async function updateAIPersona(personaKey: string, updates: AIPersonaUpdate): Promise<AIPersona> {
  return fetchAPI<AIPersona>(`/ai-personas/${personaKey}`, {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

/**
 * Upload avatar for an AI persona
 */
export async function uploadAIPersonaAvatar(
  personaKey: string,
  file: File,
  size = 128
): Promise<{ message: string; persona_key: string; size: number }> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE}/ai-personas/${personaKey}/avatar?size=${size}`,
    {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.message || error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Delete avatar for an AI persona
 */
export async function deleteAIPersonaAvatar(personaKey: string): Promise<{ message: string }> {
  return fetchAPI<{ message: string }>(`/ai-personas/${personaKey}/avatar`, {
    method: 'DELETE',
  });
}


// =============================================================================
// STRATEGY SIGNALS (QUANT OPTIMIZATION)
// =============================================================================

export interface StrategyMetrics {
  total_return_pct: number;
  sharpe_ratio: number;
  win_rate: number;
  max_drawdown_pct: number;
  n_trades: number;
}

export interface RecencyMetrics {
  weighted_return: number;
  current_year_return_pct: number;
  current_year_win_rate: number;
  current_year_trades: number;
}

export interface BenchmarkComparison {
  vs_buy_hold: number;
  vs_spy: number | null;
  beats_buy_hold: boolean;
}

export interface SignalInfo {
  type: 'BUY' | 'SELL' | 'HOLD' | 'WAIT' | 'WATCH';
  reason: string;
  has_active: boolean;
}

export interface FundamentalStatus {
  healthy: boolean;
  concerns: string[];
}

export interface TradeRecord {
  entry_date: string;
  exit_date: string | null;
  entry_price: number;
  exit_price: number | null;
  pnl_pct: number;
  exit_reason: string;
  holding_days: number | null;
}

export interface StrategySignalResponse {
  symbol: string;
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  signal: SignalInfo;
  metrics: StrategyMetrics;
  recency: RecencyMetrics;
  benchmarks: BenchmarkComparison;
  fundamentals: FundamentalStatus;
  is_statistically_valid: boolean;
  indicators_used: string[];
  recent_trades: TradeRecord[];
  optimized_at: string | null;
}

export interface StrategySignalSummary {
  symbol: string;
  strategy_name: string;
  signal_type: string;
  has_active_signal: boolean;
  win_rate: number;
  current_year_return_pct: number;
  beats_buy_hold: boolean;
  fundamentals_healthy: boolean;
  optimized_at: string | null;
}

export interface StrategySignalsListResponse {
  signals: StrategySignalSummary[];
  total: number;
  active_buy_signals: number;
  beating_market: number;
}

/**
 * Get optimized strategy signal for a symbol
 */
export async function getStrategySignal(symbol: string): Promise<StrategySignalResponse | null> {
  const cacheKey = `strategy:${symbol}`;
  
  try {
    return await apiCache.fetch(
      cacheKey,
      () => fetchAPI<StrategySignalResponse>(`/signals/strategy/${symbol}`),
      { ttl: CACHE_TTL.STOCK_INFO, staleWhileRevalidate: true }
    );
  } catch {
    return null;
  }
}

/**
 * List all strategy signals with optional filters
 */
export async function listStrategySignals(options?: {
  signalType?: string;
  activeOnly?: boolean;
  beatingMarket?: boolean;
  limit?: number;
}): Promise<StrategySignalsListResponse | null> {
  const params = new URLSearchParams();
  
  if (options?.signalType) params.append('signal_type', options.signalType);
  if (options?.activeOnly) params.append('active_only', 'true');
  if (options?.beatingMarket) params.append('beating_market', 'true');
  if (options?.limit) params.append('limit', options.limit.toString());
  
  const query = params.toString() ? `?${params.toString()}` : '';
  
  try {
    return await fetchAPI<StrategySignalsListResponse>(`/signals/strategy${query}`);
  } catch {
    return null;
  }
}

// =============================================================================
// Dip Entry Optimizer Types & API
// =============================================================================

export interface DipFrequency {
  '10_pct': number;
  '15_pct': number;
  '20_pct': number;
}

export interface DipThresholdStats {
  threshold: number;
  occurrences: number;
  per_year: number;
  win_rate_60d: number;
  avg_return_60d: number;
  recovery_rate: number;
  avg_recovery_days: number;
  entry_score: number;
}

export interface DipEntryResponse {
  symbol: string;
  current_price: number;
  recent_high: number;
  current_drawdown_pct: number;
  volatility_regime: 'low' | 'normal' | 'high';
  optimal_dip_threshold: number;
  optimal_entry_price: number;
  is_buy_now: boolean;
  buy_signal_strength: number;
  signal_reason: string;
  typical_recovery_days: number;
  avg_dips_per_year: DipFrequency;
  fundamentals_healthy: boolean;
  fundamental_notes: string[];
  threshold_analysis: DipThresholdStats[];
  analyzed_at: string;
}

/**
 * Get dip entry analysis for a symbol
 * Answers: "How much should this stock drop before I buy more?"
 */
export async function getDipEntry(symbol: string): Promise<DipEntryResponse | null> {
  const cacheKey = `dip-entry:${symbol}`;
  
  try {
    return await apiCache.fetch(
      cacheKey,
      () => fetchAPI<DipEntryResponse>(`/signals/dip-entry/${symbol}`),
      { ttl: CACHE_TTL.STOCK_INFO, staleWhileRevalidate: true }
    );
  } catch {
    return null;
  }
}
