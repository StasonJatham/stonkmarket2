import { getAuthHeaders } from './auth';
import { apiCache, CACHE_TTL } from '@/lib/cache';

const API_BASE = '/api';

async function fetchAPI<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
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
  const cacheKey = `ranking:${showAll}`;
  
  const fetcher = async () => {
    const ranking = await fetchAPI<DipStock[]>(`/dips/ranking?show_all=${showAll}`);
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
  const cacheKey = `chart:${symbol}:${days}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<ChartDataPoint[]>(`/dips/${symbol}/chart?days=${days}`),
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
    { ttl: CACHE_TTL.RANKING } // Cache for same time as ranking
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
export type BenchmarkType = 'SP500' | 'MSCI_WORLD' | null;

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

// Benchmark symbol mapping
const BENCHMARK_SYMBOLS: Record<Exclude<BenchmarkType, null>, string> = {
  SP500: '^GSPC',      // S&P 500 Index
  MSCI_WORLD: 'URTH',  // iShares MSCI World ETF (proxy for MSCI World)
};

const BENCHMARK_NAMES: Record<Exclude<BenchmarkType, null>, string> = {
  SP500: 'S&P 500',
  MSCI_WORLD: 'MSCI World',
};

export function getBenchmarkName(benchmark: BenchmarkType): string {
  if (!benchmark) return '';
  return BENCHMARK_NAMES[benchmark];
}

export async function getBenchmarkChart(
  benchmark: Exclude<BenchmarkType, null>,
  days: number = 365
): Promise<ChartDataPoint[]> {
  const symbol = BENCHMARK_SYMBOLS[benchmark];
  // Use /chart endpoint which exists, not /history
  return fetchAPI<ChartDataPoint[]>(`/dips/${symbol}/chart?days=${days}`);
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
// STOCK TINDER TYPES & API
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
  tinder_bio: string | null;
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
  const cacheKey = `tinder:cards:${includeAi}:${excludeVoted}`;
  
  return apiCache.fetch(
    cacheKey,
    () => fetchAPI<DipCardList>(`/tinder/cards?include_ai=${includeAi}&exclude_voted=${excludeVoted}`),
    { ttl: CACHE_TTL.RANKING }
  );
}

export async function getDipCard(symbol: string, refreshAi: boolean = false): Promise<DipCard> {
  return fetchAPI<DipCard>(`/tinder/cards/${symbol}?refresh_ai=${refreshAi}`);
}

export async function voteDip(symbol: string, voteType: VoteType): Promise<{ symbol: string; vote_type: string; message: string }> {
  return fetchAPI(`/tinder/cards/${symbol}/vote`, {
    method: 'PUT',
    body: JSON.stringify({ vote_type: voteType }),
  });
}

export async function getDipStats(symbol: string): Promise<DipStats> {
  return fetchAPI<DipStats>(`/tinder/cards/${symbol}/stats`);
}

export async function refreshAiAnalysis(symbol: string): Promise<DipCard> {
  return fetchAPI<DipCard>(`/tinder/cards/${symbol}/refresh-ai`, {
    method: 'POST',
  });
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
}

export async function getSymbols(): Promise<Symbol[]> {
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
  // Invalidate symbols cache on add
  apiCache.invalidate(/^symbols/);
  return result;
}

export async function updateSymbol(symbol: string, data: { min_dip_pct?: number; min_days?: number }): Promise<Symbol> {
  const result = await fetchAPI<Symbol>(`/symbols/${symbol}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
  // Invalidate symbols cache on update
  apiCache.invalidate(/^symbols/);
  return result;
}

export async function deleteSymbol(symbol: string): Promise<void> {
  await fetchAPI(`/symbols/${symbol}`, {
    method: 'DELETE',
  });
  // Invalidate symbols cache on delete
  apiCache.invalidate(/^symbols/);
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
  apiCache.invalidate(/^(runtime-settings|suggestion-settings)/);
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

export interface Suggestion {
  id: number;
  symbol: string;
  status: SuggestionStatus;
  vote_count: number;
  name: string | null;
  sector: string | null;
  summary: string | null;
  last_price: number | null;
  price_change_90d: number | null;
  created_at: string;
  updated_at: string | null;
  fetched_at: string | null;
  rejection_reason?: string | null;
  approved_by?: number | null;
  reviewed_at?: string | null;
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
}

// Public endpoints (no auth required)
export async function suggestStock(symbol: string): Promise<{ message: string; symbol: string; vote_count: number; status: string }> {
  return fetchAPI<{ message: string; symbol: string; vote_count: number; status: string }>(
    `/suggestions?symbol=${encodeURIComponent(symbol.toUpperCase())}`,
    { method: 'POST' }
  );
}

export async function voteForSuggestion(symbol: string): Promise<{ message: string; symbol: string; auto_approved?: boolean }> {
  return fetchAPI<{ message: string; symbol: string; auto_approved?: boolean }>(`/suggestions/${encodeURIComponent(symbol.toUpperCase())}/vote`, {
    method: 'PUT',
  });
}

export async function getTopSuggestions(limit: number = 10, excludeVoted: boolean = false): Promise<TopSuggestion[]> {
  return fetchAPI<TopSuggestion[]>(`/suggestions/top?limit=${limit}&exclude_voted=${excludeVoted}`);
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
  return fetchAPI<{ message: string; symbol: string }>(`/suggestions/${suggestionId}/approve`, {
    method: 'POST',
  });
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
