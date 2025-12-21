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
export async function getRanking(skipCache = false): Promise<RankingResponse> {
  const cacheKey = 'ranking';
  
  const fetcher = async () => {
    const ranking = await fetchAPI<DipStock[]>('/dips/ranking');
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
    () => fetchAPI<ChartDataPoint[]>(`/dips/${symbol}/history?days=${days}`),
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
  return fetchAPI<ChartDataPoint[]>(`/dips/${symbol}/history?days=${days}`);
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
