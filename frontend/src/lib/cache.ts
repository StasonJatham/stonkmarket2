/**
 * Simple in-memory cache with TTL support for API responses
 * Optimizes performance by reducing redundant network requests
 */

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

interface CacheOptions {
  ttl?: number; // Time to live in milliseconds
  staleWhileRevalidate?: boolean;
}

const DEFAULT_TTL = 5 * 60 * 1000; // 5 minutes
const STALE_TTL = 30 * 60 * 1000; // 30 minutes for stale-while-revalidate

class ApiCache {
  private cache = new Map<string, CacheEntry<unknown>>();
  private pendingRequests = new Map<string, Promise<unknown>>();

  /**
   * Get item from cache
   */
  get<T>(key: string): { data: T; isStale: boolean } | null {
    const entry = this.cache.get(key) as CacheEntry<T> | undefined;
    
    if (!entry) return null;
    
    const now = Date.now();
    const isExpired = now > entry.expiresAt;
    const isStale = isExpired && now < entry.expiresAt + STALE_TTL;
    
    // Return stale data if within stale window
    if (!isExpired || isStale) {
      return { data: entry.data, isStale: isExpired };
    }
    
    // Data is too old, remove it
    this.cache.delete(key);
    return null;
  }

  /**
   * Set item in cache
   */
  set<T>(key: string, data: T, options: CacheOptions = {}): void {
    const ttl = options.ttl ?? DEFAULT_TTL;
    const now = Date.now();
    
    this.cache.set(key, {
      data,
      timestamp: now,
      expiresAt: now + ttl,
    });
  }

  /**
   * Deduplicate concurrent requests to the same endpoint
   */
  async dedupe<T>(key: string, fetcher: () => Promise<T>): Promise<T> {
    // Check if there's already a pending request
    const pending = this.pendingRequests.get(key) as Promise<T> | undefined;
    if (pending) {
      return pending;
    }

    // Create new request and track it
    const request = fetcher().finally(() => {
      this.pendingRequests.delete(key);
    });
    
    this.pendingRequests.set(key, request);
    return request;
  }

  /**
   * Cached fetch with stale-while-revalidate pattern
   */
  async fetch<T>(
    key: string,
    fetcher: () => Promise<T>,
    options: CacheOptions = {}
  ): Promise<T> {
    const cached = this.get<T>(key);
    
    // Return fresh cached data immediately
    if (cached && !cached.isStale) {
      return cached.data;
    }
    
    // If we have stale data, return it and revalidate in background
    if (cached?.isStale && options.staleWhileRevalidate !== false) {
      // Revalidate in background (don't await)
      this.dedupe(key, fetcher).then(data => {
        this.set(key, data, options);
      }).catch(console.error);
      
      return cached.data;
    }
    
    // Fetch fresh data
    const data = await this.dedupe(key, fetcher);
    this.set(key, data, options);
    return data;
  }

  /**
   * Invalidate specific cache entries
   */
  invalidate(pattern?: string | RegExp): void {
    if (!pattern) {
      this.cache.clear();
      return;
    }
    
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Get cache statistics
   */
  stats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }
}

// Singleton instance
export const apiCache = new ApiCache();

// Cache TTL constants for different data types
export const CACHE_TTL = {
  RANKING: 30 * 60 * 1000,      // 30 minutes - data only changes when job runs
  CHART: 60 * 60 * 1000,        // 1 hour - historical data, invalidated when job runs
  STOCK_INFO: 60 * 60 * 1000,   // 1 hour - rarely changes
  BENCHMARK: 60 * 60 * 1000,    // 1 hour - market data
  CRON_JOBS: 60 * 1000,         // 1 minute - admin data
  SYMBOLS: 24 * 60 * 60 * 1000, // 24 hours - symbol list, invalidated on add/delete
  SETTINGS: 60 * 60 * 1000,     // 1 hour - runtime settings
  SUGGESTIONS: 5 * 60 * 1000,   // 5 minutes - suggestion settings
} as const;
