/**
 * Multi-layer cache with TTL support for API responses.
 * 
 * Features:
 * - In-memory cache for fastest access
 * - Optional localStorage persistence for surviving page reloads
 * - Stale-while-revalidate pattern for optimal UX
 * - Request deduplication to prevent duplicate network calls
 * - ETag support for efficient revalidation
 * - Data version tracking for automatic cache invalidation
 */

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
  etag?: string;
}

interface CacheOptions {
  ttl?: number; // Time to live in milliseconds
  staleWhileRevalidate?: boolean;
  persist?: boolean; // Persist to localStorage
  etag?: string; // ETag from server
}

const DEFAULT_TTL = 5 * 60 * 1000; // 5 minutes
const STALE_TTL = 30 * 60 * 1000; // 30 minutes for stale-while-revalidate
const STORAGE_PREFIX = 'stonkmarket_cache_';
const DATA_VERSION_KEY = 'stonkmarket_data_version';
const MAX_STORAGE_ENTRIES = 50; // Limit localStorage entries


class ApiCache {
  private cache = new Map<string, CacheEntry<unknown>>();
  private pendingRequests = new Map<string, Promise<unknown>>();
  private initialized = false;

  constructor() {
    // Lazy initialize from localStorage
    this.initFromStorage();
  }

  /**
   * Initialize cache from localStorage on first access
   */
  private initFromStorage(): void {
    if (this.initialized || typeof window === 'undefined') return;
    this.initialized = true;

    try {
      const now = Date.now();
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith(STORAGE_PREFIX)) {
          const cacheKey = key.slice(STORAGE_PREFIX.length);
          const raw = localStorage.getItem(key);
          if (raw) {
            const entry = JSON.parse(raw) as CacheEntry<unknown>;
            // Only restore if not completely stale
            if (now < entry.expiresAt + STALE_TTL) {
              this.cache.set(cacheKey, entry);
            } else {
              // Clean up expired entries
              localStorage.removeItem(key);
            }
          }
        }
      }
    } catch {
      // Storage might be unavailable or corrupted
      console.warn('Failed to restore cache from localStorage');
    }
  }

  /**
   * Persist entry to localStorage
   */
  private persistToStorage(key: string, entry: CacheEntry<unknown>): void {
    if (typeof window === 'undefined') return;
    
    try {
      // Check storage limit
      const storageKey = STORAGE_PREFIX + key;
      const existingKeys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k?.startsWith(STORAGE_PREFIX)) {
          existingKeys.push(k);
        }
      }
      
      // Remove oldest entries if over limit
      if (existingKeys.length >= MAX_STORAGE_ENTRIES) {
        const entries: { key: string; timestamp: number }[] = [];
        for (const k of existingKeys) {
          try {
            const raw = localStorage.getItem(k);
            if (raw) {
              const e = JSON.parse(raw) as CacheEntry<unknown>;
              entries.push({ key: k, timestamp: e.timestamp });
            }
          } catch {
            localStorage.removeItem(k);
          }
        }
        entries.sort((a, b) => a.timestamp - b.timestamp);
        // Remove oldest 10%
        const toRemove = Math.ceil(entries.length * 0.1);
        for (let i = 0; i < toRemove; i++) {
          localStorage.removeItem(entries[i].key);
        }
      }
      
      localStorage.setItem(storageKey, JSON.stringify(entry));
    } catch {
      // Storage full or unavailable
      console.warn('Failed to persist cache to localStorage');
    }
  }

  /**
   * Get item from cache
   */
  get<T>(key: string): { data: T; isStale: boolean; etag?: string } | null {
    const entry = this.cache.get(key) as CacheEntry<T> | undefined;
    
    if (!entry) return null;
    
    const now = Date.now();
    const isExpired = now > entry.expiresAt;
    const isStale = isExpired && now < entry.expiresAt + STALE_TTL;
    
    // Return stale data if within stale window
    if (!isExpired || isStale) {
      return { data: entry.data, isStale: isExpired, etag: entry.etag };
    }
    
    // Data is too old, remove it
    this.cache.delete(key);
    localStorage.removeItem(STORAGE_PREFIX + key);
    return null;
  }

  /**
   * Set item in cache
   */
  set<T>(key: string, data: T, options: CacheOptions = {}): void {
    const ttl = options.ttl ?? DEFAULT_TTL;
    
    // Skip caching if TTL is 0 (real-time mode)
    if (ttl === 0) return;
    
    const now = Date.now();
    
    const entry: CacheEntry<T> = {
      data,
      timestamp: now,
      expiresAt: now + ttl,
      etag: options.etag,
    };
    
    this.cache.set(key, entry);
    
    // Persist important data to localStorage
    if (options.persist !== false) {
      this.persistToStorage(key, entry);
    }
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
    const ttl = options.ttl ?? DEFAULT_TTL;
    
    // TTL=0 means real-time mode - skip cache entirely
    if (ttl === 0) {
      return this.dedupe(key, fetcher);
    }
    
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
      // Clear localStorage cache entries
      if (typeof window !== 'undefined') {
        const keysToRemove: string[] = [];
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key?.startsWith(STORAGE_PREFIX)) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach(k => localStorage.removeItem(k));
      }
      return;
    }
    
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
        localStorage.removeItem(STORAGE_PREFIX + key);
      }
    }
  }

  /**
   * Get cache statistics
   */
  stats(): { size: number; keys: string[]; memorySize: number; storageSize: number } {
    let storageSize = 0;
    if (typeof window !== 'undefined') {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith(STORAGE_PREFIX)) {
          storageSize++;
        }
      }
    }
    
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
      memorySize: this.cache.size,
      storageSize,
    };
  }

  /**
   * Clear all cache (memory and storage)
   */
  clear(): void {
    this.invalidate();
  }

  /**
   * Prefetch data in the background without blocking.
   * Returns immediately, caches the result when ready.
   */
  prefetch<T>(
    key: string,
    fetcher: () => Promise<T>,
    options: CacheOptions = {}
  ): void {
    // Skip if already cached and not stale
    const cached = this.get<T>(key);
    if (cached && !cached.isStale) {
      return;
    }

    // Fetch in background
    this.dedupe(key, fetcher)
      .then(data => {
        this.set(key, data, options);
      })
      .catch(() => {
        // Silently fail prefetch - it's not critical
      });
  }

  /**
   * Check if a key is cached (and not expired)
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;
    return Date.now() <= entry.expiresAt;
  }

  /**
   * Check data version from server and clear cache if version changed.
   * Called automatically when X-Data-Version header is received.
   * 
   * Only clears cache if version is NEWER (larger timestamp) to avoid
   * thrashing from multi-process inconsistencies.
   */
  checkDataVersion(serverVersion: string): void {
    if (typeof window === 'undefined') return;
    
    try {
      const storedVersion = localStorage.getItem(DATA_VERSION_KEY);
      
      if (storedVersion && storedVersion !== serverVersion) {
        // Only clear if server version is newer (larger timestamp)
        // This prevents thrashing when different server instances have slightly different versions
        const storedNum = parseInt(storedVersion, 10);
        const serverNum = parseInt(serverVersion, 10);
        
        if (!isNaN(storedNum) && !isNaN(serverNum) && serverNum > storedNum) {
          console.info(`Data version updated: ${storedVersion} → ${serverVersion}. Clearing caches.`);
          this.clear();
          localStorage.setItem(DATA_VERSION_KEY, serverVersion);
        }
        // If server version is older or same, ignore it (stale server response)
      } else if (!storedVersion) {
        // First time - just store it
        localStorage.setItem(DATA_VERSION_KEY, serverVersion);
      }
    } catch {
      // Storage unavailable
    }
  }
}

// Singleton instance
export const apiCache = new ApiCache();

// Cache TTL constants for different data types
// Short TTLs with stale-while-revalidate give instant UI + fresh data
export const CACHE_TTL = {
  RANKING: 30 * 1000,           // 30 seconds - quick refresh, stale-while-revalidate handles instant display
  CHART: 60 * 1000,             // 1 minute - chart data changes less frequently
  STOCK_INFO: 5 * 60 * 1000,    // 5 minutes - stock metadata rarely changes
  BENCHMARK: 60 * 1000,         // 1 minute - benchmark config
  CRON_JOBS: 30 * 1000,         // 30 seconds - admin wants near real-time
  SYMBOLS: 30 * 1000,           // 30 seconds - symbol list, invalidated on mutations
  SETTINGS: 5 * 60 * 1000,      // 5 minutes - runtime settings
  SUGGESTIONS: 30 * 1000,       // 30 seconds - suggestions change with votes
  RECOMMENDATIONS: 60 * 1000,   // 1 minute - quant recommendations
} as const;