# Caching Strategy

This document describes the multi-layer caching strategy used in stonkmarket.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ HTTP Cache (ETag, Cache-Control)                            ││
│  │ - 304 Not Modified responses                                ││
│  │ - Reduces network transfer for unchanged data               ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ApiCache (JavaScript)                                       ││
│  │ - In-memory + localStorage persistence                      ││
│  │ - Stale-while-revalidate pattern                           ││
│  │ - Survives page refreshes                                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Valkey (Redis-compatible)                                   ││
│  │ - Primary application cache                                 ││
│  │ - Shared across all worker processes                        ││
│  │ - Key pattern: stonkmarket:v1:{prefix}:{key}               ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Process-local cache (stock_info.py)                         ││
│  │ - Short TTL (1 hour)                                        ││
│  │ - Reduces yfinance API calls                                ││
│  │ - Acceptable drift across workers                           ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ PostgreSQL                                                  ││
│  │ - Source of truth                                           ││
│  │ - dip_state, symbols, votes tables                          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Cache Key Naming Convention

All Valkey cache keys follow this pattern:

```
stonkmarket:v1:{prefix}:{key}
```

Examples:
- `stonkmarket:v1:ranking:all:True` - Ranking with all stocks
- `stonkmarket:v1:chart:AAPL:180` - 180-day chart for AAPL
- `stonkmarket:v1:info:MSFT` - Stock info for MSFT

## TTL Configuration

| Cache Type | TTL | Rationale |
|------------|-----|-----------|
| Ranking | 1 hour | Updated daily by cron job |
| Chart | 1 hour | Historical data, rarely changes |
| Stock Info | 1 hour | Company info changes infrequently |
| Symbol validation | 30 days | Ticker symbols are stable |
| Vote cooldown | 7 days | User voting rate limit |
| Frontend ranking | 30 min | Balance freshness vs performance |
| Frontend chart | 1 hour | Match backend TTL |

## HTTP Caching

### Headers Used

- **Cache-Control**: `public, max-age={seconds}` for public endpoints
- **ETag**: Hash-based content identifier for conditional requests
- **304 Not Modified**: Returned when client ETag matches server

### Using CacheableResponse

```python
from app.cache.http_cache import CacheableResponse, CachePresets, generate_etag

@router.get("/endpoint")
async def my_endpoint(request: Request):
    data = await get_data()
    etag = generate_etag(data)
    
    if check_if_none_match(request, etag):
        return NotModifiedResponse(etag=etag)
    
    return CacheableResponse(
        data,
        etag=etag,
        **CachePresets.RANKING,  # or CHART, STOCK_INFO, etc.
    )
```

### Cache Presets

- `CachePresets.RANKING` - 30 min max-age
- `CachePresets.CHART` - 1 hour max-age  
- `CachePresets.STOCK_INFO` - 1 hour max-age
- `CachePresets.SYMBOLS` - 24 hour max-age
- `CachePresets.BENCHMARKS` - 24 hour max-age

## Frontend Cache

### ApiCache Class

Located at `frontend/src/lib/cache.ts`:

```typescript
// Fetch with caching
const data = await apiCache.fetch(
  'my:key',
  () => fetchData(),
  { ttl: CACHE_TTL.RANKING, staleWhileRevalidate: true }
);

// Invalidate cache
apiCache.invalidate('my:key');
apiCache.invalidatePrefix('chart:');  // Invalidate all chart entries
```

### Features

- **Stale-while-revalidate**: Returns cached data immediately while fetching fresh data
- **localStorage persistence**: Cache survives page refreshes
- **Automatic cleanup**: Removes expired entries periodically
- **Entry limits**: Max 100 entries in localStorage to prevent quota issues

### ETag Handling

The frontend stores ETags from responses and sends `If-None-Match` headers on subsequent requests. If the server returns 304, the cached data is used.

## Cache Invalidation

### Manual Invalidation

```python
# Single key
await cache.delete("stonkmarket:v1:ranking:all:True")

# Prefix-based (use sparingly - uses SCAN)
await cache.invalidate_prefix("ranking:")
```

### Automatic Invalidation

1. **After data_grab job**: Ranking cache is rebuilt
2. **After cache_warmup job**: Pre-populates fresh data
3. **Admin refresh endpoint**: Clears and rebuilds ranking cache

## Monitoring

### Cache Stats Endpoint

```
GET /api/health/cache
```

Returns:
```json
{
  "ranking": {
    "hits": 150,
    "misses": 12,
    "sets": 12,
    "hit_rate": 0.926,
    "avg_get_ms": 0.5,
    "avg_set_ms": 1.2
  },
  "chart": { ... },
  "totals": { ... }
}
```

### Using CacheTimer

```python
from app.cache import CacheTimer

async def expensive_operation():
    with CacheTimer("ranking"):
        # Operation timing is automatically recorded
        result = await compute_ranking()
    return result
```

## Scheduled Jobs

| Job | Schedule | Cache Effect |
|-----|----------|--------------|
| data_grab | Mon-Fri 11pm | Fetches new prices, invalidates ranking |
| cache_warmup | After data_grab | Pre-populates ranking and common charts |

## Best Practices

1. **Always use the cache module** - Don't access Valkey directly
2. **Use appropriate TTLs** - Shorter for volatile data, longer for static
3. **Add HTTP caching** - Reduces network traffic significantly
4. **Monitor hit rates** - Target >90% for frequently accessed data
5. **Invalidate surgically** - Avoid prefix-based invalidation when possible
6. **Use stale-while-revalidate** - Better UX while refreshing data
