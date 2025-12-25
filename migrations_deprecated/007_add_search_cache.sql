-- Migration: Add symbol search cache table
-- Caches yfinance search results to reduce API calls
-- Cache TTL is 7 days (company names don't change often)

CREATE TABLE IF NOT EXISTS symbol_search_cache (
    id SERIAL PRIMARY KEY,
    query VARCHAR(100) NOT NULL,
    results JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    UNIQUE(query)
);

CREATE INDEX IF NOT EXISTS idx_search_cache_query ON symbol_search_cache(query);
CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON symbol_search_cache(expires_at);

COMMENT ON TABLE symbol_search_cache IS 'Cache for yfinance symbol search results to reduce API calls';
COMMENT ON COLUMN symbol_search_cache.query IS 'Normalized uppercase search query';
COMMENT ON COLUMN symbol_search_cache.results IS 'JSON array of search results';
