-- Migration: Add symbol_search_results table
-- Stores individual search results from yfinance for future instant local lookups
-- This enables local-first search - once a symbol is found via API, future searches find it locally

CREATE TABLE IF NOT EXISTS symbol_search_results (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(50),
    quote_type VARCHAR(20) DEFAULT 'EQUITY',
    market_cap NUMERIC(20, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Indexes for efficient searching
CREATE INDEX IF NOT EXISTS idx_search_results_symbol ON symbol_search_results(UPPER(symbol));
CREATE INDEX IF NOT EXISTS idx_search_results_name ON symbol_search_results(UPPER(name) text_pattern_ops);
CREATE INDEX IF NOT EXISTS idx_search_results_expires ON symbol_search_results(expires_at);

COMMENT ON TABLE symbol_search_results IS 'Cached individual search results from yfinance API for local-first search';
COMMENT ON COLUMN symbol_search_results.symbol IS 'Stock symbol (unique)';
COMMENT ON COLUMN symbol_search_results.expires_at IS 'When this cache entry expires (default 30 days)';
