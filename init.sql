-- PostgreSQL Schema for Stonkmarket
-- Optimized for high-performance stock tracking

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- SCHEMA MIGRATIONS TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- AUTH & SECURITY TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS auth_user (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(64),
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_backup_codes TEXT,  -- JSON array of hashed backup codes
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_auth_user_username ON auth_user(username);

-- Admin-managed API keys for secure storage (OpenAI, etc.)
CREATE TABLE IF NOT EXISTS secure_api_keys (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) UNIQUE NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_hint VARCHAR(20),
    created_by INTEGER REFERENCES auth_user(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_secure_api_keys_service ON secure_api_keys(service_name);

-- User API keys for public API access
CREATE TABLE IF NOT EXISTS user_api_keys (
    id SERIAL PRIMARY KEY,
    key_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA-256 hash of the key
    key_prefix VARCHAR(16) NOT NULL,  -- First chars for identification (e.g., sm_xxxxxxxx)
    name VARCHAR(100) NOT NULL,
    description TEXT,
    user_id INTEGER REFERENCES auth_user(id),
    vote_weight INTEGER DEFAULT 10,
    rate_limit_bypass BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_user_api_keys_hash ON user_api_keys(key_hash);
CREATE INDEX idx_user_api_keys_active ON user_api_keys(is_active) WHERE is_active = TRUE;

-- ============================================================================
-- STOCK SYMBOL TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS symbols (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    market_cap BIGINT,
    summary_ai VARCHAR(350),  -- AI-generated short company description (~300 chars)
    symbol_type VARCHAR(20) DEFAULT 'stock',  -- 'stock', 'etf', 'index'
    min_dip_pct DECIMAL(5, 4) DEFAULT 0.15,
    min_days INTEGER DEFAULT 5,
    is_active BOOLEAN DEFAULT TRUE,
    -- Logo caching from Logo.dev API
    logo_light BYTEA,  -- WebP logo for light theme
    logo_dark BYTEA,   -- WebP logo for dark theme
    logo_fetched_at TIMESTAMPTZ,
    logo_source VARCHAR(50),  -- 'logo.dev' or 'favicon'
    added_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_symbols_symbol ON symbols(symbol);
CREATE INDEX idx_symbols_sector ON symbols(sector);
CREATE INDEX idx_symbols_type ON symbols(symbol_type);
CREATE INDEX idx_symbols_logo_fetched_at ON symbols(logo_fetched_at);

-- Current dip state (actively tracked dips)
CREATE TABLE IF NOT EXISTS dip_state (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    current_price DECIMAL(12, 4),
    ath_price DECIMAL(12, 4),
    dip_percentage DECIMAL(8, 4),
    dip_start_date DATE,  -- When stock first entered dip territory
    -- Legacy columns for compatibility
    ref_high DECIMAL(12, 4),
    last_price DECIMAL(12, 4),
    days_below INTEGER DEFAULT 0,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),  -- Alias for last_updated
    UNIQUE(symbol)
);

CREATE INDEX idx_dip_state_symbol ON dip_state(symbol);
CREATE INDEX idx_dip_state_percentage ON dip_state(dip_percentage DESC);
CREATE INDEX idx_dip_state_updated ON dip_state(last_updated DESC);
CREATE INDEX idx_dip_state_start_date ON dip_state(dip_start_date);

-- Dip history for tracking changes over time
CREATE TABLE IF NOT EXISTS dip_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('added', 'removed', 'updated')),
    current_price DECIMAL(12, 4),
    ath_price DECIMAL(12, 4),
    dip_percentage DECIMAL(8, 4),
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dip_history_symbol ON dip_history(symbol);
CREATE INDEX idx_dip_history_action ON dip_history(action);
CREATE INDEX idx_dip_history_recorded ON dip_history(recorded_at DESC);
CREATE INDEX idx_dip_history_time_range ON dip_history(recorded_at, action);

-- ============================================================================
-- STOCK SUGGESTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS stock_suggestions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    summary TEXT,
    website VARCHAR(255),
    ipo_year INTEGER,
    reason TEXT,
    fingerprint VARCHAR(64) NOT NULL,  -- Submitter fingerprint
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    vote_score INTEGER DEFAULT 0,
    fetch_status VARCHAR(20) DEFAULT 'pending',  -- pending, fetched, failed
    fetch_error TEXT,
    fetched_at TIMESTAMPTZ,
    current_price NUMERIC(12, 4),
    ath_price NUMERIC(12, 4),
    approved_by INTEGER REFERENCES auth_user(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX idx_suggestions_status ON stock_suggestions(status);
CREATE INDEX idx_suggestions_score ON stock_suggestions(vote_score DESC);
CREATE INDEX idx_suggestions_symbol ON stock_suggestions(symbol);

-- Votes on stock suggestions
CREATE TABLE IF NOT EXISTS suggestion_votes (
    id SERIAL PRIMARY KEY,
    suggestion_id INTEGER NOT NULL REFERENCES stock_suggestions(id) ON DELETE CASCADE,
    fingerprint VARCHAR(64) NOT NULL,
    vote_type VARCHAR(10) NOT NULL CHECK (vote_type IN ('up', 'down')),
    vote_weight INTEGER DEFAULT 1,
    api_key_id INTEGER REFERENCES user_api_keys(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(suggestion_id, fingerprint)
);

CREATE INDEX idx_suggestion_votes_suggestion ON suggestion_votes(suggestion_id);

-- ============================================================================
-- TINDER / VOTING
-- ============================================================================

-- Votes on current dips (buy/sell)
CREATE TABLE IF NOT EXISTS dip_votes (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    fingerprint VARCHAR(64) NOT NULL,
    vote_type VARCHAR(10) NOT NULL CHECK (vote_type IN ('buy', 'sell')),
    vote_weight INTEGER DEFAULT 1,
    api_key_id INTEGER REFERENCES user_api_keys(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, fingerprint)
);

CREATE INDEX idx_dip_votes_symbol ON dip_votes(symbol);
CREATE INDEX idx_dip_votes_fingerprint ON dip_votes(fingerprint);
CREATE INDEX idx_dip_votes_created ON dip_votes(created_at DESC);

-- AI-generated analysis for dips
CREATE TABLE IF NOT EXISTS dip_ai_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    swipe_bio TEXT,
    ai_rating DECIMAL(3, 1) CHECK (ai_rating >= 0 AND ai_rating <= 10),
    rating_reasoning TEXT,
    model_used VARCHAR(50),
    tokens_used INTEGER,
    is_batch_generated BOOLEAN DEFAULT FALSE,
    batch_job_id VARCHAR(100),
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_dip_ai_analysis_symbol ON dip_ai_analysis(symbol);
CREATE INDEX idx_dip_ai_analysis_rating ON dip_ai_analysis(ai_rating DESC);
CREATE INDEX idx_dip_ai_analysis_expires ON dip_ai_analysis(expires_at);

-- ============================================================================
-- API USAGE & BATCH JOBS
-- ============================================================================

-- Track API usage for cost monitoring
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    service VARCHAR(50) NOT NULL,  -- 'openai', 'yfinance', etc.
    endpoint VARCHAR(100),
    model VARCHAR(50),
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6),
    is_batch BOOLEAN DEFAULT FALSE,
    request_metadata JSONB,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_api_usage_service ON api_usage(service);
CREATE INDEX idx_api_usage_recorded ON api_usage(recorded_at DESC);
-- Note: can't create immutable date index on TIMESTAMPTZ, use recorded_at directly

-- Batch job tracking
CREATE TABLE IF NOT EXISTS batch_jobs (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100) UNIQUE NOT NULL,  -- OpenAI batch ID
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'validating', 'in_progress', 'finalizing', 'completed', 'failed', 'expired', 'cancelled')),
    total_requests INTEGER DEFAULT 0,
    completed_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    input_file_id VARCHAR(100),
    output_file_id VARCHAR(100),
    error_file_id VARCHAR(100),
    estimated_cost_usd DECIMAL(10, 6),
    actual_cost_usd DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX idx_batch_jobs_type ON batch_jobs(job_type);
CREATE INDEX idx_batch_jobs_created ON batch_jobs(created_at DESC);

-- ============================================================================
-- SCHEDULER / CRON JOBS
-- ============================================================================

CREATE TABLE IF NOT EXISTS cronjobs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    cron_expression VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMPTZ,
    next_run TIMESTAMPTZ,
    last_status VARCHAR(20),
    last_duration_ms INTEGER,
    run_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cronjobs_active ON cronjobs(is_active) WHERE is_active = TRUE;

-- Runtime settings (persisted key-value store)
CREATE TABLE IF NOT EXISTS runtime_settings (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default runtime settings
INSERT INTO runtime_settings (key, value) VALUES
    ('signal_threshold_strong_buy', '80.0'),
    ('signal_threshold_buy', '60.0'),
    ('signal_threshold_hold', '40.0'),
    ('ai_enrichment_enabled', 'true'),
    ('ai_batch_size', '0'),
    ('ai_model', '"gpt-5-mini"'),
    ('suggestion_cleanup_days', '30'),
    ('auto_approve_votes', '10'),
    ('benchmarks', '[{"id": "SP500", "symbol": "^GSPC", "name": "S&P 500", "description": "US Large Cap Index"}, {"id": "MSCI_WORLD", "symbol": "URTH", "name": "MSCI World", "description": "Global Developed Markets"}]')
ON CONFLICT (key) DO NOTHING;
CREATE INDEX idx_cronjobs_next_run ON cronjobs(next_run);

-- ============================================================================
-- RATE LIMITING
-- ============================================================================

CREATE TABLE IF NOT EXISTS rate_limit_log (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(64) NOT NULL,  -- IP hash or fingerprint
    endpoint VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 1,
    window_start TIMESTAMPTZ DEFAULT NOW(),
    last_request_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rate_limit_identifier ON rate_limit_log(identifier, endpoint);
CREATE INDEX idx_rate_limit_window ON rate_limit_log(window_start);

-- Cleanup old rate limit entries (run periodically)
CREATE OR REPLACE FUNCTION cleanup_rate_limits() RETURNS void AS $$
BEGIN
    DELETE FROM rate_limit_log WHERE window_start < NOW() - INTERVAL '1 day';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS & FUNCTIONS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_auth_user_updated
    BEFORE UPDATE ON auth_user
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_symbols_updated
    BEFORE UPDATE ON symbols
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_dip_state_updated
    BEFORE UPDATE ON dip_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_secure_api_keys_updated
    BEFORE UPDATE ON secure_api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_cronjobs_updated
    BEFORE UPDATE ON cronjobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Track dip state changes in history
CREATE OR REPLACE FUNCTION log_dip_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO dip_history (symbol, action, current_price, ath_price, dip_percentage)
        VALUES (NEW.symbol, 'added', NEW.current_price, NEW.ath_price, NEW.dip_percentage);
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO dip_history (symbol, action, current_price, ath_price, dip_percentage)
        VALUES (OLD.symbol, 'removed', OLD.current_price, OLD.ath_price, OLD.dip_percentage);
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_dip_state_history
    AFTER INSERT OR DELETE ON dip_state
    FOR EACH ROW EXECUTE FUNCTION log_dip_change();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Aggregated vote counts per dip
CREATE OR REPLACE VIEW dip_vote_summary AS
SELECT 
    symbol,
    COUNT(*) FILTER (WHERE vote_type = 'buy') as buy_votes,
    COUNT(*) FILTER (WHERE vote_type = 'sell') as sell_votes,
    SUM(vote_weight) FILTER (WHERE vote_type = 'buy') as weighted_buy,
    SUM(vote_weight) FILTER (WHERE vote_type = 'sell') as weighted_sell,
    SUM(vote_weight) FILTER (WHERE vote_type = 'buy') - 
        SUM(vote_weight) FILTER (WHERE vote_type = 'sell') as net_score
FROM dip_votes
GROUP BY symbol;

-- Full dip card data for tinder
CREATE OR REPLACE VIEW swipe_cards AS
SELECT 
    ds.symbol,
    ds.current_price,
    ds.ath_price,
    ds.dip_percentage,
    ds.first_seen,
    ds.last_updated,
    ai.swipe_bio,
    ai.ai_rating,
    ai.rating_reasoning,
    ai.generated_at as ai_generated_at,
    COALESCE(v.buy_votes, 0) as buy_votes,
    COALESCE(v.sell_votes, 0) as sell_votes,
    COALESCE(v.weighted_buy, 0) as weighted_buy,
    COALESCE(v.weighted_sell, 0) as weighted_sell,
    COALESCE(v.net_score, 0) as net_score
FROM dip_state ds
LEFT JOIN dip_ai_analysis ai ON ds.symbol = ai.symbol
LEFT JOIN dip_vote_summary v ON ds.symbol = v.symbol;

-- API usage summary by day
CREATE OR REPLACE VIEW daily_api_costs AS
SELECT 
    DATE(recorded_at) as date,
    service,
    COUNT(*) as request_count,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(cost_usd) as total_cost_usd,
    COUNT(*) FILTER (WHERE is_batch) as batch_requests
FROM api_usage
GROUP BY DATE(recorded_at), service
ORDER BY date DESC, service;

-- ============================================================================
-- DEFAULT DATA
-- ============================================================================

-- Insert default cron jobs (exactly 4)
INSERT INTO cronjobs (name, cron_expression, is_active, config) VALUES
    ('data_grab', '0 23 * * 1-5', TRUE, '{"description": "Fetch stock data from yfinance Mon-Fri 11pm"}'),
    ('batch_ai_swipe', '0 3 * * 0', TRUE, '{"description": "Generate swipe bios weekly Sunday 3am"}'),
    ('batch_ai_analysis', '0 4 * * 0', TRUE, '{"description": "Generate dip analysis weekly Sunday 4am"}'),
    ('cleanup', '0 0 * * *', TRUE, '{"description": "Clean up expired data daily midnight"}')
ON CONFLICT (name) DO NOTHING;

-- Create indexes for full-text search on stock names
CREATE INDEX idx_symbols_name_trgm ON symbols USING gin (name gin_trgm_ops);
CREATE INDEX idx_suggestions_name_trgm ON stock_suggestions USING gin (company_name gin_trgm_ops);

-- ============================================================================
-- DIP FINDER SIGNAL ENGINE
-- ============================================================================

-- Price history cache (cached yfinance price data)
CREATE TABLE IF NOT EXISTS price_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(16, 6),
    high DECIMAL(16, 6),
    low DECIMAL(16, 6),
    close DECIMAL(16, 6) NOT NULL,
    adj_close DECIMAL(16, 6),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE INDEX idx_price_history_symbol ON price_history(symbol);
CREATE INDEX idx_price_history_date ON price_history(date DESC);
CREATE INDEX idx_price_history_symbol_date ON price_history(symbol, date DESC);

-- DipFinder computed signals
CREATE TABLE IF NOT EXISTS dipfinder_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    benchmark VARCHAR(20) NOT NULL,
    window_days INTEGER NOT NULL,
    as_of_date DATE NOT NULL,
    
    -- Dip metrics
    dip_stock DECIMAL(8, 6),
    peak_stock DECIMAL(16, 6),
    dip_pctl DECIMAL(5, 2),
    dip_vs_typical DECIMAL(8, 4),
    persist_days INTEGER,
    
    -- Market context
    dip_mkt DECIMAL(8, 6),
    excess_dip DECIMAL(8, 6),
    dip_class VARCHAR(20),
    
    -- Scores
    quality_score DECIMAL(5, 2),
    stability_score DECIMAL(5, 2),
    dip_score DECIMAL(5, 2),
    final_score DECIMAL(5, 2),
    
    -- Alert
    alert_level VARCHAR(20),
    should_alert BOOLEAN DEFAULT FALSE,
    reason TEXT,
    
    -- Contributing factors (JSON)
    quality_factors JSONB,
    stability_factors JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    
    UNIQUE(ticker, benchmark, window_days, as_of_date)
);

CREATE INDEX idx_dipfinder_signals_ticker ON dipfinder_signals(ticker);
CREATE INDEX idx_dipfinder_signals_date ON dipfinder_signals(as_of_date DESC);
CREATE INDEX idx_dipfinder_signals_lookup ON dipfinder_signals(ticker, benchmark, window_days, as_of_date);
CREATE INDEX idx_dipfinder_signals_alert ON dipfinder_signals(should_alert) WHERE should_alert = TRUE;
CREATE INDEX idx_dipfinder_signals_final ON dipfinder_signals(final_score DESC);
CREATE INDEX idx_dipfinder_signals_expires ON dipfinder_signals(expires_at);

-- DipFinder configuration per symbol (optional overrides)
CREATE TABLE IF NOT EXISTS dipfinder_config (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    min_dip_abs DECIMAL(5, 4) DEFAULT 0.10,
    min_persist_days INTEGER DEFAULT 2,
    dip_percentile_threshold DECIMAL(5, 4) DEFAULT 0.80,
    dip_vs_typical_threshold DECIMAL(5, 4) DEFAULT 1.5,
    quality_gate DECIMAL(5, 2) DEFAULT 60,
    stability_gate DECIMAL(5, 2) DEFAULT 60,
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dipfinder_config_symbol ON dipfinder_config(symbol);
CREATE INDEX idx_dipfinder_config_enabled ON dipfinder_config(is_enabled) WHERE is_enabled = TRUE;

CREATE TRIGGER tr_dipfinder_config_updated
    BEFORE UPDATE ON dipfinder_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Dip state history (extended) with DipFinder fields
CREATE TABLE IF NOT EXISTS dipfinder_history (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    event_type VARCHAR(20) NOT NULL CHECK (event_type IN ('entered_dip', 'exited_dip', 'deepened', 'recovered', 'alert_triggered')),
    window_days INTEGER NOT NULL,
    dip_pct DECIMAL(8, 6),
    final_score DECIMAL(5, 2),
    dip_class VARCHAR(20),
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dipfinder_history_ticker ON dipfinder_history(ticker);
CREATE INDEX idx_dipfinder_history_event ON dipfinder_history(event_type);
CREATE INDEX idx_dipfinder_history_recorded ON dipfinder_history(recorded_at DESC);
CREATE INDEX idx_dipfinder_history_ticker_time ON dipfinder_history(ticker, recorded_at DESC);

-- YFinance info cache
CREATE TABLE IF NOT EXISTS yfinance_info_cache (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    info_data JSONB NOT NULL,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_yfinance_cache_symbol ON yfinance_info_cache(symbol);
CREATE INDEX idx_yfinance_cache_expires ON yfinance_info_cache(expires_at);

-- View for latest signals per ticker
CREATE OR REPLACE VIEW dipfinder_latest_signals AS
SELECT DISTINCT ON (ticker, benchmark, window_days)
    *
FROM dipfinder_signals
ORDER BY ticker, benchmark, window_days, as_of_date DESC;

-- View for active alerts
CREATE OR REPLACE VIEW dipfinder_active_alerts AS
SELECT 
    ds.*,
    s.name as company_name,
    s.sector,
    s.market_cap
FROM dipfinder_signals ds
JOIN symbols s ON ds.ticker = s.symbol
WHERE ds.should_alert = TRUE
  AND ds.as_of_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY ds.final_score DESC;
