-- Migration: Add stock fundamentals table
-- This stores financial metrics from Yahoo Finance that update monthly
-- Prices update daily, but fundamentals are more stable

-- ============================================================================
-- STOCK FUNDAMENTALS TABLE
-- ============================================================================
-- Stores key financial metrics for stocks (not ETFs/indexes)
-- Refreshed monthly or on-demand

CREATE TABLE IF NOT EXISTS stock_fundamentals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL REFERENCES symbols(symbol) ON DELETE CASCADE,
    
    -- Valuation Metrics
    pe_ratio DECIMAL(10, 4),              -- Trailing P/E
    forward_pe DECIMAL(10, 4),            -- Forward P/E
    peg_ratio DECIMAL(10, 4),             -- P/E to Growth ratio
    price_to_book DECIMAL(10, 4),         -- Price/Book
    price_to_sales DECIMAL(10, 4),        -- Price/Sales TTM
    enterprise_value BIGINT,              -- Enterprise Value
    ev_to_ebitda DECIMAL(10, 4),          -- EV/EBITDA
    ev_to_revenue DECIMAL(10, 4),         -- EV/Revenue
    
    -- Profitability Metrics (stored as decimals, e.g., 0.27 = 27%)
    profit_margin DECIMAL(8, 6),          -- Net profit margin
    operating_margin DECIMAL(8, 6),       -- Operating margin
    gross_margin DECIMAL(8, 6),           -- Gross margin
    ebitda_margin DECIMAL(8, 6),          -- EBITDA margin
    return_on_equity DECIMAL(8, 6),       -- ROE
    return_on_assets DECIMAL(8, 6),       -- ROA
    
    -- Financial Health
    debt_to_equity DECIMAL(10, 4),        -- D/E ratio
    current_ratio DECIMAL(8, 4),          -- Current ratio
    quick_ratio DECIMAL(8, 4),            -- Quick ratio
    total_cash BIGINT,                    -- Cash and equivalents
    total_debt BIGINT,                    -- Total debt
    free_cash_flow BIGINT,                -- Free cash flow
    operating_cash_flow BIGINT,           -- Operating cash flow
    
    -- Per Share Data
    book_value DECIMAL(10, 4),            -- Book value per share
    eps_trailing DECIMAL(10, 4),          -- Trailing EPS
    eps_forward DECIMAL(10, 4),           -- Forward EPS
    revenue_per_share DECIMAL(10, 4),     -- Revenue per share
    
    -- Growth Metrics (stored as decimals)
    revenue_growth DECIMAL(8, 6),         -- YoY revenue growth
    earnings_growth DECIMAL(8, 6),        -- YoY earnings growth
    earnings_quarterly_growth DECIMAL(8, 6), -- QoQ earnings growth
    
    -- Shares & Ownership
    shares_outstanding BIGINT,
    float_shares BIGINT,
    held_percent_insiders DECIMAL(8, 6),
    held_percent_institutions DECIMAL(8, 6),
    short_ratio DECIMAL(8, 4),
    short_percent_of_float DECIMAL(8, 6),
    
    -- Risk & Volatility
    beta DECIMAL(6, 4),
    
    -- Analyst Ratings
    recommendation VARCHAR(20),           -- 'buy', 'hold', 'sell', etc.
    recommendation_mean DECIMAL(4, 2),    -- 1.0 (strong buy) to 5.0 (strong sell)
    num_analyst_opinions INTEGER,
    target_high_price DECIMAL(12, 4),
    target_low_price DECIMAL(12, 4),
    target_mean_price DECIMAL(12, 4),
    target_median_price DECIMAL(12, 4),
    
    -- Revenue & Earnings (for context in AI analysis)
    revenue BIGINT,                       -- Total revenue TTM
    ebitda BIGINT,                        -- EBITDA
    net_income BIGINT,                    -- Net income
    
    -- Earnings Calendar
    next_earnings_date DATE,
    earnings_estimate_high DECIMAL(10, 4),
    earnings_estimate_low DECIMAL(10, 4),
    earnings_estimate_avg DECIMAL(10, 4),
    
    -- Metadata
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '30 days',
    
    UNIQUE(symbol)
);

CREATE INDEX idx_fundamentals_symbol ON stock_fundamentals(symbol);
CREATE INDEX idx_fundamentals_pe ON stock_fundamentals(pe_ratio);
CREATE INDEX idx_fundamentals_expires ON stock_fundamentals(expires_at);
CREATE INDEX idx_fundamentals_recommendation ON stock_fundamentals(recommendation);

-- Add pe_ratio to symbols table for quick access (denormalized for ranking queries)
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS pe_ratio DECIMAL(10, 4);

COMMENT ON TABLE stock_fundamentals IS 'Stock fundamental metrics from Yahoo Finance, refreshed monthly';
COMMENT ON COLUMN stock_fundamentals.recommendation_mean IS '1.0 = Strong Buy, 2.0 = Buy, 3.0 = Hold, 4.0 = Sell, 5.0 = Strong Sell';
