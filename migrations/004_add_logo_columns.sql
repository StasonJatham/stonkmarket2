-- Migration: Add logo columns to symbols table
-- Stores cached WebP logos from Logo.dev API

ALTER TABLE symbols 
ADD COLUMN IF NOT EXISTS logo_light BYTEA,
ADD COLUMN IF NOT EXISTS logo_dark BYTEA,
ADD COLUMN IF NOT EXISTS logo_fetched_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS logo_source VARCHAR(50);

-- Add index for logo refresh queries
CREATE INDEX IF NOT EXISTS idx_symbols_logo_fetched_at ON symbols(logo_fetched_at);

COMMENT ON COLUMN symbols.logo_light IS 'WebP logo for light theme from Logo.dev';
COMMENT ON COLUMN symbols.logo_dark IS 'WebP logo for dark theme from Logo.dev';
COMMENT ON COLUMN symbols.logo_fetched_at IS 'Timestamp when logos were last fetched';
COMMENT ON COLUMN symbols.logo_source IS 'Source of logo: logo.dev or favicon';
