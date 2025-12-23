-- Migration: Add fetch_status to symbols table for loading state feedback
-- Shows loading indicator when data is being enriched for a new symbol

ALTER TABLE symbols ADD COLUMN IF NOT EXISTS fetch_status VARCHAR(20) DEFAULT 'pending';
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS fetch_error TEXT;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMPTZ;

-- Valid statuses: pending, fetching, fetched, error
COMMENT ON COLUMN symbols.fetch_status IS 'pending = awaiting data, fetching = in progress, fetched = complete, error = failed';
