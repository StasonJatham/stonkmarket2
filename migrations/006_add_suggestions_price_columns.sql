-- Migration: Add price tracking columns to stock_suggestions
-- This adds current_price, ath_price, fetch_error, and fetched_at columns

-- Add new columns
ALTER TABLE stock_suggestions ADD COLUMN IF NOT EXISTS fetch_error TEXT;
ALTER TABLE stock_suggestions ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMPTZ;
ALTER TABLE stock_suggestions ADD COLUMN IF NOT EXISTS current_price NUMERIC(12, 4);
ALTER TABLE stock_suggestions ADD COLUMN IF NOT EXISTS ath_price NUMERIC(12, 4);
