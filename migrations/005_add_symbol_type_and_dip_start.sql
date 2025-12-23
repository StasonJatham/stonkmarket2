-- Migration 005: Add symbol_type column and dip_start_date to track when dips began
-- This allows distinguishing between stocks and indexes, and accurate dip duration tracking

-- Add symbol_type column to symbols table
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS symbol_type VARCHAR(20) DEFAULT 'stock';

-- Add index for symbol_type
CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type);

-- Add dip_start_date to dip_state to track when the stock entered the dip
-- This is more accurate than first_seen (which is when we added it to the table)
ALTER TABLE dip_state ADD COLUMN IF NOT EXISTS dip_start_date DATE;

-- Create index for dip_start_date
CREATE INDEX IF NOT EXISTS idx_dip_state_start_date ON dip_state(dip_start_date);
