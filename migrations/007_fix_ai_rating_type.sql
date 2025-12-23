-- Migration: Fix ai_rating column type from DECIMAL to VARCHAR
-- The AI returns string ratings like 'buy', 'hold', 'sell' not numeric values

-- Drop the check constraint and index first
DROP INDEX IF EXISTS idx_dip_ai_analysis_rating;

-- Change column type from DECIMAL to VARCHAR with valid values
ALTER TABLE dip_ai_analysis 
    ALTER COLUMN ai_rating TYPE VARCHAR(20) USING NULL;

-- Add check constraint for valid rating values
ALTER TABLE dip_ai_analysis 
    ADD CONSTRAINT chk_ai_rating 
    CHECK (ai_rating IS NULL OR ai_rating IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell'));

-- Recreate index
CREATE INDEX idx_dip_ai_analysis_rating ON dip_ai_analysis(ai_rating);
