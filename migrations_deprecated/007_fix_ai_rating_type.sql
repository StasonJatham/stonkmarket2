-- Migration: Fix ai_rating column type from DECIMAL to VARCHAR
-- The AI returns string ratings like 'buy', 'hold', 'sell' not numeric values

-- Drop views that depend on ai_rating column first
DROP VIEW IF EXISTS swipe_cards;

-- Drop the check constraint and index first
DROP INDEX IF EXISTS idx_dip_ai_analysis_rating;

-- Change column type from DECIMAL to VARCHAR with valid values
ALTER TABLE dip_ai_analysis 
    ALTER COLUMN ai_rating TYPE VARCHAR(20) USING NULL;

-- Add check constraint for valid rating values (if it doesn't exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_ai_rating'
    ) THEN
        ALTER TABLE dip_ai_analysis 
            ADD CONSTRAINT chk_ai_rating 
            CHECK (ai_rating IS NULL OR ai_rating IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell'));
    END IF;
END $$;

-- Recreate index
CREATE INDEX IF NOT EXISTS idx_dip_ai_analysis_rating ON dip_ai_analysis(ai_rating);

-- Recreate the swipe_cards view
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
