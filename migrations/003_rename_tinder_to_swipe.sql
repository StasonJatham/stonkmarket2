-- Migration 003: Rename tinder to swipe
-- Removes trademark references

-- Rename column in dip_ai_analysis
ALTER TABLE dip_ai_analysis 
RENAME COLUMN tinder_bio TO swipe_bio;

-- Drop and recreate the view with new naming
DROP VIEW IF EXISTS tinder_cards;
DROP VIEW IF EXISTS swipe_cards;

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
LEFT JOIN vote_counts v ON ds.symbol = v.symbol;

-- Also rename the cronjob if it exists (from batch_ai_tinder to batch_ai_swipe)
UPDATE cronjobs 
SET name = 'batch_ai_swipe',
    description = 'Generate swipe bios weekly Sunday 3am'
WHERE name = 'batch_ai_tinder';
