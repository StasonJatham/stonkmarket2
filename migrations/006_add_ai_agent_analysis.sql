-- Migration: Add AI agent analysis table
-- Store analysis from AI investor personas (Warren Buffett, Peter Lynch, etc.)

CREATE TABLE IF NOT EXISTS ai_agent_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    -- Verdicts from individual agents stored as JSONB array
    verdicts JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Aggregated signal
    overall_signal VARCHAR(20) NOT NULL CHECK (overall_signal IN ('bullish', 'bearish', 'neutral')),
    overall_confidence INTEGER NOT NULL CHECK (overall_confidence >= 0 AND overall_confidence <= 100),
    summary TEXT,
    -- Timestamps
    analyzed_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_ai_agent_analysis_symbol ON ai_agent_analysis(symbol);
CREATE INDEX IF NOT EXISTS idx_ai_agent_analysis_expires ON ai_agent_analysis(expires_at);
CREATE INDEX IF NOT EXISTS idx_ai_agent_analysis_signal ON ai_agent_analysis(overall_signal);

-- Comments
COMMENT ON TABLE ai_agent_analysis IS 'AI agent analysis results per stock (Warren Buffett, Peter Lynch, etc.)';
COMMENT ON COLUMN ai_agent_analysis.verdicts IS 'JSON array of agent verdicts: [{agent_id, agent_name, signal, confidence, reasoning, key_factors}]';
COMMENT ON COLUMN ai_agent_analysis.overall_signal IS 'Aggregated signal: bullish, bearish, or neutral';
COMMENT ON COLUMN ai_agent_analysis.overall_confidence IS 'Confidence level 0-100 based on agent agreement';
