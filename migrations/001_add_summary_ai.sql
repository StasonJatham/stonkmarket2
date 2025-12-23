-- Migration 001: Add summary_ai column to symbols table
-- AI-generated summaries for stock descriptions

ALTER TABLE symbols ADD COLUMN IF NOT EXISTS summary_ai VARCHAR(400);
