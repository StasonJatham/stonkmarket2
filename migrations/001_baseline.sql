-- Migration 001: Baseline
-- This migration exists to mark fresh databases as fully migrated.
-- All schema is defined in init.sql - no changes needed here.
-- Future migrations will add new features incrementally.

SELECT 'Baseline migration - schema created by init.sql' AS status;
