-- Increase summary_ai column size from 350 to 500 characters
ALTER TABLE symbols ALTER COLUMN summary_ai TYPE VARCHAR(500);
