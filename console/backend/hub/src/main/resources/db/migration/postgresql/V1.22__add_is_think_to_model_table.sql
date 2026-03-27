-- Migration script to add is_think column to model table
ALTER TABLE model ADD COLUMN IF NOT EXISTS is_think SMALLINT NOT NULL DEFAULT 0;
