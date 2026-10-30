-- Migration 001: Enable pgvector extension and add 384-dimensional embedding column to chunks table
-- Compatible with PostgreSQL 15+ and pgvector >= 0.5.0

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Add nullable embedding vector column to existing chunks table
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384);

-- 3. Comments for documentation and schema inspection
COMMENT ON COLUMN chunks.embedding IS 'BAAI/bge-small-en-v1.5 dense normalized embedding (dim=384)';
