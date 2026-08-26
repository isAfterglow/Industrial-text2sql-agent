-- Production migration contract for the long-term memory repository.
-- Enable pgvector before applying this file:
-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS long_term_memories (
  memory_id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  memory_type TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  source TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  schema_hash TEXT NOT NULL DEFAULT '',
  embedding vector(1024),
  embedding_model TEXT NOT NULL DEFAULT '',
  dedupe_key TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(namespace, memory_type, dedupe_key)
);
CREATE INDEX IF NOT EXISTS idx_ltm_namespace_type ON long_term_memories(namespace, memory_type, is_active);
CREATE INDEX IF NOT EXISTS idx_ltm_schema ON long_term_memories(namespace, schema_hash, is_active);
CREATE INDEX IF NOT EXISTS idx_ltm_embedding ON long_term_memories USING hnsw (embedding vector_cosine_ops);

