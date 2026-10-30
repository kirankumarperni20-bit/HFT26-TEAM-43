"""Centralized configuration for retrieval, embeddings, and database layers."""

# Canonical embedding model settings for BAAI/bge-small-en-v1.5
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384
MAX_SEQUENCE_LENGTH: int = 512
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

# Okapi BM25 Lexical parameters
BM25_K1: float = 1.5
BM25_B: float = 0.75

# Reciprocal Rank Fusion (RRF) parameters
RRF_K: int = 60
