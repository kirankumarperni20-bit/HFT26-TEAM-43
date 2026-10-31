"""Retrieval package containing embedding service, search algorithms, and orchestrator."""

from backend.retrieval.embeddings import (
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingService,
    InputTooLongError,
    ModelDimensionMismatchError,
    ModelLoadError,
)
from backend.retrieval.hybrid_search import (
    HybridSearchError,
    HybridSearchInputError,
    HybridSearchEngine,
    HybridSearchResult,
)
from backend.retrieval.lexical_search import (
    LexicalInputError,
    LexicalSearchEngine,
    LexicalSearchError,
    LexicalSearchResult,
    tokenize_text,
)
from backend.retrieval.ingest import ingest_pdf, parse_pdf_to_chunks
from backend.retrieval.retriever import EvidenceRetriever, EvidenceRetrieverError
from backend.retrieval.vector_search import VectorSearchEngine, VectorSearchResult

__all__ = [
    "EmbeddingService",
    "EmbeddingError",
    "InputTooLongError",
    "ModelLoadError",
    "ModelDimensionMismatchError",
    "EmbeddingInputError",
    "VectorSearchEngine",
    "VectorSearchResult",
    "LexicalSearchEngine",
    "LexicalSearchResult",
    "LexicalSearchError",
    "LexicalInputError",
    "tokenize_text",
    "HybridSearchEngine",
    "HybridSearchResult",
    "HybridSearchError",
    "HybridSearchInputError",
    "EvidenceRetriever",
    "EvidenceRetrieverError",
    "parse_pdf_to_chunks",
    "ingest_pdf",
]

