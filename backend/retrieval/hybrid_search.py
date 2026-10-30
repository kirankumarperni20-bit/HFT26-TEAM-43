"""Hybrid retrieval engine combining vector and lexical search via Reciprocal Rank Fusion (RRF)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from backend.config import RRF_K
from backend.retrieval.embeddings import EmbeddingService
from backend.retrieval.lexical_search import LexicalSearchEngine
from backend.retrieval.vector_search import VectorSearchEngine
from backend.schemas.chunk import DocumentChunk


class HybridSearchError(Exception):
    """Base exception for hybrid retrieval errors."""

    pass


class HybridSearchInputError(HybridSearchError, ValueError):
    """Raised when hybrid search query or parameters are invalid."""

    pass


@dataclass(frozen=True)
class HybridSearchResult:
    """Fused candidate evidence record retaining original chunk metadata and retrieval ranks."""

    chunk: DocumentChunk
    score: float
    vector_rank: int | None
    lexical_rank: int | None


class HybridSearchEngine:
    """Orchestrates hybrid retrieval combining dense vector search and sparse BM25 lexical search.

    Combines candidate ranked lists using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum_{m in {vector, lexical}} (1.0 / (RRF_K + rank_m(d)))
    where rank positions are 1-indexed.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_search: VectorSearchEngine,
        lexical_search: LexicalSearchEngine,
        rrf_k: int = RRF_K,
    ) -> None:
        if isinstance(rrf_k, bool) or rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k}.")

        self.embedding_service = embedding_service
        self.vector_search = vector_search
        self.lexical_search = lexical_search
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        paper_ids: Sequence[str],
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[HybridSearchResult]:
        """Perform hybrid retrieval using Reciprocal Rank Fusion over selected paper IDs.

        Args:
            query: The atomic claim or search query string.
            paper_ids: Paper identifiers to restrict retrieval to.
            top_k: Maximum number of fused results to return (must be > 0).
            candidate_k: Number of candidates to retrieve from each retriever (must be > 0).

        Returns:
            A list of HybridSearchResult items ordered by RRF score descending,
            with deterministic tie-breaking on chunk_id ascending.

        Raises:
            HybridSearchInputError: If query is empty or whitespace-only.
            ValueError: If top_k <= 0 or candidate_k <= 0.
        """
        if not isinstance(query, str) or not query.strip():
            raise HybridSearchInputError("Query cannot be empty or whitespace-only.")

        if isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if isinstance(candidate_k, bool) or candidate_k <= 0:
            raise ValueError(f"candidate_k must be a positive integer, got {candidate_k}.")

        if not paper_ids:
            return []

        unique_paper_ids = list(dict.fromkeys(paper_ids))

        # 1. Generate query embedding (propagates InputTooLongError if query exceeds limit)
        query_vector = self.embedding_service.embed_query(query)

        # 2. Execute vector and lexical search sequentially to protect session thread-safety
        vector_results = self.vector_search.search(
            query_embedding=query_vector,
            paper_ids=unique_paper_ids,
            top_k=candidate_k,
        )

        lexical_results = self.lexical_search.search(
            query=query,
            paper_ids=unique_paper_ids,
            top_k=candidate_k,
        )

        # 3. Fuse candidate rankings using Reciprocal Rank Fusion
        fused: dict[str, dict[str, Any]] = {}

        # Accumulate vector ranks (1-indexed)
        seen_vector_chunks: set[str] = set()
        v_rank = 1
        for v_res in vector_results:
            cid = v_res.chunk.chunk_id
            if cid in seen_vector_chunks:
                continue
            seen_vector_chunks.add(cid)

            v_score = 1.0 / (self.rrf_k + v_rank)
            fused[cid] = {
                "chunk": v_res.chunk,
                "score": v_score,
                "vector_rank": v_rank,
                "lexical_rank": None,
            }
            v_rank += 1

        # Accumulate lexical ranks (1-indexed)
        seen_lexical_chunks: set[str] = set()
        l_rank = 1
        for l_res in lexical_results:
            cid = l_res.chunk.chunk_id
            if cid in seen_lexical_chunks:
                continue
            seen_lexical_chunks.add(cid)

            l_score = 1.0 / (self.rrf_k + l_rank)
            if cid in fused:
                if fused[cid]["chunk"] != l_res.chunk:
                    raise HybridSearchError(
                        f"Conflicting chunk metadata detected for chunk ID '{cid}' between vector and lexical retrievers."
                    )
                fused[cid]["score"] += l_score
                fused[cid]["lexical_rank"] = l_rank
            else:
                fused[cid] = {
                    "chunk": l_res.chunk,
                    "score": l_score,
                    "vector_rank": None,
                    "lexical_rank": l_rank,
                }
            l_rank += 1

        if not fused:
            return []

        # Sort by combined RRF score descending, tie-breaking by chunk_id ascending
        ranked_candidates = list(fused.values())
        ranked_candidates.sort(key=lambda item: (-item["score"], item["chunk"].chunk_id))

        return [
            HybridSearchResult(
                chunk=item["chunk"],
                score=item["score"],
                vector_rank=item["vector_rank"],
                lexical_rank=item["lexical_rank"],
            )
            for item in ranked_candidates[:top_k]
        ]
