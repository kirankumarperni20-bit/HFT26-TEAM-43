"""Vector search engine providing cosine similarity retrieval over chunk embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from backend.config import EMBEDDING_DIMENSION
from backend.database.models import ChunkRecord
from backend.schemas.chunk import DocumentChunk


@dataclass(frozen=True)
class VectorSearchResult:
    """Internal result container representing a ranked chunk with its cosine similarity score."""

    chunk: DocumentChunk
    score: float


class VectorSearchEngine:
    """Performs semantic vector search over document chunks using cosine similarity.

    Supports:
        - Production PostgreSQL retrieval using pgvector cosine distance (<=>).
        - Explicit SQLite test adapter computing exact cosine similarity via NumPy.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def validate_query_vector(
        query_vector: Sequence[float] | np.ndarray,
        expected_dim: int = EMBEDDING_DIMENSION,
    ) -> list[float]:
        """Validate that the query vector is 1D, has correct dimensions, and has finite values.

        Raises:
            ValueError: If the query vector is invalid, non-finite, or sized incorrectly.
        """
        arr = np.asarray(query_vector, dtype=np.float32)
        if arr.ndim != 1 or arr.shape != (expected_dim,):
            raise ValueError(
                f"Query vector must be 1-dimensional with shape ({expected_dim},); "
                f"got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Query vector contains NaN or infinite values.")
        return arr.tolist()

    @staticmethod
    def build_postgres_search_query(
        query_vector: list[float],
        paper_ids: Sequence[str],
        top_k: int = 20,
    ) -> Select[tuple[ChunkRecord, Any]]:
        """Construct a PostgreSQL pgvector cosine similarity select statement."""
        unique_paper_ids = list(dict.fromkeys(paper_ids))
        cos_distance = ChunkRecord.embedding.cosine_distance(query_vector)
        similarity = (1.0 - cos_distance).label("similarity")

        return (
            select(ChunkRecord, similarity)
            .where(ChunkRecord.paper_id.in_(unique_paper_ids))
            .where(ChunkRecord.embedding.is_not(None))
            .order_by(cos_distance.asc(), ChunkRecord.chunk_id.asc())
            .limit(top_k)
        )

    def search(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        paper_ids: Sequence[str],
        top_k: int = 20,
    ) -> list[VectorSearchResult]:
        """Retrieve candidate document chunks matching the query embedding using cosine similarity.

        Args:
            query_embedding: Pre-generated 384-dimensional query vector.
            paper_ids: List of paper IDs to restrict retrieval to.
            top_k: Maximum number of candidate chunks to return (must be > 0).

        Returns:
            Ranked list of VectorSearchResult objects ordered by similarity descending,
            with deterministic tie-breaking on chunk_id ascending.
            Returns an empty list immediately if paper_ids is empty.

        Raises:
            ValueError: If query_embedding is invalid or top_k <= 0.
        """
        if isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not paper_ids:
            return []

        q_list = self.validate_query_vector(query_embedding)
        unique_paper_ids = list(dict.fromkeys(paper_ids))

        # Check database dialect to determine retrieval path
        bind = self.session.get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""

        if dialect_name == "sqlite":
            return self._search_sqlite_adapter(q_list, unique_paper_ids, top_k)
        else:
            return self._search_postgres_pgvector(q_list, unique_paper_ids, top_k)

    def _search_postgres_pgvector(
        self,
        query_vector: list[float],
        paper_ids: list[str],
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Execute PostgreSQL pgvector search using the cosine distance operator."""
        stmt = self.build_postgres_search_query(query_vector, paper_ids, top_k)
        rows = self.session.execute(stmt).all()

        results: list[VectorSearchResult] = []
        for record, sim_score in rows:
            results.append(VectorSearchResult(chunk=record.to_pydantic(), score=float(sim_score)))
        return results

    def _search_sqlite_adapter(
        self,
        query_vector: list[float],
        paper_ids: list[str],
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Execute vector search against SQLite test database computing cosine similarity via NumPy.

        Note: Strictly a unit-testing adapter. Production PostgreSQL utilizes pgvector natively.
        """
        stmt = (
            select(ChunkRecord)
            .where(ChunkRecord.paper_id.in_(paper_ids))
            .where(ChunkRecord.embedding.is_not(None))
        )
        records = self.session.scalars(stmt).all()
        if not records:
            return []

        q_arr = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm == 0:
            q_norm = 1.0
        q_unit = q_arr / q_norm

        scored_candidates: list[tuple[float, str, DocumentChunk]] = []
        for record in records:
            if record.embedding is None:
                continue

            emb_arr = np.asarray(record.embedding, dtype=np.float32)
            if emb_arr.ndim != 1 or emb_arr.shape != (EMBEDDING_DIMENSION,):
                continue

            emb_norm = float(np.linalg.norm(emb_arr))
            if emb_norm == 0:
                emb_norm = 1.0
            emb_unit = emb_arr / emb_norm

            sim = float(np.dot(q_unit, emb_unit))
            scored_candidates.append((sim, record.chunk_id, record.to_pydantic()))

        # Sort by similarity descending, with deterministic tie-breaking on chunk_id ascending
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        return [
            VectorSearchResult(chunk=item[2], score=item[0])
            for item in scored_candidates[:top_k]
        ]
