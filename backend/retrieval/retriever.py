"""Evidence Retriever orchestrating document chunk indexing and claim evidence retrieval (Step 9)."""

from collections.abc import Sequence
from typing import Any

from backend.database.repository import ChunkRepository, DuplicateChunkError
from backend.retrieval.embeddings import EmbeddingService
from backend.retrieval.hybrid_search import HybridSearchEngine
from backend.schemas.chunk import DocumentChunk
from backend.schemas.claim import Claim
from backend.schemas.evidence import ClaimEvidenceResponse, EvidenceItem


class EvidenceRetrieverError(Exception):
    """Base exception for evidence retrieval and indexing errors."""

    pass


class EvidenceRetriever:
    """High-level facade orchestrating document chunk indexing and claim evidence retrieval.

    Coordinates:
    1. Chunk indexing: Generates dense embeddings via EmbeddingService, persists metadata
       and vectors via ChunkRepository without committing transactions.
    2. Evidence retrieval: Evaluates a Claim by delegating to HybridSearchEngine and
       maps fused results to standard EvidenceItem and ClaimEvidenceResponse schemas.

    Transaction Ownership:
        EvidenceRetriever operates strictly on the caller's shared SQLAlchemy session.
        It does NOT commit or roll back transactions. The transaction lifecycle must be
        managed by the caller using `get_db_session()` or equivalent session management.
    """

    def __init__(
        self,
        repository: ChunkRepository,
        embedding_service: EmbeddingService,
        hybrid_search: HybridSearchEngine,
    ) -> None:
        """Initialize EvidenceRetriever with required repository and search dependencies.

        Args:
            repository: ChunkRepository instance bound to the caller's active database session.
            embedding_service: EmbeddingService instance for document and query vectorization.
            hybrid_search: HybridSearchEngine instance for Reciprocal Rank Fusion retrieval.
        """
        self.repository = repository
        self.embedding_service = embedding_service
        self.hybrid_search = hybrid_search

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Index a batch of document chunks by generating embeddings and persisting records.

        Workflow:
        1. Validates input; returns 0 immediately if chunks sequence is empty.
        2. Fast-fails if input batch contains duplicate chunk IDs.
        3. Extracts raw chunk text without alteration.
        4. Generates document embeddings using EmbeddingService.embed_documents().
           Oversized passages propagate InputTooLongError.
        5. Verifies embedding count matches chunk count.
        6. Associates each chunk_id with its corresponding vector (as list of floats).
        7. Inserts chunk metadata into database via ChunkRepository.insert_chunks().
        8. Persists embedding vectors via ChunkRepository.update_embeddings().
        9. Verifies update count matches inserted count and returns the count of indexed chunks.

        Args:
            chunks: Sequence of DocumentChunk Pydantic models to index.

        Returns:
            The number of successfully indexed chunks (0 for empty input).

        Raises:
            InputTooLongError: If any chunk text exceeds the embedding model's maximum sequence length.
            DuplicateChunkError: If any chunk_id is duplicated in the batch or already in the database.
            ValueError: If embedding generation produces a count mismatch or invalid inputs.
            EvidenceRetrieverError: If database persistence counts mismatch or inference fails.
        """
        if not chunks:
            return 0

        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        # Fast-fail on intra-batch duplicate chunk IDs before generating embeddings
        incoming_ids = [c.chunk_id for c in chunk_list]
        if len(incoming_ids) != len(set(incoming_ids)):
            seen = set()
            duplicates = {x for x in incoming_ids if x in seen or seen.add(x)}
            raise DuplicateChunkError(
                f"Duplicate chunk IDs found within the input batch: {list(duplicates)}"
            )

        # Extract verbatim chunk texts
        texts = [chunk.text for chunk in chunk_list]

        # Generate dense embeddings prior to modifying database state
        vectors = self.embedding_service.embed_documents(texts)

        if len(vectors) != len(chunk_list):
            raise ValueError(
                f"Generated embedding count ({len(vectors)}) does not match input chunk count ({len(chunk_list)})."
            )

        # Associate each original chunk ID with its float vector
        embeddings_by_chunk_id: dict[str, list[float]] = {}
        for chunk, vec in zip(chunk_list, vectors):
            float_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            embeddings_by_chunk_id[chunk.chunk_id] = float_list

        # Persist chunk records and embeddings in the repository (flushes only; no commit)
        inserted_count = self.repository.insert_chunks(chunk_list)
        updated_count = self.repository.update_embeddings(embeddings_by_chunk_id)

        if updated_count != inserted_count:
            raise EvidenceRetrieverError(
                f"Embedding update count ({updated_count}) does not match inserted chunk count ({inserted_count})."
            )

        return inserted_count

    def retrieve_evidence(
        self,
        claim: Claim,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> ClaimEvidenceResponse:
        """Retrieve rank-fused evidence items supporting or refuting an atomic claim.

        Workflow:
        1. Validates claim type and positive top_k / candidate_k parameters.
        2. Passes claim_text and target paper_ids to HybridSearchEngine.
        3. Maps each HybridSearchResult to an EvidenceItem preserving verbatim text and metadata.
        4. Deduplicates evidence items by chunk_id.
        5. Wraps items in a ClaimEvidenceResponse preserving the original claim_id.

        Args:
            claim: Pydantic Claim instance containing claim_id, claim_text, and paper_ids.
            top_k: Maximum number of evidence items to return (must be > 0).
            candidate_k: Number of candidates to fetch from vector and lexical retrievers (must be > 0).

        Returns:
            ClaimEvidenceResponse containing the claim_id and list of EvidenceItem objects.

        Raises:
            ValueError: If claim is invalid, top_k <= 0, or candidate_k <= 0.
            HybridSearchError: If hybrid retrieval encounters a failure.
        """
        if not isinstance(claim, Claim):
            raise ValueError(f"Expected Claim instance, got {type(claim).__name__}")

        if isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if isinstance(candidate_k, bool) or candidate_k <= 0:
            raise ValueError(f"candidate_k must be a positive integer, got {candidate_k}.")

        # Retrieve fused candidates from hybrid search
        hybrid_results = self.hybrid_search.search(
            query=claim.claim_text,
            paper_ids=claim.paper_ids,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        # Map each HybridSearchResult into EvidenceItem, ensuring unique chunk IDs
        evidence_items: list[EvidenceItem] = []
        seen_chunk_ids: set[str] = set()

        for res in hybrid_results:
            cid = res.chunk.chunk_id
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)

            evidence_items.append(
                EvidenceItem(
                    evidence_id=cid,
                    chunk_id=cid,
                    paper_id=res.chunk.paper_id,
                    page_number=res.chunk.page_number,
                    paragraph_number=res.chunk.paragraph_number,
                    text=res.chunk.text,
                    retrieval_score=float(res.score),
                    retrieval_method="hybrid",
                )
            )

        return ClaimEvidenceResponse(
            claim_id=claim.claim_id,
            evidence=evidence_items,
        )
