"""Unit and integration tests for EvidenceRetriever (Step 9)."""

from typing import Any
import numpy as np
import pytest
from sqlalchemy.orm import Session

from backend.config import EMBEDDING_DIMENSION
from backend.database.connection import create_db_engine, get_session_factory
from backend.database.models import Base, ChunkRecord
from backend.database.repository import ChunkRepository, DuplicateChunkError
from backend.retrieval.embeddings import InputTooLongError
from backend.retrieval.hybrid_search import HybridSearchError, HybridSearchResult
from backend.retrieval.retriever import EvidenceRetriever, EvidenceRetrieverError
from backend.schemas.chunk import DocumentChunk
from backend.schemas.claim import Claim
from backend.schemas.evidence import ClaimEvidenceResponse, EvidenceItem


def make_dummy_chunk(
    chunk_id: str,
    paper_id: str = "P001",
    page_number: int = 1,
    paragraph_number: int = 1,
    section: str | None = "Results",
    text: str = "Sample research chunk text.",
) -> DocumentChunk:
    """Helper to construct a valid DocumentChunk."""
    return DocumentChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        page_number=page_number,
        paragraph_number=paragraph_number,
        section=section,
        text=text,
    )


def make_dummy_claim(
    claim_id: str = "C001",
    claim_text: str = "Transformer self-attention scales quadratically with sequence length.",
    paper_ids: list[str] | None = None,
) -> Claim:
    """Helper to construct a valid Claim."""
    return Claim(
        claim_id=claim_id,
        claim_text=claim_text,
        paper_ids=paper_ids or ["P001"],
    )


class MockEmbeddingService:
    """Mock EmbeddingService for controlling embedding outputs without downloading weights."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.last_texts: list[str] | None = None
        self.fail_with: Exception | None = None
        self.return_mismatch: bool = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.last_texts = list(texts)
        if self.fail_with:
            raise self.fail_with

        if self.return_mismatch:
            return np.ones((max(0, len(texts) - 1), self.dimension), dtype=np.float32)

        arr = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i in range(len(texts)):
            arr[i, i % self.dimension] = 1.0
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        if self.fail_with:
            raise self.fail_with
        arr = np.zeros(self.dimension, dtype=np.float32)
        arr[0] = 1.0
        return arr


class MockHybridSearchEngine:
    """Mock HybridSearchEngine for controlling hybrid search results."""

    def __init__(self) -> None:
        self.results: list[HybridSearchResult] = []
        self.last_query: str | None = None
        self.last_paper_ids: list[str] | None = None
        self.last_top_k: int | None = None
        self.last_candidate_k: int | None = None
        self.fail_with: Exception | None = None

    def search(
        self,
        query: str,
        paper_ids: list[str],
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[HybridSearchResult]:
        self.last_query = query
        self.last_paper_ids = list(paper_ids)
        self.last_top_k = top_k
        self.last_candidate_k = candidate_k

        if self.fail_with:
            raise self.fail_with

        return self.results


@pytest.fixture
def sqlite_engine():
    """Create a temporary in-memory SQLite engine with tables for retriever tests."""
    engine = create_db_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine):
    """Yield a transactional session connected to the temporary SQLite test database."""
    session_factory = get_session_factory(sqlite_engine)
    session: Session = session_factory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def repository(db_session):
    return ChunkRepository(session=db_session)


@pytest.fixture
def mock_embedding():
    return MockEmbeddingService()


@pytest.fixture
def mock_hybrid():
    return MockHybridSearchEngine()


@pytest.fixture
def retriever(repository, mock_embedding, mock_hybrid):
    return EvidenceRetriever(
        repository=repository,
        embedding_service=mock_embedding,  # type: ignore[arg-type]
        hybrid_search=mock_hybrid,        # type: ignore[arg-type]
    )


# ==============================================================================
# Indexing Unit Tests (1 - 11)
# ==============================================================================


def test_empty_chunks_return_zero(retriever, mock_embedding):
    """1. Verify indexing an empty chunk sequence returns zero immediately."""
    result = retriever.index_chunks([])
    assert result == 0
    assert mock_embedding.last_texts is None


def test_valid_chunks_generate_embeddings(retriever, mock_embedding):
    """2. Verify valid chunks trigger document embedding generation with their text."""
    chunks = [
        make_dummy_chunk("P001_C001", text="First passage on neural attention mechanisms."),
        make_dummy_chunk("P001_C002", text="Second passage describing multi-head projection."),
    ]
    count = retriever.index_chunks(chunks)
    assert count == 2
    assert mock_embedding.last_texts == [
        "First passage on neural attention mechanisms.",
        "Second passage describing multi-head projection.",
    ]


def test_original_chunk_ids_map_to_correct_vectors(retriever, db_session):
    """3. Verify each chunk ID in the database is associated with its correct vector."""
    chunks = [
        make_dummy_chunk("P001_C001", text="Text 1"),
        make_dummy_chunk("P001_C002", text="Text 2"),
    ]
    retriever.index_chunks(chunks)

    r1 = db_session.get(ChunkRecord, "P001_C001")
    r2 = db_session.get(ChunkRecord, "P001_C002")
    assert r1 is not None and r2 is not None

    # Check vector values match mock's formula (index 0 gets dim 0 = 1.0, index 1 gets dim 1 = 1.0)
    assert r1.embedding[0] == 1.0
    assert r1.embedding[1] == 0.0
    assert r2.embedding[1] == 1.0
    assert r2.embedding[0] == 0.0


def test_original_text_remains_unchanged(retriever, db_session):
    """4. Verify raw extracted chunk text is stored verbatim without stripping or alterations."""
    raw_text = "\n\t  Empirical accuracy: 94.2% ± 0.3% & non-ASCII: λ, μ \n"
    chunk = make_dummy_chunk("P001_EXACT", text=raw_text)
    retriever.index_chunks([chunk])

    record = db_session.get(ChunkRecord, "P001_EXACT")
    assert record is not None
    assert record.text == raw_text


def test_multiple_chunks_indexed_correctly(retriever, db_session):
    """5. Verify multiple chunks spanning multiple papers are all persisted with embeddings."""
    chunks = [
        make_dummy_chunk("P001_C001", paper_id="P001", text="P1 passage 1"),
        make_dummy_chunk("P001_C002", paper_id="P001", text="P1 passage 2"),
        make_dummy_chunk("P002_C001", paper_id="P002", text="P2 passage 1"),
    ]
    count = retriever.index_chunks(chunks)
    assert count == 3

    for c in chunks:
        record = db_session.get(ChunkRecord, c.chunk_id)
        assert record is not None
        assert record.paper_id == c.paper_id
        assert record.text == c.text
        assert record.embedding is not None


def test_existing_repository_methods_reused(retriever, monkeypatch):
    """6. Verify EvidenceRetriever delegates persistence directly to ChunkRepository methods."""
    called_methods: list[str] = []

    orig_insert = retriever.repository.insert_chunks
    orig_update = retriever.repository.update_embeddings

    def spy_insert(chunks):
        called_methods.append("insert_chunks")
        return orig_insert(chunks)

    def spy_update(mapping):
        called_methods.append("update_embeddings")
        return orig_update(mapping)

    monkeypatch.setattr(retriever.repository, "insert_chunks", spy_insert)
    monkeypatch.setattr(retriever.repository, "update_embeddings", spy_update)

    retriever.index_chunks([make_dummy_chunk("P001_C001")])
    assert called_methods == ["insert_chunks", "update_embeddings"]


def test_oversized_chunks_propagate_input_too_long_error(retriever, mock_embedding, db_session):
    """7. Verify oversized chunk text propagates InputTooLongError without partial persistence."""
    mock_embedding.fail_with = InputTooLongError(position=0, token_count=650, max_limit=512)

    with pytest.raises(InputTooLongError, match="exceeds model maximum token limit"):
        retriever.index_chunks([make_dummy_chunk("P001_LONG", text="Excessively long text...")])

    assert db_session.get(ChunkRecord, "P001_LONG") is None


def test_duplicate_ids_raise_repository_error(retriever):
    """8. Verify duplicate chunk IDs in batch or database propagate DuplicateChunkError."""
    chunk = make_dummy_chunk("P001_DUPE")
    # Intra-batch duplicate
    with pytest.raises(DuplicateChunkError):
        retriever.index_chunks([chunk, chunk])

    # Pre-existing duplicate
    retriever.index_chunks([chunk])
    with pytest.raises(DuplicateChunkError):
        retriever.index_chunks([chunk])


def test_embedding_count_mismatch_rejected(retriever, mock_embedding):
    """9. Verify an embedding-count mismatch from the embedding service is rejected."""
    mock_embedding.return_mismatch = True
    chunks = [
        make_dummy_chunk("P001_C001"),
        make_dummy_chunk("P001_C002"),
    ]
    with pytest.raises(ValueError, match="does not match input chunk count"):
        retriever.index_chunks(chunks)


def test_index_chunks_does_not_commit_transaction(retriever, db_session):
    """10. Verify index_chunks does not commit, leaving transaction lifecycle to caller."""
    retriever.index_chunks([make_dummy_chunk("P001_C001")])

    # Transaction is still open/pending; rollback should revert all changes
    db_session.rollback()
    assert db_session.get(ChunkRecord, "P001_C001") is None


def test_no_commit_if_embedding_generation_fails(retriever, mock_embedding, db_session):
    """11. Verify failure during embedding generation leaves database untouched."""
    mock_embedding.fail_with = RuntimeError("Embedding service unavailable.")
    chunk = make_dummy_chunk("P001_FAIL")

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        retriever.index_chunks([chunk])

    assert db_session.get(ChunkRecord, "P001_FAIL") is None


# ==============================================================================
# Evidence Retrieval Unit Tests (12 - 24)
# ==============================================================================


def test_valid_claim_calls_hybrid_search_with_correct_query(retriever, mock_hybrid):
    """12. Verify retrieve_evidence passes claim_text to HybridSearchEngine."""
    claim = make_dummy_claim(claim_text="Residual connections prevent vanishing gradients.")
    retriever.retrieve_evidence(claim)

    assert mock_hybrid.last_query == "Residual connections prevent vanishing gradients."


def test_only_requested_paper_ids_passed(retriever, mock_hybrid):
    """13. Verify exactly the claim's paper_ids are forwarded to hybrid retrieval."""
    claim = make_dummy_claim(paper_ids=["P001", "P002"])
    retriever.retrieve_evidence(claim)

    assert mock_hybrid.last_paper_ids == ["P001", "P002"]


def test_top_k_and_candidate_k_passed_correctly(retriever, mock_hybrid):
    """14. Verify top_k and candidate_k parameters are passed through to HybridSearchEngine."""
    claim = make_dummy_claim()
    retriever.retrieve_evidence(claim, top_k=8, candidate_k=25)

    assert mock_hybrid.last_top_k == 8
    assert mock_hybrid.last_candidate_k == 25


def test_evidence_item_fields_mapped_correctly(retriever, mock_hybrid):
    """15. Verify all fields from HybridSearchResult map accurately to EvidenceItem."""
    chunk = DocumentChunk(
        chunk_id="P001_C042",
        paper_id="P001",
        page_number=5,
        paragraph_number=3,
        section="Experiments",
        text="The proposed model achieved 89.2% BLEU score on WMT 2014.",
    )
    mock_hybrid.results = [
        HybridSearchResult(chunk=chunk, score=0.032258, vector_rank=1, lexical_rank=2)
    ]

    claim = make_dummy_claim()
    response = retriever.retrieve_evidence(claim)

    assert len(response.evidence) == 1
    item = response.evidence[0]
    assert item.evidence_id == "P001_C042"
    assert item.chunk_id == "P001_C042"
    assert item.paper_id == "P001"
    assert item.page_number == 5
    assert item.paragraph_number == 3
    assert item.text == chunk.text
    assert item.retrieval_score == pytest.approx(0.032258)
    assert item.retrieval_method == "hybrid"


def test_evidence_id_equals_chunk_id(retriever, mock_hybrid):
    """16. Verify the provisional contract evidence_id == chunk_id is strictly upheld."""
    chunk = make_dummy_chunk("P009_C999")
    mock_hybrid.results = [
        HybridSearchResult(chunk=chunk, score=0.016, vector_rank=1, lexical_rank=None)
    ]

    response = retriever.retrieve_evidence(make_dummy_claim())
    assert response.evidence[0].evidence_id == "P009_C999"
    assert response.evidence[0].evidence_id == response.evidence[0].chunk_id


def test_rrf_score_preserved_as_retrieval_score(retriever, mock_hybrid):
    """17. Verify the RRF score is preserved as a float without alteration."""
    chunk = make_dummy_chunk("P001_C001")
    exact_score = 0.03278688524590164
    mock_hybrid.results = [
        HybridSearchResult(chunk=chunk, score=exact_score, vector_rank=1, lexical_rank=1)
    ]

    response = retriever.retrieve_evidence(make_dummy_claim())
    assert response.evidence[0].retrieval_score == exact_score


def test_retrieval_method_is_hybrid(retriever, mock_hybrid):
    """18. Verify retrieval_method on retrieved EvidenceItem is 'hybrid'."""
    chunk = make_dummy_chunk("P001_C001")
    mock_hybrid.results = [
        HybridSearchResult(chunk=chunk, score=0.02, vector_rank=2, lexical_rank=3)
    ]

    response = retriever.retrieve_evidence(make_dummy_claim())
    assert response.evidence[0].retrieval_method == "hybrid"


def test_original_passage_text_and_metadata_preserved(retriever, mock_hybrid):
    """19. Verify original chunk passage text with special formatting is preserved verbatim."""
    special_text = "\n\t  Results (Table 3): Precision = 91.5% ± 0.4% \n"
    chunk = DocumentChunk(
        chunk_id="P001_PRESERVE",
        paper_id="P001",
        page_number=12,
        paragraph_number=4,
        section="Ablation Study",
        text=special_text,
    )
    mock_hybrid.results = [
        HybridSearchResult(chunk=chunk, score=0.025, vector_rank=1, lexical_rank=4)
    ]

    response = retriever.retrieve_evidence(make_dummy_claim())
    item = response.evidence[0]
    assert item.text == special_text
    assert item.page_number == 12
    assert item.paragraph_number == 4


def test_duplicate_evidence_ids_not_returned(retriever, mock_hybrid):
    """20. Verify that even if candidate list has duplicate chunk IDs, evidence is deduplicated."""
    chunk = make_dummy_chunk("P001_DUPE_CHUNK")
    mock_hybrid.results = [
        HybridSearchResult(chunk=chunk, score=0.03, vector_rank=1, lexical_rank=2),
        HybridSearchResult(chunk=chunk, score=0.01, vector_rank=5, lexical_rank=None),
    ]

    response = retriever.retrieve_evidence(make_dummy_claim())
    assert len(response.evidence) == 1
    assert response.evidence[0].evidence_id == "P001_DUPE_CHUNK"


def test_empty_hybrid_results_produce_empty_evidence_list(retriever, mock_hybrid):
    """21. Verify empty hybrid search result produces ClaimEvidenceResponse with empty evidence."""
    mock_hybrid.results = []
    response = retriever.retrieve_evidence(make_dummy_claim(claim_id="C_EMPTY"))

    assert response.claim_id == "C_EMPTY"
    assert response.evidence == []


def test_claim_id_preserved(retriever, mock_hybrid):
    """22. Verify input Claim.claim_id is preserved in ClaimEvidenceResponse."""
    claim = make_dummy_claim(claim_id="CLAIM_UNIQUE_42")
    response = retriever.retrieve_evidence(claim)

    assert response.claim_id == "CLAIM_UNIQUE_42"


def test_hybrid_retrieval_errors_propagated(retriever, mock_hybrid):
    """23. Verify failures in hybrid retrieval propagate without suppression."""
    mock_hybrid.fail_with = HybridSearchError("Retrieval backend failure.")

    with pytest.raises(HybridSearchError, match="Retrieval backend failure"):
        retriever.retrieve_evidence(make_dummy_claim())


def test_invalid_top_k_and_candidate_k_rejected(retriever):
    """24. Verify non-positive top_k and candidate_k parameters raise ValueError."""
    claim = make_dummy_claim()

    for k in [0, -1, -5]:
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retriever.retrieve_evidence(claim, top_k=k)

    for k in [0, -1, -5]:
        with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
            retriever.retrieve_evidence(claim, candidate_k=k)

    with pytest.raises(ValueError, match="Expected Claim instance"):
        retriever.retrieve_evidence("invalid query string")  # type: ignore[arg-type]


# ==============================================================================
# End-to-End Integration Test
# ==============================================================================


def test_end_to_end_indexing_and_evidence_retrieval(repository, db_session):
    """End-to-end integration: Index real DocumentChunks into SQLite, then retrieve evidence for a Claim."""
    # 1. Setup realistic chunks
    c1 = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=2,
        section="Abstract",
        text="Dense retrieval using dual encoders outperforms classical keyword search on open-domain QA.",
    )
    c2 = DocumentChunk(
        chunk_id="P001_C002",
        paper_id="P001",
        page_number=3,
        paragraph_number=1,
        section="Methodology",
        text="BM25 scoring leverages term frequency and inverse document frequency with length normalization.",
    )
    c3 = DocumentChunk(
        chunk_id="P002_C001",
        paper_id="P002",
        page_number=2,
        paragraph_number=4,
        section="Results",
        text="Reciprocal Rank Fusion effectively combines ranked lists from dense and sparse retrieval models.",
    )

    mock_emb = MockEmbeddingService()
    mock_hyb = MockHybridSearchEngine()

    retriever = EvidenceRetriever(
        repository=repository,
        embedding_service=mock_emb,  # type: ignore[arg-type]
        hybrid_search=mock_hyb,      # type: ignore[arg-type]
    )

    # 2. Index chunks
    indexed_count = retriever.index_chunks([c1, c2, c3])
    assert indexed_count == 3

    # Verify chunks are stored in SQLite
    stored_c1 = db_session.get(ChunkRecord, "P001_C001")
    assert stored_c1 is not None
    assert stored_c1.text == c1.text
    assert len(stored_c1.embedding) == EMBEDDING_DIMENSION

    # 3. Configure hybrid search results simulating search for RRF
    mock_hyb.results = [
        HybridSearchResult(chunk=c3, score=(1.0 / 61 + 1.0 / 62), vector_rank=1, lexical_rank=2),
        HybridSearchResult(chunk=c1, score=(1.0 / 62), vector_rank=2, lexical_rank=None),
    ]

    # 4. Retrieve evidence for a Claim
    claim = Claim(
        claim_id="CLAIM_E2E_01",
        claim_text="Hybrid retrieval combining dense vectors and BM25 improves QA accuracy.",
        paper_ids=["P001", "P002"],
    )

    response = retriever.retrieve_evidence(claim, top_k=5, candidate_k=20)

    # 5. Verify ClaimEvidenceResponse contract
    assert isinstance(response, ClaimEvidenceResponse)
    assert response.claim_id == "CLAIM_E2E_01"
    assert len(response.evidence) == 2

    e1, e2 = response.evidence
    assert e1.evidence_id == "P002_C001"
    assert e1.chunk_id == "P002_C001"
    assert e1.paper_id == "P002"
    assert e1.text == c3.text
    assert e1.retrieval_method == "hybrid"
    assert e1.retrieval_score == pytest.approx(1.0 / 61 + 1.0 / 62)

    assert e2.evidence_id == "P001_C001"
    assert e2.chunk_id == "P001_C001"
    assert e2.paper_id == "P001"
    assert e2.text == c1.text
    assert e2.retrieval_method == "hybrid"
    assert e2.retrieval_score == pytest.approx(1.0 / 62)


def test_index_chunks_fast_fails_on_intra_batch_duplicates_before_embedding(retriever, mock_embedding):
    """Verify index_chunks detects intra-batch duplicates early without generating embeddings."""
    chunk = make_dummy_chunk("P001_DUPE_EARLY")

    def fail_if_called(texts):
        raise AssertionError("embed_documents should not be called when duplicate IDs exist in the batch!")

    mock_embedding.embed_documents = fail_if_called

    with pytest.raises(DuplicateChunkError):
        retriever.index_chunks([chunk, chunk])


def test_retrieve_evidence_rejects_boolean_parameters(retriever):
    """Verify retrieve_evidence rejects boolean top_k and candidate_k."""
    claim = make_dummy_claim()
    for bool_val in [True, False]:
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retriever.retrieve_evidence(claim, top_k=bool_val)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
            retriever.retrieve_evidence(claim, candidate_k=bool_val)  # type: ignore[arg-type]
