"""Unit tests for HybridSearchEngine using Reciprocal Rank Fusion (RRF) (Step 8)."""

from typing import Any
import pytest

from backend.config import RRF_K
from backend.retrieval.embeddings import InputTooLongError
from backend.retrieval.hybrid_search import (
    HybridSearchEngine,
    HybridSearchError,
    HybridSearchInputError,
    HybridSearchResult,
)
from backend.retrieval.lexical_search import LexicalSearchResult
from backend.retrieval.vector_search import VectorSearchResult
from backend.schemas.chunk import DocumentChunk


def make_dummy_chunk(chunk_id: str, paper_id: str = "P001", text: str = "Sample text") -> DocumentChunk:
    """Helper to construct a valid DocumentChunk."""
    return DocumentChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        page_number=1,
        paragraph_number=1,
        section="Results",
        text=text,
    )


class CallOrderTracker:
    def __init__(self) -> None:
        self.calls: list[str] = []


class MockEmbeddingService:
    def __init__(self, tracker: CallOrderTracker | None = None) -> None:
        self.tracker = tracker
        self.last_query: str | None = None

    def embed_query(self, query: str) -> list[float]:
        self.last_query = query
        if self.tracker:
            self.tracker.calls.append("embed_query")
        return [1.0] + [0.0] * 383


class MockVectorSearchEngine:
    def __init__(self, tracker: CallOrderTracker | None = None) -> None:
        self.tracker = tracker
        self.results: list[VectorSearchResult] = []
        self.last_paper_ids: list[str] | None = None
        self.last_top_k: int | None = None

    def search(self, query_embedding: Any, paper_ids: list[str], top_k: int = 20) -> list[VectorSearchResult]:
        if self.tracker:
            self.tracker.calls.append("vector_search")
        self.last_paper_ids = paper_ids
        self.last_top_k = top_k
        return self.results


class MockLexicalSearchEngine:
    def __init__(self, tracker: CallOrderTracker | None = None) -> None:
        self.tracker = tracker
        self.results: list[LexicalSearchResult] = []
        self.last_paper_ids: list[str] | None = None
        self.last_top_k: int | None = None

    def search(self, query: str, paper_ids: list[str], top_k: int = 20) -> list[LexicalSearchResult]:
        if self.tracker:
            self.tracker.calls.append("lexical_search")
        self.last_paper_ids = paper_ids
        self.last_top_k = top_k
        return self.results


@pytest.fixture
def call_tracker():
    return CallOrderTracker()


@pytest.fixture
def mock_embedding(call_tracker):
    return MockEmbeddingService(tracker=call_tracker)


@pytest.fixture
def mock_vector(call_tracker):
    return MockVectorSearchEngine(tracker=call_tracker)


@pytest.fixture
def mock_lexical(call_tracker):
    return MockLexicalSearchEngine(tracker=call_tracker)


@pytest.fixture
def hybrid_engine(mock_embedding, mock_vector, mock_lexical):
    return HybridSearchEngine(
        embedding_service=mock_embedding,  # type: ignore[arg-type]
        vector_search=mock_vector,        # type: ignore[arg-type]
        lexical_search=mock_lexical,      # type: ignore[arg-type]
        rrf_k=60,
    )


def test_chunk_in_both_engines_receives_contributions_from_both(hybrid_engine, mock_vector, mock_lexical):
    """1 & 2 & 3. Verify chunks in both engines sum reciprocal rank contributions with ranks starting at 1."""
    chunk_a = make_dummy_chunk("P001_C001")
    # Vector rank 1 (1-indexed) -> contribution: 1 / (60 + 1) = 1 / 61
    mock_vector.results = [VectorSearchResult(chunk=chunk_a, score=0.95)]
    # Lexical rank 3 (1-indexed) -> contribution: 1 / (60 + 3) = 1 / 63
    chunk_other1 = make_dummy_chunk("P001_OTHER1")
    chunk_other2 = make_dummy_chunk("P001_OTHER2")
    mock_lexical.results = [
        LexicalSearchResult(chunk=chunk_other1, score=15.0),
        LexicalSearchResult(chunk=chunk_other2, score=12.0),
        LexicalSearchResult(chunk=chunk_a, score=10.0),
    ]

    results = hybrid_engine.search(query="sample query", paper_ids=["P001"])
    fused_a = next(r for r in results if r.chunk.chunk_id == "P001_C001")

    expected_score = (1.0 / 61) + (1.0 / 63)
    assert fused_a.score == pytest.approx(expected_score, rel=1e-6)
    assert fused_a.vector_rank == 1
    assert fused_a.lexical_rank == 3


def test_chunk_in_only_one_engine_retained(hybrid_engine, mock_vector, mock_lexical):
    """4. Verify results appearing in only one engine are retained with other rank None."""
    chunk_v = make_dummy_chunk("P001_V_ONLY")
    chunk_l = make_dummy_chunk("P001_L_ONLY")
    mock_vector.results = [VectorSearchResult(chunk=chunk_v, score=0.9)]
    mock_lexical.results = [LexicalSearchResult(chunk=chunk_l, score=10.0)]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 2

    res_v = next(r for r in results if r.chunk.chunk_id == "P001_V_ONLY")
    assert res_v.vector_rank == 1
    assert res_v.lexical_rank is None
    assert res_v.score == pytest.approx(1.0 / 61)

    res_l = next(r for r in results if r.chunk.chunk_id == "P001_L_ONLY")
    assert res_l.vector_rank is None
    assert res_l.lexical_rank == 1
    assert res_l.score == pytest.approx(1.0 / 61)


def test_chunk_ranked_well_in_both_wins(hybrid_engine, mock_vector, mock_lexical):
    """5. Verify a chunk ranked #1 in both engines achieves the top combined rank."""
    chunk_top = make_dummy_chunk("P001_TOP")
    chunk_v_only = make_dummy_chunk("P001_V_ONLY")
    chunk_l_only = make_dummy_chunk("P001_L_ONLY")

    # Vector: TOP (#1), V_ONLY (#2)
    mock_vector.results = [
        VectorSearchResult(chunk=chunk_top, score=0.99),
        VectorSearchResult(chunk=chunk_v_only, score=0.95),
    ]
    # Lexical: TOP (#1), L_ONLY (#2)
    mock_lexical.results = [
        LexicalSearchResult(chunk=chunk_top, score=25.0),
        LexicalSearchResult(chunk=chunk_l_only, score=20.0),
    ]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert results[0].chunk.chunk_id == "P001_TOP"
    assert results[0].score == pytest.approx((1.0 / 61) + (1.0 / 61))


def test_results_deduplicated_by_chunk_id(hybrid_engine, mock_vector, mock_lexical):
    """6. Verify each unique chunk ID appears at most once in the final output."""
    chunk_shared = make_dummy_chunk("P001_SHARED")
    mock_vector.results = [VectorSearchResult(chunk=chunk_shared, score=0.8)]
    mock_lexical.results = [LexicalSearchResult(chunk=chunk_shared, score=12.0)]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "P001_SHARED"


def test_intra_engine_duplicates_not_double_counted(hybrid_engine, mock_vector, mock_lexical):
    """7. Verify duplicate entries from the same engine do not double-count rank contributions."""
    chunk = make_dummy_chunk("P001_DUPE")
    # Vector returns same chunk twice
    mock_vector.results = [
        VectorSearchResult(chunk=chunk, score=0.9),
        VectorSearchResult(chunk=chunk, score=0.7),
    ]
    mock_lexical.results = []

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 1
    # Only rank 1 should be counted
    assert results[0].score == pytest.approx(1.0 / 61)
    assert results[0].vector_rank == 1


def test_results_sorted_by_rrf_score_descending(hybrid_engine, mock_vector, mock_lexical):
    """8. Verify final fused results are sorted monotonically descending by RRF score."""
    c1 = make_dummy_chunk("C1")
    c2 = make_dummy_chunk("C2")
    c3 = make_dummy_chunk("C3")

    mock_vector.results = [
        VectorSearchResult(chunk=c1, score=0.9),  # rank 1
        VectorSearchResult(chunk=c2, score=0.8),  # rank 2
    ]
    mock_lexical.results = [
        VectorSearchResult(chunk=c1, score=10.0),  # rank 1 -> c1 gets 1/61 + 1/61
        VectorSearchResult(chunk=c3, score=8.0),   # rank 2 -> c3 gets 1/62
    ]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.chunk_id == "C1"


def test_tied_scores_deterministic_chunk_id_ascending(hybrid_engine, mock_vector, mock_lexical):
    """9. Verify tied RRF scores break ties deterministically by chunk_id ascending."""
    chunk_z = make_dummy_chunk("P001_ZETA")
    chunk_a = make_dummy_chunk("P001_ALPHA")
    chunk_m = make_dummy_chunk("P001_MU")

    # Both chunks appear at rank 1 in their respective single engines -> identical RRF score 1/61
    mock_vector.results = [VectorSearchResult(chunk=chunk_z, score=0.9)]
    mock_lexical.results = [
        LexicalSearchResult(chunk=chunk_a, score=10.0),
        LexicalSearchResult(chunk=chunk_m, score=10.0),
    ]

    # Let's give each exactly one rank 1 contribution (e.g. 1 / 61)
    mock_vector.results = [VectorSearchResult(chunk=chunk_z, score=0.9)]  # rank 1 (1/61)
    mock_lexical.results = [VectorSearchResult(chunk=chunk_a, score=10.0)]  # rank 1 (1/61)

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 2
    assert results[0].score == pytest.approx(results[1].score)
    assert results[0].chunk.chunk_id == "P001_ALPHA"
    assert results[1].chunk.chunk_id == "P001_ZETA"


def test_top_k_limits_result_count(hybrid_engine, mock_vector, mock_lexical):
    """10. Verify top_k parameter limits the number of returned items."""
    chunks = [make_dummy_chunk(f"C{i}") for i in range(10)]
    mock_vector.results = [VectorSearchResult(chunk=c, score=0.9 - i * 0.05) for i, c in enumerate(chunks)]
    mock_lexical.results = []

    results = hybrid_engine.search(query="query", paper_ids=["P001"], top_k=3)
    assert len(results) == 3


def test_same_paper_ids_reach_both_engines(hybrid_engine, mock_vector, mock_lexical):
    """11. Verify identical paper_ids list is passed to both retrievers."""
    hybrid_engine.search(query="query", paper_ids=["P001", "P002"])

    assert mock_vector.last_paper_ids == ["P001", "P002"]
    assert mock_lexical.last_paper_ids == ["P001", "P002"]


def test_duplicate_paper_ids_removed(hybrid_engine, mock_vector, mock_lexical):
    """12. Verify duplicate paper IDs in search input are deduplicated."""
    hybrid_engine.search(query="query", paper_ids=["P001", "P002", "P001", "P002"])

    assert mock_vector.last_paper_ids == ["P001", "P002"]
    assert mock_lexical.last_paper_ids == ["P001", "P002"]


def test_empty_paper_ids_returns_empty_list_without_calling_engines(hybrid_engine, mock_vector, mock_lexical, call_tracker):
    """13. Verify empty paper_ids returns empty list immediately without calling engines."""
    results = hybrid_engine.search(query="query", paper_ids=[])

    assert results == []
    assert len(call_tracker.calls) == 0


def test_invalid_query_rejected(hybrid_engine):
    """14. Verify empty or whitespace-only query raises HybridSearchInputError."""
    for invalid_q in ["", "   ", "\t\n"]:
        with pytest.raises(HybridSearchInputError):
            hybrid_engine.search(query=invalid_q, paper_ids=["P001"])


def test_invalid_top_k_rejected(hybrid_engine):
    """15. Verify non-positive top_k raises ValueError."""
    for invalid_k in [0, -1, -10]:
        with pytest.raises(ValueError):
            hybrid_engine.search(query="query", paper_ids=["P001"], top_k=invalid_k)


def test_invalid_candidate_k_rejected(hybrid_engine):
    """16. Verify non-positive candidate_k raises ValueError."""
    for invalid_k in [0, -1, -5]:
        with pytest.raises(ValueError):
            hybrid_engine.search(query="query", paper_ids=["P001"], candidate_k=invalid_k)


def test_invalid_rrf_k_rejected(mock_embedding, mock_vector, mock_lexical):
    """17. Verify non-positive rrf_k constant raises ValueError during initialization."""
    for invalid_k in [0, -1, -60]:
        with pytest.raises(ValueError):
            HybridSearchEngine(
                embedding_service=mock_embedding,
                vector_search=mock_vector,
                lexical_search=mock_lexical,
                rrf_k=invalid_k,
            )


def test_vector_only_retrieval_works(hybrid_engine, mock_vector, mock_lexical):
    """18. Verify search functions cleanly when lexical retriever returns zero candidates."""
    chunk = make_dummy_chunk("P001_V1")
    mock_vector.results = [VectorSearchResult(chunk=chunk, score=0.88)]
    mock_lexical.results = []

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "P001_V1"
    assert results[0].vector_rank == 1
    assert results[0].lexical_rank is None
    assert results[0].score == pytest.approx(1.0 / 61)


def test_lexical_only_retrieval_works(hybrid_engine, mock_vector, mock_lexical):
    """19. Verify search functions cleanly when vector retriever returns zero candidates."""
    chunk = make_dummy_chunk("P001_L1")
    mock_vector.results = []
    mock_lexical.results = [LexicalSearchResult(chunk=chunk, score=14.5)]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "P001_L1"
    assert results[0].vector_rank is None
    assert results[0].lexical_rank == 1
    assert results[0].score == pytest.approx(1.0 / 61)


def test_both_engines_returning_empty_gives_empty_list(hybrid_engine, mock_vector, mock_lexical):
    """20. Verify that when both retrievers return empty lists, hybrid search returns empty list."""
    mock_vector.results = []
    mock_lexical.results = []

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert results == []


def test_original_text_and_metadata_preserved(hybrid_engine, mock_vector, mock_lexical):
    """21. Verify original text with spaces/symbols and metadata fields are strictly preserved."""
    raw_text = "\n\t   Raw research text: 92% ± 0.5% & non-ASCII: α, β \n"
    chunk = DocumentChunk(
        chunk_id="P001_PRESERVED",
        paper_id="P001",
        page_number=8,
        paragraph_number=3,
        section="Validation",
        text=raw_text,
    )
    mock_vector.results = [VectorSearchResult(chunk=chunk, score=0.9)]
    mock_lexical.results = [LexicalSearchResult(chunk=chunk, score=12.0)]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 1
    res_chunk = results[0].chunk
    assert res_chunk.chunk_id == "P001_PRESERVED"
    assert res_chunk.paper_id == "P001"
    assert res_chunk.page_number == 8
    assert res_chunk.paragraph_number == 3
    assert res_chunk.section == "Validation"
    assert res_chunk.text == raw_text


def test_vector_and_lexical_invoked_sequentially(hybrid_engine, mock_vector, mock_lexical, call_tracker):
    """22. Verify that query embedding, vector search, and lexical search are executed in sequential order."""
    hybrid_engine.search(query="query", paper_ids=["P001"])

    assert call_tracker.calls == [
        "embed_query",
        "vector_search",
        "lexical_search",
    ]


def test_rrf_score_is_not_raw_similarity_or_bm25(hybrid_engine, mock_vector, mock_lexical):
    """23. Verify RRF score is distinct from cosine similarity (0.95) and BM25 score (18.5)."""
    chunk = make_dummy_chunk("P001_DISTINCT")
    mock_vector.results = [VectorSearchResult(chunk=chunk, score=0.95)]
    mock_lexical.results = [LexicalSearchResult(chunk=chunk, score=18.5)]

    results = hybrid_engine.search(query="query", paper_ids=["P001"])
    assert len(results) == 1
    score = results[0].score

    # Score must equal (1/61 + 1/61) = 2/61 ≈ 0.03278688
    assert score != 0.95
    assert score != 18.5
    assert score == pytest.approx(2.0 / 61, rel=1e-5)


def test_conflicting_chunk_metadata_raises_hybrid_search_error(hybrid_engine, mock_vector, mock_lexical):
    """Verify that chunks with identical chunk_id but differing metadata raise HybridSearchError."""
    chunk_v = DocumentChunk(
        chunk_id="P001_CONFLICT",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        section="Methods",
        text="Version A text from vector search",
    )
    chunk_l = DocumentChunk(
        chunk_id="P001_CONFLICT",
        paper_id="P001",
        page_number=2,
        paragraph_number=5,
        section="Results",
        text="Conflicting version B text from lexical search",
    )
    mock_vector.results = [VectorSearchResult(chunk=chunk_v, score=0.9)]
    mock_lexical.results = [LexicalSearchResult(chunk=chunk_l, score=12.0)]

    with pytest.raises(HybridSearchError, match="Conflicting chunk metadata detected"):
        hybrid_engine.search(query="query", paper_ids=["P001"])


def test_embedding_token_limit_error_propagated(hybrid_engine, mock_embedding):
    """Verify that query embedding token limit validation error (InputTooLongError) propagates."""
    def fail_embed(query: str):
        raise InputTooLongError(position="query", token_count=600, max_limit=512)

    mock_embedding.embed_query = fail_embed
    with pytest.raises(InputTooLongError, match="exceeds model maximum token limit"):
        hybrid_engine.search(query="long query", paper_ids=["P001"])


def test_vector_or_lexical_failure_propagated(hybrid_engine, mock_vector, mock_lexical):
    """Verify that vector or lexical engine retrieval failures propagate without suppression."""
    def fail_vector(*args, **kwargs):
        raise RuntimeError("Vector database connection lost.")

    mock_vector.search = fail_vector
    with pytest.raises(RuntimeError, match="Vector database connection lost"):
        hybrid_engine.search(query="query", paper_ids=["P001"])

    mock_vector.search = lambda *args, **kwargs: []

    def fail_lexical(*args, **kwargs):
        raise RuntimeError("Lexical index corrupted.")

    mock_lexical.search = fail_lexical
    with pytest.raises(RuntimeError, match="Lexical index corrupted"):
        hybrid_engine.search(query="query", paper_ids=["P001"])


def test_boolean_and_non_string_parameters_rejected(hybrid_engine, mock_embedding, mock_vector, mock_lexical):
    """Verify boolean values for rrf_k, top_k, candidate_k and non-string query are rejected."""
    for bool_val in [True, False]:
        with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
            HybridSearchEngine(
                embedding_service=mock_embedding,
                vector_search=mock_vector,
                lexical_search=mock_lexical,
                rrf_k=bool_val,  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            hybrid_engine.search(query="valid query", paper_ids=["P001"], top_k=bool_val)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
            hybrid_engine.search(query="valid query", paper_ids=["P001"], candidate_k=bool_val)  # type: ignore[arg-type]

    for invalid_q in [123, None, ["query"]]:
        with pytest.raises(HybridSearchInputError):
            hybrid_engine.search(query=invalid_q, paper_ids=["P001"])  # type: ignore[arg-type]
