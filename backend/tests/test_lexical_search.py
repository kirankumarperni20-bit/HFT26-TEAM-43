"""Unit tests for the Okapi BM25 LexicalSearchEngine (Step 7)."""

import pytest
from sqlalchemy.orm import Session

from backend.database.connection import create_db_engine, get_session_factory
from backend.database.models import Base
from backend.database.repository import ChunkRepository
from backend.retrieval.lexical_search import (
    LexicalInputError,
    LexicalSearchEngine,
    LexicalSearchResult,
    tokenize_text,
)
from backend.schemas.chunk import DocumentChunk


@pytest.fixture
def sqlite_engine():
    """Create a temporary in-memory SQLite engine with tables for lexical tests."""
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
def search_engine(repository):
    return LexicalSearchEngine(repository=repository)


RESEARCH_CHUNKS = [
    DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        section="Abstract",
        text="We introduce the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism.",
    ),
    DocumentChunk(
        chunk_id="P001_C002",
        paper_id="P001",
        page_number=5,
        paragraph_number=3,
        section="Results",
        text="On the WMT 2014 translation benchmark, the proposed model achieved an accuracy of 92% and BLEU score of 28.4.",
    ),
    DocumentChunk(
        chunk_id="P001_C003",
        paper_id="P001",
        page_number=6,
        paragraph_number=2,
        section="Conclusion",
        text="Self-attention networks without recurrence demonstrate faster training times across cross-lingual evaluation tasks.",
    ),
    DocumentChunk(
        chunk_id="P002_C001",
        paper_id="P002",
        page_number=2,
        paragraph_number=1,
        section="Introduction",
        text="Deep residual networks with 50 layers solve the degradation problem on ImageNet classification.",
    ),
    DocumentChunk(
        chunk_id="P002_C002",
        paper_id="P002",
        page_number=4,
        paragraph_number=3,
        section="Architecture",
        text="Residual connections ensure that gradient signals propagate backward through deep layers without vanishing.",
    ),
    DocumentChunk(
        chunk_id="P002_C003",
        paper_id="P002",
        page_number=6,
        paragraph_number=1,
        section="Experiments",
        text="Experiments on CIFAR-10 demonstrate that 110-layer networks achieve lower training error and competitive test accuracy.",
    ),
]


def test_valid_lexical_search_returns_results(repository, search_engine):
    """1. Verify valid search retrieves matching research chunks."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results = search_engine.search(query="Transformer attention mechanism", paper_ids=["P001"])
    assert len(results) > 0
    assert all(isinstance(r, LexicalSearchResult) for r in results)
    assert results[0].chunk.chunk_id == "P001_C001"


def test_exact_keyword_matches_retrieved(repository, search_engine):
    """2. Verify exact keyword terms like 'BLEU' retrieve the target chunk."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results = search_engine.search(query="BLEU score", paper_ids=["P001"])
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "P001_C002"
    assert "BLEU score of 28.4" in results[0].chunk.text


def test_relevant_passages_rank_higher(repository, search_engine):
    """3. Verify highly relevant passages with multiple term occurrences rank above peripheral matches."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    # Query focused on residual connections and vanishing gradients
    results = search_engine.search(
        query="residual connections gradient propagate vanishing",
        paper_ids=["P002"],
    )
    assert len(results) >= 2
    # P002_C002 has residual, connections, gradient, propagate, vanishing (all 5 query terms)
    # P002_C001 only has residual
    assert results[0].chunk.chunk_id == "P002_C002"
    assert results[0].score > results[1].score


def test_search_only_returns_chunks_from_requested_papers(repository, search_engine):
    """4. Verify search strictly respects paper_ids filter and excludes unrequested papers."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    # Search for term present in P001 and P002
    results_p1 = search_engine.search(query="model", paper_ids=["P001"])
    assert len(results_p1) > 0
    assert all(r.chunk.paper_id == "P001" for r in results_p1)

    results_p2 = search_engine.search(query="residual", paper_ids=["P002"])
    assert len(results_p2) > 0
    assert all(r.chunk.paper_id == "P002" for r in results_p2)


def test_empty_paper_ids_returns_empty_list(repository, search_engine):
    """5. Verify empty paper_ids returns empty list immediately."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results = search_engine.search(query="Transformer", paper_ids=[])
    assert results == []


def test_unknown_paper_ids_returns_empty_list(repository, search_engine):
    """6. Verify querying an unknown paper ID returns an empty list."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results = search_engine.search(query="Transformer", paper_ids=["P999_UNKNOWN"])
    assert results == []


def test_empty_query_rejected(search_engine):
    """7. Verify empty query string raises LexicalInputError."""
    with pytest.raises(LexicalInputError):
        search_engine.search(query="", paper_ids=["P001"])


def test_whitespace_only_query_rejected(search_engine):
    """8. Verify whitespace-only query raises LexicalInputError."""
    for ws in ["   ", "\t", "\n\r  \n"]:
        with pytest.raises(LexicalInputError):
            search_engine.search(query=ws, paper_ids=["P001"])


def test_invalid_top_k_rejected(search_engine):
    """9. Verify non-positive top_k raises ValueError."""
    for invalid_k in [0, -1, -50]:
        with pytest.raises(ValueError):
            search_engine.search(query="valid query", paper_ids=["P001"], top_k=invalid_k)


def test_query_with_no_matching_terms_returns_empty_list(repository, search_engine):
    """10. Verify query containing terms not present in the corpus returns empty list."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results = search_engine.search(
        query="photosynthesis chlorophyll chloroplast",
        paper_ids=["P001", "P002"],
    )
    assert results == []


def test_empty_candidate_corpus_handled(repository, search_engine):
    """11. Verify searching a database with no chunks returns empty list without error."""
    # Repository has zero chunks inserted
    results = search_engine.search(query="Transformer", paper_ids=["P001"])
    assert results == []


def test_tokenization_is_case_insensitive(repository, search_engine):
    """12. Verify queries in all-caps or mixed-case retrieve matching chunks."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results_lower = search_engine.search(query="transformer attention", paper_ids=["P001"])
    results_upper = search_engine.search(query="TRANSFORMER ATTENTION", paper_ids=["P001"])

    assert len(results_lower) == len(results_upper)
    assert results_lower[0].chunk.chunk_id == results_upper[0].chunk.chunk_id
    assert results_lower[0].score == pytest.approx(results_upper[0].score)


def test_punctuation_handled_consistently(repository, search_engine):
    """13. Verify punctuation in query and text is parsed consistently."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    # Hyphenated, quoted, and punctuated query terms
    results = search_engine.search(query="'cross-lingual' (evaluation)!", paper_ids=["P001"])
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "P001_C003"


def test_numeric_tokens_are_searchable(repository, search_engine):
    """14. Verify numeric terms like '92' or '50' can be searched specifically."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results_92 = search_engine.search(query="accuracy 92", paper_ids=["P001"])
    assert len(results_92) >= 1
    assert results_92[0].chunk.chunk_id == "P001_C002"

    results_50 = search_engine.search(query="50 layers", paper_ids=["P002"])
    assert len(results_50) >= 1
    assert results_50[0].chunk.chunk_id == "P002_C001"


def test_original_source_text_and_metadata_preserved(repository, search_engine):
    """15. Verify original text with newlines, spaces, and metadata is preserved in results."""
    raw_text = "\n\t  Raw text with special symbols: 92% ± 0.5% \n"
    chunk = DocumentChunk(
        chunk_id="P003_C001",
        paper_id="P003",
        page_number=10,
        paragraph_number=4,
        section="Detailed Methodology",
        text=raw_text,
    )
    repository.insert_chunks([chunk])

    results = search_engine.search(query="symbols 92", paper_ids=["P003"])
    assert len(results) == 1
    res = results[0].chunk
    assert res.chunk_id == "P003_C001"
    assert res.paper_id == "P003"
    assert res.page_number == 10
    assert res.paragraph_number == 4
    assert res.section == "Detailed Methodology"
    assert res.text == raw_text


def test_top_k_limits_result_count(repository, search_engine):
    """16. Verify top_k parameter strictly caps returned results."""
    repository.insert_chunks(RESEARCH_CHUNKS)

    results = search_engine.search(query="model network architecture", paper_ids=["P001", "P002"], top_k=2)
    assert len(results) == 2


def test_identical_scores_have_deterministic_tie_breaking(repository, search_engine):
    """17. Verify chunks with identical BM25 scores break ties deterministically by chunk_id ascending."""
    chunks = [
        DocumentChunk(
            chunk_id="P004_C_ZETA",
            paper_id="P004",
            page_number=1,
            paragraph_number=3,
            text="Identical benchmark text sentence.",
        ),
        DocumentChunk(
            chunk_id="P004_C_ALPHA",
            paper_id="P004",
            page_number=1,
            paragraph_number=1,
            text="Identical benchmark text sentence.",
        ),
        DocumentChunk(
            chunk_id="P004_C_BETA",
            paper_id="P004",
            page_number=1,
            paragraph_number=2,
            text="Identical benchmark text sentence.",
        ),
    ]
    repository.insert_chunks(chunks)

    results = search_engine.search(query="benchmark", paper_ids=["P004"])
    assert len(results) == 3
    # Scores are all identical
    assert results[0].score == pytest.approx(results[1].score)
    assert results[1].score == pytest.approx(results[2].score)
    # Order must be alphabetically sorted by chunk_id
    assert [r.chunk.chunk_id for r in results] == [
        "P004_C_ALPHA",
        "P004_C_BETA",
        "P004_C_ZETA",
    ]


def test_repository_transaction_not_committed_by_lexical_search(repository, search_engine, db_session):
    """18. Verify that lexical search does not commit caller's transaction."""
    chunk = DocumentChunk(
        chunk_id="P005_C001",
        paper_id="P005",
        page_number=1,
        paragraph_number=1,
        text="Transaction test uncommitted passage.",
    )
    repository.insert_chunks([chunk])

    # Search against uncommitted chunk
    results = search_engine.search(query="transaction", paper_ids=["P005"])
    assert len(results) == 1

    # Session is still active and uncommitted
    assert db_session.is_active

    # Rolling back session discards the chunk, proving lexical search did not commit
    db_session.rollback()
    assert repository.get_chunks_by_paper_ids(["P005"]) == []


def test_unicode_scientific_terms_tokenized_and_retrieved(repository, search_engine):
    """Verify Unicode scientific terms (Greek characters, accented names) are tokenized and retrieved."""
    chunks = [
        DocumentChunk(
            chunk_id="P010_C001",
            paper_id="P010",
            page_number=1,
            paragraph_number=1,
            text="The Schrödinger wave equation governs quantum mechanical systems.",
        ),
        DocumentChunk(
            chunk_id="P010_C002",
            paper_id="P010",
            page_number=2,
            paragraph_number=1,
            text="Radioactive α-decay exhibits quantum tunneling through a potential barrier.",
        ),
        DocumentChunk(
            chunk_id="P010_C003",
            paper_id="P010",
            page_number=3,
            paragraph_number=1,
            text="Classical dynamical systems satisfy the Poincaré recurrence theorem.",
        ),
    ]
    repository.insert_chunks(chunks)

    # Search for Schrödinger
    res_schrodinger = search_engine.search(query="Schrödinger equation", paper_ids=["P010"])
    assert len(res_schrodinger) >= 1
    assert res_schrodinger[0].chunk.chunk_id == "P010_C001"

    # Search for α-decay
    res_alpha = search_engine.search(query="α-decay tunneling", paper_ids=["P010"])
    assert len(res_alpha) >= 1
    assert res_alpha[0].chunk.chunk_id == "P010_C002"

    # Search for Poincaré
    res_poincare = search_engine.search(query="Poincaré recurrence", paper_ids=["P010"])
    assert len(res_poincare) >= 1
    assert res_poincare[0].chunk.chunk_id == "P010_C003"


def test_invalid_query_type_and_boolean_top_k_rejected(search_engine):
    """Verify non-string query raises LexicalInputError and boolean top_k raises ValueError."""
    for invalid_q in [123, None, ["query"]]:
        with pytest.raises(LexicalInputError):
            search_engine.search(query=invalid_q, paper_ids=["P001"])  # type: ignore[arg-type]

    for bool_val in [True, False]:
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_engine.search(query="valid query", paper_ids=["P001"], top_k=bool_val)  # type: ignore[arg-type]
