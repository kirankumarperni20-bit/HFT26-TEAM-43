"""Unit tests for persistent embeddings and VectorSearchEngine (Step 6)."""

import numpy as np
import pytest
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from backend.config import EMBEDDING_DIMENSION
from backend.database.connection import create_db_engine, get_session_factory
from backend.database.models import Base, ChunkRecord
from backend.database.repository import ChunkRepository
from backend.retrieval.vector_search import VectorSearchEngine, VectorSearchResult
from backend.schemas.chunk import DocumentChunk


@pytest.fixture
def sqlite_engine():
    """Create a temporary in-memory SQLite engine with tables for vector tests."""
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
def search_engine(db_session):
    return VectorSearchEngine(session=db_session)


def make_unit_vector(direction_index: int, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    """Create a unit vector with a 1.0 at direction_index and 0.0 elsewhere."""
    vec = [0.0] * dim
    vec[direction_index] = 1.0
    return vec


def test_chunk_table_has_embedding_column(sqlite_engine):
    """1. Verify that the 'chunks' table includes the 'embedding' column."""
    inspector = inspect(sqlite_engine)
    columns = {col["name"]: col for col in inspector.get_columns("chunks")}
    assert "embedding" in columns
    assert columns["embedding"]["nullable"] is True


def test_embedding_persistence_and_retrieval(repository, db_session):
    """2 & 3. Verify successful embedding persistence and correct mapping to chunk ID."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Sample chunk text",
    )
    repository.insert_chunks([chunk])

    # Update with 384-d vector
    vector = make_unit_vector(0)
    updated_count = repository.update_embeddings({"P001_C001": vector})
    assert updated_count == 1

    record = db_session.get(ChunkRecord, "P001_C001")
    assert record is not None
    assert record.embedding is not None
    assert len(record.embedding) == EMBEDDING_DIMENSION
    assert record.embedding[0] == pytest.approx(1.0)
    assert record.embedding[1] == pytest.approx(0.0)


def test_empty_embedding_update_returns_zero(repository):
    """4. Verify updating an empty mapping returns 0 immediately."""
    count = repository.update_embeddings({})
    assert count == 0


def test_incorrect_vector_dimension_rejected(repository):
    """5. Verify updating with wrong vector dimension raises ValueError."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Sample chunk text",
    )
    repository.insert_chunks([chunk])

    # Pass 128-d vector instead of 384-d
    wrong_dim_vector = [0.1] * 128
    with pytest.raises(ValueError) as exc_info:
        repository.update_embeddings({"P001_C001": wrong_dim_vector})

    assert "invalid dimension" in str(exc_info.value)


def test_nan_or_infinite_vector_values_rejected(repository):
    """6. Verify vector containing NaN or Inf is rejected with ValueError."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Sample chunk text",
    )
    repository.insert_chunks([chunk])

    nan_vector = make_unit_vector(0)
    nan_vector[10] = float("nan")
    with pytest.raises(ValueError) as exc_info:
        repository.update_embeddings({"P001_C001": nan_vector})
    assert "NaN or infinite" in str(exc_info.value)

    inf_vector = make_unit_vector(0)
    inf_vector[20] = float("inf")
    with pytest.raises(ValueError) as exc_info:
        repository.update_embeddings({"P001_C001": inf_vector})
    assert "NaN or infinite" in str(exc_info.value)


def test_unknown_chunk_id_rejected(repository):
    """7. Verify updating a chunk_id that does not exist raises KeyError."""
    vector = make_unit_vector(0)
    with pytest.raises(KeyError) as exc_info:
        repository.update_embeddings({"NON_EXISTENT_ID": vector})

    assert "NON_EXISTENT_ID" in str(exc_info.value)


def test_search_empty_paper_ids_returns_empty_list(search_engine):
    """8. Verify vector search returns empty list immediately when paper_ids is empty."""
    q_vec = make_unit_vector(0)
    results = search_engine.search(query_embedding=q_vec, paper_ids=[])
    assert results == []


def test_search_returns_only_selected_paper_ids(repository, search_engine):
    """9. Verify search filters strictly by selected paper IDs."""
    chunks = [
        DocumentChunk(
            chunk_id="P001_C001",
            paper_id="P001",
            page_number=1,
            paragraph_number=1,
            text="Paper 1 chunk",
        ),
        DocumentChunk(
            chunk_id="P002_C001",
            paper_id="P002",
            page_number=1,
            paragraph_number=1,
            text="Paper 2 chunk",
        ),
    ]
    repository.insert_chunks(chunks)

    v1 = make_unit_vector(0)
    v2 = make_unit_vector(0)
    repository.update_embeddings({"P001_C001": v1, "P002_C001": v2})

    # Search only paper P001
    results = search_engine.search(query_embedding=v1, paper_ids=["P001"])
    assert len(results) == 1
    assert results[0].chunk.paper_id == "P001"
    assert results[0].chunk.chunk_id == "P001_C001"


def test_search_excludes_chunks_without_embeddings(repository, search_engine):
    """10. Verify chunks with NULL embeddings are excluded from vector search."""
    chunks = [
        DocumentChunk(
            chunk_id="P001_C001",
            paper_id="P001",
            page_number=1,
            paragraph_number=1,
            text="Chunk with embedding",
        ),
        DocumentChunk(
            chunk_id="P001_C002",
            paper_id="P001",
            page_number=1,
            paragraph_number=2,
            text="Chunk without embedding (NULL)",
        ),
    ]
    repository.insert_chunks(chunks)
    repository.update_embeddings({"P001_C001": make_unit_vector(0)})

    results = search_engine.search(query_embedding=make_unit_vector(0), paper_ids=["P001"])
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "P001_C001"


def test_correct_ranking_for_known_cosine_similarity(repository, search_engine):
    """11. Verify exact ranking based on known cosine similarity angles."""
    chunks = [
        DocumentChunk(
            chunk_id="P001_ORTHOGONAL",
            paper_id="P001",
            page_number=1,
            paragraph_number=1,
            text="Orthogonal vector",
        ),
        DocumentChunk(
            chunk_id="P001_EXACT",
            paper_id="P001",
            page_number=1,
            paragraph_number=2,
            text="Identical vector",
        ),
        DocumentChunk(
            chunk_id="P001_PARTIAL",
            paper_id="P001",
            page_number=1,
            paragraph_number=3,
            text="45 degree vector",
        ),
    ]
    repository.insert_chunks(chunks)

    # Query vector along dim 0
    q_vec = make_unit_vector(0)

    # EXACT: identical to q_vec -> cos_sim = 1.0
    v_exact = make_unit_vector(0)

    # PARTIAL: 45 degrees between dim 0 and dim 1 -> cos_sim ≈ 0.7071
    v_partial = [0.0] * EMBEDDING_DIMENSION
    v_partial[0] = float(np.cos(np.pi / 4))
    v_partial[1] = float(np.sin(np.pi / 4))

    # ORTHOGONAL: along dim 1 -> cos_sim = 0.0
    v_orthogonal = make_unit_vector(1)

    repository.update_embeddings({
        "P001_EXACT": v_exact,
        "P001_PARTIAL": v_partial,
        "P001_ORTHOGONAL": v_orthogonal,
    })

    results = search_engine.search(query_embedding=q_vec, paper_ids=["P001"])
    assert len(results) == 3
    assert results[0].chunk.chunk_id == "P001_EXACT"
    assert results[0].score == pytest.approx(1.0, rel=1e-5)

    assert results[1].chunk.chunk_id == "P001_PARTIAL"
    assert results[1].score == pytest.approx(float(np.cos(np.pi / 4)), rel=1e-5)

    assert results[2].chunk.chunk_id == "P001_ORTHOGONAL"
    assert results[2].score == pytest.approx(0.0, abs=1e-5)


def test_top_k_limits_results(repository, search_engine):
    """12. Verify top_k parameter strictly caps the returned result count."""
    chunks = [
        DocumentChunk(
            chunk_id=f"P001_C{i:03d}",
            paper_id="P001",
            page_number=1,
            paragraph_number=i,
            text=f"Passage {i}",
        )
        for i in range(10)
    ]
    repository.insert_chunks(chunks)
    embeddings = {f"P001_C{i:03d}": make_unit_vector(0) for i in range(10)}
    repository.update_embeddings(embeddings)

    results = search_engine.search(query_embedding=make_unit_vector(0), paper_ids=["P001"], top_k=3)
    assert len(results) == 3


def test_deterministic_tie_breaking(repository, search_engine):
    """13. Verify that chunks with identical similarity scores break ties by chunk_id ascending."""
    chunks = [
        DocumentChunk(
            chunk_id="P001_C_GAMMA",
            paper_id="P001",
            page_number=1,
            paragraph_number=3,
            text="Gamma chunk",
        ),
        DocumentChunk(
            chunk_id="P001_C_ALPHA",
            paper_id="P001",
            page_number=1,
            paragraph_number=1,
            text="Alpha chunk",
        ),
        DocumentChunk(
            chunk_id="P001_C_BETA",
            paper_id="P001",
            page_number=1,
            paragraph_number=2,
            text="Beta chunk",
        ),
    ]
    repository.insert_chunks(chunks)

    # Identical vectors for all three chunks
    shared_vec = make_unit_vector(0)
    repository.update_embeddings({
        "P001_C_GAMMA": shared_vec,
        "P001_C_ALPHA": shared_vec,
        "P001_C_BETA": shared_vec,
    })

    results = search_engine.search(query_embedding=shared_vec, paper_ids=["P001"])
    assert len(results) == 3
    # Scores are all identical (1.0)
    assert all(r.score == pytest.approx(1.0) for r in results)
    # Ordering must be alphabetically ascending by chunk_id
    assert [r.chunk.chunk_id for r in results] == [
        "P001_C_ALPHA",
        "P001_C_BETA",
        "P001_C_GAMMA",
    ]


def test_source_metadata_and_text_preserved(repository, search_engine):
    """14. Verify exact raw text and metadata fields are preserved in VectorSearchResult."""
    raw_text = "\n\t  Precise unmodified chunk with special characters: 92% ± 0.5% \n"
    chunk = DocumentChunk(
        chunk_id="P001_C099",
        paper_id="P001",
        page_number=7,
        paragraph_number=4,
        section="Discussion & Results",
        text=raw_text,
    )
    repository.insert_chunks([chunk])
    repository.update_embeddings({"P001_C099": make_unit_vector(0)})

    results = search_engine.search(query_embedding=make_unit_vector(0), paper_ids=["P001"])
    assert len(results) == 1
    res = results[0].chunk
    assert res.chunk_id == "P001_C099"
    assert res.paper_id == "P001"
    assert res.page_number == 7
    assert res.paragraph_number == 4
    assert res.section == "Discussion & Results"
    assert res.text == raw_text


def test_vector_search_does_not_commit_transactions(repository, search_engine, db_session):
    """15. Verify vector search leaves transactions open and does not commit automatically."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Uncommitted search test",
    )
    repository.insert_chunks([chunk])
    repository.update_embeddings({"P001_C001": make_unit_vector(0)})

    # Execute search
    results = search_engine.search(query_embedding=make_unit_vector(0), paper_ids=["P001"])
    assert len(results) == 1

    # Verify session transaction is still active (uncommitted)
    assert db_session.is_active

    # Rolling back session discards the inserted chunk and embedding
    db_session.rollback()
    assert repository.get_chunks_by_paper_ids(["P001"]) == []


def test_postgres_sql_statement_compilation():
    """16. Verify PostgreSQL pgvector query compiles with '<=>' operator and paper filtering."""
    q_vec = make_unit_vector(0)
    stmt = VectorSearchEngine.build_postgres_search_query(q_vec, paper_ids=["P001", "P002"], top_k=10)

    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))

    # Verify pgvector cosine-distance operator '<=>' is present in the SQL
    assert "<=>" in compiled_sql, f"Expected pgvector '<=>' operator in compiled SQL:\n{compiled_sql}"

    # Verify filtering conditions
    assert "chunks.paper_id IN" in compiled_sql
    assert "chunks.embedding IS NOT NULL" in compiled_sql
    assert "LIMIT" in compiled_sql
    assert "ORDER BY" in compiled_sql


def test_boolean_top_k_rejected_in_vector_search(search_engine):
    """Verify boolean top_k is rejected with ValueError."""
    q_vec = make_unit_vector(0)
    for bool_val in [True, False]:
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search_engine.search(query_embedding=q_vec, paper_ids=["P001"], top_k=bool_val)  # type: ignore[arg-type]
