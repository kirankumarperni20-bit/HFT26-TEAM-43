"""Unit tests for SQLAlchemy models, connection utilities, and ChunkRepository (Step 4)."""

import os
import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from backend.database.connection import (
    DatabaseConfigurationError,
    create_db_engine,
    get_database_url,
    get_db_session,
    get_session_factory,
)
from backend.database.models import Base, ChunkRecord
from backend.database.repository import (
    ChunkRepository,
    DatabaseIntegrityError,
    DuplicateChunkError,
    is_duplicate_chunk_id_violation,
)
from backend.schemas.chunk import DocumentChunk


@pytest.fixture
def sqlite_engine():
    """Create a temporary in-memory SQLite engine with fresh tables for each test."""
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
    """Provide a ChunkRepository instance bound to the active test session."""
    return ChunkRepository(session=db_session)


def test_table_creation(sqlite_engine):
    """1. Verify that SQLAlchemy creates the 'chunks' table and required columns successfully."""
    inspector = inspect(sqlite_engine)
    table_names = inspector.get_table_names()
    assert "chunks" in table_names

    columns = {col["name"]: col for col in inspector.get_columns("chunks")}
    assert "chunk_id" in columns
    assert "paper_id" in columns
    assert "page_number" in columns
    assert "paragraph_number" in columns
    assert "text" in columns
    assert "section" in columns

    pk_constraint = inspector.get_pk_constraint("chunks")
    assert pk_constraint["constrained_columns"] == ["chunk_id"]


def test_insert_valid_document_chunk(repository, db_session):
    """2. Verify that a valid DocumentChunk can be inserted and returned count is 1."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        section="Abstract",
        text="Sample abstract text.",
    )
    count = repository.insert_chunks([chunk])
    assert count == 1

    record = db_session.get(ChunkRecord, "P001_C001")
    assert record is not None
    assert record.chunk_id == "P001_C001"


def test_metadata_fields_persistence(repository):
    """3. Verify all metadata fields (page, paragraph, section) are persisted and mapped back correctly."""
    chunk = DocumentChunk(
        chunk_id="P001_C002",
        paper_id="P001",
        page_number=5,
        paragraph_number=3,
        section="Results",
        text="The proposed model achieved an accuracy of 92%.",
    )
    repository.insert_chunks([chunk])

    retrieved = repository.get_chunks_by_paper_ids(["P001"])
    assert len(retrieved) == 1
    c = retrieved[0]
    assert c.chunk_id == "P001_C002"
    assert c.paper_id == "P001"
    assert c.page_number == 5
    assert c.paragraph_number == 3
    assert c.section == "Results"
    assert c.text == "The proposed model achieved an accuracy of 92%."


def test_exact_original_text_preservation(repository):
    """4. Verify that original text with leading/trailing spaces, newlines, tabs is preserved exactly."""
    raw_text = "\n\t   Raw extracted chunk text with special symbols: 92% ± 0.5%   \n\n"
    chunk = DocumentChunk(
        chunk_id="P001_C003",
        paper_id="P001",
        page_number=2,
        paragraph_number=4,
        section=None,
        text=raw_text,
    )
    repository.insert_chunks([chunk])

    retrieved = repository.get_chunks_by_paper_ids(["P001"])
    assert len(retrieved) == 1
    assert retrieved[0].text == raw_text
    assert retrieved[0].section is None


def test_retrieval_filters_by_paper_ids(repository):
    """5. Verify retrieval returns only chunks belonging to the requested paper IDs."""
    chunks = [
        DocumentChunk(
            chunk_id="P001_C001",
            paper_id="P001",
            page_number=1,
            paragraph_number=1,
            text="Paper 1 chunk 1",
        ),
        DocumentChunk(
            chunk_id="P001_C002",
            paper_id="P001",
            page_number=2,
            paragraph_number=1,
            text="Paper 1 chunk 2",
        ),
        DocumentChunk(
            chunk_id="P002_C001",
            paper_id="P002",
            page_number=1,
            paragraph_number=1,
            text="Paper 2 chunk 1",
        ),
        DocumentChunk(
            chunk_id="P003_C001",
            paper_id="P003",
            page_number=1,
            paragraph_number=1,
            text="Paper 3 chunk 1",
        ),
    ]
    repository.insert_chunks(chunks)

    # Filter for P001 only
    results_p1 = repository.get_chunks_by_paper_ids(["P001"])
    assert len(results_p1) == 2
    assert {c.chunk_id for c in results_p1} == {"P001_C001", "P001_C002"}

    # Filter for P001 and P003
    results_p1_p3 = repository.get_chunks_by_paper_ids(["P001", "P003"])
    assert len(results_p1_p3) == 3
    assert {c.chunk_id for c in results_p1_p3} == {"P001_C001", "P001_C002", "P003_C001"}


def test_retrieval_empty_paper_ids_returns_empty_list(repository):
    """6. Verify retrieval returns an empty list immediately when paper_ids is empty."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Some text",
    )
    repository.insert_chunks([chunk])

    assert repository.get_chunks_by_paper_ids([]) == []


def test_retrieval_non_matching_paper_ids_returns_empty_list(repository):
    """7. Verify retrieval returns an empty list when no matching papers exist."""
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Some text",
    )
    repository.insert_chunks([chunk])

    assert repository.get_chunks_by_paper_ids(["NON_EXISTENT_PAPER"]) == []


def test_duplicate_chunk_id_raises_error(repository):
    """8. Verify duplicate chunk IDs raise DuplicateChunkError without overwriting existing data."""
    chunk1 = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Initial text.",
    )
    repository.insert_chunks([chunk1])

    # Attempt to insert same chunk_id with modified text
    duplicate_chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Overwritten text.",
    )
    with pytest.raises(DuplicateChunkError) as exc_info:
        repository.insert_chunks([duplicate_chunk])

    assert "P001_C001" in str(exc_info.value)

    # Confirm original text remains intact
    retrieved = repository.get_chunks_by_paper_ids(["P001"])
    assert len(retrieved) == 1
    assert retrieved[0].text == "Initial text."


def test_intra_batch_duplicate_chunk_id_raises_error(repository):
    """Verify that multiple chunks with identical chunk_id in the same batch raise DuplicateChunkError."""
    chunk1 = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="First instance.",
    )
    chunk2 = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=2,
        text="Second instance with same ID.",
    )
    with pytest.raises(DuplicateChunkError) as exc_info:
        repository.insert_chunks([chunk1, chunk2])

    assert "P001_C001" in str(exc_info.value)


def test_empty_insertion_returns_zero(repository):
    """9. Verify that inserting an empty collection returns 0."""
    count = repository.insert_chunks([])
    assert count == 0


def test_database_configuration_error_when_url_missing(monkeypatch):
    """10. Verify DatabaseConfigurationError is raised when no URL is supplied or in environment."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "DATABASE_URL" in str(exc_info.value)


def test_get_database_url_prefers_explicit_argument(monkeypatch):
    """Verify explicit URL argument takes precedence over environment variable."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://env_user:env_pass@localhost:5432/env_db")
    explicit_url = "sqlite:///explicit.db"
    assert get_database_url(explicit_url) == explicit_url


def test_stable_chunk_ordering(repository):
    """Verify chunks are returned with deterministic ordering: paper_id, page_number, paragraph_number, chunk_id."""
    # Insert in reverse order
    chunks = [
        DocumentChunk(
            chunk_id="P001_C003",
            paper_id="P001",
            page_number=3,
            paragraph_number=1,
            text="Page 3",
        ),
        DocumentChunk(
            chunk_id="P001_C001",
            paper_id="P001",
            page_number=1,
            paragraph_number=2,
            text="Page 1 Para 2",
        ),
        DocumentChunk(
            chunk_id="P001_C000",
            paper_id="P001",
            page_number=1,
            paragraph_number=1,
            text="Page 1 Para 1",
        ),
    ]
    repository.insert_chunks(chunks)

    results = repository.get_chunks_by_paper_ids(["P001"])
    assert [c.chunk_id for c in results] == ["P001_C000", "P001_C001", "P001_C003"]


def test_insert_leaves_transaction_open_until_caller_commits(sqlite_engine):
    """Verify that insert_chunks() leaves the transaction open and does not commit prematurely."""
    session_factory = get_session_factory(sqlite_engine)
    session = session_factory()
    repo = ChunkRepository(session)

    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Uncommitted chunk text",
    )
    repo.insert_chunks([chunk])

    # The transaction in session is active and uncommitted
    assert session.is_active

    # Flushed record is visible in the active uncommitted transaction
    assert len(repo.get_chunks_by_paper_ids(["P001"])) == 1

    # Because repository did NOT commit, caller rollback discards the flushed changes
    session.rollback()

    # After rollback, the record is gone, proving repository did not commit
    assert repo.get_chunks_by_paper_ids(["P001"]) == []
    session.close()


def test_get_db_session_commits_on_successful_exit(sqlite_engine):
    """Verify that exiting get_db_session() context manager successfully commits all inserted records."""
    session_factory = get_session_factory(sqlite_engine)
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Committed via get_db_session",
    )

    with get_db_session(session_factory) as session:
        repo = ChunkRepository(session)
        count = repo.insert_chunks([chunk])
        assert count == 1

    # Verify persistence using a fresh session
    with get_db_session(session_factory) as check_session:
        repo_check = ChunkRepository(check_session)
        records = repo_check.get_chunks_by_paper_ids(["P001"])
        assert len(records) == 1
        assert records[0].chunk_id == "P001_C001"
        assert records[0].text == "Committed via get_db_session"


def test_get_db_session_rolls_back_on_exception(sqlite_engine):
    """Verify that an exception raised inside get_db_session() rolls back the transaction."""
    session_factory = get_session_factory(sqlite_engine)
    chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Should be rolled back",
    )

    with pytest.raises(RuntimeError) as exc_info:
        with get_db_session(session_factory) as session:
            repo = ChunkRepository(session)
            repo.insert_chunks([chunk])
            raise RuntimeError("Simulated transaction failure")

    assert "Simulated transaction failure" in str(exc_info.value)

    # Verify nothing was persisted
    with get_db_session(session_factory) as check_session:
        repo_check = ChunkRepository(check_session)
        assert repo_check.get_chunks_by_paper_ids(["P001"]) == []


def test_failed_batch_does_not_partially_persist(sqlite_engine):
    """Verify that a batch with duplicate errors rolls back cleanly without partial persistence."""
    session_factory = get_session_factory(sqlite_engine)

    # Pre-seed one record
    seed_chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Seed chunk",
    )
    with get_db_session(session_factory) as session:
        ChunkRepository(session).insert_chunks([seed_chunk])

    # Now attempt batch insertion containing one new valid chunk and one duplicate chunk
    new_chunk = DocumentChunk(
        chunk_id="P001_C002",
        paper_id="P001",
        page_number=1,
        paragraph_number=2,
        text="New chunk that should not persist",
    )
    duplicate_chunk = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=3,
        text="Duplicate chunk ID",
    )

    with pytest.raises(DuplicateChunkError):
        with get_db_session(session_factory) as session:
            repo = ChunkRepository(session)
            repo.insert_chunks([new_chunk, duplicate_chunk])

    # Verify that new_chunk was NOT partially persisted
    with get_db_session(session_factory) as check_session:
        repo_check = ChunkRepository(check_session)
        chunks = repo_check.get_chunks_by_paper_ids(["P001"])
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "P001_C001"


class MockPsycopgDiagnostic:
    def __init__(
        self,
        constraint_name: str | None = None,
        column_name: str | None = None,
        sqlstate: str = "23505",
        table_name: str = "chunks",
        message_primary: str = "",
    ) -> None:
        self.constraint_name = constraint_name
        self.column_name = column_name
        self.sqlstate = sqlstate
        self.table_name = table_name
        self.message_primary = message_primary


class MockPsycopgError(Exception):
    def __init__(
        self,
        message: str,
        sqlstate: str = "23505",
        diag: MockPsycopgDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate
        self.diag = diag


class MockSqliteDriverError(Exception):
    def __init__(
        self,
        message: str,
        sqlite_errorname: str,
        sqlite_errorcode: int,
    ) -> None:
        super().__init__(message)
        self.sqlite_errorname = sqlite_errorname
        self.sqlite_errorcode = sqlite_errorcode


def test_postgres_psycopg_sqlstate_23505_with_chunks_pkey_raises_duplicate_chunk_error(repository, monkeypatch):
    """1. PostgreSQL 23505 with chunks_pkey -> DuplicateChunkError with chained cause."""
    from sqlalchemy.exc import IntegrityError

    diag = MockPsycopgDiagnostic(constraint_name="chunks_pkey", sqlstate="23505", table_name="chunks")
    driver_err = MockPsycopgError("duplicate key value violates unique constraint 'chunks_pkey'", sqlstate="23505", diag=diag)
    exc = IntegrityError("statement", orig=driver_err, params={})

    assert is_duplicate_chunk_id_violation(exc) is True

    monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc))

    chunk = DocumentChunk(chunk_id="P001_CONCURRENT", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
    with pytest.raises(DuplicateChunkError) as exc_info:
        repository.insert_chunks([chunk])

    assert exc_info.value.__cause__ is exc


def test_postgres_psycopg_sqlstate_23505_with_different_constraint_raises_database_integrity_error(repository, monkeypatch):
    """2. PostgreSQL 23505 with another constraint -> DatabaseIntegrityError."""
    from sqlalchemy.exc import IntegrityError

    diag = MockPsycopgDiagnostic(constraint_name="uq_some_other_unique_index", sqlstate="23505", table_name="chunks")
    driver_err = MockPsycopgError("duplicate key on other column", sqlstate="23505", diag=diag)
    exc = IntegrityError("statement", orig=driver_err, params={})

    assert is_duplicate_chunk_id_violation(exc) is False

    monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc))

    chunk = DocumentChunk(chunk_id="P001_OTHER_UQ", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
    with pytest.raises(DatabaseIntegrityError) as exc_info:
        repository.insert_chunks([chunk])

    assert exc_info.value.__cause__ is exc


def test_postgres_psycopg_sqlstate_23505_without_sufficient_diagnostics_raises_database_integrity_error(repository, monkeypatch):
    """3. PostgreSQL 23505 without sufficient diagnostics, even when message mentions chunk_id -> DatabaseIntegrityError."""
    from sqlalchemy.exc import IntegrityError

    diag = MockPsycopgDiagnostic(constraint_name=None, column_name=None, sqlstate="23505", table_name="chunks", message_primary="Key (chunk_id)=(P001_C001) already exists.")
    driver_err = MockPsycopgError("Key (chunk_id)=(P001_C001) already exists.", sqlstate="23505", diag=diag)
    exc = IntegrityError("statement", orig=driver_err, params={})

    assert is_duplicate_chunk_id_violation(exc) is False

    monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc))

    chunk = DocumentChunk(chunk_id="P001_NO_DIAG", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
    with pytest.raises(DatabaseIntegrityError) as exc_info:
        repository.insert_chunks([chunk])

    assert exc_info.value.__cause__ is exc


def test_postgres_psycopg_not_null_foreign_key_and_check_raise_database_integrity_error(repository, monkeypatch):
    """4. PostgreSQL NOT NULL, foreign-key and check violations -> DatabaseIntegrityError."""
    from sqlalchemy.exc import IntegrityError

    for state, name in [("23502", "not null"), ("23503", "foreign key"), ("23514", "check constraint")]:
        diag = MockPsycopgDiagnostic(sqlstate=state, message_primary=f"violates {name}")
        driver_err = MockPsycopgError(f"violates {name}", sqlstate=state, diag=diag)
        exc = IntegrityError("statement", orig=driver_err, params={})

        assert is_duplicate_chunk_id_violation(exc) is False

        monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc))
        chunk = DocumentChunk(chunk_id="P001_C_ERR", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
        with pytest.raises(DatabaseIntegrityError) as exc_info:
            repository.insert_chunks([chunk])
        assert exc_info.value.__cause__ is exc


def test_sqlite_primary_key_violation_raises_duplicate_chunk_error(repository, monkeypatch):
    """5. SQLite primary-key violation on chunks -> DuplicateChunkError."""
    from sqlalchemy.exc import IntegrityError

    orig_pk = MockSqliteDriverError(
        "UNIQUE constraint failed: chunks.chunk_id",
        sqlite_errorname="SQLITE_CONSTRAINT_PRIMARYKEY",
        sqlite_errorcode=1555,
    )
    exc_pk = IntegrityError("statement", orig=orig_pk, params={})
    assert is_duplicate_chunk_id_violation(exc_pk) is True

    monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc_pk))
    chunk = DocumentChunk(chunk_id="P001_SQLITE_PK", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
    with pytest.raises(DuplicateChunkError) as exc_info:
        repository.insert_chunks([chunk])
    assert exc_info.value.__cause__ is exc_pk


def test_sqlite_unique_violation_without_reliable_chunk_id_raises_database_integrity_error(repository, monkeypatch):
    """6. SQLite unique violation without reliable chunk ID identification -> DatabaseIntegrityError."""
    from sqlalchemy.exc import IntegrityError

    orig_uq = MockSqliteDriverError(
        "UNIQUE constraint failed: chunks.chunk_id",
        sqlite_errorname="SQLITE_CONSTRAINT_UNIQUE",
        sqlite_errorcode=2067,
    )
    exc_uq = IntegrityError("statement", orig=orig_uq, params={})
    assert is_duplicate_chunk_id_violation(exc_uq) is False

    monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc_uq))
    chunk = DocumentChunk(chunk_id="P001_SQLITE_UQ", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
    with pytest.raises(DatabaseIntegrityError) as exc_info:
        repository.insert_chunks([chunk])
    assert exc_info.value.__cause__ is exc_uq


def test_sqlite_not_null_foreign_key_and_check_raise_database_integrity_error(repository, monkeypatch):
    """7. SQLite NOT NULL, foreign-key and check violations -> DatabaseIntegrityError."""
    from sqlalchemy.exc import IntegrityError

    for err_name, code, msg in [
        ("SQLITE_CONSTRAINT_NOTNULL", 1299, "NOT NULL constraint failed: chunks.text"),
        ("SQLITE_CONSTRAINT_CHECK", 275, "CHECK constraint failed: chunks"),
        ("SQLITE_CONSTRAINT_FOREIGNKEY", 787, "FOREIGN KEY constraint failed"),
    ]:
        orig = MockSqliteDriverError(msg, sqlite_errorname=err_name, sqlite_errorcode=code)
        exc = IntegrityError("statement", orig=orig, params={})

        assert is_duplicate_chunk_id_violation(exc) is False

        monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc))
        chunk = DocumentChunk(chunk_id="P001_NN", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
        with pytest.raises(DatabaseIntegrityError) as exc_info:
            repository.insert_chunks([chunk])
        assert exc_info.value.__cause__ is exc


def test_unrecognized_integrity_error_defaults_to_database_integrity_error(repository, monkeypatch):
    """Verify an integrity error that cannot be confidently classified raises DatabaseIntegrityError."""
    from sqlalchemy.exc import IntegrityError

    exc = IntegrityError("statement", orig=Exception("Unknown driver error"), params={})
    assert is_duplicate_chunk_id_violation(exc) is False

    monkeypatch.setattr(repository.session, "flush", lambda: (_ for _ in ()).throw(exc))
    chunk = DocumentChunk(chunk_id="P001_UNREC", paper_id="P001", page_number=1, paragraph_number=1, text="Text")
    with pytest.raises(DatabaseIntegrityError):
        repository.insert_chunks([chunk])


def test_chunk_record_to_pydantic_conversion():
    """Verify ChunkRecord.to_pydantic() converts an ORM model accurately to a DocumentChunk."""
    record = ChunkRecord(
        chunk_id="P001_C005",
        paper_id="P001",
        page_number=3,
        paragraph_number=2,
        text="Direct conversion passage.",
        section="Methods",
    )
    chunk = record.to_pydantic()
    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == "P001_C005"
    assert chunk.paper_id == "P001"
    assert chunk.page_number == 3
    assert chunk.paragraph_number == 2
    assert chunk.text == "Direct conversion passage."
    assert chunk.section == "Methods"


def test_create_db_engine_configures_pool_pre_ping_for_postgres():
    """Verify create_db_engine defaults pool_pre_ping=True for non-sqlite engines."""
    engine = create_db_engine("postgresql+psycopg://user:pass@localhost:5432/db")
    assert engine.pool._pre_ping is True
