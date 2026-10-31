"""Unit tests for backend/database/init_db.py (PostgreSQL and pgvector database initialization)."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from backend.database.connection import create_db_engine, get_session_factory
from backend.database.init_db import (
    init_database,
    main,
    sanitize_error_message,
    sanitize_url,
)
from backend.database.models import Base, ChunkRecord
from backend.database.repository import ChunkRepository
from backend.schemas.chunk import DocumentChunk


def test_sanitize_error_message_masks_passwords():
    """Verify sanitize_error_message masks sensitive password credentials."""
    msg1 = "Connection failed to postgresql+psycopg://admin:my_secret_password@db.example.com:5432/research"
    sanitized1 = sanitize_error_message(msg1)
    assert "my_secret_password" not in sanitized1
    assert "admin:***@" in sanitized1

    msg2 = 'driver error: password="top_secret_token" authentication failed'
    sanitized2 = sanitize_error_message(msg2)
    assert "top_secret_token" not in sanitized2
    assert "password='***'" in sanitized2 or "password=***" in sanitized2


def test_init_database_sqlite_creates_tables_and_indexes(tmp_path):
    """Verify init_database creates the chunks table and indexes on SQLite."""
    db_file = tmp_path / "test_init.db"
    db_url = f"sqlite:///{db_file}"

    success = init_database(database_url=db_url)
    assert success is True

    # Inspect the newly initialized database
    engine = create_db_engine(db_url)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "chunks" in tables

    columns = {col["name"]: col for col in inspector.get_columns("chunks")}
    assert "chunk_id" in columns
    assert "paper_id" in columns
    assert "page_number" in columns
    assert "paragraph_number" in columns
    assert "text" in columns
    assert "section" in columns
    assert "embedding" in columns

    indexes = {idx["name"] for idx in inspector.get_indexes("chunks")}
    assert "ix_chunks_paper_page_para" in indexes

    engine.dispose()


def test_init_database_never_drops_tables_or_deletes_data(tmp_path):
    """Verify init_database is strictly non-destructive: existing tables and rows are preserved."""
    db_file = tmp_path / "test_preserve.db"
    db_url = f"sqlite:///{db_file}"

    # First initialization
    assert init_database(database_url=db_url) is True

    engine = create_db_engine(db_url)
    session_factory = get_session_factory(engine)
    session = session_factory()
    repo = ChunkRepository(session)

    # Insert a sample chunk record
    chunk = DocumentChunk(
        chunk_id="chunk_keep_01",
        paper_id="paper_keep_01",
        page_number=1,
        paragraph_number=1,
        text="Critical persisted text that must never be deleted.",
        section="Introduction",
    )
    repo.insert_chunks([chunk])
    session.commit()
    session.close()

    # Run init_database a SECOND time
    assert init_database(database_url=db_url) is True

    # Verify the chunk is still present and intact
    check_session = session_factory()
    check_repo = ChunkRepository(check_session)
    retrieved = check_repo.get_chunks_by_paper_ids(["paper_keep_01"])
    assert len(retrieved) == 1
    assert retrieved[0].chunk_id == "chunk_keep_01"
    assert retrieved[0].text == "Critical persisted text that must never be deleted."
    check_session.close()
    engine.dispose()


def test_init_database_fails_when_database_url_missing(monkeypatch, caplog):
    """Verify init_database returns False and logs an error when DATABASE_URL is unconfigured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with caplog.at_level(logging.ERROR):
        result = init_database(database_url=None)

    assert result is False
    assert any("DATABASE_URL" in record.message and "not configured" in record.message for record in caplog.records)


def test_init_database_fails_gracefully_on_unreachable_host(caplog):
    """Verify init_database fails gracefully and masks password when host is unreachable."""
    unreachable_url = "postgresql+psycopg://user:test_secret_pw@127.0.0.1:59998/test_db"

    with caplog.at_level(logging.ERROR):
        result = init_database(database_url=unreachable_url)

    assert result is False
    assert any("Could not establish database connection" in record.message for record in caplog.records)
    # Ensure password is never printed in logs
    for record in caplog.records:
        assert "test_secret_pw" not in record.message


def test_init_database_postgres_enables_pgvector_extension():
    """Verify pgvector extension command is executed when dialect is postgresql."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_engine.url.render_as_string.return_value = "postgresql+psycopg://user:***@localhost:5432/db"

    # Context managers for connect() and begin()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    # Inspector mock to report all tables exist
    with patch("backend.database.init_db.inspect") as mock_inspect:
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = list(Base.metadata.tables.keys())
        mock_inspect.return_value = mock_inspector

        with patch.object(Base.metadata, "create_all") as mock_create_all:
            result = init_database(engine=mock_engine)
            assert result is True

            # Verify CREATE EXTENSION IF NOT EXISTS vector was called
            calls = [str(call[0][0]) for call in mock_conn.execute.call_args_list]
            assert any("CREATE EXTENSION IF NOT EXISTS vector" in c for c in calls)
            mock_create_all.assert_called_once()


def test_init_database_postgres_extension_failure_returns_false(caplog):
    """Verify init_database returns False and logs error when pgvector extension cannot be enabled."""
    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_engine.url.render_as_string.return_value = "postgresql+psycopg://user:***@localhost:5432/db"

    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    # Simulate permission denied when running CREATE EXTENSION
    mock_begin = MagicMock()
    mock_begin.__enter__.return_value = mock_conn
    mock_conn.execute.side_effect = [
        None,  # SELECT 1 succeeds
        ProgrammingError("statement", {}, Exception("permission denied to create extension 'vector'")),
    ]
    mock_engine.begin.return_value = mock_begin

    with caplog.at_level(logging.ERROR):
        result = init_database(engine=mock_engine)

    assert result is False
    assert any("Failed to enable 'vector' extension" in record.message for record in caplog.records)


def test_cli_main_success_returns_zero(tmp_path):
    """Verify CLI main entrypoint returns 0 on successful initialization."""
    db_file = tmp_path / "cli_test.db"
    exit_code = main(["--url", f"sqlite:///{db_file}"])
    assert exit_code == 0


def test_cli_main_failure_returns_one(monkeypatch):
    """Verify CLI main entrypoint returns 1 on failure."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    exit_code = main(["--url", "postgresql+psycopg://user:pass@127.0.0.1:59997/nonexistent"])
    assert exit_code == 1
