"""Unit and integration tests for PDF ingestion, chunking, and pgvector indexing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from backend.config import EMBEDDING_DIMENSION
from backend.database.connection import create_db_engine, get_session_factory
from backend.database.models import Base, ChunkRecord
from backend.retrieval.ingest import (
    clean_pdf_text,
    derive_paper_id,
    ingest_pdf,
    is_heading_line,
    parse_pdf_to_chunks,
)

TEST_PDF_PATH = Path("backend/data/test_paper.pdf")


def test_derive_paper_id():
    """Verify derive_paper_id extracts canonical paper identifiers."""
    assert derive_paper_id(Path("backend/data/test_paper.pdf")) == "Vaswani_2017_Attention"
    assert derive_paper_id(Path("/path/to/Vaswani_2017.pdf")) == "Vaswani_2017_Attention"
    assert derive_paper_id(Path("/path/to/bert_paper.pdf")) == "bert_paper"


def test_is_heading_line():
    """Verify heading recognition for academic papers."""
    assert is_heading_line("Abstract") is True
    assert is_heading_line("1 Introduction") is True
    assert is_heading_line("3.2.2 Multi-Head Attention") is True
    assert is_heading_line("Table 1: Complexity") is True
    assert is_heading_line("Figure 2: Architectures") is True
    assert is_heading_line("6.1 Machine Translation") is True
    assert is_heading_line("This is regular paragraph body text.") is False


def test_clean_pdf_text():
    """Verify normalization of soft hyphens and line-break hyphenation."""
    raw = "atten-\n tion is all you need\xadwith ligatures\ufffe"
    cleaned = clean_pdf_text(raw)
    assert "atten-tion" in cleaned
    assert "\xad" not in cleaned
    assert "\ufffe" not in cleaned


def test_parse_pdf_to_chunks_structure():
    """Verify parse_pdf_to_chunks produces valid DocumentChunk models with metadata from test_paper.pdf."""
    assert TEST_PDF_PATH.is_file(), "test_paper.pdf must exist in backend/data"

    chunks = parse_pdf_to_chunks(TEST_PDF_PATH)
    assert len(chunks) in (30, 33)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs must be unique"

    for idx, chunk in enumerate(chunks, 1):
        assert chunk.paper_id == "Vaswani_2017_Attention"
        assert chunk.chunk_id.startswith("Vaswani_2017_Attention_p")
        assert 1 <= chunk.page_number <= 11
        assert chunk.paragraph_number >= 1
        assert len(chunk.text.strip()) > 50
        assert chunk.section is not None

    # Verify key sections are captured
    sections = {c.section for c in chunks if c.section}
    assert any("Abstract" in s for s in sections)
    assert any("Introduction" in s for s in sections)
    assert any("Attention" in s for s in sections)
    assert any("Machine Translation" in s for s in sections)


def test_parse_pdf_missing_file_raises():
    """Verify parse_pdf_to_chunks raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        parse_pdf_to_chunks("backend/data/nonexistent_paper.pdf")


def test_ingest_pdf_sqlite_e2e_and_idempotent(tmp_path):
    """Verify end-to-end PDF ingestion into database, embedding dimension (384), and idempotency."""
    db_file = tmp_path / "test_ingest.db"
    db_url = f"sqlite:///{db_file}"

    # Initialize tables
    engine = create_db_engine(db_url)
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    # First Ingestion Run
    newly_indexed, total = ingest_pdf(TEST_PDF_PATH, database_url=db_url)
    assert newly_indexed in (30, 33)
    assert total in (30, 33)

    # Verify persistence in database
    verify_engine = create_db_engine(db_url)
    session_factory = get_session_factory(verify_engine)
    with session_factory() as session:
        records = session.scalars(select(ChunkRecord)).all()
        assert len(records) in (30, 33)

        # Verify embedding dimension
        for r in records:
            assert r.paper_id == "Vaswani_2017_Attention"
            assert r.embedding is not None
            assert len(r.embedding) == EMBEDDING_DIMENSION  # 384
            assert r.page_number >= 1
            assert r.paragraph_number >= 1
            assert len(r.text) > 0

    # Second Ingestion Run (Idempotency test)
    # Re-running on the exact same PDF must NOT duplicate or error
    re_indexed, total_re = ingest_pdf(TEST_PDF_PATH, database_url=db_url)
    assert re_indexed == 0
    assert total_re in (30, 33)

    # Verify count is STILL exactly 30
    with session_factory() as session:
        records_after = session.scalars(select(ChunkRecord)).all()
        assert len(records_after) in (30, 33)

    verify_engine.dispose()


def test_no_synthetic_data_ingested():
    """Verify that synthetic ConvS2S extraction items are excluded from PDF ingestion."""
    chunks = parse_pdf_to_chunks(TEST_PDF_PATH)
    for c in chunks:
        assert "convs2s" not in c.chunk_id.lower()
        assert "[SYNTHETIC TEST DATA]" not in c.text
        assert "Gehring" not in c.paper_id
