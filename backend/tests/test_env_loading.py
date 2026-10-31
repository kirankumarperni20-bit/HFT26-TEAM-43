"""Tests for environment variable loading and missing DATABASE_URL handling."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.database.connection import (
    DatabaseConfigurationError,
    get_database_url,
    get_db_session_dep,
    is_database_configured,
    load_environment,
    reset_cached_engine,
)
from backend.llm.schemas.claim import AtomicClaim
from backend.llm.schemas.verification import RetrievedEvidence
from backend.llm.services.retrieval_service import EvidenceRetrievalService
from backend.main import app


@pytest.fixture(autouse=True)
def clean_database_env(monkeypatch):
    """Ensure DATABASE_URL environment is cleanly isolated and reset cached engine."""
    reset_cached_engine()
    yield
    reset_cached_engine()


def test_load_environment_from_custom_path(tmp_path, monkeypatch):
    """Verify load_environment correctly parses a .env file and does not overwrite existing env."""
    custom_env = tmp_path / "custom.env"
    custom_db = tmp_path / "custom_loaded.db"
    custom_env.write_text(f"DATABASE_URL=sqlite:///{custom_db}\nEXISTING_VAR=from_file\n")

    monkeypatch.setenv("EXISTING_VAR", "already_set")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    loaded = load_environment(force_reload=True, env_path=custom_env)
    assert loaded is True
    assert os.environ.get("DATABASE_URL") == f"sqlite:///{custom_db}"
    # override=False must preserve the existing OS environment variable
    assert os.environ.get("EXISTING_VAR") == "already_set"


def test_load_environment_handles_nonexistent_file():
    """Verify load_environment returns False without crashing for a non-existent file."""
    loaded = load_environment(force_reload=True, env_path="/path/to/nonexistent/.env")
    assert loaded is False


def test_is_database_configured(monkeypatch):
    """Verify is_database_configured returns False when unset and True when configured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # When unset, should return False
    assert is_database_configured() is False

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/db")
    assert is_database_configured() is True

    # Whitespace only should return False
    monkeypatch.setenv("DATABASE_URL", "   ")
    assert is_database_configured() is False


def test_get_database_url_missing_raises_configuration_error(monkeypatch):
    """Verify get_database_url raises DatabaseConfigurationError when DATABASE_URL is not set."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        get_database_url()

    assert "Production database URL is not configured" in str(exc_info.value)
    assert "DATABASE_URL" in str(exc_info.value)


def test_get_database_url_explicit_argument(monkeypatch):
    """Verify get_database_url prefers an explicit argument and strips whitespace."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    url = get_database_url("  sqlite:///test.db  ")
    assert url == "sqlite:///test.db"


def test_get_db_session_dep_yields_none_when_unconfigured(monkeypatch, caplog):
    """Verify get_db_session_dep yields None and logs clear warning when DATABASE_URL is missing."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_cached_engine()

    with caplog.at_level(logging.WARNING):
        gen = get_db_session_dep()
        session = next(gen)
        assert session is None
        with pytest.raises(StopIteration):
            next(gen)

    assert any(
        "DATABASE_URL is not configured" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    )


def test_get_db_session_dep_yields_none_and_logs_error_on_connection_failure(monkeypatch, caplog):
    """Verify get_db_session_dep yields None and logs an error without exposing secrets when connection fails."""
    # Point to an unreachable host with simulated secret password
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:super_secret_password@127.0.0.1:59999/nonexistent_db")
    reset_cached_engine()

    with caplog.at_level(logging.ERROR):
        gen = get_db_session_dep()
        session = next(gen)
        assert session is None
        with pytest.raises(StopIteration):
            next(gen)

    # Confirm error logged
    assert any(
        "Failed to initialize database session" in record.message for record in caplog.records
    )
    # Confirm secret credentials are not leaked in log output
    for record in caplog.records:
        assert "super_secret_password" not in record.message


def test_get_db_session_dep_yields_active_session_and_closes_it(monkeypatch):
    """Verify get_db_session_dep yields an active session for valid SQLite and closes cleanly."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    reset_cached_engine()

    gen = get_db_session_dep()
    session = next(gen)
    assert session is not None

    with patch.object(session, "close", wraps=session.close) as mock_close:
        # Complete the generator
        with pytest.raises(StopIteration):
            next(gen)

        mock_close.assert_called_once()


def test_evidence_retrieval_service_is_database_backed_flag():
    """Verify is_database_backed reflects whether an EvidenceRetriever is bound."""
    service_fallback = EvidenceRetrievalService(retriever=None)
    assert service_fallback.is_database_backed is False

    mock_retriever = MagicMock()
    service_backed = EvidenceRetrievalService(retriever=mock_retriever)
    assert service_backed.is_database_backed is True


def test_evidence_retrieval_service_logs_fallback_mode(caplog):
    """Verify retrieve() clearly logs that it is using local PDF fallback when retriever is None."""
    service = EvidenceRetrievalService(retriever=None)
    claim = AtomicClaim(
        claim_id="C001",
        claim_text="The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
        source_text="The Transformer model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.",
        involved_papers=["Vaswani_2017_Attention"],
    )

    with caplog.at_level(logging.INFO):
        results = service.retrieve(claim, allowed_paper_ids=["Vaswani_2017_Attention"])

    assert any(
        "EvidenceRetriever is inactive" in record.message
        and "Falling back to local PDF evidence corpus" in record.message
        for record in caplog.records
    )
    assert len(results) > 0
    assert results[0].retrieval_method in ("reference_hint", "lexical_overlap")


def test_verify_endpoint_gracefully_degrades_when_db_is_none(monkeypatch):
    """Verify POST /api/research/verify works seamlessly using local fallback when DB is unconfigured."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_cached_engine()

    client = TestClient(app)
    payload = {
        "claims": [
            {
                "claim_id": "claim_test_01",
                "claim_text": "The Transformer uses multi-head attention.",
                "source_text": "The Transformer model uses multi-head attention mechanisms.",
                "involved_papers": ["Vaswani_2017_Attention"],
            }
        ],
        "allowed_paper_ids": ["Vaswani_2017_Attention"],
    }

    mock_llm_json = json.dumps({
        "verifications": [
            {
                "claim_id": "claim_test_01",
                "verification_status": "SUPPORTED",
                "explanation": "Verified against attention text.",
                "supporting_evidence": [
                    {
                        "evidence_id": "ev_vaswani_p1_01",
                        "quote": "multi-head attention",
                    }
                ],
            }
        ]
    })

    with patch("backend.llm.client.LLMClient.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = mock_llm_json
        response = client.post("/api/research/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_claims"] == 1
        assert len(data["verifications"]) == 1
        assert data["verifications"][0]["claim_id"] == "claim_test_01"


