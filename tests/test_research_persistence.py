"""
Integration and persistence tests for Research Summary & Job lifecycle.
Verifies:
1. GET /api/research returns past jobs sorted newest-first.
2. GET /api/research/{id} successfully restores past job state across page transitions.
3. InMemoryResearchJobRepository persists snapshots to disk and reloads on server restart.
4. Jobs with research focus queries preserve focus across serialization.
5. Missing or invalid job IDs are gracefully rejected with 404.
"""

import copy
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.documents import get_repository as get_doc_repo
from backend.api.research import get_research_job_repository, get_research_service
from backend.ingestion.repository import InMemoryDocumentRepository
from backend.research.repository import InMemoryResearchJobRepository, ResearchJobRecord
from backend.research.service import ResearchJobService
from backend.schemas.document import DocumentMetadata, DocumentStatus
from backend.schemas.research import (
    ResearchStatus,
    ResearchTask,
    GroundedField,
    PaperSummary,
    PaperSynthesisItem,
    EvidenceReference,
)


@pytest.fixture
def persistence_setup(tmp_path):
    storage_file = tmp_path / "saved_jobs.json"
    doc_repo = InMemoryDocumentRepository()
    job_repo = InMemoryResearchJobRepository(storage_path=storage_file)
    service = ResearchJobService(job_repository=job_repo, document_repository=doc_repo)

    app.dependency_overrides[get_doc_repo] = lambda: doc_repo
    app.dependency_overrides[get_research_job_repository] = lambda: job_repo
    app.dependency_overrides[get_research_service] = lambda: service

    client = TestClient(app)

    # Seed test documents
    doc1 = DocumentMetadata(
        paper_id="P_pers_001",
        filename="persistence_paper_1.pdf",
        status=DocumentStatus.COMPLETED,
    )
    doc2 = DocumentMetadata(
        paper_id="P_pers_002",
        filename="persistence_paper_2.pdf",
        status=DocumentStatus.COMPLETED,
    )
    doc_repo.save_document(doc1)
    doc_repo.save_document(doc2)

    yield {
        "client": client,
        "doc_repo": doc_repo,
        "job_repo": job_repo,
        "service": service,
        "storage_file": storage_file,
    }

    app.dependency_overrides.clear()


def test_list_research_jobs_empty(persistence_setup):
    """GET /api/research returns empty list when no jobs exist."""
    client = persistence_setup["client"]
    resp = client.get("/api/research")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_research_jobs_newest_first(persistence_setup):
    """GET /api/research returns all saved jobs sorted newest-first."""
    client = persistence_setup["client"]
    job_repo = persistence_setup["job_repo"]

    # Save three jobs with distinct creation timestamps
    job1 = ResearchJobRecord(
        research_id="job_old",
        status=ResearchStatus.COMPLETED,
        paper_ids=["P_pers_001"],
        tasks=[ResearchTask.SUMMARIZE],
        created_at="2026-10-31T10:00:00.000000Z",
        completed_at="2026-10-31T10:01:00.000000Z",
    )
    job2 = ResearchJobRecord(
        research_id="job_mid",
        status=ResearchStatus.COMPLETED,
        paper_ids=["P_pers_001", "P_pers_002"],
        tasks=[ResearchTask.SUMMARIZE, ResearchTask.COMPARE],
        created_at="2026-10-31T11:00:00.000000Z",
        completed_at="2026-10-31T11:01:00.000000Z",
        research_focus="Evaluation metrics",
    )
    job3 = ResearchJobRecord(
        research_id="job_newest",
        status=ResearchStatus.PROCESSING,
        paper_ids=["P_pers_002"],
        tasks=[ResearchTask.SUMMARIZE],
        created_at="2026-10-31T12:00:00.000000Z",
    )

    job_repo.save_job(job1)
    job_repo.save_job(job2)
    job_repo.save_job(job3)

    resp = client.get("/api/research")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) == 3

    # Verify descending order by created_at
    assert jobs[0]["research_id"] == "job_newest"
    assert jobs[1]["research_id"] == "job_mid"
    assert jobs[2]["research_id"] == "job_old"
    assert jobs[1]["research_focus"] == "Evaluation metrics"


def test_restore_job_by_id_after_navigation(persistence_setup):
    """
    Simulates navigating away from Research Workspace to Documents and returning.
    Frontend fetches GET /api/research/{job_id} using saved localStorage ID.
    """
    client = persistence_setup["client"]
    job_repo = persistence_setup["job_repo"]

    # Seed completed synthesis result
    test_summary = PaperSynthesisItem(
        paper_id="P_pers_001",
        summary=PaperSummary(
            objective=GroundedField(
                text="Investigate persistence in distributed vector storage.",
                evidence=[
                    EvidenceReference(
                        chunk_id="chk_01",
                        paper_id="P_pers_001",
                        page_number=1,
                        paragraph_number=2,
                        excerpt="We propose a persistent indexing mechanism.",
                    )
                ],
            ),
            methodology=GroundedField(
                text="Empirical evaluation across synthetic and real benchmark suites.",
            ),
            findings=[
                GroundedField(
                    text="Persistence reduces reload latency by 94%.",
                    evidence=[
                        EvidenceReference(
                            chunk_id="chk_02",
                            paper_id="P_pers_001",
                            page_number=4,
                            paragraph_number=1,
                            excerpt="Reload latency was measured at 6% of baseline.",
                        )
                    ],
                )
            ],
            limitations=[
                GroundedField(text="Storage overhead is 12% higher on disk.")
            ],
        ),
    )

    record = ResearchJobRecord(
        research_id="job_persist_test",
        status=ResearchStatus.COMPLETED,
        paper_ids=["P_pers_001"],
        tasks=[ResearchTask.SUMMARIZE],
        created_at="2026-10-31T11:30:00.000000Z",
        completed_at="2026-10-31T11:30:15.000000Z",
        papers=[test_summary],
        research_focus="Latency reduction",
    )
    job_repo.save_job(record)

    # 1. User navigates away to Documents page (simulated by listing documents)
    doc_resp = client.get("/api/documents")
    assert doc_resp.status_code == 200

    # 2. User returns to Research Workspace (simulated by fetching job by saved ID)
    restore_resp = client.get("/api/research/job_persist_test")
    assert restore_resp.status_code == 200

    data = restore_resp.json()
    assert data["research_id"] == "job_persist_test"
    assert data["status"] == "completed"
    assert data["research_focus"] == "Latency reduction"
    assert len(data["papers"]) == 1
    assert data["papers"][0]["summary"]["objective"]["text"] == (
        "Investigate persistence in distributed vector storage."
    )
    assert len(data["papers"][0]["summary"]["findings"][0]["evidence"]) == 1
    assert data["papers"][0]["summary"]["findings"][0]["evidence"][0]["page_number"] == 4


def test_server_restart_persistence(tmp_path):
    """
    Simulates a backend server restart:
    Instance 1 writes a research job to disk snapshot.
    Instance 2 boots up from that same disk file and verifies state is intact.
    """
    storage_path = tmp_path / "restart_test_jobs.json"

    # Instance 1
    repo1 = InMemoryResearchJobRepository(storage_path=storage_path)
    job = ResearchJobRecord(
        research_id="job_restart_01",
        status=ResearchStatus.COMPLETED,
        paper_ids=["P_001", "P_002"],
        tasks=[ResearchTask.SUMMARIZE, ResearchTask.COMPARE],
        focus_query="Distributed consensus",
        research_focus="Distributed consensus",
        created_at="2026-10-31T09:00:00.000000Z",
        completed_at="2026-10-31T09:01:00.000000Z",
        is_claim_verified=True,
    )
    repo1.save_job(job)

    # Verify JSON file on disk
    assert storage_path.is_file()
    with open(storage_path, "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert len(saved_json) == 1
    assert saved_json[0]["research_id"] == "job_restart_01"

    # Instance 2 (Simulating server reboot with same storage path)
    repo2 = InMemoryResearchJobRepository(storage_path=storage_path)
    restored = repo2.get_job("job_restart_01")

    assert restored is not None
    assert restored.research_id == "job_restart_01"
    assert restored.status == ResearchStatus.COMPLETED
    assert restored.paper_ids == ["P_001", "P_002"]
    assert restored.research_focus == "Distributed consensus"
    assert restored.is_claim_verified is True


def test_restore_missing_job_returns_404(persistence_setup):
    """A missing or expired research job ID returns 404 allowing frontend to cleanly reset."""
    client = persistence_setup["client"]
    resp = client.get("/api/research/non_existent_job_12345")
    assert resp.status_code == 404
    body = resp.json()
    assert "detail" in body
    assert "non_existent_job_12345" in body["detail"]

