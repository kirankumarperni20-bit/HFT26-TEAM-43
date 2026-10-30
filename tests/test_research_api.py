import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.documents import get_repository as get_doc_repo
from backend.api.research import get_research_job_repository, get_research_service
from backend.ingestion.repository import InMemoryDocumentRepository
from backend.research.repository import InMemoryResearchJobRepository
from backend.research.service import ResearchJobService
from backend.schemas.document import DocumentMetadata, DocumentStatus
from backend.schemas.research import (
    ResearchStatus,
    ResearchTask,
    ResearchJobError,
    GroundedField,
    PaperSummary,
    PaperSynthesisItem,
    ComparisonCell,
    ComparisonRow,
    ComparisonResult,
    EvidenceReference,
)


@pytest.fixture
def test_setup():
    """Sets up isolated in-memory repositories and FastAPI test client."""
    doc_repo = InMemoryDocumentRepository()
    job_repo = InMemoryResearchJobRepository()
    service = ResearchJobService(job_repository=job_repo, document_repository=doc_repo)

    app.dependency_overrides[get_doc_repo] = lambda: doc_repo
    app.dependency_overrides[get_research_job_repository] = lambda: job_repo
    app.dependency_overrides[get_research_service] = lambda: service

    client = TestClient(app)

    # Seed two completed test documents
    doc1 = DocumentMetadata(
        paper_id="P_test_001",
        filename="paper_one.pdf",
        status=DocumentStatus.COMPLETED,
    )
    doc2 = DocumentMetadata(
        paper_id="P_test_002",
        filename="paper_two.pdf",
        status=DocumentStatus.COMPLETED,
    )
    # Seed one in-progress document
    doc_processing = DocumentMetadata(
        paper_id="P_test_proc",
        filename="paper_processing.pdf",
        status=DocumentStatus.PROCESSING,
    )

    doc_repo.save_document(doc1)
    doc_repo.save_document(doc2)
    doc_repo.save_document(doc_processing)

    yield {
        "client": client,
        "doc_repo": doc_repo,
        "job_repo": job_repo,
        "service": service,
    }

    app.dependency_overrides.clear()


def test_valid_summary_job_creation(test_setup):
    """1. Test valid summary-job creation returns 202 and queued status."""
    client = test_setup["client"]
    payload = {
        "paper_ids": ["P_test_001"],
        "tasks": ["summarize"],
        "focus_query": "Key findings regarding accuracy",
    }
    response = client.post("/api/research/generate", json=payload)
    assert response.status_code == 202

    data = response.json()
    assert "research_id" in data
    assert data["research_id"].startswith("res_")
    assert data["status"] == "queued"
    assert data["paper_ids"] == ["P_test_001"]
    assert data["poll_url"] == f"/api/research/{data['research_id']}"
    assert "created_at" in data


def test_valid_comparison_job_creation(test_setup):
    """2. Test valid comparison-job creation with at least two papers."""
    client = test_setup["client"]
    payload = {
        "paper_ids": ["P_test_001", "P_test_002"],
        "tasks": ["compare"],
    }
    response = client.post("/api/research/generate", json=payload)
    assert response.status_code == 202

    data = response.json()
    assert data["status"] == "queued"
    assert set(data["paper_ids"]) == {"P_test_001", "P_test_002"}


def test_rejection_comparison_with_only_one_paper(test_setup):
    """3. Test rejection of comparison when fewer than two papers are selected."""
    client = test_setup["client"]
    payload = {
        "paper_ids": ["P_test_001"],
        "tasks": ["compare"],
    }
    response = client.post("/api/research/generate", json=payload)
    assert response.status_code == 422
    assert "Comparison requires at least two papers" in str(response.json())


def test_rejection_duplicate_paper_ids(test_setup):
    """4. Test rejection of duplicate paper IDs in payload."""
    client = test_setup["client"]
    payload = {
        "paper_ids": ["P_test_001", "P_test_001"],
        "tasks": ["summarize"],
    }
    response = client.post("/api/research/generate", json=payload)
    assert response.status_code == 422
    assert "Duplicate paper IDs are not permitted" in str(response.json())


def test_rejection_nonexistent_paper(test_setup):
    """5. Test rejection when requested paper does not exist."""
    client = test_setup["client"]
    payload = {
        "paper_ids": ["P_nonexistent"],
        "tasks": ["summarize"],
    }
    response = client.post("/api/research/generate", json=payload)
    assert response.status_code == 400
    assert "Paper 'P_nonexistent' does not exist" in response.json()["detail"]


def test_rejection_incomplete_paper_ingestion(test_setup):
    """6. Test rejection when requested paper has not finished ingestion."""
    client = test_setup["client"]
    payload = {
        "paper_ids": ["P_test_proc"],
        "tasks": ["summarize"],
    }
    response = client.post("/api/research/generate", json=payload)
    assert response.status_code == 400
    assert "has not completed ingestion" in response.json()["detail"]


def test_unknown_research_job_id(test_setup):
    """7. Test GET returns 404 for an unknown research-job ID."""
    client = test_setup["client"]
    response = client.get("/api/research/res_unknown_999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_valid_status_transitions(test_setup):
    """8. Test job transitions through queued -> processing -> completed."""
    client = test_setup["client"]
    service = test_setup["service"]

    # 1. Create job
    create_resp = client.post(
        "/api/research/generate",
        json={"paper_ids": ["P_test_001"], "tasks": ["summarize"]},
    )
    research_id = create_resp.json()["research_id"]

    # Initial state is queued
    get_resp1 = client.get(f"/api/research/{research_id}")
    assert get_resp1.json()["status"] == "queued"

    # 2. Transition to processing
    service.transition_to_processing(research_id)
    get_resp2 = client.get(f"/api/research/{research_id}")
    assert get_resp2.json()["status"] == "processing"

    # 3. Complete job
    service.complete_job(research_id=research_id)
    get_resp3 = client.get(f"/api/research/{research_id}")
    assert get_resp3.json()["status"] == "completed"
    assert get_resp3.json()["completed_at"] is not None


def test_structured_failure_handling(test_setup):
    """9. Test structured error details when a job fails."""
    client = test_setup["client"]
    service = test_setup["service"]

    create_resp = client.post(
        "/api/research/generate",
        json={"paper_ids": ["P_test_001"], "tasks": ["summarize"]},
    )
    research_id = create_resp.json()["research_id"]

    service.transition_to_processing(research_id)

    # Fail job with structured error
    err = ResearchJobError(
        code="LLM_CONTEXT_WINDOW_EXCEEDED",
        message="Document token length exceeds model context capacity.",
        retryable=False,
        retry_after_seconds=None,
    )
    service.fail_job(research_id, err)

    get_resp = client.get(f"/api/research/{research_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["status"] == "failed"
    assert data["error"]["code"] == "LLM_CONTEXT_WINDOW_EXCEEDED"
    assert data["error"]["retryable"] is False
    assert data["completed_at"] is not None


def test_result_retrieval_after_successful_processing(test_setup):
    """10. Test retrieving full synthesis results after processing completes."""
    client = test_setup["client"]
    service = test_setup["service"]

    create_resp = client.post(
        "/api/research/generate",
        json={
            "paper_ids": ["P_test_001", "P_test_002"],
            "tasks": ["summarize", "compare"],
        },
    )
    research_id = create_resp.json()["research_id"]

    # Controlled test synthesis payload
    test_summary = PaperSummary(
        objective=GroundedField(
            text="Evaluate hybrid retrieval performance.",
            evidence=[
                EvidenceReference(
                    chunk_id="chunk_001",
                    paper_id="P_test_001",
                    page_number=1,
                    paragraph_number=2,
                    excerpt="We evaluate hybrid dense-sparse retrieval.",
                )
            ],
        ),
        methodology=GroundedField(
            text="BM25 combined with dense embeddings.",
            evidence=[],
        ),
        findings=[
            GroundedField(
                text="Hybrid retrieval outperforms BM25 by 14% MRR@10.",
                evidence=[],
            )
        ],
        limitations=[
            GroundedField(
                text="Evaluated only on computer science preprints.",
                evidence=[],
            )
        ],
    )

    test_item = PaperSynthesisItem(
        paper_id="P_test_001",
        filename="paper_one.pdf",
        summary=test_summary,
    )

    test_comparison = ComparisonResult(
        synthesis_narrative="Both papers explore semantic search with complementary approaches.",
        matrix=[
            ComparisonRow(
                dimension="Retrieval Technique",
                cells=[
                    ComparisonCell(
                        paper_id="P_test_001",
                        value=GroundedField(text="Dense-sparse hybrid"),
                    ),
                    ComparisonCell(
                        paper_id="P_test_002",
                        value=GroundedField(text="Pure dense vector indexing"),
                    ),
                ],
            )
        ],
    )

    service.transition_to_processing(research_id)
    service.complete_job(
        research_id=research_id,
        papers=[test_item],
        comparison=test_comparison,
    )

    get_resp = client.get(f"/api/research/{research_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()

    assert data["status"] == "completed"
    assert data["is_claim_verified"] is False
    assert len(data["papers"]) == 1
    assert data["papers"][0]["summary"]["objective"]["text"] == "Evaluate hybrid retrieval performance."
    assert len(data["papers"][0]["summary"]["objective"]["evidence"]) == 1
    assert data["comparison"]["synthesis_narrative"].startswith("Both papers")
    assert len(data["comparison"]["matrix"]) == 1


def test_no_fabricated_result_when_processing_has_not_run(test_setup):
    """11. Confirm no fabricated results or mock text exist while queued."""
    client = test_setup["client"]
    create_resp = client.post(
        "/api/research/generate",
        json={"paper_ids": ["P_test_001"], "tasks": ["summarize"]},
    )
    research_id = create_resp.json()["research_id"]

    get_resp = client.get(f"/api/research/{research_id}")
    data = get_resp.json()
    assert data["status"] == "queued"
    assert data["papers"] is None
    assert data["comparison"] is None
    assert data["completed_at"] is None
    assert data["error"] is None


def test_regression_existing_document_apis(test_setup):
    """12. Confirm existing document listing and health check APIs remain functional."""
    client = test_setup["client"]

    # Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "healthy"}

    # Document list
    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) == 3
    assert any(d["paper_id"] == "P_test_001" for d in docs)

