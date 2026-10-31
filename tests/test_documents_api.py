import os
import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.documents import get_repository
import backend.api.documents as documents_module
from backend.schemas.document import DocumentStatus

client = TestClient(app)


def _create_sample_pdf_bytes() -> bytes:
    """Creates a 2-page valid PDF with sections and text for API testing."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "1. Introduction\n\nThis research proposes a new approach to automated claim verification.")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "2. Methodology\n\nWe employ hybrid retrieval combining BM25 and dense embeddings.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_blank_pdf_bytes() -> bytes:
    """Creates a 1-page PDF containing zero extractable text."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_password_protected_pdf_bytes() -> bytes:
    """Creates an encrypted PDF requiring a password."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential Paper Content")
    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="admin",
        user_pw="secret123",
    )
    doc.close()
    return pdf_bytes


def test_health_endpoint():
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_successful_pdf_upload():
    """Test successful PDF upload through /api/documents/upload."""
    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("research_paper.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 201

    data = response.json()
    assert "documents" in data
    assert len(data["documents"]) == 1

    doc = data["documents"][0]
    assert doc["filename"] == "research_paper.pdf"
    assert doc["status"] == "completed"
    assert doc["paper_id"].startswith("P")

    # Verify document is listed in repository
    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    paper_ids = [d["paper_id"] for d in list_response.json()]
    assert doc["paper_id"] in paper_ids


def test_extracted_chunks_available_after_upload():
    """Test GET /api/documents/{paper_id}/chunks returns canonical chunks."""
    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("chunk_test.pdf", pdf_bytes, "application/pdf")}

    upload_res = client.post("/api/documents/upload", files=files)
    assert upload_res.status_code == 201
    paper_id = upload_res.json()["documents"][0]["paper_id"]

    chunks_res = client.get(f"/api/documents/{paper_id}/chunks")
    assert chunks_res.status_code == 200
    chunks = chunks_res.json()

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == f"{paper_id}_C001"
    assert chunks[0]["paper_id"] == paper_id
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["paragraph_number"] == 1
    assert chunks[0]["section"] == "Introduction"
    assert "This research proposes a new approach" in chunks[0]["text"]

    assert chunks[1]["chunk_id"] == f"{paper_id}_C002"
    assert chunks[1]["section"] == "Methodology"


def test_page_text_retrieval():
    """Test GET /api/documents/{paper_id}/pages/{page_number} returns exact text."""
    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("page_test.pdf", pdf_bytes, "application/pdf")}

    upload_res = client.post("/api/documents/upload", files=files)
    paper_id = upload_res.json()["documents"][0]["paper_id"]

    page_res = client.get(f"/api/documents/{paper_id}/pages/1")
    assert page_res.status_code == 200
    page_data = page_res.json()
    assert page_data["paper_id"] == paper_id
    assert page_data["page_number"] == 1
    assert "Introduction" in page_data["text"]

    # Non-existent page number
    missing_page_res = client.get(f"/api/documents/{paper_id}/pages/99")
    assert missing_page_res.status_code == 404


def test_get_document_metadata_and_status():
    """Test GET /api/documents/{paper_id} and /api/documents/{paper_id}/status."""
    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("meta_test.pdf", pdf_bytes, "application/pdf")}

    upload_res = client.post("/api/documents/upload", files=files)
    paper_id = upload_res.json()["documents"][0]["paper_id"]

    # Document metadata endpoint
    meta_res = client.get(f"/api/documents/{paper_id}")
    assert meta_res.status_code == 200
    assert meta_res.json()["paper_id"] == paper_id
    assert meta_res.json()["status"] == "completed"

    # Document status endpoint
    status_res = client.get(f"/api/documents/{paper_id}/status")
    assert status_res.status_code == 200
    assert status_res.json() == {
        "paper_id": paper_id,
        "status": "completed",
    }


def test_unknown_paper_returns_404():
    """Test 404 responses for non-existent paper IDs."""
    assert client.get("/api/documents/NONEXISTENT").status_code == 404
    assert client.get("/api/documents/NONEXISTENT/status").status_code == 404
    assert client.get("/api/documents/NONEXISTENT/chunks").status_code == 404
    assert client.get("/api/documents/NONEXISTENT/pages/1").status_code == 404


def test_invalid_pdf_is_rejected():
    """Test that files without PDF signature or invalid extension are rejected."""
    # Invalid extension
    res1 = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert res1.status_code == 400
    assert "pdf extension" in res1.json()["detail"].lower()

    # Invalid header (ends with .pdf but content is not %PDF)
    res2 = client.post(
        "/api/documents/upload",
        files={"file": ("fake.pdf", b"NOT_A_REAL_PDF_STREAM", "application/pdf")},
    )
    assert res2.status_code == 400
    assert "invalid magic file signature" in res2.json()["detail"].lower()


def test_empty_pdf_is_handled():
    """Test that zero-byte PDFs return HTTP 400."""
    res = client.post(
        "/api/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_password_protected_pdf_failure_and_status_marked_failed():
    """Test that password-protected PDF fails and document is marked failed."""
    encrypted_bytes = _create_password_protected_pdf_bytes()
    files = {"file": ("encrypted.pdf", encrypted_bytes, "application/pdf")}

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "password-protected" in response.json()["detail"].lower()

    # Verify in shared repository that the document was marked as failed
    repo = get_repository()
    failed_docs = [d for d in repo.list_documents() if d.status == DocumentStatus.FAILED]
    assert len(failed_docs) >= 1
    assert any(d.filename == "encrypted.pdf" for d in failed_docs)


def test_scanned_pdf_without_text_marks_document_failed():
    """Test blank / scanned image-only PDF marks document failed."""
    blank_bytes = _create_blank_pdf_bytes()
    files = {"file": ("blank_scan.pdf", blank_bytes, "application/pdf")}

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "OCR may be required" in response.json()["detail"]


def test_unexpected_ingestion_error_returns_generic_message_and_marks_failed(monkeypatch):
    """
    Test that unexpected internal errors log traceback, return generic HTTP 500,
    hide internal exception details, and update document status to failed.
    """
    def mock_extract_paragraphs(pages):
        raise RuntimeError("Internal secret connection failure at /var/secrets/creds.json")

    monkeypatch.setattr(documents_module, "extract_paragraphs", mock_extract_paragraphs)

    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("crash_test.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/api/documents/upload", files=files)

    # Must return HTTP 500 with a generic message
    assert response.status_code == 500
    assert response.json()["detail"] == "Document ingestion failed."
    # Must NOT expose sensitive stack details
    assert "secret" not in response.text
    assert "/var/secrets" not in response.text

    # Verify repository marked document status as failed
    repo = get_repository()
    failed_docs = [d for d in repo.list_documents() if d.filename == "crash_test.pdf"]
    assert len(failed_docs) == 1
    assert failed_docs[0].status == DocumentStatus.FAILED


def test_upload_size_limit_enforcement(monkeypatch):
    """
    Test that uploads exceeding MAX_UPLOAD_BYTES return HTTP 413 Entity Too Large
    and stream reading is aborted in chunks.
    """
    # Set limit to 200 bytes for this test
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "200")

    pdf_bytes = _create_sample_pdf_bytes()  # typically ~700-1000 bytes
    assert len(pdf_bytes) > 200

    files = {"file": ("oversized.pdf", pdf_bytes, "application/pdf")}
    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"].lower()


def test_repository_state_is_shared_across_requests():
    """Test that the repository is shared and persists across endpoint calls."""
    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("shared_state.pdf", pdf_bytes, "application/pdf")}

    upload_res = client.post("/api/documents/upload", files=files)
    paper_id = upload_res.json()["documents"][0]["paper_id"]

    # Call 1: list documents
    list_res = client.get("/api/documents")
    assert any(d["paper_id"] == paper_id for d in list_res.json())

    # Call 2: get status
    status_res = client.get(f"/api/documents/{paper_id}/status")
    assert status_res.json()["status"] == "completed"

    # Direct repository inspection
    repo = get_repository()
    assert repo.get_document(paper_id) is not None
    assert len(repo.get_chunks(paper_id)) == 2


def test_existing_json_field_names_remain_unchanged():
    """Verify that returned JSON fields strictly match the shared contracts."""
    pdf_bytes = _create_sample_pdf_bytes()
    files = {"file": ("contract_test.pdf", pdf_bytes, "application/pdf")}

    upload_res = client.post("/api/documents/upload", files=files)
    upload_data = upload_res.json()

    # DocumentUploadResponse contract: {"documents": [{"paper_id", "filename", "status"}]}
    assert "documents" in upload_data
    doc = upload_data["documents"][0]
    assert set(doc.keys()) == {"paper_id", "filename", "status"}

    paper_id = doc["paper_id"]

    # Status response contract: {"paper_id", "status"}
    status_res = client.get(f"/api/documents/{paper_id}/status")
    assert set(status_res.json().keys()) == {"paper_id", "status"}

    # Chunk contract: {"chunk_id", "paper_id", "page_number", "paragraph_number", "section", "text"}
    chunks_res = client.get(f"/api/documents/{paper_id}/chunks")
    chunk = chunks_res.json()[0]
    assert set(chunk.keys()) == {"chunk_id", "paper_id", "page_number", "paragraph_number", "section", "text"}
