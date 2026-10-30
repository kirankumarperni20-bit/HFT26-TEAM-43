"""Unit tests for shared Pydantic schemas (Step 3)."""

import pytest
from pydantic import ValidationError

from backend.schemas.chunk import DocumentChunk
from backend.schemas.claim import Claim
from backend.schemas.evidence import ClaimEvidenceResponse, EvidenceItem


def test_valid_document_chunk_creation():
    """Verify DocumentChunk instantiates properly with valid contract fields."""
    chunk = DocumentChunk(
        chunk_id="P001_C002",
        paper_id="P001",
        page_number=5,
        paragraph_number=3,
        section="Results",
        text="The proposed model achieved an accuracy of 92%.",
    )
    assert chunk.chunk_id == "P001_C002"
    assert chunk.paper_id == "P001"
    assert chunk.page_number == 5
    assert chunk.paragraph_number == 3
    assert chunk.section == "Results"
    assert chunk.text == "The proposed model achieved an accuracy of 92%."


def test_optional_section_handling():
    """Verify section defaults to None when omitted or explicitly passed as None."""
    chunk_without_section = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Abstract text without explicit section.",
    )
    assert chunk_without_section.section is None

    chunk_with_none_section = DocumentChunk(
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        section=None,
        text="Abstract text with None section.",
    )
    assert chunk_with_none_section.section is None


def test_original_chunk_text_preservation():
    """Verify that original chunk text preserves exact characters including leading/trailing whitespace, tabs, and newlines."""
    raw_text = "   \n\tExact unformatted raw text with special characters: 92% ± 0.5% & non-ASCII: α, β. \n  "
    chunk = DocumentChunk(
        chunk_id="P001_C003",
        paper_id="P001",
        page_number=2,
        paragraph_number=4,
        text=raw_text,
    )
    assert chunk.text == raw_text
    assert len(chunk.text) == len(raw_text)


def test_valid_claim_creation():
    """Verify Claim instantiates correctly with single and multiple paper IDs."""
    claim = Claim(
        claim_id="C001",
        claim_text="Paper A achieved 92% accuracy.",
        paper_ids=["P001", "P002"],
    )
    assert claim.claim_id == "C001"
    assert claim.claim_text == "Paper A achieved 92% accuracy."
    assert claim.paper_ids == ["P001", "P002"]


def test_claim_empty_paper_ids_rejected():
    """Verify that a Claim with an empty paper_ids list raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Claim(
            claim_id="C002",
            claim_text="Invalid claim without target papers.",
            paper_ids=[],
        )
    assert "paper_ids" in str(exc_info.value)


def test_valid_evidence_item_creation():
    """Verify EvidenceItem instantiates correctly across all allowed retrieval methods."""
    methods = ["hybrid", "vector", "lexical"]
    for method in methods:
        item = EvidenceItem(
            evidence_id="P001_C002",
            chunk_id="P001_C002",
            paper_id="P001",
            page_number=5,
            paragraph_number=3,
            text="The proposed model achieved an accuracy of 92%.",
            retrieval_score=0.032,
            retrieval_method=method,
        )
        assert item.evidence_id == "P001_C002"
        assert item.chunk_id == "P001_C002"
        assert item.retrieval_score == 0.032
        assert item.retrieval_method == method


def test_invalid_retrieval_method_rejected():
    """Verify that EvidenceItem rejects retrieval methods outside ('hybrid', 'vector', 'lexical')."""
    invalid_methods = ["keyword", "bm25", "dense", "random", ""]
    for method in invalid_methods:
        with pytest.raises(ValidationError) as exc_info:
            EvidenceItem(
                evidence_id="P001_C002",
                chunk_id="P001_C002",
                paper_id="P001",
                page_number=5,
                paragraph_number=3,
                text="The proposed model achieved an accuracy of 92%.",
                retrieval_score=0.032,
                retrieval_method=method,  # type: ignore[arg-type]
            )
        assert "retrieval_method" in str(exc_info.value)


def test_claim_evidence_response_with_multiple_items():
    """Verify ClaimEvidenceResponse can hold multiple EvidenceItem objects."""
    item1 = EvidenceItem(
        evidence_id="P001_C001",
        chunk_id="P001_C001",
        paper_id="P001",
        page_number=1,
        paragraph_number=1,
        text="Transformer models utilize self-attention mechanisms.",
        retrieval_score=0.045,
        retrieval_method="hybrid",
    )
    item2 = EvidenceItem(
        evidence_id="P001_C002",
        chunk_id="P001_C002",
        paper_id="P001",
        page_number=5,
        paragraph_number=3,
        text="The proposed model achieved an accuracy of 92%.",
        retrieval_score=0.032,
        retrieval_method="hybrid",
    )

    response = ClaimEvidenceResponse(
        claim_id="C001",
        evidence=[item1, item2],
    )
    assert response.claim_id == "C001"
    assert len(response.evidence) == 2
    assert response.evidence[0].chunk_id == "P001_C001"
    assert response.evidence[1].chunk_id == "P001_C002"


def test_claim_evidence_response_empty_list_support():
    """Verify ClaimEvidenceResponse defaults to an empty list and accepts an explicit empty list."""
    default_response = ClaimEvidenceResponse(claim_id="C003")
    assert default_response.claim_id == "C003"
    assert default_response.evidence == []

    explicit_response = ClaimEvidenceResponse(claim_id="C004", evidence=[])
    assert explicit_response.claim_id == "C004"
    assert explicit_response.evidence == []
