"""
test_claim_verification_milestone.py — Comprehensive tests for Milestone F5:
Evidence-Based Claim Verification.

Validates:
1. Supported claim directly substantiated by evidence.
2. Contradicted claim with conflicting numerical value or negation.
3. Insufficient evidence outcome when evidence does not establish truth.
4. Verification failure handling without crashing.
5. Empty retrieval / no candidate passage handling.
6. Provenance validation: quote substring check and allowed paper restriction.
7. Verification across research job synthesis: /api/research/{research_id}/verify.
8. Cross-paper evidence isolation.
"""

import asyncio
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend.llm.schemas.claim import AtomicClaim
from backend.llm.schemas.verification import (
    RetrievedEvidence,
    VerificationRequest,
    VerificationStatus,
)
from backend.llm.services.verification_service import ClaimVerificationService
from backend.llm.services.retrieval_service import EvidenceRetrievalService
from backend.schemas.chunk import Chunk
from backend.schemas.document import DocumentMetadata, DocumentStatus
from backend.api.documents import get_repository as get_doc_repo
from backend.api.research import get_research_job_repository as get_job_repo
from backend.schemas.research import (
    StartResearchPayload,
    ResearchTask,
)
from backend.research.service import ResearchJobService


def make_passage(
    ev_id: str,
    paper_id: str,
    text: str,
    page: int = 1,
    section: str = "Introduction",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=ev_id,
        paper_id=paper_id,
        chunk_id=f"chk_{ev_id}",
        text=text,
        page=page,
        section=section,
        retrieval_method="lexical_overlap",
        retrieval_score=0.92,
    )


def test_supported_claim_evaluated_accurately():
    """Verify that a claim with matching factual assertions is classified as SUPPORTED."""
    claim = AtomicClaim(
        claim_id="c_supp_1",
        claim_text="The model achieved an accuracy of 94.6% on the benchmark dataset.",
        source_text="The model achieved an accuracy of 94.6% on the benchmark dataset.",
        involved_papers=["paper_alpha"],
    )
    passage = make_passage(
        ev_id="ev_supp_1",
        paper_id="paper_alpha",
        text="On the benchmark dataset, the model achieved an accuracy of 94.6%, surpassing all baselines.",
        page=5,
        section="Results",
    )

    service = ClaimVerificationService(client=None)
    req = VerificationRequest(claims=[claim], passages=[passage])
    res = asyncio.run(service.verify(req))

    assert res.total_claims == 1
    assert res.supported_count == 1
    assert res.verifications[0].verification_status == VerificationStatus.SUPPORTED
    assert res.verifications[0].provenance_validated is True
    assert len(res.verifications[0].supporting_evidence) == 1
    assert res.verifications[0].supporting_evidence[0].page == 5
    assert res.verifications[0].supporting_evidence[0].is_verbatim is True


def test_contradicted_claim_with_conflicting_numbers():
    """Verify that a claim with a conflicting metric is classified as CONTRADICTED."""
    claim = AtomicClaim(
        claim_id="c_contra_1",
        claim_text="The method achieved a BLEU score of 35.0 on English-German translation.",
        source_text="The method achieved a BLEU score of 35.0 on English-German translation.",
        involved_papers=["paper_beta"],
    )
    passage = make_passage(
        ev_id="ev_contra_1",
        paper_id="paper_beta",
        text="The method achieved a BLEU score of 25.16 on English-German translation, which is lower than expected.",
        page=7,
        section="Experiments",
    )

    service = ClaimVerificationService(client=None)
    req = VerificationRequest(claims=[claim], passages=[passage])
    res = asyncio.run(service.verify(req))

    assert res.total_claims == 1
    assert res.contradicted_count == 1
    verif = res.verifications[0]
    assert verif.verification_status == VerificationStatus.CONTRADICTED
    assert len(verif.contradicting_evidence) == 1
    assert verif.contradicting_evidence[0].is_verbatim is True
    assert "25.16" in verif.contradicting_evidence[0].quote


def test_insufficient_evidence_when_facts_not_established():
    """Verify that when retrieved passages do not confirm the claim, outcome is INSUFFICIENT EVIDENCE."""
    claim = AtomicClaim(
        claim_id="c_inconcl_1",
        claim_text="The researchers deployed the system across 50 production hospital clusters.",
        source_text="The researchers deployed the system across 50 production hospital clusters.",
        involved_papers=["paper_gamma"],
    )
    passage = make_passage(
        ev_id="ev_inconcl_1",
        paper_id="paper_gamma",
        text="We plan to explore clinical hospital deployment in future collaborative studies.",
        page=11,
        section="Future Work",
    )

    service = ClaimVerificationService(client=None)
    req = VerificationRequest(claims=[claim], passages=[passage])
    res = asyncio.run(service.verify(req))

    assert res.total_claims == 1
    verif = res.verifications[0]
    assert verif.verification_status in (
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.UNRESOLVED,
    )
    assert verif.unresolved_reason is not None
    assert "insufficient" in verif.unresolved_reason.lower() or "no direct" in verif.unresolved_reason.lower()


def test_empty_retrieval_classified_as_unresolved():
    """Verify that claims with zero candidate passages return UNRESOLVED without failure."""
    claim = AtomicClaim(
        claim_id="c_empty_1",
        claim_text="Quantum annealing accelerates the training pipeline by 10x.",
        source_text="Quantum annealing accelerates the training pipeline by 10x.",
        involved_papers=["paper_delta"],
    )
    service = ClaimVerificationService(client=None)
    req = VerificationRequest(claims=[claim], passages=[])
    res = asyncio.run(service.verify(req))

    assert res.total_claims == 1
    assert res.unresolved_count == 1
    assert res.verifications[0].verification_status == VerificationStatus.UNRESOLVED


def test_cross_paper_evidence_isolation():
    """Verify that candidate passages from paper_a are not used to substantiate paper_b."""
    claim_a = AtomicClaim(
        claim_id="c_iso_a",
        claim_text="Algorithm A utilizes 12 layers with multi-head attention.",
        source_text="Algorithm A utilizes 12 layers with multi-head attention.",
        involved_papers=["paper_a"],
    )
    passages = [
        make_passage("ev_a1", "paper_a", "Algorithm A utilizes 12 layers with multi-head attention.", page=2),
        make_passage("ev_b1", "paper_b", "Algorithm B utilizes convolutional gated units.", page=4),
    ]

    service = ClaimVerificationService(client=None)
    req_a = VerificationRequest(claims=[claim_a], passages=passages, allowed_paper_ids=["paper_a"])
    res_a = asyncio.run(service.verify(req_a))

    assert res_a.supported_count == 1
    assert res_a.verifications[0].supporting_evidence[0].paper_id == "paper_a"


def test_document_repository_chunk_retrieval_fallback():
    """Verify that EvidenceRetrievalService retrieves from DocumentRepository chunks when database is inactive."""
    doc_repo = get_doc_repo()
    paper_id = "test_doc_repo_paper_01"

    doc = DocumentMetadata(
        paper_id=paper_id,
        filename="test_paper.pdf",
        status=DocumentStatus.COMPLETED,
    )
    try:
        doc_repo.save_document(doc)
    except ValueError:
        pass

    chunks = [
        Chunk(
            chunk_id="chk_repo_1",
            paper_id=paper_id,
            page_number=1,
            paragraph_number=1,
            text="We present a scalable deep neural network achieving 99.2% accuracy on image classification.",
            section="Abstract",
        ),
        Chunk(
            chunk_id="chk_repo_2",
            paper_id=paper_id,
            page_number=2,
            paragraph_number=1,
            text="The network was trained on 4 NVIDIA A100 GPUs for 6 hours.",
            section="Methods",
        ),
    ]
    doc_repo.save_chunks(paper_id, chunks)

    claim = AtomicClaim(
        claim_id="c_repo_1",
        claim_text="A scalable deep neural network achieves 99.2% accuracy on image classification.",
        source_text="A scalable deep neural network achieves 99.2% accuracy on image classification.",
        involved_papers=[paper_id],
    )

    retrieval_service = EvidenceRetrievalService(document_repository=doc_repo)
    candidates = retrieval_service.retrieve(claim=claim, allowed_paper_ids=[paper_id])

    assert len(candidates) > 0
    assert candidates[0].paper_id == paper_id
    assert "99.2%" in candidates[0].text


def test_research_job_verification_end_to_end():
    """Test full integration of research generation and claim verification via ResearchJobService."""
    doc_repo = get_doc_repo()
    job_repo = get_job_repo()
    service = ResearchJobService(job_repository=job_repo, document_repository=doc_repo)

    paper_id = "e2e_verify_paper_01"
    doc = DocumentMetadata(
        paper_id=paper_id,
        filename="e2e_study.pdf",
        status=DocumentStatus.COMPLETED,
    )
    try:
        doc_repo.save_document(doc)
    except ValueError:
        pass

    chunks = [
        Chunk(
            chunk_id="e2e_chk_1",
            paper_id=paper_id,
            page_number=1,
            paragraph_number=1,
            text="The primary research objective is developing robust self-supervised representations for biological sequences.",
            section="1 Introduction",
        ),
        Chunk(
            chunk_id="e2e_chk_2",
            paper_id=paper_id,
            page_number=2,
            paragraph_number=1,
            text="Our approach employs a masked sequence modeling objective over 500 million protein sequences.",
            section="2 Methodology",
        ),
    ]
    doc_repo.save_chunks(paper_id, chunks)

    payload = StartResearchPayload(
        paper_ids=[paper_id],
        tasks=[ResearchTask.SUMMARIZE],
    )
    create_resp = service.create_job(payload)
    research_id = create_resp.research_id

    # Process job synchronously
    asyncio.run(service.process_job(research_id))

    job_before = service.get_job(research_id)
    assert job_before.status.value == "completed"
    assert job_before.is_claim_verified is False
    assert job_before.verifications is None

    # Now run claim verification
    verified_result = asyncio.run(service.verify_job(research_id))

    assert verified_result.is_claim_verified is True
    assert verified_result.verifications is not None
    assert len(verified_result.verifications) >= 2
    for verif in verified_result.verifications:
        assert verif.verification_status in [
            VerificationStatus.SUPPORTED,
            VerificationStatus.PARTIALLY_SUPPORTED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.UNRESOLVED,
            VerificationStatus.VERIFICATION_FAILED,
        ]


def test_verify_research_endpoint_fastapi():
    """Verify that POST /api/research/{research_id}/verify endpoint functions correctly via FastAPI client."""
    client = TestClient(app)

    doc_repo = get_doc_repo()
    paper_id = "api_client_paper_01"
    doc = DocumentMetadata(
        paper_id=paper_id,
        filename="api_test.pdf",
        status=DocumentStatus.COMPLETED,
    )
    try:
        doc_repo.save_document(doc)
    except ValueError:
        pass

    chunks = [
        Chunk(
            chunk_id="api_chk_1",
            paper_id=paper_id,
            page_number=1,
            paragraph_number=1,
            text="We investigate transfer learning efficiency across multi-modal transformers.",
            section="Introduction",
        )
    ]
    doc_repo.save_chunks(paper_id, chunks)

    # 1. Create job
    create_res = client.post(
        "/api/research/generate",
        json={"paper_ids": [paper_id], "tasks": ["summarize"]},
    )
    assert create_res.status_code == 202
    research_id = create_res.json()["research_id"]

    # 2. Process job directly
    job_repo = get_job_repo()
    svc = ResearchJobService(job_repository=job_repo, document_repository=doc_repo)
    asyncio.run(svc.process_job(research_id))

    # 3. Verify claims via API
    verify_res = client.post(f"/api/research/{research_id}/verify")
    assert verify_res.status_code == 200
    data = verify_res.json()
    assert data["research_id"] == research_id
    assert data["is_claim_verified"] is True
    assert "verifications" in data
    assert isinstance(data["verifications"], list)
