"""
verification.py — Pydantic schemas for evidence retrieval and claim verification.

Defines schemas for:
- Retrieved candidate passages with exact provenance.
- Atomic claim verification outcomes (SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNRESOLVED, INSUFFICIENT EVIDENCE, VERIFICATION FAILED).
- Verbatim evidence citations and provenance audit metadata.
- Request and response contracts for the verification API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

try:
    from backend.llm.schemas.claim import AtomicClaim
except ModuleNotFoundError:
    from llm.schemas.claim import AtomicClaim


class VerificationStatus(str, Enum):
    """Classification of claim verification outcome against evidence."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"
    VERIFICATION_FAILED = "VERIFICATION FAILED"


class RetrievedEvidence(BaseModel):
    """Candidate evidence passage retrieved for a claim."""

    evidence_id: str = Field(
        ...,
        description="Unique identifier for the passage (e.g., 'ev_p1_001').",
    )
    paper_id: str = Field(
        ...,
        description="Source paper identifier.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Chunk identifier if partitioned.",
    )
    text: str = Field(
        ...,
        description="Exact raw text of the retrieved passage.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number (1-indexed) if identified.",
    )
    section: str | None = Field(
        default=None,
        description="Section heading or context metadata.",
    )
    retrieval_method: str = Field(
        default="lexical_overlap",
        description="Method used for retrieval (e.g., 'lexical_overlap', 'reference_hint', 'bm25').",
    )
    retrieval_score: float = Field(
        default=0.0,
        description="Retrieval relevance score (0.0 to 1.0).",
    )


class VerifiedEvidenceItem(BaseModel):
    """An exact cited excerpt from a candidate passage used as verification evidence."""

    evidence_id: str = Field(
        ...,
        description="Identifier of the candidate passage cited.",
    )
    paper_id: str = Field(
        ...,
        description="Identifier of the source paper.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Chunk identifier from source passage.",
    )
    page: int | None = Field(
        default=None,
        description="Page number from source passage.",
    )
    section: str | None = Field(
        default=None,
        description="Section heading from source passage.",
    )
    quote: str = Field(
        ...,
        description="Exact verbatim excerpt from the passage text.",
    )
    retrieval_method: str | None = Field(
        default=None,
        description="Inherited retrieval method.",
    )
    is_verbatim: bool = Field(
        default=True,
        description="True if quote is an exact character-for-character substring of passage text.",
    )
    validation_notes: str | None = Field(
        default=None,
        description="Notes if provenance or verbatim validation required adjustment.",
    )


class ClaimVerification(BaseModel):
    """Verification outcome and evidence audit certificate for a single atomic claim."""

    claim_id: str = Field(
        ...,
        description="ID of the verified atomic claim.",
    )
    claim_text: str = Field(
        ...,
        description="Exact text of the atomic proposition evaluated.",
    )
    verification_status: VerificationStatus = Field(
        ...,
        description="Verification classification: SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNRESOLVED, INSUFFICIENT EVIDENCE, or VERIFICATION FAILED.",
    )
    explanation: str = Field(
        ...,
        description="Detailed reasoning justifying the verification status.",
    )
    supporting_evidence: list[VerifiedEvidenceItem] = Field(
        default_factory=list,
        description="Passage citations that substantiate the claim.",
    )
    contradicting_evidence: list[VerifiedEvidenceItem] = Field(
        default_factory=list,
        description="Passage citations that refute or contradict the claim.",
    )
    unresolved_reason: str | None = Field(
        default=None,
        description="Explicit rationale when evidence is insufficient or inconclusive.",
    )
    confidence: float | None = Field(
        default=None,
        description="Model self-reported confidence score (0.0 to 1.0; not proof).",
    )
    candidate_evidence_ids: list[str] = Field(
        default_factory=list,
        description="List of all candidate evidence IDs considered during evaluation.",
    )
    provenance_validated: bool = Field(
        default=True,
        description="True if all evidence IDs and quotes passed deterministic provenance checks.",
    )


class VerificationRequest(BaseModel):
    """Request payload for claim verification."""

    claims: list[AtomicClaim] = Field(
        ...,
        min_length=1,
        description="List of atomic claims to verify.",
    )
    allowed_paper_ids: list[str] | None = Field(
        default=None,
        description="Optional list of paper IDs to restrict evidence retrieval and citations to.",
    )
    passages: list[RetrievedEvidence] | None = Field(
        default=None,
        description="Optional explicit candidate passages. If omitted, built-in retrieval service is used.",
    )
    parent_statement_context: dict[str, str] | None = Field(
        default=None,
        description="Mapping from statement_id or claim_id to original synthesis statement for pronoun resolution.",
    )


class VerificationResponse(BaseModel):
    """Response payload containing claim verifications and audit metrics."""

    verifications: list[ClaimVerification] = Field(
        ...,
        description="List of claim verification results.",
    )
    total_claims: int = Field(
        ...,
        description="Total number of claims evaluated.",
    )
    supported_count: int = Field(
        default=0,
        description="Number of claims classified as SUPPORTED.",
    )
    partially_supported_count: int = Field(
        default=0,
        description="Number of claims classified as PARTIALLY_SUPPORTED.",
    )
    contradicted_count: int = Field(
        default=0,
        description="Number of claims classified as CONTRADICTED.",
    )
    unresolved_count: int = Field(
        default=0,
        description="Number of claims classified as UNRESOLVED or INSUFFICIENT EVIDENCE.",
    )
    insufficient_evidence_count: int = Field(
        default=0,
        description="Number of claims classified as INSUFFICIENT EVIDENCE.",
    )
    failed_count: int = Field(
        default=0,
        description="Number of claims classified as VERIFICATION FAILED.",
    )
    provenance_failure_count: int = Field(
        default=0,
        description="Number of claims where cited evidence failed deterministic provenance validation.",
    )
    model: str | None = Field(
        default=None,
        description="LLM model identifier used for semantic verification.",
    )
