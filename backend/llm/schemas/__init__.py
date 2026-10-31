"""
schemas — Pydantic models for LLM request / response validation.

Define all API-facing data contracts here so that FastAPI can
auto-generate OpenAPI docs and validate payloads at the boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SummarizeRequest(BaseModel):
    """Payload sent by the caller to request a summary."""

    text: str = Field(
        ...,
        min_length=1,
        description="The source text to summarise.",
    )
    max_length: int | None = Field(
        default=None,
        gt=0,
        description="Optional cap on the summary length (in tokens).",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class SummarizeResponse(BaseModel):
    """Payload returned after a successful summarisation."""

    summary: str = Field(
        ...,
        description="The generated summary.",
    )
    model: str = Field(
        ...,
        description="Identifier of the model that produced the summary.",
    )


# ---------------------------------------------------------------------------
# Extraction schemas re-export
# ---------------------------------------------------------------------------

try:
    from backend.llm.schemas.extraction import (
        ComputationalCostItem,
        DatasetItem,
        EvaluationMetricItem,
        EvidenceReference,
        ExtractionRequest,
        ExtractionResponse,
        KeyFindingItem,
        MethodologyItem,
        ResearchPaperExtraction,
        TokenizationItem,
        TrainingConfigItem,
    )
    from backend.llm.schemas.synthesis import (
        ComparisonType,
        CrossPaperSynthesis,
        ExperimentalComparison,
        FindingComparison,
        FindingRelationship,
        FutureDirection,
        MethodologicalComparison,
        ResearchGap,
        SourceReference,
        SynthesisRequest,
        SynthesisResponse,
    )
    from backend.llm.schemas.claim import (
        AlignmentStatus,
        AtomicClaim,
        ClaimDecompositionRequest,
        ClaimDecompositionResponse,
        ClaimType,
        StatementInputItem,
    )
    from backend.llm.schemas.verification import (
        ClaimVerification,
        RetrievedEvidence,
        VerificationRequest,
        VerificationResponse,
        VerificationStatus,
        VerifiedEvidenceItem,
    )
except ModuleNotFoundError:
    from llm.schemas.extraction import (
        ComputationalCostItem,
        DatasetItem,
        EvaluationMetricItem,
        EvidenceReference,
        ExtractionRequest,
        ExtractionResponse,
        KeyFindingItem,
        MethodologyItem,
        ResearchPaperExtraction,
        TokenizationItem,
        TrainingConfigItem,
    )
    from llm.schemas.synthesis import (
        ComparisonType,
        CrossPaperSynthesis,
        ExperimentalComparison,
        FindingComparison,
        FindingRelationship,
        FutureDirection,
        MethodologicalComparison,
        ResearchGap,
        SourceReference,
        SynthesisRequest,
        SynthesisResponse,
    )
    from llm.schemas.claim import (
        AlignmentStatus,
        AtomicClaim,
        ClaimDecompositionRequest,
        ClaimDecompositionResponse,
        ClaimType,
        StatementInputItem,
    )
    from llm.schemas.verification import (
        ClaimVerification,
        RetrievedEvidence,
        VerificationRequest,
        VerificationResponse,
        VerificationStatus,
        VerifiedEvidenceItem,
    )

__all__ = [
    "SummarizeRequest",
    "SummarizeResponse",
    "EvidenceReference",
    "MethodologyItem",
    "DatasetItem",
    "EvaluationMetricItem",
    "KeyFindingItem",
    "ResearchPaperExtraction",
    "ExtractionRequest",
    "ExtractionResponse",
    "TokenizationItem",
    "TrainingConfigItem",
    "ComputationalCostItem",
    "FindingRelationship",
    "ComparisonType",
    "SourceReference",
    "MethodologicalComparison",
    "ExperimentalComparison",
    "FindingComparison",
    "ResearchGap",
    "FutureDirection",
    "CrossPaperSynthesis",
    "SynthesisRequest",
    "SynthesisResponse",
    "ClaimType",
    "AlignmentStatus",
    "AtomicClaim",
    "StatementInputItem",
    "ClaimDecompositionRequest",
    "ClaimDecompositionResponse",
    "VerificationStatus",
    "RetrievedEvidence",
    "VerifiedEvidenceItem",
    "ClaimVerification",
    "VerificationRequest",
    "VerificationResponse",
]
