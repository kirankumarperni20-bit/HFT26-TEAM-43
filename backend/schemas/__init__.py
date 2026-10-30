"""Shared Pydantic schemas for the CBIT Hacktoberfest 2026 Team 43 AI Research Assistant."""

from backend.schemas.chunk import DocumentChunk
from backend.schemas.claim import Claim
from backend.schemas.evidence import ClaimEvidenceResponse, EvidenceItem

__all__ = [
    "DocumentChunk",
    "Claim",
    "EvidenceItem",
    "ClaimEvidenceResponse",
]
