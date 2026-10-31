"""
claim.py — Pydantic schemas for atomic claim decomposition.

Defines data contracts for breaking compound cross-paper synthesis statements
into independently verifiable, atomic propositions with strict Unicode character
offsets, paper attribution, and citation provenance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from backend.llm.schemas.synthesis import CrossPaperSynthesis, SourceReference
except ModuleNotFoundError:
    from llm.schemas.synthesis import CrossPaperSynthesis, SourceReference


# ---------------------------------------------------------------------------
# Enums for Atomic Claims
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    """Categorization of atomic claim proposition types."""

    FACTUAL = "FACTUAL"
    NUMERICAL = "NUMERICAL"
    METHODOLOGICAL = "METHODOLOGICAL"
    COMPARATIVE = "COMPARATIVE"
    LIMITATION = "LIMITATION"
    FUTURE_WORK = "FUTURE_WORK"
    OTHER = "OTHER"


class AlignmentStatus(str, Enum):
    """Fidelity classification of character offset alignment against source text."""

    EXACT = "EXACT"                  # Exact character-for-character contiguous substring
    PARAPHRASED = "PARAPHRASED"      # Text altered/paraphrased; offsets cannot be confirmed
    AMBIGUOUS = "AMBIGUOUS"          # Multiple occurrences cannot be uniquely disambiguated
    UNALIGNED = "UNALIGNED"          # Claim text not found in source text


# ---------------------------------------------------------------------------
# Atomic Claim Model
# ---------------------------------------------------------------------------


class AtomicClaim(BaseModel):
    """An independently verifiable atomic proposition extracted from a synthesis statement."""

    claim_id: str = Field(
        ...,
        description="Unique stable identifier for the atomic claim (e.g., 'claim_001').",
    )
    claim_text: str = Field(
        ...,
        description="Exact text of the atomic proposition.",
    )
    source_text: str = Field(
        ...,
        description="Exact original synthesis statement from which this claim was extracted.",
    )
    start_offset: int | None = Field(
        default=None,
        description="Starting Python Unicode character offset in source_text (0-indexed). None if unaligned.",
    )
    end_offset: int | None = Field(
        default=None,
        description="Ending Python Unicode character offset in source_text. None if unaligned.",
    )
    claim_type: ClaimType = Field(
        default=ClaimType.FACTUAL,
        description="Category of proposition: FACTUAL, NUMERICAL, METHODOLOGICAL, COMPARATIVE, etc.",
    )
    involved_papers: list[str] = Field(
        default_factory=list,
        description="Paper IDs associated with this proposition.",
    )
    source_references: list[SourceReference] = Field(
        default_factory=list,
        description="Source citations inherited from the original synthesis statement.",
    )
    alignment_status: AlignmentStatus = Field(
        default=AlignmentStatus.EXACT,
        description="Fidelity status of the substring offset calculation.",
    )
    is_exact_substring: bool = Field(
        default=True,
        description="True if source_text[start_offset:end_offset] == claim_text exactly.",
    )
    statement_id: str | None = Field(
        default=None,
        description="Identifier of the original source statement (e.g., 'stmt_001').",
    )
    validation_notes: str | None = Field(
        default=None,
        description="Optional diagnostic notes if claim was flagged or unaligned.",
    )

    @model_validator(mode="after")
    def validate_offset_consistency(self) -> AtomicClaim:
        """Verify that when marked exact, offsets reproduce the claim text."""
        if self.alignment_status == AlignmentStatus.EXACT and self.start_offset is not None and self.end_offset is not None:
            if not (0 <= self.start_offset <= self.end_offset <= len(self.source_text)):
                raise ValueError(
                    f"Invalid offset bounds [{self.start_offset}:{self.end_offset}] for source_text length {len(self.source_text)}"
                )
            slice_text = self.source_text[self.start_offset:self.end_offset]
            if slice_text != self.claim_text:
                raise ValueError(
                    f"Offset mismatch: source_text[{self.start_offset}:{self.end_offset}] '{slice_text}' != claim_text '{self.claim_text}'"
                )
        return self


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class StatementInputItem(BaseModel):
    """An individual statement provided directly for decomposition."""

    statement_id: str = Field(..., description="Unique ID for the statement (e.g. 'stmt_001').")
    text: str = Field(..., description="The compound synthesis statement text.")
    involved_papers: list[str] = Field(default_factory=list, description="Paper IDs associated with this statement.")
    source_references: list[SourceReference] = Field(
        default_factory=list, description="Source references associated with this statement."
    )


class ClaimDecompositionRequest(BaseModel):
    """Request payload for claim decomposition.

    Accepts either a full CrossPaperSynthesis object, or a list of StatementInputItem statements.
    """

    synthesis: CrossPaperSynthesis | None = Field(
        default=None,
        description="Complete structured synthesis from Phase 2.",
    )
    statements: list[StatementInputItem] | None = Field(
        default=None,
        description="Optional explicit list of statements to decompose.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_input_present(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("synthesis") is None and data.get("statements") is None:
                raise ValueError("Must provide either 'synthesis' or 'statements' for decomposition.")
        return data


class ClaimDecompositionResponse(BaseModel):
    """Structured response containing all decomposed atomic claims and alignment statistics."""

    claims: list[AtomicClaim] = Field(
        ...,
        description="List of all decomposed atomic claims.",
    )
    total_claims: int = Field(
        ...,
        description="Total number of atomic claims generated.",
    )
    source_statement_count: int = Field(
        ...,
        description="Total number of original synthesis statements processed.",
    )
    exact_alignment_count: int = Field(
        default=0,
        description="Number of claims with verified exact substring Unicode character offsets.",
    )
    flagged_claims_count: int = Field(
        default=0,
        description="Number of claims flagged as paraphrased, ambiguous, or unaligned.",
    )
    model: str | None = Field(
        default=None,
        description="LLM model identifier used for decomposition.",
    )
