"""
synthesis.py — Pydantic schemas for structured cross-paper synthesis.

Data contracts for multi-paper comparative analysis, capturing:
- Methodological similarities and differences
- Dataset and experimental condition differences
- Findings agreement vs. contextual divergence vs. genuine contradictions
- Cross-cutting research limitations, gaps, and future research directions
- Strict source provenance citations grounding all comparative statements
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from backend.llm.schemas.extraction import ResearchPaperExtraction
except ModuleNotFoundError:
    from llm.schemas.extraction import ResearchPaperExtraction


# ---------------------------------------------------------------------------
# Enums for Comparative Classification
# ---------------------------------------------------------------------------


class FindingRelationship(str, Enum):
    """Categorization of finding relationships across papers."""

    AGREEMENT = "AGREEMENT"
    CONTEXTUAL_DIFFERENCE = "CONTEXTUAL_DIFFERENCE"
    CONTRADICTION = "CONTRADICTION"


class ComparisonType(str, Enum):
    """Categorization of methodological alignment."""

    SIMILARITY = "SIMILARITY"
    DIFFERENCE = "DIFFERENCE"


# ---------------------------------------------------------------------------
# Grounding / Source Reference Model
# ---------------------------------------------------------------------------


class SourceReference(BaseModel):
    """Citation referencing evidence from a specific source paper."""

    paper_id: str = Field(
        ...,
        description="Identifier of the source paper where this evidence originates.",
    )
    section: str | None = Field(
        default=None,
        description="Section heading or table in the source paper.",
    )
    quote: str | None = Field(
        default=None,
        description="Verbatim excerpt from the source paper grounding this claim.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Chunk or page provenance identifier.",
    )
    page: int | None = Field(
        default=None,
        description="PDF page number if identified.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_reference(cls, data: Any) -> Any:
        """Coerce strings or loose dictionaries into valid SourceReference objects."""
        if isinstance(data, str):
            return {"paper_id": "unknown", "quote": data.strip()}
        return data


# ---------------------------------------------------------------------------
# Comparative Item Models
# ---------------------------------------------------------------------------


class MethodologicalComparison(BaseModel):
    """Comparison of algorithmic, architectural, or procedural choices."""

    aspect: str = Field(
        ...,
        description="Architectural or algorithmic topic (e.g. 'Attention Mechanism', 'Recurrence vs Self-Attention').",
    )
    comparison_type: ComparisonType = Field(
        default=ComparisonType.SIMILARITY,
        description="Whether this represents a shared design choice ('SIMILARITY') or divergence ('DIFFERENCE').",
    )
    description: str = Field(
        ...,
        description="Comparative explanation of how the papers compare on this aspect.",
    )
    paper_details: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from paper_id to its specific implementation or design choice.",
    )
    evidence: list[SourceReference] = Field(
        default_factory=list,
        description="Evidence references grounding this comparison across papers.",
    )


class ExperimentalComparison(BaseModel):
    """Comparison of datasets, benchmarks, hardware, or training settings."""

    aspect: str = Field(
        ...,
        description="Experimental dimension (e.g. 'Evaluation Benchmark', 'Hardware Infrastructure', 'Optimizer').",
    )
    description: str = Field(
        ...,
        description="Comparative synthesis of experimental setups across the papers.",
    )
    paper_details: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from paper_id to its specific benchmark, hardware, or setting.",
    )
    evidence: list[SourceReference] = Field(
        default_factory=list,
        description="Evidence references grounding this experimental comparison.",
    )


class FindingComparison(BaseModel):
    """Comparative synthesis of empirical results and claims across papers."""

    topic: str = Field(
        ...,
        description="Core empirical question or task (e.g. 'Machine Translation Quality', 'Training Speed').",
    )
    relationship: FindingRelationship = Field(
        default=FindingRelationship.AGREEMENT,
        description="Whether results agree, diverge due to contextual factors, or form a genuine contradiction.",
    )
    description: str = Field(
        ...,
        description="Comparative synthesis of findings across papers.",
    )
    contextual_factors: list[str] = Field(
        default_factory=list,
        description="Contextual variables explaining differences (e.g. 'Different datasets', 'Different metrics', 'Model size differences').",
    )
    paper_findings: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from paper_id to its specific finding, score, or empirical observation.",
    )
    evidence: list[SourceReference] = Field(
        default_factory=list,
        description="Evidence references grounding the comparative findings.",
    )


class ResearchGap(BaseModel):
    """Synthesized research limitation, bottleneck, or unaddressed challenge."""

    gap: str = Field(
        ...,
        description="Description of the shared or unaddressed research gap/limitation.",
    )
    affected_papers: list[str] = Field(
        default_factory=list,
        description="List of paper_ids affected by or acknowledging this limitation.",
    )
    details: str | None = Field(
        default=None,
        description="Detailed explanation of the constraint, failure mode, or theoretical bound.",
    )
    evidence: list[SourceReference] = Field(
        default_factory=list,
        description="Evidence references grounding this research gap.",
    )


class FutureDirection(BaseModel):
    """Synthesized future research avenue or open direction."""

    direction: str = Field(
        ...,
        description="Proposed research direction or open opportunity.",
    )
    proposed_by: list[str] = Field(
        default_factory=list,
        description="List of paper_ids that explicitly motivate or propose this direction.",
    )
    potential_impact: str | None = Field(
        default=None,
        description="Anticipated benefits, theoretical importance, or practical applications.",
    )
    evidence: list[SourceReference] = Field(
        default_factory=list,
        description="Evidence references grounding this future work suggestion.",
    )


# ---------------------------------------------------------------------------
# Top-Level Cross-Paper Synthesis Output Model
# ---------------------------------------------------------------------------


class CrossPaperSynthesis(BaseModel):
    """Complete structured synthesis comparing two or more research papers."""

    overview: str = Field(
        ...,
        description="Executive cross-paper overview summarizing the collective problem domain and overarching findings.",
    )
    papers_analyzed: list[dict[str, str]] = Field(
        default_factory=list,
        description="Metadata for all synthesized papers: [{'paper_id': '...', 'paper_title': '...'}]",
    )
    common_themes: list[str] = Field(
        default_factory=list,
        description="High-level themes or shared objectives uniting the analyzed papers.",
    )
    methodological_similarities: list[MethodologicalComparison] = Field(
        default_factory=list,
        description="Shared architectural, mathematical, or procedural methods.",
    )
    methodological_differences: list[MethodologicalComparison] = Field(
        default_factory=list,
        description="Divergent architectural or procedural approaches.",
    )
    dataset_and_experimental_differences: list[ExperimentalComparison] = Field(
        default_factory=list,
        description="Comparative analysis of datasets, benchmarks, and experimental conditions.",
    )
    findings_agreement: list[FindingComparison] = Field(
        default_factory=list,
        description="Empirical or theoretical findings that concur across the papers.",
    )
    findings_disagreement: list[FindingComparison] = Field(
        default_factory=list,
        description="Findings that diverge or show contextual variance (carefully distinguishing context from contradiction).",
    )
    research_limitations_and_gaps: list[ResearchGap] = Field(
        default_factory=list,
        description="Collective limitations, computational bottlenecks, or unaddressed gaps.",
    )
    future_research_directions: list[FutureDirection] = Field(
        default_factory=list,
        description="Promising cross-cutting directions for future investigation.",
    )


# ---------------------------------------------------------------------------
# API Request / Response Schemas
# ---------------------------------------------------------------------------


class SynthesisRequest(BaseModel):
    """Request payload for multi-paper synthesis."""

    papers: list[ResearchPaperExtraction] = Field(
        ...,
        min_length=2,
        description="At least two structured research paper extractions to synthesize.",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Optional list of specific topics or dimensions to prioritize in the synthesis.",
    )

    @field_validator("papers")
    @classmethod
    def validate_min_papers(cls, papers: list[ResearchPaperExtraction]) -> list[ResearchPaperExtraction]:
        if len(papers) < 2:
            raise ValueError("Cross-paper synthesis requires at least two extracted research papers.")
        return papers


class SynthesisResponse(BaseModel):
    """Response payload returned by the cross-paper synthesis service."""

    data: CrossPaperSynthesis = Field(
        ...,
        description="Structured comparative cross-paper synthesis.",
    )
    model: str = Field(
        ...,
        description="LLM model identifier that produced the synthesis.",
    )
    papers_count: int = Field(
        ...,
        description="Number of papers successfully synthesized.",
    )
