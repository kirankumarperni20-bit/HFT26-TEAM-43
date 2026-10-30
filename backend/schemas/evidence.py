"""Evidence schemas representing retrieval results returned to Member 3/4."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    """Represents a retrieved evidence item corresponding to a document chunk."""

    model_config = ConfigDict(str_strip_whitespace=False, extra="forbid")

    evidence_id: str = Field(..., description="Evidence identifier, provisionally mapped to chunk_id")
    chunk_id: str = Field(..., description="Identifier of the source chunk")
    paper_id: str = Field(..., description="Identifier of the source paper")
    page_number: int = Field(..., description="Page number of the source chunk")
    paragraph_number: int = Field(..., description="Paragraph number of the source chunk")
    text: str = Field(..., description="Exact chunk text preserved from the source chunk")
    retrieval_score: float = Field(..., description="Normalized or rank-fused retrieval relevance score")
    retrieval_method: Literal["hybrid", "vector", "lexical"] = Field(
        ..., description="Retrieval method used to fetch this evidence item ('hybrid', 'vector', or 'lexical')"
    )


class ClaimEvidenceResponse(BaseModel):
    """Container response returning all retrieved evidence for a given claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., description="Identifier of the claim being addressed")
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="List of retrieved evidence items, defaults to empty list if no results are found",
    )
