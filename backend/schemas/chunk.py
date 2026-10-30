"""DocumentChunk schema representing an extracted chunk from Member 1's ingestion pipeline."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """Represents a discrete text chunk extracted from a research paper by Member 1."""

    model_config = ConfigDict(str_strip_whitespace=False, extra="forbid")

    chunk_id: str = Field(..., description="Unique chunk identifier assigned by Member 1 (e.g. 'P001_C002')")
    paper_id: str = Field(..., description="Identifier of the source paper (e.g. 'P001')")
    page_number: int = Field(..., description="Page number where the chunk appears")
    paragraph_number: int = Field(..., description="Paragraph position within the page")
    text: str = Field(..., description="Exact raw text content of the chunk, preserved without alteration")
    section: str | None = Field(default=None, description="Optional section heading name (e.g. 'Results')")
