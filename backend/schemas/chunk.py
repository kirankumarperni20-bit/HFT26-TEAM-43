from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    """Canonical chunk schema used by the PDF ingestion pipeline."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    paper_id: str = Field(..., min_length=1)
    page_number: int = Field(..., gt=0)
    paragraph_number: int = Field(..., gt=0)
    text: str = Field(..., min_length=1)
    section: str | None = None

    @field_validator("chunk_id", "paper_id", mode="after")
    @classmethod
    def validate_non_empty_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identifier cannot be empty or whitespace only")
        return value

    @field_validator("text", mode="after")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return value


# Backward-compatible name required by Member 2's retrieval and database modules.
DocumentChunk = Chunk