from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    paper_id: str = Field(..., min_length=1)
    page_number: int = Field(..., gt=0)
    paragraph_number: int = Field(..., gt=0)
    text: str = Field(..., min_length=1)
    section: str | None = None

    @field_validator("chunk_id", "paper_id", mode="after")
    @classmethod
    def validate_non_empty_identifiers(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Identifier cannot be empty or whitespace only")
        return v

    @field_validator("text", mode="after")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace only")
        # Return exact string untouched: do not strip, lowercase, or normalize
        return v
