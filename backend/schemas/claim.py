"""Claim schema representing an atomic claim input from Member 3."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Claim(BaseModel):
    """Represents an atomic claim extracted by Member 3 to be evaluated against specified papers."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., description="Unique identifier of the claim (e.g. 'C001')")
    claim_text: str = Field(..., description="Text content of the atomic claim")
    paper_ids: list[str] = Field(..., min_length=1, description="Non-empty list of paper identifiers to search within")

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure paper_ids contains at least one target paper."""
        if not v:
            raise ValueError("paper_ids must contain at least one paper identifier")
        return v
