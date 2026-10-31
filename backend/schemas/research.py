from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    from backend.llm.schemas.verification import ClaimVerification
except ModuleNotFoundError:
    try:
        from llm.schemas.verification import ClaimVerification
    except ModuleNotFoundError:
        ClaimVerification = Any


class ResearchTask(str, Enum):
    SUMMARIZE = "summarize"
    COMPARE = "compare"


class ResearchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    paper_id: str = Field(..., min_length=1)
    page_number: int = Field(..., gt=0)
    paragraph_number: int = Field(..., gt=0)
    excerpt: str = Field(..., min_length=1)


class GroundedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class PaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: GroundedField
    methodology: GroundedField
    findings: list[GroundedField] = Field(default_factory=list)
    limitations: list[GroundedField] = Field(default_factory=list)


class PaperSynthesisItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    filename: Optional[str] = None
    summary: PaperSummary


class ComparisonCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    value: GroundedField


class ComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    cells: list[ComparisonCell]


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    synthesis_narrative: Optional[str] = None
    matrix: list[ComparisonRow] = Field(default_factory=list)


class ResearchJobError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    retryable: bool
    retry_after_seconds: Optional[int] = None


class StartResearchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_ids: list[str] = Field(..., min_length=1, max_length=4)
    tasks: list[ResearchTask] = Field(..., min_length=1)
    focus_query: Optional[str] = Field(default=None, max_length=500)
    research_focus: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def sync_focus_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            fq = data.get("focus_query")
            rf = data.get("research_focus")
            if rf is not None and fq is None:
                data["focus_query"] = rf
            elif fq is not None and rf is None:
                data["research_focus"] = fq
        return data

    @field_validator("paper_ids")
    @classmethod
    def validate_unique_papers(cls, v: list[str]) -> list[str]:
        if not (1 <= len(v) <= 4):
            raise ValueError("Must provide between 1 and 4 papers.")
        if len(set(v)) != len(v):
            raise ValueError("Duplicate paper IDs are not permitted.")
        for pid in v:
            if not pid or not pid.strip():
                raise ValueError("Paper IDs cannot be empty or whitespace.")
        return v

    @field_validator("tasks")
    @classmethod
    def validate_unique_tasks(cls, v: list[ResearchTask]) -> list[ResearchTask]:
        if not v:
            raise ValueError("At least one task must be selected.")
        if len(set(v)) != len(v):
            raise ValueError("Duplicate tasks are not permitted.")
        return v

    @model_validator(mode="after")
    def validate_comparison_paper_count(self) -> "StartResearchPayload":
        if ResearchTask.COMPARE in self.tasks and len(self.paper_ids) < 2:
            raise ValueError("Comparison requires at least two papers.")
        return self


class StartResearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_id: str
    status: ResearchStatus
    paper_ids: list[str]
    created_at: str
    poll_url: Optional[str] = None
    message: Optional[str] = None


class ResearchResultResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    research_id: str
    status: ResearchStatus
    created_at: str
    completed_at: Optional[str] = None
    paper_ids: list[str]
    tasks: list[ResearchTask]
    focus_query: Optional[str] = None
    research_focus: Optional[str] = None
    poll_url: Optional[str] = None
    papers: Optional[list[PaperSynthesisItem]] = None
    comparison: Optional[ComparisonResult] = None
    is_claim_verified: bool = False
    verifications: Optional[list[ClaimVerification]] = None
    error: Optional[ResearchJobError] = None
