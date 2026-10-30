import copy
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, ConfigDict

from backend.schemas.research import (
    ResearchStatus,
    ResearchTask,
    PaperSynthesisItem,
    ComparisonResult,
    ResearchJobError,
)


class ResearchJobRecord(BaseModel):
    """Internal persistence record for an asynchronous research job."""
    model_config = ConfigDict(extra="forbid")

    research_id: str
    status: ResearchStatus
    paper_ids: list[str]
    tasks: list[ResearchTask]
    focus_query: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    papers: Optional[list[PaperSynthesisItem]] = None
    comparison: Optional[ComparisonResult] = None
    is_claim_verified: bool = False
    error: Optional[ResearchJobError] = None


class ResearchJobRepository(ABC):
    """
    Abstract interface for research job persistence.
    Provides a clean boundary allowing Member 2 to replace the in-memory
    implementation with PostgreSQL without impacting research service logic.
    """

    @abstractmethod
    def save_job(self, job: ResearchJobRecord) -> None:
        """Saves a newly created research job record. Raises ValueError if job already exists."""
        pass

    @abstractmethod
    def get_job(self, research_id: str) -> Optional[ResearchJobRecord]:
        """Retrieves a research job record by ID, or None if not found."""
        pass

    @abstractmethod
    def update_job_status(
        self, research_id: str, status: ResearchStatus
    ) -> Optional[ResearchJobRecord]:
        """Updates the status of an existing job."""
        pass

    @abstractmethod
    def update_job_result(
        self,
        research_id: str,
        papers: Optional[list[PaperSynthesisItem]],
        comparison: Optional[ComparisonResult],
        completed_at: str,
    ) -> Optional[ResearchJobRecord]:
        """Stores the completed synthesis result and marks the job completed."""
        pass

    @abstractmethod
    def update_job_failed(
        self,
        research_id: str,
        error: ResearchJobError,
        completed_at: str,
    ) -> Optional[ResearchJobRecord]:
        """Records error details and marks the job failed."""
        pass

    @abstractmethod
    def list_jobs(self) -> list[ResearchJobRecord]:
        """Lists all stored research jobs."""
        pass


class InMemoryResearchJobRepository(ResearchJobRepository):
    """
    Temporary in-memory implementation of ResearchJobRepository with deep-copy data isolation.
    NOTE: All data is stored strictly in-process memory and will not persist across restarts.
    Designed for replacement by Member 2 with PostgreSQL/SQLAlchemy.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, ResearchJobRecord] = {}

    def save_job(self, job: ResearchJobRecord) -> None:
        if not isinstance(job, ResearchJobRecord):
            raise TypeError(f"Expected ResearchJobRecord, got {type(job).__name__}")
        if job.research_id in self._jobs:
            raise ValueError(f"Job with ID '{job.research_id}' already exists.")
        self._jobs[job.research_id] = copy.deepcopy(job)

    def get_job(self, research_id: str) -> Optional[ResearchJobRecord]:
        job = self._jobs.get(research_id)
        return copy.deepcopy(job) if job is not None else None

    def update_job_status(
        self, research_id: str, status: ResearchStatus
    ) -> Optional[ResearchJobRecord]:
        job = self._jobs.get(research_id)
        if job is None:
            return None
        updated = job.model_copy(update={"status": status})
        self._jobs[research_id] = updated
        return copy.deepcopy(updated)

    def update_job_result(
        self,
        research_id: str,
        papers: Optional[list[PaperSynthesisItem]],
        comparison: Optional[ComparisonResult],
        completed_at: str,
    ) -> Optional[ResearchJobRecord]:
        job = self._jobs.get(research_id)
        if job is None:
            return None
        updated = job.model_copy(
            update={
                "status": ResearchStatus.COMPLETED,
                "papers": copy.deepcopy(papers) if papers is not None else None,
                "comparison": copy.deepcopy(comparison) if comparison is not None else None,
                "completed_at": completed_at,
                "is_claim_verified": False,
                "error": None,
            }
        )
        self._jobs[research_id] = updated
        return copy.deepcopy(updated)

    def update_job_failed(
        self,
        research_id: str,
        error: ResearchJobError,
        completed_at: str,
    ) -> Optional[ResearchJobRecord]:
        job = self._jobs.get(research_id)
        if job is None:
            return None
        updated = job.model_copy(
            update={
                "status": ResearchStatus.FAILED,
                "error": copy.deepcopy(error),
                "completed_at": completed_at,
            }
        )
        self._jobs[research_id] = updated
        return copy.deepcopy(updated)

    def list_jobs(self) -> list[ResearchJobRecord]:
        return [copy.deepcopy(j) for j in self._jobs.values()]

