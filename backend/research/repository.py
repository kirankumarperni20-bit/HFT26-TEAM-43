import copy
import json
import logging
import os
from pathlib import Path
logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

from typing import Optional, Any

from pydantic import BaseModel, ConfigDict



from backend.schemas.research import (

    ResearchStatus,

    ResearchTask,

    PaperSynthesisItem,

    ComparisonResult,

    ResearchJobError,

)



try:

    from backend.llm.schemas.verification import ClaimVerification

except ModuleNotFoundError:

    try:

        from llm.schemas.verification import ClaimVerification

    except ModuleNotFoundError:

        ClaimVerification = Any





class ResearchJobRecord(BaseModel):

    """Internal persistence record for an asynchronous research job."""

    model_config = ConfigDict(extra="forbid")



    research_id: str

    status: ResearchStatus

    paper_ids: list[str]

    tasks: list[ResearchTask]

    focus_query: Optional[str] = None

    research_focus: Optional[str] = None

    created_at: str

    completed_at: Optional[str] = None

    papers: Optional[list[PaperSynthesisItem]] = None

    comparison: Optional[ComparisonResult] = None

    is_claim_verified: bool = False

    verifications: Optional[list[ClaimVerification]] = None

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

        is_claim_verified: bool = False,

    ) -> Optional[ResearchJobRecord]:

        """Stores the completed synthesis result and marks the job completed."""

        pass



    @abstractmethod

    def update_job_verifications(

        self,

        research_id: str,

        verifications: list[ClaimVerification],

        is_claim_verified: bool = True,

    ) -> Optional[ResearchJobRecord]:

        """Stores claim verification results for a research job."""

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
    In-memory implementation of ResearchJobRepository with deep-copy data isolation
    and automatic file-backed snapshot persistence across server restarts.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._jobs: dict[str, ResearchJobRecord] = {}
        self._storage_path: Optional[Path] = storage_path
        if self._storage_path is None and not os.getenv("PYTEST_CURRENT_TEST"):
            self._storage_path = Path(__file__).resolve().parent.parent / "data" / "research_jobs.json"
        self._load_from_storage()

    def _load_from_storage(self) -> None:
        if not self._storage_path or not self._storage_path.is_file():
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, list):
                for item in raw_data:
                    record = ResearchJobRecord.model_validate(item)
                    self._jobs[record.research_id] = record
        except Exception as exc:
            logger.warning("Failed to load research jobs from storage: %s", exc)

    def _persist_to_storage(self) -> None:
        if not self._storage_path:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = [job.model_dump() for job in self._jobs.values()]
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to persist research jobs to storage: %s", exc)

    def save_job(self, job: ResearchJobRecord) -> None:
        if not isinstance(job, ResearchJobRecord):
            raise TypeError(f"Expected ResearchJobRecord, got {type(job).__name__}")
        if job.research_id in self._jobs:
            raise ValueError(f"Job with ID '{job.research_id}' already exists.")
        self._jobs[job.research_id] = copy.deepcopy(job)
        self._persist_to_storage()

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
        self._persist_to_storage()
        return copy.deepcopy(updated)

    def update_job_result(
        self,
        research_id: str,
        papers: Optional[list[PaperSynthesisItem]],
        comparison: Optional[ComparisonResult],
        completed_at: str,
        is_claim_verified: bool = False,
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
                "is_claim_verified": is_claim_verified,
                "error": None,
            }
        )
        self._jobs[research_id] = updated
        self._persist_to_storage()
        return copy.deepcopy(updated)

    def update_job_verifications(
        self,
        research_id: str,
        verifications: list[ClaimVerification],
        is_claim_verified: bool = True,
    ) -> Optional[ResearchJobRecord]:
        job = self._jobs.get(research_id)
        if job is None:
            return None
        updated = job.model_copy(
            update={
                "verifications": copy.deepcopy(verifications),
                "is_claim_verified": is_claim_verified,
            }
        )
        self._jobs[research_id] = updated
        self._persist_to_storage()
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
        self._persist_to_storage()
        return copy.deepcopy(updated)

    def list_jobs(self) -> list[ResearchJobRecord]:
        return [copy.deepcopy(j) for j in self._jobs.values()]
