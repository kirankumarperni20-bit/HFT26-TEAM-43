import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import HTTPException, status

from backend.ingestion.repository import DocumentRepository
from backend.schemas.document import DocumentStatus
from backend.schemas.research import (
    StartResearchPayload,
    StartResearchResponse,
    ResearchResultResponse,
    ResearchStatus,
    PaperSynthesisItem,
    ComparisonResult,
    ResearchJobError,
)
from backend.research.repository import ResearchJobRepository, ResearchJobRecord


class ResearchJobService:
    """
    Explicit service layer coordinating research job creation, document validation,
    status transitions, and result retrieval.
    """

    def __init__(
        self,
        job_repository: ResearchJobRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self.job_repository = job_repository
        self.document_repository = document_repository

    def create_job(self, payload: StartResearchPayload) -> StartResearchResponse:
        """
        Validates requested papers and persists a new asynchronous research job in 'queued' status.
        Raises HTTPException(400) if any paper does not exist or has not completed ingestion.
        """
        # Validate that all requested papers exist and have completed ingestion
        for paper_id in payload.paper_ids:
            doc = self.document_repository.get_document(paper_id)
            if doc is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Paper '{paper_id}' does not exist.",
                )
            if doc.status != DocumentStatus.COMPLETED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Paper '{paper_id}' has not completed ingestion "
                        f"(current status: '{doc.status.value}')."
                    ),
                )

        research_id = f"res_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        poll_url = f"/api/research/{research_id}"

        record = ResearchJobRecord(
            research_id=research_id,
            status=ResearchStatus.QUEUED,
            paper_ids=payload.paper_ids,
            tasks=payload.tasks,
            focus_query=payload.focus_query,
            created_at=now_iso,
            completed_at=None,
            papers=None,
            comparison=None,
            is_claim_verified=False,
            error=None,
        )

        self.job_repository.save_job(record)

        return StartResearchResponse(
            research_id=research_id,
            status=ResearchStatus.QUEUED,
            paper_ids=payload.paper_ids,
            created_at=now_iso,
            poll_url=poll_url,
            message="Research job accepted and queued for processing.",
        )

    def get_job(self, research_id: str) -> ResearchResultResponse:
        """
        Retrieves current status and results for a research job.
        Raises HTTPException(404) if the job ID is unknown.
        """
        job = self.job_repository.get_job(research_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )

        return ResearchResultResponse(
            research_id=job.research_id,
            status=job.status,
            created_at=job.created_at,
            completed_at=job.completed_at,
            paper_ids=job.paper_ids,
            tasks=job.tasks,
            focus_query=job.focus_query,
            poll_url=f"/api/research/{job.research_id}",
            papers=job.papers,
            comparison=job.comparison,
            is_claim_verified=job.is_claim_verified,
            error=job.error,
        )

    def transition_to_processing(self, research_id: str) -> ResearchJobRecord:
        """Transitions job to 'processing' status."""
        job = self.job_repository.get_job(research_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )
        updated = self.job_repository.update_job_status(
            research_id, ResearchStatus.PROCESSING
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )
        return updated

    def complete_job(
        self,
        research_id: str,
        papers: Optional[list[PaperSynthesisItem]] = None,
        comparison: Optional[ComparisonResult] = None,
    ) -> ResearchJobRecord:
        """Stores synthesis results and marks the job 'completed'."""
        now_iso = datetime.now(timezone.utc).isoformat()
        updated = self.job_repository.update_job_result(
            research_id=research_id,
            papers=papers,
            comparison=comparison,
            completed_at=now_iso,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )
        return updated

    def fail_job(
        self, research_id: str, error: ResearchJobError
    ) -> ResearchJobRecord:
        """Records error information and marks the job 'failed'."""
        now_iso = datetime.now(timezone.utc).isoformat()
        updated = self.job_repository.update_job_failed(
            research_id=research_id,
            error=error,
            completed_at=now_iso,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )
        return updated

