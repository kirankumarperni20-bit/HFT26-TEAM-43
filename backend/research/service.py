import logging
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
    ResearchTask,
    PaperSynthesisItem,
    ComparisonResult,
    ComparisonRow,
    ComparisonCell,
    GroundedField,
    EvidenceReference,
    PaperSummary,
    ResearchJobError,
)
from backend.research.repository import ResearchJobRepository, ResearchJobRecord

try:
    from backend.llm.schemas.claim import AtomicClaim
except ModuleNotFoundError:
    from llm.schemas.claim import AtomicClaim

logger = logging.getLogger(__name__)


class ResearchJobService:
    """
    Explicit service layer coordinating research job creation, document validation,
    status transitions, result retrieval, and evidence-based claim verification.
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
            verifications=None,
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
            verifications=job.verifications,
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
        is_claim_verified: bool = False,
    ) -> ResearchJobRecord:
        """Stores synthesis results and marks the job 'completed'."""
        now_iso = datetime.now(timezone.utc).isoformat()
        updated = self.job_repository.update_job_result(
            research_id=research_id,
            papers=papers,
            comparison=comparison,
            completed_at=now_iso,
            is_claim_verified=is_claim_verified,
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

    async def verify_job(self, research_id: str) -> ResearchResultResponse:
        """
        Extracts claims from a completed research job's summary and comparative matrix,
        verifies them against candidate evidence passages with strict provenance validation,
        updates the job record with the verification results, and marks is_claim_verified=True.
        """
        job = self.job_repository.get_job(research_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )
        if job.status != ResearchStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot verify claims for job with status '{job.status.value}'. Must be completed.",
            )

        claims: list[AtomicClaim] = []
        claim_idx = 1

        if job.papers:
            for paper in job.papers:
                pid = paper.paper_id
                # Objective claim
                if paper.summary.objective and paper.summary.objective.text and paper.summary.objective.text.strip():
                    claims.append(
                        AtomicClaim(
                            claim_id=f"clm_{pid}_{claim_idx}",
                            claim_text=paper.summary.objective.text.strip(),
                            source_text=paper.summary.objective.text.strip(),
                            involved_papers=[pid],
                        )
                    )
                    claim_idx += 1
                # Methodology claim
                if paper.summary.methodology and paper.summary.methodology.text and paper.summary.methodology.text.strip():
                    claims.append(
                        AtomicClaim(
                            claim_id=f"clm_{pid}_{claim_idx}",
                            claim_text=paper.summary.methodology.text.strip(),
                            source_text=paper.summary.methodology.text.strip(),
                            involved_papers=[pid],
                        )
                    )
                    claim_idx += 1
                # Findings claims
                for f in paper.summary.findings:
                    if f.text and f.text.strip():
                        claims.append(
                            AtomicClaim(
                                claim_id=f"clm_{pid}_{claim_idx}",
                                claim_text=f.text.strip(),
                                source_text=f.text.strip(),
                                involved_papers=[pid],
                            )
                        )
                        claim_idx += 1
                # Limitations claims
                for lim in paper.summary.limitations:
                    if lim.text and lim.text.strip():
                        claims.append(
                            AtomicClaim(
                                claim_id=f"clm_{pid}_{claim_idx}",
                                claim_text=lim.text.strip(),
                                source_text=lim.text.strip(),
                                involved_papers=[pid],
                            )
                        )
                        claim_idx += 1

        if job.comparison and job.comparison.matrix:
            for row in job.comparison.matrix:
                for cell in row.cells:
                    if cell.value and cell.value.text and cell.value.text.strip():
                        claims.append(
                            AtomicClaim(
                                claim_id=f"clm_comp_{cell.paper_id}_{claim_idx}",
                                claim_text=cell.value.text.strip(),
                                source_text=cell.value.text.strip(),
                                involved_papers=[cell.paper_id],
                            )
                        )
                        claim_idx += 1

        if not claims:
            # If no text claims were present to verify
            updated = self.job_repository.update_job_verifications(
                research_id=research_id,
                verifications=[],
                is_claim_verified=True,
            )
            return self.get_job(research_id)

        from backend.llm.services.verification_service import ClaimVerificationService
        from backend.llm.services.retrieval_service import EvidenceRetrievalService
        from backend.llm.schemas.verification import VerificationRequest

        retrieval_service = EvidenceRetrievalService(document_repository=self.document_repository)
        verif_service = ClaimVerificationService(retrieval_service=retrieval_service)

        req = VerificationRequest(claims=claims, allowed_paper_ids=job.paper_ids)
        verif_resp = await verif_service.verify(req)

        updated = self.job_repository.update_job_verifications(
            research_id=research_id,
            verifications=verif_resp.verifications,
            is_claim_verified=True,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research job '{research_id}' not found.",
            )

        return self.get_job(research_id)

    async def process_job(self, research_id: str) -> None:
        """
        Asynchronous background task to process a queued research job.
        Extracts grounded summary information for each paper with strict
        publication-metadata filtering, generates verifiable cross-paper synthesis
        if requested, and transitions the job to 'completed'.
        """
        try:
            job = self.job_repository.get_job(research_id)
            if not job:
                return

            self.transition_to_processing(research_id)

            from backend.research.cleaner import (
                synthesize_paper_summary,
                synthesize_cross_paper_comparison,
            )

            papers_synthesis: list[PaperSynthesisItem] = []
            for paper_id in job.paper_ids:
                doc = self.document_repository.get_document(paper_id)
                filename = doc.filename if doc and doc.filename else f"{paper_id}.pdf"
                chunks = self.document_repository.get_chunks(paper_id)

                summary = synthesize_paper_summary(chunks, paper_id, filename)
                papers_synthesis.append(
                    PaperSynthesisItem(
                        paper_id=paper_id,
                        filename=filename,
                        summary=summary,
                    )
                )

            comparison: Optional[ComparisonResult] = None
            if ResearchTask.COMPARE in job.tasks or len(job.paper_ids) >= 2:
                comparison = synthesize_cross_paper_comparison(papers_synthesis)

            self.complete_job(
                research_id=research_id,
                papers=papers_synthesis,
                comparison=comparison,
                is_claim_verified=False,
            )
        except Exception as exc:
            logger.exception("Error processing research job %s: %s", research_id, exc)
            err = ResearchJobError(
                code="PROCESSING_ERROR",
                message=f"Research synthesis encountered an unexpected error: {exc}",
                retryable=True,
                retry_after_seconds=5,
            )
            self.fail_job(research_id, err)
