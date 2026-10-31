import logging
import os
from fastapi import APIRouter, BackgroundTasks, Depends, status

from backend.api.documents import get_repository as get_document_repository
from backend.ingestion.repository import DocumentRepository
from backend.schemas.research import (
    StartResearchPayload,
    StartResearchResponse,
    ResearchResultResponse,
)
from backend.research.repository import (
    ResearchJobRepository,
    InMemoryResearchJobRepository,
)
from backend.research.service import ResearchJobService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

# Single shared research job repository across application lifecycle
_shared_job_repository: ResearchJobRepository = InMemoryResearchJobRepository()


def get_research_job_repository() -> ResearchJobRepository:
    """Dependency provider for ResearchJobRepository."""
    return _shared_job_repository


def get_research_service(
    job_repo: ResearchJobRepository = Depends(get_research_job_repository),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> ResearchJobService:
    """Dependency provider for ResearchJobService."""
    return ResearchJobService(job_repository=job_repo, document_repository=doc_repo)


@router.post(
    "/generate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=StartResearchResponse,
    summary="Initiate asynchronous research synthesis",
)
def generate_research(
    payload: StartResearchPayload,
    background_tasks: BackgroundTasks,
    service: ResearchJobService = Depends(get_research_service),
) -> StartResearchResponse:
    """
    Initiates an asynchronous research synthesis job for 1–4 ingested papers.
    Validates paper existence and completion status, persists the job in 'queued' status,
    and returns HTTP 202 Accepted with the polling URL.
    """
    logger.info(
        "Received research request: papers=%s, tasks=%s, focus=%s",
        payload.paper_ids,
        payload.tasks,
        bool(payload.focus_query),
    )
    response = service.create_job(payload)
    if not os.getenv("PYTEST_CURRENT_TEST"):
        background_tasks.add_task(service.process_job, response.research_id)
    return response


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=list[ResearchResultResponse],
    summary="List all research jobs",
)
@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=list[ResearchResultResponse],
    include_in_schema=False,
)
def list_research_jobs(
    service: ResearchJobService = Depends(get_research_service),
) -> list[ResearchResultResponse]:
    """Retrieves all stored research jobs sorted with the most recent first."""
    return service.list_jobs()


@router.get(
    "/{research_id}",
    status_code=status.HTTP_200_OK,
    response_model=ResearchResultResponse,
    summary="Retrieve research job status and results",
)
def get_research(
    research_id: str,
    service: ResearchJobService = Depends(get_research_service),
) -> ResearchResultResponse:
    """
    Retrieves the status, parameters, and generated results of an asynchronous research job.
    Returns HTTP 404 if the research job does not exist.
    """
    return service.get_job(research_id)


@router.post(
    "/{research_id}/verify",
    status_code=status.HTTP_200_OK,
    response_model=ResearchResultResponse,
    summary="Verify claims in generated research summary",
)
async def verify_research(
    research_id: str,
    service: ResearchJobService = Depends(get_research_service),
) -> ResearchResultResponse:
    """
    Evaluates claims from a completed research summary against source document evidence.
    Classifies each claim as SUPPORTED, CONTRADICTED, INSUFFICIENT EVIDENCE, or VERIFICATION FAILED.
    """
    return await service.verify_job(research_id)
