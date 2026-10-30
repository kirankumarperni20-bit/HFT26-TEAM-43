import logging
from fastapi import APIRouter, Depends, status

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
    return service.create_job(payload)


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

