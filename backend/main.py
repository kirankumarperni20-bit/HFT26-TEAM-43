"""
main.py — FastAPI application entry point.

Unified entry point integrating:
- Document ingestion & chunk API (/api/documents)
- Asynchronous Research Generation & Result Polling API (/api/research/generate, /api/research/{research_id})
- GenAI LLM Research Endpoints (/api/research/synthesize, /api/research/decompose, /api/research/verify)
- Dedicated LLM Module Router (/llm)
- Health check (/health)
- CORS Middleware for React frontend
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Ensure environment variables are loaded
from backend.database.connection import get_db_session_dep, load_environment

load_environment()

# Document & Research Ingestion/Job APIs
from backend.api.documents import router as documents_router, get_repository
from backend.api.research import router as research_router, get_research_job_repository

# GenAI LLM Services & Schemas
from backend.llm import router as llm_router
from backend.llm.schemas.claim import ClaimDecompositionRequest, ClaimDecompositionResponse
from backend.llm.schemas.synthesis import SynthesisRequest, SynthesisResponse
from backend.llm.schemas.verification import VerificationRequest, VerificationResponse
from backend.llm.services.claim_service import ClaimDecompositionService
from backend.llm.services.synthesis_service import SynthesisService
from backend.llm.services.verification_service import ClaimVerificationService
from backend.llm.services.retrieval_service import (
    EvidenceRetrievalService,
    create_retriever_from_session,
)

app = FastAPI(
    title="AI Research Assistant",
    description="Backend service for verifiable summaries, cross-paper synthesis, and document exploration.",
    version="0.1.0",
)

# CORS configuration
DEFAULT_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

frontend_origins_env = os.getenv("FRONTEND_ORIGINS")
if frontend_origins_env:
    allowed_origins = [orig.strip() for orig in frontend_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = DEFAULT_FRONTEND_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint to verify backend service status."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# GenAI Endpoints mounted under /api/research
# ---------------------------------------------------------------------------
genai_research_router = APIRouter(prefix="/api/research", tags=["Research GenAI"])


@genai_research_router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_endpoint(request: SynthesisRequest) -> SynthesisResponse:
    """Perform cross-paper comparative synthesis across multiple extracted research papers."""
    try:
        service = SynthesisService()
        return await service.synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Synthesis failed: {exc}")


@genai_research_router.post("/decompose", response_model=ClaimDecompositionResponse)
async def decompose_endpoint(request: ClaimDecompositionRequest) -> ClaimDecompositionResponse:
    """Decompose cross-paper synthesis statements into independently verifiable atomic claims."""
    try:
        service = ClaimDecompositionService()
        return await service.decompose(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Claim decomposition failed: {exc}")


@genai_research_router.post("/verify", response_model=VerificationResponse)
async def verify_endpoint(
    request: VerificationRequest,
    db: Any = Depends(get_db_session_dep),
) -> VerificationResponse:
    """Verify atomic claims against retrieved candidate evidence passages with strict provenance validation."""
    try:
        if db is not None:
            retriever = create_retriever_from_session(db)
        else:
            logger.info("Database session is inactive. Claim verification will use local PDF fallback corpus.")
            retriever = None
        retrieval_service = EvidenceRetrievalService(retriever=retriever)
        service = ClaimVerificationService(retrieval_service=retrieval_service)
        return await service.verify(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Claim verification failed: {exc}")


# ---------------------------------------------------------------------------
# Mount Routers
# ---------------------------------------------------------------------------
app.include_router(documents_router, prefix="/api")
app.include_router(research_router, prefix="/api")
app.include_router(genai_research_router)
app.include_router(llm_router, prefix="/llm", tags=["LLM"])


def get_shared_repository():
    """Access to the application-level shared document repository."""
    return get_repository()


def get_shared_research_repository():
    """Access to the application-level shared research job repository."""
    return get_research_job_repository()
