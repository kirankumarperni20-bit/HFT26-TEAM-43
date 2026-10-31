"""
main.py — FastAPI application entry point.

Responsibilities:
    - Creates the FastAPI app instance.
    - Mounts the /llm router from the llm module.
    - Exposes a root health-check endpoint.
"""

import logging
import os
from collections.abc import Generator
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

logger = logging.getLogger(__name__)

try:
    from backend.database.connection import get_db_session_dep, load_environment
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
except ModuleNotFoundError:
    from database.connection import get_db_session_dep, load_environment
    from llm import router as llm_router
    from llm.schemas.claim import ClaimDecompositionRequest, ClaimDecompositionResponse
    from llm.schemas.synthesis import SynthesisRequest, SynthesisResponse
    from llm.schemas.verification import VerificationRequest, VerificationResponse
    from llm.services.claim_service import ClaimDecompositionService
    from llm.services.synthesis_service import SynthesisService
    from llm.services.verification_service import ClaimVerificationService
    from llm.services.retrieval_service import (
        EvidenceRetrievalService,
        create_retriever_from_session,
    )

# Ensure backend/.env is loaded upon application startup
load_environment()

app = FastAPI(
    title="AI Research Assistant — GenAI Module",
    description="Backend service for verifiable summaries and cross-paper synthesis.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Research API Router
# ---------------------------------------------------------------------------

research_router = APIRouter(prefix="/api/research", tags=["Research"])


@research_router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_endpoint(request: SynthesisRequest) -> SynthesisResponse:
    """Perform cross-paper comparative synthesis across multiple extracted research papers."""
    try:
        service = SynthesisService()
        return await service.synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Synthesis failed: {exc}")


@research_router.post("/decompose", response_model=ClaimDecompositionResponse)
async def decompose_endpoint(request: ClaimDecompositionRequest) -> ClaimDecompositionResponse:
    """Decompose cross-paper synthesis statements into independently verifiable atomic claims."""
    try:
        service = ClaimDecompositionService()
        return await service.decompose(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Claim decomposition failed: {exc}")


@research_router.post("/verify", response_model=VerificationResponse)
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
# Mount sub-routers
# ---------------------------------------------------------------------------

app.include_router(research_router)
app.include_router(llm_router, prefix="/llm", tags=["LLM"])

