"""
llm — GenAI module package.

Exports the FastAPI APIRouter so that main.py can mount it
under the /llm prefix.
"""

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

logger = logging.getLogger(__name__)

try:
    from backend.database.connection import get_db_session_dep
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
    from database.connection import get_db_session_dep
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

router = APIRouter()


@router.get("/status")
async def llm_status():
    """Return the current status of the LLM module."""
    return {"module": "llm", "status": "initialised"}


@router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_papers_endpoint(request: SynthesisRequest) -> SynthesisResponse:
    """Perform cross-paper comparative synthesis across multiple extracted research papers."""
    try:
        service = SynthesisService()
        return await service.synthesize(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Synthesis failed: {exc}")


@router.post("/decompose", response_model=ClaimDecompositionResponse)
async def decompose_claims_endpoint(request: ClaimDecompositionRequest) -> ClaimDecompositionResponse:
    """Decompose synthesis statements into atomic claims."""
    try:
        service = ClaimDecompositionService()
        return await service.decompose(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Claim decomposition failed: {exc}")


@router.post("/verify", response_model=VerificationResponse)
async def verify_claims_endpoint(
    request: VerificationRequest,
    db: Any = Depends(get_db_session_dep),
) -> VerificationResponse:
    """Verify atomic claims against candidate evidence."""
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


