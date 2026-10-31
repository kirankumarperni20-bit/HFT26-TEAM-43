"""
services — Business-logic layer for the LLM module.

Each public function in this module represents a discrete capability
(e.g. summarise, synthesise).  Services orchestrate:
    prompt construction  →  LLM call  →  output validation  →  response
"""

from __future__ import annotations

try:
    from backend.llm.client import LLMClient
    from backend.llm.config import get_settings
    from backend.llm.schemas import SummarizeRequest, SummarizeResponse
    from backend.llm.services.extraction_service import ExtractionService
    from backend.llm.services.synthesis_service import SynthesisService, synthesize_papers
    from backend.llm.services.claim_service import ClaimDecompositionService, decompose_claims
    from backend.llm.services.retrieval_service import EvidenceRetrievalService
    from backend.llm.services.verification_service import ClaimVerificationService, verify_claims
except ModuleNotFoundError:
    from llm.client import LLMClient
    from llm.config import get_settings
    from llm.schemas import SummarizeRequest, SummarizeResponse
    from llm.services.extraction_service import ExtractionService
    from llm.services.synthesis_service import SynthesisService, synthesize_papers
    from llm.services.claim_service import ClaimDecompositionService, decompose_claims
    from llm.services.retrieval_service import EvidenceRetrievalService
    from llm.services.verification_service import ClaimVerificationService, verify_claims

__all__ = [
    "summarize",
    "ExtractionService",
    "SynthesisService",
    "synthesize_papers",
    "ClaimDecompositionService",
    "decompose_claims",
    "EvidenceRetrievalService",
    "ClaimVerificationService",
    "verify_claims",
]
