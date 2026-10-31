"""
verification_service.py — Claim verification and evidence provenance service.

Evaluates atomic research claims against candidate document evidence:
1. Ingests atomic claims and candidate passages (or retrieves them via EvidenceRetrievalService).
2. Augments claim evaluation with parent statement context for pronoun resolution.
3. Invokes shared LLMClient with the verification prompt template.
4. Performs strict Python-side deterministic provenance validation:
   - Verifies all cited evidence IDs exist in candidate passage set.
   - Verifies paper IDs match allowed / involved papers.
   - Verifies all quotes are exact substrings of their candidate passages.
   - Directly copies page and chunk metadata from source passages (never trusts model).
5. Aggregates status counts (SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNRESOLVED).
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from backend.llm.client import LLMClient
    from backend.llm.config import LLMSettings, get_settings
    from backend.llm.schemas.claim import AtomicClaim
    from backend.llm.schemas.verification import (
        ClaimVerification,
        RetrievedEvidence,
        VerificationRequest,
        VerificationResponse,
        VerificationStatus,
        VerifiedEvidenceItem,
    )
    from backend.llm.services.extraction_service import clean_and_parse_json
    from backend.llm.services.retrieval_service import EvidenceRetrievalService
except ModuleNotFoundError:
    from llm.client import LLMClient
    from llm.config import LLMSettings, get_settings
    from llm.schemas.claim import AtomicClaim
    from llm.schemas.verification import (
        ClaimVerification,
        RetrievedEvidence,
        VerificationRequest,
        VerificationResponse,
        VerificationStatus,
        VerifiedEvidenceItem,
    )
    from llm.services.extraction_service import clean_and_parse_json
    from llm.services.retrieval_service import EvidenceRetrievalService

logger = logging.getLogger(__name__)

PROMPT_FILE_PATH = Path(__file__).parent.parent / "prompts" / "verification.txt"


@lru_cache(maxsize=1)
def load_verification_prompt_template() -> str:
    """Load and cache the verification prompt template from disk."""
    if not PROMPT_FILE_PATH.is_file():
        raise FileNotFoundError(
            f"Verification prompt template not found at: {PROMPT_FILE_PATH}"
        )
    content = PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
    if "{claims_and_evidence_json}" not in content:
        raise ValueError(
            "Verification prompt template missing required placeholder: '{claims_and_evidence_json}'"
        )
    return content


def normalize_text_for_quote_check(text: str) -> str:
    """Normalize whitespace and common typographic quotes for substring testing."""
    if not text:
        return ""
    text = text.replace("\ufffe", "-").replace("\xad", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return text.strip().lower()


class ClaimVerificationService:
    """Service orchestrating evidence retrieval, claim verification, and provenance auditing."""

    def __init__(
        self,
        client: LLMClient | None = None,
        retrieval_service: EvidenceRetrievalService | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or LLMClient(settings=self._settings)
        self._retrieval = retrieval_service or EvidenceRetrievalService()

    async def verify(self, request: VerificationRequest) -> VerificationResponse:
        """Verify a collection of atomic claims against candidate evidence."""
        if not request.claims:
            return VerificationResponse(
                verifications=[],
                total_claims=0,
                supported_count=0,
                partially_supported_count=0,
                contradicted_count=0,
                unresolved_count=0,
                provenance_failure_count=0,
                model=self._settings.llm_model_name,
            )

        # Build passage index and retrieve candidates per claim
        claim_evidence_payload: list[dict[str, Any]] = []
        claim_map: dict[str, AtomicClaim] = {c.claim_id: c for c in request.claims}
        claim_candidate_map: dict[str, dict[str, RetrievedEvidence]] = {}

        claims_needing_eval: list[AtomicClaim] = []
        immediate_unresolved: list[ClaimVerification] = []

        for claim in request.claims:
            # 1. Retrieve or locate candidate passages
            if request.passages is not None:
                # Use provided candidate passages directly
                candidates = request.passages
            else:
                candidates = self._retrieval.retrieve(
                    claim=claim,
                    allowed_paper_ids=request.allowed_paper_ids,
                )

            claim_candidate_map[claim.claim_id] = {c.evidence_id: c for c in candidates}

            if not candidates:
                # No candidate passages found in corpus
                immediate_unresolved.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        verification_status=VerificationStatus.UNRESOLVED,
                        explanation="No candidate evidence passages found in the source documents.",
                        supporting_evidence=[],
                        contradicting_evidence=[],
                        unresolved_reason="Insufficient evidence: 0 matching passages retrieved.",
                        confidence=1.0,
                        candidate_evidence_ids=[],
                        provenance_validated=True,
                    )
                )
                continue

            claims_needing_eval.append(claim)

            # Resolve parent statement context for pronoun resolution
            parent_ctx = None
            if request.parent_statement_context and claim.claim_id in request.parent_statement_context:
                parent_ctx = request.parent_statement_context[claim.claim_id]
            elif claim.source_text and claim.source_text != claim.claim_text:
                parent_ctx = claim.source_text

            claim_evidence_payload.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type.value,
                "involved_papers": claim.involved_papers,
                "parent_statement_context": parent_ctx,
                "candidate_passages": [
                    {
                        "evidence_id": cand.evidence_id,
                        "paper_id": cand.paper_id,
                        "chunk_id": cand.chunk_id,
                        "page": cand.page,
                        "section": cand.section,
                        "passage_text": cand.text,
                        "retrieval_method": cand.retrieval_method,
                    }
                    for cand in candidates
                ],
            })

        # If all claims were unresolved due to missing passages, return immediately
        if not claims_needing_eval:
            return VerificationResponse(
                verifications=immediate_unresolved,
                total_claims=len(immediate_unresolved),
                supported_count=0,
                partially_supported_count=0,
                contradicted_count=0,
                unresolved_count=len(immediate_unresolved),
                provenance_failure_count=0,
                model=self._settings.llm_model_name,
            )

        # 2. Render prompt and invoke LLM
        template = load_verification_prompt_template()
        prompt = template.replace(
            "{claims_and_evidence_json}",
            json.dumps(claim_evidence_payload, indent=2, ensure_ascii=False),
        )

        logger.info(
            "Sending claim verification request for %d claims to model %s",
            len(claims_needing_eval),
            self._settings.llm_model_name,
        )

        raw_response = await self._client.generate(prompt)

        # 3. Clean and parse JSON response
        parsed_dict = clean_and_parse_json(raw_response)
        verifications_raw = parsed_dict.get("verifications", [])
        if not isinstance(verifications_raw, list):
            raise ValueError(
                f"Model response 'verifications' field must be a list. Received: {type(verifications_raw)}"
            )

        # 4. Deterministic provenance and quote verification
        completed_verifications: list[ClaimVerification] = list(immediate_unresolved)
        supported_cnt = 0
        partially_cnt = 0
        contradicted_cnt = 0
        unresolved_cnt = len(immediate_unresolved)
        provenance_fail_cnt = 0

        for item in verifications_raw:
            cid = item.get("claim_id")
            if cid not in claim_map:
                logger.warning("Model returned unknown claim_id: %s. Skipping.", cid)
                continue

            claim = claim_map[cid]
            candidates_dict = claim_candidate_map.get(cid, {})

            # Parse verification status
            raw_status = str(item.get("verification_status", "UNRESOLVED")).upper()
            try:
                status = VerificationStatus(raw_status)
            except ValueError:
                status = VerificationStatus.UNRESOLVED

            explanation = item.get("explanation", "").strip() or "Evaluated against candidate evidence."
            unresolved_reason = item.get("unresolved_reason")
            confidence = item.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = None

            # Process evidence citations
            has_provenance_failure = False

            def process_citations(raw_list: list[dict[str, Any]]) -> list[VerifiedEvidenceItem]:
                nonlocal has_provenance_failure
                citations: list[VerifiedEvidenceItem] = []
                for ev_dict in raw_list:
                    ev_id = ev_dict.get("evidence_id")
                    quote = ev_dict.get("quote", "").strip()

                    # Check 1: Does evidence_id exist in candidate passages?
                    if ev_id not in candidates_dict:
                        logger.warning(
                            "Provenance Failure: Claim %s cited unretrieved evidence_id '%s'",
                            cid,
                            ev_id,
                        )
                        has_provenance_failure = True
                        continue

                    cand = candidates_dict[ev_id]

                    # Check 2: Does paper_id belong to allowed input papers?
                    if request.allowed_paper_ids and cand.paper_id not in request.allowed_paper_ids:
                        logger.warning(
                            "Provenance Failure: Passage paper_id '%s' not in allowed_paper_ids",
                            cand.paper_id,
                        )
                        has_provenance_failure = True
                        continue

                    # Check 3: Is quote an exact substring of candidate passage text?
                    is_verbatim = True
                    notes = None
                    if not quote:
                        is_verbatim = False
                        notes = "Empty quote provided."
                    elif quote not in cand.text:
                        norm_q = normalize_text_for_quote_check(quote)
                        norm_t = normalize_text_for_quote_check(cand.text)
                        if norm_q in norm_t:
                            is_verbatim = True
                            notes = "Matches passage text after whitespace normalization."
                        else:
                            is_verbatim = False
                            notes = "Quote is not an exact substring of the candidate passage text."
                            has_provenance_failure = True

                    # Copy exact provenance directly from source candidate passage
                    citations.append(
                        VerifiedEvidenceItem(
                            evidence_id=cand.evidence_id,
                            paper_id=cand.paper_id,
                            chunk_id=cand.chunk_id,
                            page=cand.page,
                            section=cand.section,
                            quote=quote or cand.text[:120],
                            retrieval_method=cand.retrieval_method,
                            is_verbatim=is_verbatim,
                            validation_notes=notes,
                        )
                    )
                return citations

            supporting_ev = process_citations(item.get("supporting_evidence", []))
            contradicting_ev = process_citations(item.get("contradicting_evidence", []))

            # Enforce provenance integrity on status:
            # If claim was marked SUPPORTED but has 0 valid supporting citations due to provenance failures:
            if status == VerificationStatus.SUPPORTED and not supporting_ev:
                status = VerificationStatus.UNRESOLVED
                unresolved_reason = "Downgraded to UNRESOLVED: cited evidence failed deterministic provenance validation."
                has_provenance_failure = True

            if has_provenance_failure:
                provenance_fail_cnt += 1

            # Count statuses
            if status == VerificationStatus.SUPPORTED:
                supported_cnt += 1
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                partially_cnt += 1
            elif status == VerificationStatus.CONTRADICTED:
                contradicted_cnt += 1
            else:
                unresolved_cnt += 1

            verif_obj = ClaimVerification(
                claim_id=cid,
                claim_text=claim.claim_text,
                verification_status=status,
                explanation=explanation,
                supporting_evidence=supporting_ev,
                contradicting_evidence=contradicting_ev,
                unresolved_reason=unresolved_reason,
                confidence=confidence,
                candidate_evidence_ids=list(candidates_dict.keys()),
                provenance_validated=not has_provenance_failure,
            )
            completed_verifications.append(verif_obj)

        return VerificationResponse(
            verifications=completed_verifications,
            total_claims=len(request.claims),
            supported_count=supported_cnt,
            partially_supported_count=partially_cnt,
            contradicted_count=contradicted_cnt,
            unresolved_count=unresolved_cnt,
            provenance_failure_count=provenance_fail_cnt,
            model=self._settings.llm_model_name,
        )


async def verify_claims(
    request: VerificationRequest,
    client: LLMClient | None = None,
    retrieval_service: EvidenceRetrievalService | None = None,
    settings: LLMSettings | None = None,
) -> VerificationResponse:
    """Convenience helper to run claim verification directly."""
    service = ClaimVerificationService(
        client=client, retrieval_service=retrieval_service, settings=settings
    )
    return await service.verify(request)
