"""
verification_service.py — Claim verification and evidence provenance service.

Evaluates atomic research claims against candidate document evidence:
1. Ingests atomic claims and candidate passages (or retrieves them via EvidenceRetrievalService).
2. Augments claim evaluation with parent statement context for pronoun resolution.
3. Invokes shared LLMClient with the verification prompt template, or executes deterministic
   semantic and numerical alignment when LLM is unavailable.
4. Produces verifiable outcomes:
   - SUPPORTED: The available evidence sufficiently supports the claim.
   - CONTRADICTED: The available evidence directly conflicts with the claim.
   - INSUFFICIENT EVIDENCE (UNRESOLVED): The available evidence does not establish whether the claim is true.
   - VERIFICATION FAILED: Verification could not be completed because of a technical error.
5. Performs strict Python-side deterministic provenance validation:
   - Verifies all cited evidence IDs exist in candidate passage set.
   - Verifies paper IDs match allowed / involved papers.
   - Verifies all quotes are exact substrings of their candidate passages.
   - Directly copies page and chunk metadata from source passages (never trusts model).
6. Aggregates status counts and returns a complete audit certificate.
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
    from backend.llm.services.retrieval_service import EvidenceRetrievalService, STOPWORDS
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
    from llm.services.retrieval_service import EvidenceRetrievalService, STOPWORDS

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
        if client is not None:
            self._client = client
        else:
            try:
                self._client = LLMClient(settings=self._settings)
            except Exception as exc:
                logger.info(
                    "LLMClient initialization deferred/unavailable (%s). Deterministic fallback verification active.",
                    exc,
                )
                self._client = None
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
                insufficient_evidence_count=0,
                failed_count=0,
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
                insufficient_evidence_count=len(immediate_unresolved),
                failed_count=0,
                provenance_failure_count=0,
                model=self._settings.llm_model_name,
            )

        # 2. Invoke LLM if available, otherwise execute deterministic semantic evaluation
        verifications_raw: list[dict[str, Any]] = []
        llm_succeeded = False

        if self._client is not None:
            try:
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
                parsed_dict = clean_and_parse_json(raw_response)
                verifications_raw = parsed_dict.get("verifications", [])
                if not isinstance(verifications_raw, list):
                    raise ValueError(
                        f"Model response 'verifications' field must be a list. Received: {type(verifications_raw)}"
                    )
            except ValueError:
                raise
            except Exception as exc:
                logger.warning(
                    "LLM provider error (%s). Falling back to deterministic evidence evaluation.",
                    exc,
                )
                verifications_raw = self._deterministic_eval(
                    claims=claims_needing_eval,
                    claim_candidate_map=claim_candidate_map,
                    allowed_paper_ids=request.allowed_paper_ids,
                )
        else:
            logger.info("LLMClient is inactive. Executing deterministic semantic evaluation fallback.")
            verifications_raw = self._deterministic_eval(
                claims=claims_needing_eval,
                claim_candidate_map=claim_candidate_map,
                allowed_paper_ids=request.allowed_paper_ids,
            )

        # 3. Deterministic provenance and quote verification
        completed_verifications: list[ClaimVerification] = list(immediate_unresolved)
        supported_cnt = 0
        partially_cnt = 0
        contradicted_cnt = 0
        unresolved_cnt = len(immediate_unresolved)
        insufficient_cnt = len(immediate_unresolved)
        failed_cnt = 0
        provenance_fail_cnt = 0

        for item in verifications_raw:
            cid = item.get("claim_id")
            if cid not in claim_map:
                logger.warning("Model returned unknown claim_id: %s. Skipping.", cid)
                continue

            claim = claim_map[cid]
            candidates_dict = claim_candidate_map.get(cid, {})

            # Parse verification status
            raw_status_str = str(item.get("verification_status", "UNRESOLVED")).strip().upper()
            clean_status = raw_status_str.replace("-", "_").replace(" ", "_")

            if clean_status in ("INSUFFICIENT_EVIDENCE", "UNRESOLVED"):
                if "INSUFFICIENT" in raw_status_str:
                    status = VerificationStatus.INSUFFICIENT_EVIDENCE
                else:
                    status = VerificationStatus.UNRESOLVED
            elif clean_status in ("VERIFICATION_FAILED", "FAILED"):
                status = VerificationStatus.VERIFICATION_FAILED
            elif clean_status == "CONTRADICTED":
                status = VerificationStatus.CONTRADICTED
            elif clean_status == "PARTIALLY_SUPPORTED":
                status = VerificationStatus.PARTIALLY_SUPPORTED
            elif clean_status == "SUPPORTED":
                status = VerificationStatus.SUPPORTED
            else:
                try:
                    status = VerificationStatus(raw_status_str)
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
            elif status in (VerificationStatus.UNRESOLVED, VerificationStatus.INSUFFICIENT_EVIDENCE):
                unresolved_cnt += 1
                insufficient_cnt += 1
            elif status == VerificationStatus.VERIFICATION_FAILED:
                failed_cnt += 1
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
            insufficient_evidence_count=insufficient_cnt,
            failed_count=failed_cnt,
            provenance_failure_count=provenance_fail_cnt,
            model=self._settings.llm_model_name,
        )

    def _deterministic_eval(
        self,
        claims: list[AtomicClaim],
        claim_candidate_map: dict[str, dict[str, RetrievedEvidence]],
        allowed_paper_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        """
        Deterministic evidence evaluation fallback when LLM is unavailable or offline.
        Evaluates lexical overlap, numerical assertions, and contradiction indicators
        to classify claims into SUPPORTED, CONTRADICTED, or INSUFFICIENT EVIDENCE.
        """
        results: list[dict[str, Any]] = []

        for claim in claims:
            cid = claim.claim_id
            candidates = list(claim_candidate_map.get(cid, {}).values())
            if not candidates:
                results.append({
                    "claim_id": cid,
                    "verification_status": "INSUFFICIENT EVIDENCE",
                    "explanation": "No candidate evidence passages found in the source documents.",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "unresolved_reason": "Insufficient evidence: 0 matching passages retrieved.",
                    "confidence": 1.0,
                })
                continue

            claim_text = claim.claim_text.strip()
            clean_claim = re.sub(r"[^\w\s\.-]", " ", claim_text.lower())
            claim_words = [w for w in clean_claim.split() if w not in STOPWORDS and len(w) > 2]
            claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_text))

            best_match: tuple[str, float, RetrievedEvidence | None, str] = ("INSUFFICIENT EVIDENCE", 0.0, None, "")
            contradiction_found = False
            contradiction_cand = None
            contradiction_quote = ""
            contradiction_reason = ""

            for cand in candidates:
                cand_text = cand.text
                cand_lower = cand_text.lower()

                # Split candidate passage into distinct sentences
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cand_text) if len(s.strip()) > 8]
                if not sentences:
                    sentences = [cand_text.strip()]

                for sent in sentences:
                    sent_clean = re.sub(r"[^\w\s\.-]", " ", sent.lower())
                    sent_words = set(sent_clean.split())
                    sent_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", sent))

                    overlap = 0.0
                    if claim_words:
                        overlap = sum(1 for w in claim_words if w in sent_words) / len(claim_words)

                    # 1. Contradiction Detection:
                    # Check if sentence shares topic/domain keywords but reports conflicting numbers
                    # or contradictory polarity
                    if overlap >= 0.25 and claim_numbers:
                        conflicting_nums = sent_numbers - claim_numbers
                        neg_cues = ["lower than", "inferior", "worse than", "fails to", "cannot", "does not", "contradict", "below", "outperformed by"]
                        has_neg = any(cue in sent.lower() for cue in neg_cues)
                        has_metric_overlap = any(m in claim_text.lower() for m in ["bleu", "accuracy", "f1", "%", "score", "rate"]) and any(m in sent.lower() for m in ["bleu", "accuracy", "f1", "%", "score", "rate"])

                        if (conflicting_nums and (has_neg or has_metric_overlap)):
                            contradiction_found = True
                            contradiction_cand = cand
                            contradiction_quote = sent
                            contradiction_reason = f"The source text reports '{sent}', directly conflicting with the claim."
                            break

                    # 2. Support Detection:
                    # If all claimed numbers are present (or claim has no numbers) and word overlap is high
                    if (not claim_numbers or claim_numbers.issubset(sent_numbers)) and overlap >= 0.40:
                        if overlap > best_match[1]:
                            best_match = ("SUPPORTED", overlap, cand, sent)
                    # 3. Partial Support:
                    elif claim_numbers and claim_numbers.intersection(sent_numbers) and overlap >= 0.30:
                        if best_match[0] != "SUPPORTED" and overlap > best_match[1]:
                            best_match = ("PARTIALLY_SUPPORTED", overlap, cand, sent)

                if contradiction_found:
                    break

            if contradiction_found and contradiction_cand:
                results.append({
                    "claim_id": cid,
                    "verification_status": "CONTRADICTED",
                    "explanation": contradiction_reason,
                    "supporting_evidence": [],
                    "contradicting_evidence": [{
                        "evidence_id": contradiction_cand.evidence_id,
                        "quote": contradiction_quote,
                    }],
                    "confidence": 0.95,
                })
            elif best_match[0] == "SUPPORTED" and best_match[2]:
                cand = best_match[2]
                results.append({
                    "claim_id": cid,
                    "verification_status": "SUPPORTED",
                    "explanation": f"The available evidence in the paper supports the claim: '{best_match[3]}'",
                    "supporting_evidence": [{
                        "evidence_id": cand.evidence_id,
                        "quote": best_match[3],
                    }],
                    "contradicting_evidence": [],
                    "confidence": round(min(0.99, max(0.80, best_match[1])), 2),
                })
            elif best_match[0] == "PARTIALLY_SUPPORTED" and best_match[2]:
                cand = best_match[2]
                results.append({
                    "claim_id": cid,
                    "verification_status": "PARTIALLY_SUPPORTED",
                    "explanation": f"The source document confirms part of the claim: '{best_match[3]}', but does not substantiate all assertions.",
                    "supporting_evidence": [{
                        "evidence_id": cand.evidence_id,
                        "quote": best_match[3],
                    }],
                    "contradicting_evidence": [],
                    "confidence": round(min(0.85, max(0.65, best_match[1])), 2),
                })
            else:
                results.append({
                    "claim_id": cid,
                    "verification_status": "INSUFFICIENT EVIDENCE",
                    "explanation": "The available evidence does not establish whether the claim is true.",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "unresolved_reason": "No direct experimental or factual evidence in the retrieved passages confirms or refutes this claim.",
                    "confidence": 0.85,
                })

        return results


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
