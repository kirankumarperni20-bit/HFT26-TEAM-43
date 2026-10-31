"""
run_live_verification_review.py — Focused real-corpus validation for Phase 4.

Tests four claims individually against backend/data/test_paper.pdf:
1. "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German."
2. "The Transformer achieves 41.0 BLEU on WMT 2014 English-to-French."
3. "The Transformer uses multi-head attention."
4. "The Transformer uses recurrent neural networks."

Uses the actual shared LLMClient and ClaimVerificationService (no mocks).
Prints complete retrieved passages, quotes, statuses, explanations, and exact substring validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.client import LLMClient
from llm.schemas.claim import AtomicClaim, ClaimType
from llm.schemas.verification import VerificationRequest, VerificationResponse
from llm.services.retrieval_service import EvidenceRetrievalService
from llm.services.verification_service import ClaimVerificationService, normalize_text_for_quote_check

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("live_verification_review")


TARGET_CLAIMS = [
    {
        "claim_id": "claim_01_bleu_ende",
        "claim_text": "The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
        "claim_type": ClaimType.NUMERICAL,
        "expected_status": "SUPPORTED",
    },
    {
        "claim_id": "claim_02_bleu_enfr",
        "claim_text": "The Transformer achieves 41.0 BLEU on WMT 2014 English-to-French.",
        "claim_type": ClaimType.NUMERICAL,
        "expected_status": "SUPPORTED",
    },
    {
        "claim_id": "claim_03_multi_head_attention",
        "claim_text": "The Transformer uses multi-head attention.",
        "claim_type": ClaimType.METHODOLOGICAL,
        "expected_status": "SUPPORTED",
    },
    {
        "claim_id": "claim_04_recurrent_neural_networks",
        "claim_text": "The Transformer uses recurrent neural networks.",
        "claim_type": ClaimType.METHODOLOGICAL,
        "expected_status": "CONTRADICTED",
    },
]


async def run_review() -> dict:
    retrieval_service = EvidenceRetrievalService()
    llm_client = LLMClient()
    verification_service = ClaimVerificationService(
        client=llm_client, retrieval_service=retrieval_service
    )

    review_results = []

    print("=" * 80)
    print("PHASE 4 REVIEW: FOCUSED REAL-CORPUS CLAIM VERIFICATION")
    print("Corpus Source: backend/data/test_paper.pdf (Vaswani et al., 2017)")
    print("Synthetic Data Note: backend/data/convs2s_extraction.json is synthetic test data only.")
    print("Model: Featherless AI shared LLMClient (Live Evaluation)")
    print("=" * 80)

    for spec in TARGET_CLAIMS:
        claim_id = spec["claim_id"]
        claim_text = spec["claim_text"]
        claim_type = spec["claim_type"]

        claim_obj = AtomicClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_type=claim_type,
            source_text=claim_text,
            involved_papers=["Vaswani_2017_Attention"],
        )

        # 1. Retrieve candidate passages
        retrieved_passages = retrieval_service.retrieve(
            claim=claim_obj,
            allowed_paper_ids=["Vaswani_2017_Attention"],
            max_candidates=3,
        )

        print("\n" + "#" * 80)
        print(f"CLAIM: \"{claim_text}\"")
        print(f"Claim ID: {claim_id} | Type: {claim_type.value} | Expected: {spec['expected_status']}")
        print("-" * 80)
        print("RETRIEVED CANDIDATE PASSAGES (COMPLETE TEXT, UNTRUNCATED):")

        passages_dump = []
        for idx, p in enumerate(retrieved_passages, 1):
            print(f"\n  [Passage {idx}]")
            print(f"  Evidence ID:      {p.evidence_id}")
            print(f"  Paper ID:         {p.paper_id}")
            print(f"  Page Number:      {p.page}")
            print(f"  Chunk ID:         {p.chunk_id}")
            print(f"  Section:          {p.section}")
            print(f"  Retrieval Method: {p.retrieval_method}")
            print(f"  Retrieval Score:  {p.retrieval_score:.4f}")
            print("  Complete Passage Text:")
            print("  " + "-" * 40)
            for line in p.text.splitlines():
                print(f"    {line}")
            print("  " + "-" * 40)

            passages_dump.append({
                "evidence_id": p.evidence_id,
                "paper_id": p.paper_id,
                "page": p.page,
                "chunk_id": p.chunk_id,
                "section": p.section,
                "retrieval_method": p.retrieval_method,
                "retrieval_score": p.retrieval_score,
                "complete_text": p.text,
            })

        # 2. Run Live LLM Verification
        print("\nRUNNING LIVE LLM VERIFICATION SERVICE...")
        req = VerificationRequest(
            claims=[claim_obj],
            passages=retrieved_passages,
            allowed_paper_ids=["Vaswani_2017_Attention"],
        )
        resp: VerificationResponse = await verification_service.verify(req)

        verif = resp.verifications[0]

        print("-" * 80)
        print("VERIFICATION RESULTS:")
        print(f"  Status:             {verif.verification_status.value}")
        print(f"  Explanation:        {verif.explanation}")
        if verif.unresolved_reason:
            print(f"  Unresolved Reason:  {verif.unresolved_reason}")
        print(f"  Confidence:         {verif.confidence}")
        print(f"  Provenance Valid:   {verif.provenance_validated}")

        print("\n  CITATIONS & EXACT QUOTE VALIDATION:")
        evidence_citations = []
        citations_to_check = verif.supporting_evidence + verif.contradicting_evidence
        if not citations_to_check:
            print("    (No evidence passages cited)")

        for c_idx, cit in enumerate(citations_to_check, 1):
            # Locate corresponding passage in retrieved passages
            matching_passage = next((p for p in retrieved_passages if p.evidence_id == cit.evidence_id), None)
            exact_substring = False
            norm_substring = False
            if matching_passage:
                exact_substring = cit.quote in matching_passage.text
                norm_q = normalize_text_for_quote_check(cit.quote)
                norm_t = normalize_text_for_quote_check(matching_passage.text)
                norm_substring = norm_q in norm_t

            print(f"    [{c_idx}] Evidence ID: {cit.evidence_id} | Page: {cit.page} | Chunk: {cit.chunk_id} | Section: {cit.section}")
            print(f"        Selected Quote:           \"{cit.quote}\"")
            print(f"        Exact Substring of Text:  {exact_substring}")
            print(f"        Normalized Substring:     {norm_substring}")
            print(f"        Provenance is_verbatim:   {cit.is_verbatim}")
            if cit.validation_notes:
                print(f"        Validation Notes:         {cit.validation_notes}")

            evidence_citations.append({
                "evidence_id": cit.evidence_id,
                "paper_id": cit.paper_id,
                "page": cit.page,
                "chunk_id": cit.chunk_id,
                "section": cit.section,
                "quote": cit.quote,
                "exact_substring": exact_substring,
                "normalized_substring": norm_substring,
                "is_verbatim": cit.is_verbatim,
                "validation_notes": cit.validation_notes,
            })

        review_results.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_type": claim_type.value,
            "expected_status": spec["expected_status"],
            "actual_status": verif.verification_status.value,
            "explanation": verif.explanation,
            "confidence": verif.confidence,
            "provenance_validated": verif.provenance_validated,
            "retrieved_passages": passages_dump,
            "evidence_citations": evidence_citations,
        })

    out_file = BACKEND_DIR / "data" / "live_verification_review_results.json"
    out_file.write_text(json.dumps(review_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Complete review results saved to: {out_file}")

    return review_results


if __name__ == "__main__":
    asyncio.run(run_review())
