"""
test_verification.py — Comprehensive unit and integration tests for Phase 4: Evidence Retrieval and Claim Verification.

Covers:
1. Correctly supported claim.
2. Partially supported claim.
3. Contradicted claim.
4. Unresolved claim when no sufficient evidence exists.
5. Topically relevant but non-supporting passage (relevance != support).
6. Invalid evidence ID (rejected by deterministic provenance check).
7. Fabricated paper ID (rejected by provenance check).
8. Non-verbatim quote (flagged as non-verbatim / provenance failure).
9. Exact page and chunk provenance copied directly from source.
10. Multiple candidate passages evaluation.
11. Numerical claim with different experimental conditions (benchmark mismatch).
12. Parent statement context for pronoun-dependent atomic claims.
13. API endpoint integration: POST /api/research/verify and POST /llm/verify.
14. Invalid model JSON handling.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.llm.schemas.claim import AlignmentStatus, AtomicClaim, ClaimType
from backend.llm.schemas.verification import (
    ClaimVerification,
    RetrievedEvidence,
    VerificationRequest,
    VerificationResponse,
    VerificationStatus,
    VerifiedEvidenceItem,
)
from backend.llm.services.retrieval_service import EvidenceRetrievalService
from backend.llm.services.verification_service import ClaimVerificationService


class TestClaimVerification(unittest.TestCase):
    """Test suite for evidence retrieval, claim verification, and provenance auditing."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def _sample_passage(
        self,
        ev_id: str = "ev_001",
        paper_id: str = "Vaswani_2017_Attention",
        page: int = 8,
        section: str = "6.1 Machine Translation",
        text: str = "On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.",
    ) -> RetrievedEvidence:
        return RetrievedEvidence(
            evidence_id=ev_id,
            paper_id=paper_id,
            chunk_id="chunk_3",
            page=page,
            section=section,
            text=text,
            retrieval_method="lexical_overlap",
            retrieval_score=0.95,
        )

    def test_01_correctly_supported_claim(self) -> None:
        """Verify that a claim directly substantiated by evidence is classified as SUPPORTED."""
        claim = AtomicClaim(
            claim_id="c_001",
            claim_text="Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
            source_text="Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = self._sample_passage()

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_001",
                    "verification_status": "SUPPORTED",
                    "explanation": "The passage confirms the big Transformer achieves a state-of-the-art BLEU score of 28.4 on WMT 2014 English-to-German.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_001",
                            "quote": "establishing a new state-of-the-art BLEU score of 28.4.",
                        }
                    ],
                    "contradicting_evidence": [],
                    "confidence": 0.98,
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        req = VerificationRequest(claims=[claim], passages=[passage])
        res = asyncio.run(service.verify(req))

        self.assertEqual(res.total_claims, 1)
        self.assertEqual(res.supported_count, 1)
        verif = res.verifications[0]
        self.assertEqual(verif.verification_status, VerificationStatus.SUPPORTED)
        self.assertTrue(verif.provenance_validated)
        self.assertEqual(len(verif.supporting_evidence), 1)
        self.assertTrue(verif.supporting_evidence[0].is_verbatim)
        self.assertEqual(verif.supporting_evidence[0].page, 8)

    def test_02_partially_supported_claim(self) -> None:
        """Verify that a claim with only partial or weaker evidence is classified as PARTIALLY_SUPPORTED."""
        claim = AtomicClaim(
            claim_id="c_002",
            claim_text="Transformer trains in 12 hours on 8 GPUs with 100 million parameters.",
            source_text="Transformer trains in 12 hours on 8 GPUs with 100 million parameters.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = self._sample_passage(
            ev_id="ev_002",
            page=7,
            section="5.2 Hardware and Schedule",
            text="We trained the base models for 100,000 steps or 12 hours on 8 NVIDIA P100 GPUs.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_002",
                    "verification_status": "PARTIALLY_SUPPORTED",
                    "explanation": "The passage confirms training in 12 hours on 8 GPUs, but does not confirm 100 million parameters.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_002",
                            "quote": "12 hours on 8 NVIDIA P100 GPUs",
                        }
                    ],
                    "confidence": 0.85,
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        self.assertEqual(res.partially_supported_count, 1)
        self.assertEqual(res.verifications[0].verification_status, VerificationStatus.PARTIALLY_SUPPORTED)

    def test_03_contradicted_claim(self) -> None:
        """Verify that a claim refuted by empirical evidence is classified as CONTRADICTED."""
        claim = AtomicClaim(
            claim_id="c_003",
            claim_text="ConvS2S outperforms the Transformer with 35.0 BLEU on WMT 2014 En-De.",
            source_text="ConvS2S outperforms the Transformer with 35.0 BLEU on WMT 2014 En-De.",
            involved_papers=["Gehring_2017_ConvS2S"],
        )
        passage = self._sample_passage(
            ev_id="ev_003",
            paper_id="Gehring_2017_ConvS2S",
            page=6,
            section="Section 4",
            text="ConvS2S achieves 25.16 BLEU on WMT 2014 English-German, which is lower than the Transformer.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_003",
                    "verification_status": "CONTRADICTED",
                    "explanation": "ConvS2S achieves 25.16 BLEU, directly contradicting the claimed 35.0 BLEU score.",
                    "supporting_evidence": [],
                    "contradicting_evidence": [
                        {
                            "evidence_id": "ev_003",
                            "quote": "ConvS2S achieves 25.16 BLEU on WMT 2014 English-German",
                        }
                    ],
                    "confidence": 0.99,
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        self.assertEqual(res.contradicted_count, 1)
        verif = res.verifications[0]
        self.assertEqual(verif.verification_status, VerificationStatus.CONTRADICTED)
        self.assertEqual(len(verif.contradicting_evidence), 1)

    def test_04_unresolved_claim_when_insufficient_evidence(self) -> None:
        """Verify that when evidence is absent, claim is classified as UNRESOLVED with clear reason."""
        claim = AtomicClaim(
            claim_id="c_004",
            claim_text="The authors tested the Transformer on speech recognition with 98% accuracy.",
            source_text="The authors tested the Transformer on speech recognition with 98% accuracy.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = self._sample_passage(
            ev_id="ev_004",
            page=10,
            section="7 Conclusion",
            text="We plan to extend the Transformer to problems involving input and output modalities other than text, such as images, audio and video.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_004",
                    "verification_status": "UNRESOLVED",
                    "explanation": "The paper only mentions plans to test audio in the future; it does not report speech recognition experiments or 98% accuracy.",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "unresolved_reason": "No speech recognition accuracy experiments reported in the text.",
                    "confidence": 0.90,
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        self.assertEqual(res.unresolved_count, 1)
        self.assertEqual(res.verifications[0].verification_status, VerificationStatus.UNRESOLVED)
        self.assertIsNotNone(res.verifications[0].unresolved_reason)

    def test_05_topically_relevant_but_non_supporting_passage(self) -> None:
        """Verify that topical keyword overlap alone is not treated as evidence support."""
        claim = AtomicClaim(
            claim_id="c_005",
            claim_text="Transformer base model uses 16 parallel attention heads.",
            source_text="Transformer base model uses 16 parallel attention heads.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        # Passage mentions multi-head attention, but does not confirm 16 heads
        passage = self._sample_passage(
            ev_id="ev_005",
            page=4,
            section="3.2.2 Multi-Head Attention",
            text="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_005",
                    "verification_status": "UNRESOLVED",
                    "explanation": "The passage describes multi-head attention qualitatively but does not specify the number of heads (h=16).",
                    "supporting_evidence": [],
                    "unresolved_reason": "Topically relevant to attention, but head count is missing.",
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        self.assertEqual(res.unresolved_count, 1)
        self.assertEqual(res.verifications[0].verification_status, VerificationStatus.UNRESOLVED)

    def test_06_invalid_evidence_id_rejected_by_provenance_check(self) -> None:
        """Verify that citing an unretrieved evidence ID triggers a provenance failure."""
        claim = AtomicClaim(
            claim_id="c_006",
            claim_text="Transformer achieved 28.4 BLEU.",
            source_text="Transformer achieved 28.4 BLEU.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = self._sample_passage(ev_id="ev_001")

        # LLM invents an evidence_id "ev_hallucinated_999"
        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_006",
                    "verification_status": "SUPPORTED",
                    "explanation": "Supported by paper.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_hallucinated_999",
                            "quote": "score of 28.4",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        verif = res.verifications[0]
        # Must be downgraded to UNRESOLVED because supporting citation was invalid
        self.assertEqual(verif.verification_status, VerificationStatus.UNRESOLVED)
        self.assertFalse(verif.provenance_validated)
        self.assertEqual(len(verif.supporting_evidence), 0)
        self.assertEqual(res.provenance_failure_count, 1)

    def test_07_fabricated_paper_id_filtered(self) -> None:
        """Verify that evidence with paper IDs outside allowed_paper_ids is rejected."""
        claim = AtomicClaim(
            claim_id="c_007",
            claim_text="ConvS2S uses gated linear units.",
            source_text="ConvS2S uses gated linear units.",
            involved_papers=["Gehring_2017_ConvS2S"],
        )
        # Passage belongs to disallowed paper ID
        passage = self._sample_passage(
            ev_id="ev_007",
            paper_id="Unallowed_Paper_XYZ",
            text="Our architecture uses gated linear units.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_007",
                    "verification_status": "SUPPORTED",
                    "explanation": "Supported by passage.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_007",
                            "quote": "uses gated linear units",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        req = VerificationRequest(
            claims=[claim],
            passages=[passage],
            allowed_paper_ids=["Gehring_2017_ConvS2S"],  # Unallowed_Paper_XYZ not allowed
        )
        res = asyncio.run(service.verify(req))

        verif = res.verifications[0]
        self.assertEqual(verif.verification_status, VerificationStatus.UNRESOLVED)
        self.assertFalse(verif.provenance_validated)

    def test_08_non_verbatim_quote_flagged(self) -> None:
        """Verify that altered quotes not appearing in candidate passage are flagged as non-verbatim."""
        claim = AtomicClaim(
            claim_id="c_008",
            claim_text="Transformer achieved 28.4 BLEU.",
            source_text="Transformer achieved 28.4 BLEU.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = self._sample_passage(
            ev_id="ev_008",
            text="The big model achieves a score of 28.4 on English-to-German.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_008",
                    "verification_status": "SUPPORTED",
                    "explanation": "Supported.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_008",
                            "quote": "Completely altered quote that does not appear in text",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        verif = res.verifications[0]
        self.assertFalse(verif.provenance_validated)
        self.assertEqual(len(verif.supporting_evidence), 1)
        self.assertFalse(verif.supporting_evidence[0].is_verbatim)

    def test_09_exact_page_and_chunk_provenance_copied_from_source(self) -> None:
        """Verify that page, section, and chunk_id metadata are copied directly from candidate passages."""
        claim = AtomicClaim(
            claim_id="c_009",
            claim_text="Transformer achieved 28.4 BLEU.",
            source_text="Transformer achieved 28.4 BLEU.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = RetrievedEvidence(
            evidence_id="ev_009",
            paper_id="Vaswani_2017_Attention",
            chunk_id="chunk_actual_7",
            page=8,
            section="Section 5.1",
            text="Our model achieves 28.4 BLEU.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_009",
                    "verification_status": "SUPPORTED",
                    "explanation": "Matches text.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_009",
                            "quote": "achieves 28.4 BLEU",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        item = res.verifications[0].supporting_evidence[0]
        # Metadata must strictly match source passage
        self.assertEqual(item.page, 8)
        self.assertEqual(item.chunk_id, "chunk_actual_7")
        self.assertEqual(item.section, "Section 5.1")
        self.assertEqual(item.paper_id, "Vaswani_2017_Attention")

    def test_10_multiple_candidate_passages_selection(self) -> None:
        """Verify that the verifier correctly selects the supporting passage among multiple candidates."""
        claim = AtomicClaim(
            claim_id="c_010",
            claim_text="Transformer achieved 41.0 BLEU on English-to-French.",
            source_text="Transformer achieved 41.0 BLEU on English-to-French.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        p1 = self._sample_passage(ev_id="ev_irrelevant", text="Self-attention has O(1) sequential operations.")
        p2 = self._sample_passage(ev_id="ev_german", text="Achieves 28.4 BLEU on English-to-German.")
        p3 = self._sample_passage(ev_id="ev_french", text="On the English-to-French task, our model establishes 41.0 BLEU.")

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_010",
                    "verification_status": "SUPPORTED",
                    "explanation": "ev_french substantiates 41.0 BLEU on English-to-French.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_french",
                            "quote": "our model establishes 41.0 BLEU.",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[p1, p2, p3])))

        verif = res.verifications[0]
        self.assertEqual(verif.verification_status, VerificationStatus.SUPPORTED)
        self.assertEqual(verif.supporting_evidence[0].evidence_id, "ev_french")

    def test_11_numerical_claim_with_different_conditions(self) -> None:
        """Verify that matching numbers on different benchmarks/tasks are not falsely accepted."""
        claim = AtomicClaim(
            claim_id="c_011",
            claim_text="Transformer achieves 41.0 BLEU on English-to-German.",
            source_text="Transformer achieves 41.0 BLEU on English-to-German.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        # 41.0 BLEU is on French, not German
        passage = self._sample_passage(
            ev_id="ev_french_only",
            text="Our model establishes a state-of-the-art score of 41.0 on English-to-French.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_011",
                    "verification_status": "CONTRADICTED",
                    "explanation": "The score of 41.0 was achieved on English-to-French, not English-to-German (which scored 28.4).",
                    "contradicting_evidence": [
                        {
                            "evidence_id": "ev_french_only",
                            "quote": "41.0 on English-to-French",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        self.assertEqual(res.contradicted_count, 1)

    def test_12_parent_statement_context_for_pronoun_dependent_claim(self) -> None:
        """Verify that pronoun-dependent propositions use parent context to identify the subject."""
        claim = AtomicClaim(
            claim_id="c_012",
            claim_text="uses self-attention to parallelize sequence processing",
            source_text="Transformer achieves 28.4 BLEU and uses self-attention to parallelize sequence processing.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        passage = self._sample_passage(
            ev_id="ev_012",
            text="The Transformer relies entirely on self-attention to draw global dependencies and enable parallelization.",
        )

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_012",
                    "verification_status": "SUPPORTED",
                    "explanation": "Using parent context 'Transformer', passage confirms the model uses self-attention to enable parallelization.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_012",
                            "quote": "relies entirely on self-attention to draw global dependencies and enable parallelization",
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimVerificationService(client=mock_client)
        res = asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))

        self.assertEqual(res.supported_count, 1)

    def test_13_api_endpoint_integration(self) -> None:
        """Verify FastAPI endpoints POST /api/research/verify and POST /llm/verify."""
        claim_dict = {
            "claim_id": "c_api_01",
            "claim_text": "Transformer achieves 28.4 BLEU.",
            "source_text": "Transformer achieves 28.4 BLEU.",
            "involved_papers": ["Vaswani_2017_Attention"],
        }
        passage_dict = {
            "evidence_id": "ev_api_01",
            "paper_id": "Vaswani_2017_Attention",
            "text": "Transformer achieves 28.4 BLEU.",
            "page": 8,
            "section": "6.1",
        }

        mock_llm_json = json.dumps({
            "verifications": [
                {
                    "claim_id": "c_api_01",
                    "verification_status": "SUPPORTED",
                    "explanation": "Direct match.",
                    "supporting_evidence": [
                        {
                            "evidence_id": "ev_api_01",
                            "quote": "Transformer achieves 28.4 BLEU.",
                        }
                    ],
                }
            ]
        })

        with patch("backend.llm.client.LLMClient.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_llm_json

            # 1. Test POST /api/research/verify
            resp1 = self.client.post(
                "/api/research/verify",
                json={"claims": [claim_dict], "passages": [passage_dict]},
            )
            self.assertEqual(resp1.status_code, 200, resp1.text)
            data1 = resp1.json()
            self.assertEqual(data1["total_claims"], 1)
            self.assertEqual(data1["supported_count"], 1)
            self.assertEqual(data1["verifications"][0]["verification_status"], "SUPPORTED")

            # 2. Test POST /llm/verify
            resp2 = self.client.post(
                "/llm/verify",
                json={"claims": [claim_dict], "passages": [passage_dict]},
            )
            self.assertEqual(resp2.status_code, 200, resp2.text)

            # 3. Test 422 for empty claims
            resp_err = self.client.post("/api/research/verify", json={"claims": []})
            self.assertEqual(resp_err.status_code, 422)

    def test_14_invalid_model_json_handling(self) -> None:
        """Verify that malformed or non-JSON output from the LLM raises ValueError cleanly."""
        claim = AtomicClaim(
            claim_id="c_014",
            claim_text="Test claim",
            source_text="Test claim",
        )
        passage = self._sample_passage()

        mock_client = AsyncMock()
        mock_client.generate.return_value = "Broken response: <not_json>"

        service = ClaimVerificationService(client=mock_client)
        with self.assertRaises(ValueError):
            asyncio.run(service.verify(VerificationRequest(claims=[claim], passages=[passage])))


if __name__ == "__main__":
    unittest.main()
