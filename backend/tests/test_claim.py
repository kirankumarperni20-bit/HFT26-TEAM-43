"""
test_claim.py — Comprehensive unit and integration tests for Phase 3: Atomic Claim Decomposition.

Covers:
1. Compound sentence splits into multiple atomic claims.
2. Already atomic sentence remains one claim.
3. Numerical values (metrics, parameters) are preserved exactly.
4. Exact Unicode character offsets: source_text[start:end] == claim_text.
5. Altered or paraphrased claims are flagged (not marked exact).
6. Duplicate substring text in source statement handled safely.
7. Unknown / hallucinated paper IDs are rejected / filtered out.
8. Source references inheritance and provenance verification.
9. Invalid model JSON handling (safe failure).
10. Missing / empty source statements handling.
11. FastAPI endpoints: POST /api/research/decompose and POST /llm/decompose.
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
from backend.llm.schemas.claim import (
    AlignmentStatus,
    AtomicClaim,
    ClaimDecompositionRequest,
    ClaimDecompositionResponse,
    ClaimType,
    StatementInputItem,
)
from backend.llm.schemas.synthesis import (
    ComparisonType,
    CrossPaperSynthesis,
    FindingComparison,
    FindingRelationship,
    MethodologicalComparison,
    SourceReference,
)
from backend.llm.services.claim_service import (
    ClaimDecompositionService,
    compute_deterministic_offsets,
    extract_statements_from_synthesis,
)


class TestClaimDecomposition(unittest.TestCase):
    """Test suite for atomic claim decomposition."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_01_compound_sentence_splits_into_multiple_claims(self) -> None:
        """Verify that a multi-clause compound statement splits into multiple atomic claims."""
        compound_text = (
            "Transformer achieved 28.4 BLEU on WMT 2014 English-to-German and "
            "uses self-attention to parallelize sequence processing."
        )
        stmt = StatementInputItem(
            statement_id="stmt_001",
            text=compound_text,
            involved_papers=["Vaswani_2017_Attention"],
        )

        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_001",
                    "atomic_claims": [
                        {
                            "claim_text": "Transformer achieved 28.4 BLEU",
                            "claim_type": "NUMERICAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        },
                        {
                            "claim_text": "WMT 2014 English-to-German",
                            "claim_type": "FACTUAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        },
                        {
                            "claim_text": "uses self-attention to parallelize sequence processing",
                            "claim_type": "METHODOLOGICAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        },
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        req = ClaimDecompositionRequest(statements=[stmt])
        res = asyncio.run(service.decompose(req))

        self.assertEqual(res.source_statement_count, 1)
        self.assertEqual(res.total_claims, 3)
        self.assertEqual(res.exact_alignment_count, 3)
        self.assertEqual(res.flagged_claims_count, 0)

        # Check each claim has exact offsets
        for claim in res.claims:
            self.assertEqual(claim.alignment_status, AlignmentStatus.EXACT)
            self.assertTrue(claim.is_exact_substring)
            self.assertIsNotNone(claim.start_offset)
            self.assertIsNotNone(claim.end_offset)
            self.assertEqual(
                compound_text[claim.start_offset : claim.end_offset],
                claim.claim_text,
            )

    def test_02_already_atomic_sentence_remains_one_claim(self) -> None:
        """Verify that an atomic sentence is not fragmented into meaningless words."""
        atomic_text = "Vaswani et al. evaluate on WMT 2014 English-to-French."
        stmt = StatementInputItem(
            statement_id="stmt_002",
            text=atomic_text,
            involved_papers=["Vaswani_2017_Attention"],
        )

        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_002",
                    "atomic_claims": [
                        {
                            "claim_text": "Vaswani et al. evaluate on WMT 2014 English-to-French.",
                            "claim_type": "FACTUAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        }
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        req = ClaimDecompositionRequest(statements=[stmt])
        res = asyncio.run(service.decompose(req))

        self.assertEqual(res.total_claims, 1)
        claim = res.claims[0]
        self.assertEqual(claim.claim_text, atomic_text)
        self.assertEqual(claim.start_offset, 0)
        self.assertEqual(claim.end_offset, len(atomic_text))
        self.assertEqual(claim.alignment_status, AlignmentStatus.EXACT)

    def test_03_numerical_values_preserved_exactly(self) -> None:
        """Verify that numerical metrics and decimal points (28.4, 25.16) are preserved."""
        text = "Transformer scored 28.4 BLEU while ConvS2S scored 25.16 BLEU."
        stmt = StatementInputItem(
            statement_id="stmt_003",
            text=text,
            involved_papers=["Vaswani_2017_Attention", "Gehring_2017_ConvS2S"],
        )

        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_003",
                    "atomic_claims": [
                        {
                            "claim_text": "Transformer scored 28.4 BLEU",
                            "claim_type": "NUMERICAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        },
                        {
                            "claim_text": "ConvS2S scored 25.16 BLEU",
                            "claim_type": "NUMERICAL",
                            "involved_papers": ["Gehring_2017_ConvS2S"],
                        },
                    ],
                }
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        req = ClaimDecompositionRequest(statements=[stmt])
        res = asyncio.run(service.decompose(req))

        self.assertEqual(res.total_claims, 2)
        self.assertIn("28.4", res.claims[0].claim_text)
        self.assertIn("25.16", res.claims[1].claim_text)

    def test_04_exact_unicode_offsets_and_indexing(self) -> None:
        """Verify Unicode character offset accuracy with non-ASCII mathematical notation."""
        source_text = "The attention mechanism divides scores by √dk and applies a softmax."
        claim_text = "divides scores by √dk"

        start, end, status, _ = compute_deterministic_offsets(claim_text, source_text)
        self.assertEqual(status, AlignmentStatus.EXACT)
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        # Check Python Unicode slicing
        self.assertEqual(source_text[start:end], claim_text)

    def test_05_paraphrased_or_altered_claim_flagged(self) -> None:
        """Verify that when a model paraphrases text, it is flagged as PARAPHRASED with None offsets."""
        source_text = "Transformer achieves state-of-the-art results on English-to-German translation."
        paraphrased_claim = "The Transformer reached top scores in German translation."

        start, end, status, notes = compute_deterministic_offsets(paraphrased_claim, source_text)
        self.assertEqual(status, AlignmentStatus.PARAPHRASED)
        self.assertIsNone(start)
        self.assertIsNone(end)
        self.assertIn("paraphrased", notes.lower())

        # Test end-to-end service handling of paraphrased claim
        stmt = StatementInputItem(statement_id="stmt_005", text=source_text)
        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_005",
                    "atomic_claims": [
                        {
                            "claim_text": paraphrased_claim,
                            "claim_type": "FACTUAL",
                        }
                    ],
                }
            ]
        })
        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        res = asyncio.run(service.decompose(ClaimDecompositionRequest(statements=[stmt])))

        self.assertEqual(res.total_claims, 1)
        self.assertEqual(res.exact_alignment_count, 0)
        self.assertEqual(res.flagged_claims_count, 1)
        self.assertFalse(res.claims[0].is_exact_substring)
        self.assertEqual(res.claims[0].alignment_status, AlignmentStatus.PARAPHRASED)
        self.assertIsNone(res.claims[0].start_offset)
        self.assertIsNone(res.claims[0].end_offset)

    def test_06_duplicate_substring_text_handled_safely(self) -> None:
        """Verify that when a term appears multiple times, monotonic alignment resolves it."""
        source_text = "First self-attention layer computes keys, and second self-attention layer computes values."
        # First occurrence
        start1, end1, status1, _ = compute_deterministic_offsets("self-attention layer", source_text, 0)
        self.assertEqual(status1, AlignmentStatus.EXACT)
        self.assertEqual(source_text[start1:end1], "self-attention layer")

        # Second occurrence using end1 as search start
        start2, end2, status2, _ = compute_deterministic_offsets("self-attention layer", source_text, end1)
        self.assertEqual(status2, AlignmentStatus.EXACT)
        self.assertGreater(start2, start1)
        self.assertEqual(source_text[start2:end2], "self-attention layer")

    def test_07_unknown_hallucinated_paper_ids_filtered(self) -> None:
        """Verify that paper IDs not belonging to the statement are rejected."""
        stmt = StatementInputItem(
            statement_id="stmt_007",
            text="The model uses multi-head attention.",
            involved_papers=["Vaswani_2017_Attention"],
        )
        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_007",
                    "atomic_claims": [
                        {
                            "claim_text": "The model uses multi-head attention.",
                            "claim_type": "METHODOLOGICAL",
                            "involved_papers": ["Hallucinated_Paper_999", "Vaswani_2017_Attention"],
                        }
                    ],
                }
            ]
        })
        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        res = asyncio.run(service.decompose(ClaimDecompositionRequest(statements=[stmt])))

        self.assertEqual(res.total_claims, 1)
        # Hallucinated ID must be filtered out
        self.assertEqual(res.claims[0].involved_papers, ["Vaswani_2017_Attention"])

    def test_08_source_references_inherited(self) -> None:
        """Verify that citations attached to parent statements are inherited by atomic claims."""
        ref = SourceReference(paper_id="Vaswani_2017_Attention", section="Section 3.2", quote="we found it beneficial")
        stmt = StatementInputItem(
            statement_id="stmt_008",
            text="Multi-head attention projects queries h times.",
            involved_papers=["Vaswani_2017_Attention"],
            source_references=[ref],
        )
        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_008",
                    "atomic_claims": [
                        {
                            "claim_text": "Multi-head attention projects queries h times.",
                            "claim_type": "METHODOLOGICAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        }
                    ],
                }
            ]
        })
        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        res = asyncio.run(service.decompose(ClaimDecompositionRequest(statements=[stmt])))

        self.assertEqual(len(res.claims[0].source_references), 1)
        self.assertEqual(res.claims[0].source_references[0].section, "Section 3.2")
        self.assertEqual(res.claims[0].source_references[0].paper_id, "Vaswani_2017_Attention")

    def test_09_invalid_model_json_handling(self) -> None:
        """Verify that malformed or non-JSON model output raises ValueError cleanly."""
        stmt = StatementInputItem(statement_id="stmt_009", text="Some test text.")
        mock_client = AsyncMock()
        mock_client.generate.return_value = "Broken output: <xml>not json</xml>"

        service = ClaimDecompositionService(client=mock_client)
        with self.assertRaises(ValueError):
            asyncio.run(service.decompose(ClaimDecompositionRequest(statements=[stmt])))

    def test_10_missing_and_empty_statements_handling(self) -> None:
        """Verify that decomposing an empty list returns 0 claims without error."""
        service = ClaimDecompositionService()
        req = ClaimDecompositionRequest(statements=[])
        res = asyncio.run(service.decompose(req))
        self.assertEqual(res.total_claims, 0)
        self.assertEqual(res.source_statement_count, 0)

        # Test request validation failure when neither is provided
        with self.assertRaises(ValueError):
            ClaimDecompositionRequest()

    def test_11_fastapi_endpoints_integration(self) -> None:
        """Verify FastAPI endpoints POST /api/research/decompose and POST /llm/decompose."""
        stmt = {
            "statement_id": "stmt_api",
            "text": "Transformer achieves 28.4 BLEU.",
            "involved_papers": ["Vaswani_2017_Attention"],
            "source_references": [],
        }

        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_api",
                    "atomic_claims": [
                        {
                            "claim_text": "Transformer achieves 28.4 BLEU.",
                            "claim_type": "NUMERICAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        }
                    ],
                }
            ]
        })

        with patch("backend.llm.client.LLMClient.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_llm_json

            # 1. Test POST /api/research/decompose
            resp1 = self.client.post("/api/research/decompose", json={"statements": [stmt]})
            self.assertEqual(resp1.status_code, 200, resp1.text)
            data1 = resp1.json()
            self.assertEqual(data1["total_claims"], 1)
            self.assertEqual(data1["claims"][0]["claim_text"], "Transformer achieves 28.4 BLEU.")
            self.assertEqual(data1["claims"][0]["start_offset"], 0)
            self.assertEqual(data1["claims"][0]["end_offset"], 31)

            # 2. Test POST /llm/decompose
            resp2 = self.client.post("/llm/decompose", json={"statements": [stmt]})
            self.assertEqual(resp2.status_code, 200, resp2.text)
            data2 = resp2.json()
            self.assertEqual(data2["total_claims"], 1)

            # 3. Test 422 for empty payload
            resp_err = self.client.post("/api/research/decompose", json={})
            self.assertEqual(resp_err.status_code, 422)

    def test_12_decomposition_from_crosspapersynthesis(self) -> None:
        """Verify full CrossPaperSynthesis decomposition into atomic claims with citation inheritance."""
        synthesis = CrossPaperSynthesis(
            overview="Transformer relies on self-attention.",
            papers_analyzed=[{"paper_id": "Vaswani_2017_Attention", "paper_title": "Attention Is All You Need"}],
            methodological_similarities=[
                MethodologicalComparison(
                    aspect="Attention Mechanism",
                    comparison_type=ComparisonType.SIMILARITY,
                    description="Both models utilize an attention mechanism.",
                    paper_details={"Vaswani_2017_Attention": "Self-attention"},
                    evidence=[
                        SourceReference(
                            paper_id="Vaswani_2017_Attention",
                            section="Section 3.2",
                            quote="attention mechanism",
                        )
                    ],
                )
            ],
        )

        mock_llm_json = json.dumps({
            "decompositions": [
                {
                    "statement_id": "stmt_001",
                    "atomic_claims": [
                        {
                            "claim_text": "Transformer relies on self-attention.",
                            "claim_type": "METHODOLOGICAL",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        }
                    ],
                },
                {
                    "statement_id": "stmt_002",
                    "atomic_claims": [
                        {
                            "claim_text": "Both models utilize an attention mechanism.",
                            "claim_type": "COMPARATIVE",
                            "involved_papers": ["Vaswani_2017_Attention"],
                        }
                    ],
                },
            ]
        })

        mock_client = AsyncMock()
        mock_client.generate.return_value = mock_llm_json

        service = ClaimDecompositionService(client=mock_client)
        req = ClaimDecompositionRequest(synthesis=synthesis)
        res = asyncio.run(service.decompose(req))

        self.assertEqual(res.source_statement_count, 2)
        self.assertEqual(res.total_claims, 2)
        self.assertEqual(res.exact_alignment_count, 2)

        # Check statement 2 inherited citation
        claim2 = res.claims[1]
        self.assertEqual(claim2.statement_id, "stmt_002")
        self.assertEqual(len(claim2.source_references), 1)
        self.assertEqual(claim2.source_references[0].section, "Section 3.2")
        self.assertEqual(claim2.source_references[0].paper_id, "Vaswani_2017_Attention")


if __name__ == "__main__":
    unittest.main()
