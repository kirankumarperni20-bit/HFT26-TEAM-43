"""
test_retrieval_adapter.py — Comprehensive unit tests for the EvidenceRetrievalService adapter.

Tests all required integration requirements:
1. Successful conversion of AtomicClaim to retrieval Claim.
2. Successful conversion of EvidenceItem to RetrievedEvidence.
3. Filtering using involved_papers.
4. Filtering using allowed_paper_ids when involved_papers is empty.
5. Empty paper IDs return no evidence without querying database.
6. No synthetic evidence enters production retrieval.
7. Database retrieval errors do not silently trigger synthetic fallback.
8. Existing candidate_pool test behavior remains intact.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.llm.schemas.claim import AtomicClaim
from backend.llm.schemas.verification import RetrievedEvidence
from backend.llm.services.retrieval_service import (
    EvidenceRetrievalService,
    convert_atomic_claim_to_retrieval_claim,
    convert_evidence_item_to_retrieved_evidence,
    resolve_target_paper_ids,
)
from backend.retrieval.retriever import EvidenceRetriever, EvidenceRetrieverError
from backend.schemas.claim import Claim as RetrievalClaim
from backend.schemas.evidence import ClaimEvidenceResponse, EvidenceItem


class TestRetrievalAdapter(unittest.TestCase):
    """Test suite validating the EvidenceRetrievalService adapter integration."""

    def test_01_conversion_atomic_claim_to_retrieval_claim(self) -> None:
        """1. Successful conversion of AtomicClaim to retrieval Claim schema."""
        atomic = AtomicClaim(
            claim_id="C_101",
            claim_text="The Transformer employs multi-head self-attention mechanisms.",
            source_text="The Transformer employs multi-head self-attention mechanisms.",
            involved_papers=["Vaswani_2017_Attention"],
        )

        retrieval_claim = convert_atomic_claim_to_retrieval_claim(
            claim=atomic,
            target_paper_ids=["Vaswani_2017_Attention"],
        )

        self.assertIsInstance(retrieval_claim, RetrievalClaim)
        self.assertEqual(retrieval_claim.claim_id, "C_101")
        self.assertEqual(retrieval_claim.claim_text, atomic.claim_text)
        self.assertEqual(retrieval_claim.paper_ids, ["Vaswani_2017_Attention"])

        # Validate that constructing with empty target_paper_ids raises ValueError
        with self.assertRaises(ValueError):
            convert_atomic_claim_to_retrieval_claim(atomic, [])

    def test_02_conversion_evidence_item_to_retrieved_evidence(self) -> None:
        """2. Successful conversion of EvidenceItem to RetrievedEvidence schema."""
        item = EvidenceItem(
            evidence_id="chunk_v_042",
            chunk_id="chunk_v_042",
            paper_id="Vaswani_2017_Attention",
            page_number=3,
            paragraph_number=2,
            text="Multi-head attention allows the model to jointly attend to information...",
            retrieval_score=0.8875,
            retrieval_method="hybrid",
            section="3.2 Attention",
        )

        retrieved = convert_evidence_item_to_retrieved_evidence(item)

        self.assertIsInstance(retrieved, RetrievedEvidence)
        self.assertEqual(retrieved.evidence_id, "chunk_v_042")
        self.assertEqual(retrieved.chunk_id, "chunk_v_042")
        self.assertEqual(retrieved.paper_id, "Vaswani_2017_Attention")
        self.assertEqual(retrieved.page, 3)
        self.assertEqual(retrieved.text, item.text)
        self.assertEqual(retrieved.retrieval_score, 0.8875)
        self.assertEqual(retrieved.retrieval_method, "hybrid")
        self.assertEqual(retrieved.section, "3.2 Attention")

    def test_03_filtering_using_involved_papers(self) -> None:
        """3. Filtering using involved_papers intersected with allowed_paper_ids."""
        claim = AtomicClaim(
            claim_id="C_103",
            claim_text="Attention mechanism removes recurrence entirely.",
            source_text="Attention mechanism removes recurrence entirely.",
            involved_papers=["Vaswani_2017_Attention", "Excluded_Paper_XYZ"],
        )

        mock_retriever = MagicMock(spec=EvidenceRetriever)
        mock_retriever.retrieve_evidence.return_value = ClaimEvidenceResponse(
            claim_id="C_103",
            evidence=[
                EvidenceItem(
                    evidence_id="chunk_01",
                    chunk_id="chunk_01",
                    paper_id="Vaswani_2017_Attention",
                    page_number=1,
                    paragraph_number=1,
                    text="We propose the Transformer.",
                    retrieval_score=0.9,
                    retrieval_method="hybrid",
                )
            ],
        )

        service = EvidenceRetrievalService(retriever=mock_retriever)
        results = service.retrieve(
            claim=claim,
            allowed_paper_ids=["Vaswani_2017_Attention"],  # Excluded_Paper_XYZ not allowed
        )

        # Assert retriever was called with strictly permitted paper IDs
        mock_retriever.retrieve_evidence.assert_called_once()
        called_claim: RetrievalClaim = mock_retriever.retrieve_evidence.call_args[1]["claim"]
        self.assertEqual(called_claim.paper_ids, ["Vaswani_2017_Attention"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].paper_id, "Vaswani_2017_Attention")

    def test_04_filtering_using_allowed_paper_ids_when_involved_papers_is_empty(self) -> None:
        """4. Filtering using allowed_paper_ids when involved_papers is empty."""
        claim = AtomicClaim(
            claim_id="C_104",
            claim_text="The architecture achieves state-of-the-art results.",
            source_text="The architecture achieves state-of-the-art results.",
            involved_papers=[],  # empty
        )

        mock_retriever = MagicMock(spec=EvidenceRetriever)
        mock_retriever.retrieve_evidence.return_value = ClaimEvidenceResponse(
            claim_id="C_104",
            evidence=[],
        )

        service = EvidenceRetrievalService(retriever=mock_retriever)
        service.retrieve(
            claim=claim,
            allowed_paper_ids=["Vaswani_2017_Attention"],
        )

        mock_retriever.retrieve_evidence.assert_called_once()
        called_claim: RetrievalClaim = mock_retriever.retrieve_evidence.call_args[1]["claim"]
        self.assertEqual(called_claim.paper_ids, ["Vaswani_2017_Attention"])

    def test_05_empty_paper_ids_return_no_evidence(self) -> None:
        """5. Empty paper IDs return no evidence and do not query the database."""
        mock_retriever = MagicMock(spec=EvidenceRetriever)
        service = EvidenceRetrievalService(retriever=mock_retriever)

        # Case A: both involved_papers and allowed_paper_ids are empty
        claim_a = AtomicClaim(
            claim_id="C_105_A",
            claim_text="General claim without paper association.",
            source_text="General claim without paper association.",
            involved_papers=[],
        )
        res_a = service.retrieve(claim=claim_a, allowed_paper_ids=[])
        self.assertEqual(res_a, [])
        mock_retriever.retrieve_evidence.assert_not_called()

        # Case B: disjoint involved_papers and allowed_paper_ids
        claim_b = AtomicClaim(
            claim_id="C_105_B",
            claim_text="Claim citing Paper X.",
            source_text="Claim citing Paper X.",
            involved_papers=["Paper_X"],
        )
        res_b = service.retrieve(claim=claim_b, allowed_paper_ids=["Paper_Y"])
        self.assertEqual(res_b, [])
        mock_retriever.retrieve_evidence.assert_not_called()

    def test_06_no_synthetic_evidence_enters_production_retrieval(self) -> None:
        """6. No synthetic ConvS2S data enters production evidence retrieval."""
        # Default production service (retriever=None, include_synthetic_fallback=False)
        service = EvidenceRetrievalService()

        corpus = service._get_or_load_local_corpus()
        for passage in corpus:
            self.assertNotIn(
                "[SYNTHETIC TEST DATA]",
                passage.text,
                f"Synthetic marker found in production corpus passage: {passage.evidence_id}",
            )
            self.assertNotEqual(
                passage.retrieval_method,
                "synthetic_test_corpus",
                f"Synthetic retrieval method found in production corpus: {passage.evidence_id}",
            )

        # Retrieve for a claim querying ConvS2S: must return empty because no synthetic data is ingested
        claim = AtomicClaim(
            claim_id="C_106",
            claim_text="ConvS2S uses gated linear units.",
            source_text="ConvS2S uses gated linear units.",
            involved_papers=["Gehring_2017_ConvS2S"],
        )
        results = service.retrieve(claim=claim, allowed_paper_ids=["Gehring_2017_ConvS2S"])
        self.assertEqual(results, [])

    def test_07_database_retrieval_errors_do_not_silently_trigger_synthetic_fallback(self) -> None:
        """7. Database retrieval errors do not silently trigger synthetic fallback."""
        mock_retriever = MagicMock(spec=EvidenceRetriever)
        mock_retriever.retrieve_evidence.side_effect = EvidenceRetrieverError(
            "FATAL: database connection refused at postgresql://localhost:5432"
        )

        service = EvidenceRetrievalService(retriever=mock_retriever)
        claim = AtomicClaim(
            claim_id="C_107",
            claim_text="Transformer achieves 28.4 BLEU.",
            source_text="Transformer achieves 28.4 BLEU.",
            involved_papers=["Vaswani_2017_Attention"],
        )

        # Must raise the database error directly rather than masking it with synthetic or local fallback
        with self.assertRaises(EvidenceRetrieverError):
            service.retrieve(claim=claim, allowed_paper_ids=["Vaswani_2017_Attention"])

    def test_08_existing_candidate_pool_test_behavior_remains_intact(self) -> None:
        """8. Existing candidate_pool test behavior remains completely intact."""
        custom_pool = [
            RetrievedEvidence(
                evidence_id="pool_ev_01",
                paper_id="Vaswani_2017_Attention",
                chunk_id="chunk_custom_1",
                text="The Transformer achieves 28.4 BLEU score on WMT 2014 English-to-German.",
                page=8,
                section="Table 2",
                retrieval_score=1.0,
                retrieval_method="candidate_pool",
            ),
            RetrievedEvidence(
                evidence_id="pool_ev_02",
                paper_id="Vaswani_2017_Attention",
                chunk_id="chunk_custom_2",
                text="An irrelevant passage about hardware specifications.",
                page=7,
                section="Hardware",
                retrieval_score=0.1,
                retrieval_method="candidate_pool",
            ),
        ]

        # Service configured with a mock retriever that should NOT be called when candidate_pool is provided
        mock_retriever = MagicMock(spec=EvidenceRetriever)
        service = EvidenceRetrievalService(retriever=mock_retriever)

        claim = AtomicClaim(
            claim_id="C_108",
            claim_text="The Transformer achieves 28.4 BLEU score on WMT 2014 English-to-German.",
            source_text="The Transformer achieves 28.4 BLEU score on WMT 2014 English-to-German.",
            involved_papers=["Vaswani_2017_Attention"],
        )

        results = service.retrieve(
            claim=claim,
            allowed_paper_ids=["Vaswani_2017_Attention"],
            candidate_pool=custom_pool,
        )

        # Mock retriever must not have been queried
        mock_retriever.retrieve_evidence.assert_not_called()

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].evidence_id, "pool_ev_01")
        self.assertEqual(results[0].page, 8)
        self.assertEqual(results[0].section, "Table 2")


if __name__ == "__main__":
    unittest.main()
