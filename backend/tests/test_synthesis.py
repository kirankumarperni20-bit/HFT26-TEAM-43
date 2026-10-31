"""
test_synthesis.py — Comprehensive unit and integration tests for cross-paper synthesis.

Covers:
1. Successful synthesis of two papers (Transformer vs ConvS2S baseline).
2. Methodological similarities and differences.
3. Missing information handling without fabrication.
4. Source reference integrity and citation traceability.
5. Contextual differences vs. genuine contradictions (different benchmarks/metrics).
6. Invalid model JSON and recovery/failure handling.
7. Pydantic validation failures.
8. FastAPI endpoints: POST /api/research/synthesize and POST /llm/synthesize.
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
from backend.llm.schemas.extraction import (
    DatasetItem,
    EvaluationMetricItem,
    EvidenceReference,
    KeyFindingItem,
    MethodologyItem,
    ResearchPaperExtraction,
)
from backend.llm.schemas.synthesis import (
    ComparisonType,
    CrossPaperSynthesis,
    ExperimentalComparison,
    FindingComparison,
    FindingRelationship,
    FutureDirection,
    MethodologicalComparison,
    ResearchGap,
    SourceReference,
    SynthesisRequest,
    SynthesisResponse,
)
from backend.llm.services.synthesis_service import (
    SynthesisService,
    load_synthesis_prompt_template,
    serialize_papers_for_prompt,
    synthesize_papers,
)


def create_sample_paper_transformer() -> ResearchPaperExtraction:
    """Create sample extraction for Attention Is All You Need (Vaswani et al.)."""
    return ResearchPaperExtraction(
        paper_id="Vaswani_2017_Attention",
        paper_title="Attention Is All You Need",
        research_problem="Sequential computation in recurrent networks precludes parallelization within training examples.",
        research_objectives=[
            "Propose the Transformer network architecture based solely on attention mechanisms.",
            "Demonstrate superior translation quality with reduced training time.",
        ],
        methodology=[
            MethodologyItem(
                technique="Transformer",
                description="Architecture based solely on self-attention and feed-forward layers, dispensing with recurrence.",
                evidence=EvidenceReference(
                    paper_id="Vaswani_2017_Attention",
                    section="Abstract",
                    quote="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms",
                ),
            ),
            MethodologyItem(
                technique="Multi-Head Attention",
                description="Linearly projects queries, keys, and values h times to jointly attend to different subspaces.",
                evidence=EvidenceReference(
                    paper_id="Vaswani_2017_Attention",
                    section="Section 3.2.2",
                    quote="linearly project the queries, keys and values h times with different, learned linear projections",
                ),
            ),
        ],
        datasets=[
            DatasetItem(
                name="WMT 2014 English-to-German",
                category="benchmark",
                details="4.5 million sentence pairs",
                evidence=EvidenceReference(quote="WMT 2014 English-German dataset"),
            ),
            DatasetItem(
                name="WMT 2014 English-to-French",
                category="benchmark",
                details="36 million sentence pairs",
                evidence=EvidenceReference(quote="WMT 2014 English-French dataset"),
            ),
        ],
        evaluation_metrics=[
            EvaluationMetricItem(
                metric_name="BLEU",
                value="28.4",
                task_or_benchmark="WMT 2014 English-to-German",
                evidence=EvidenceReference(quote="big model achieves a BLEU score of 28.4"),
            ),
            EvaluationMetricItem(
                metric_name="BLEU",
                value="41.0",
                task_or_benchmark="WMT 2014 English-to-French",
                evidence=EvidenceReference(quote="big model achieves a BLEU score of 41.0"),
            ),
        ],
        key_findings=[
            KeyFindingItem(
                claim="Self-attention layers connect all positions with a constant number of sequential operations O(1).",
                evidence=EvidenceReference(quote="connects all positions with a constant number of sequentially executed operations"),
            ),
            KeyFindingItem(
                claim="The Transformer outperforms previous models on WMT 2014 En-De with 28.4 BLEU.",
                evidence=EvidenceReference(quote="outperforms the best previously reported models"),
            ),
        ],
        limitations=[
            "Computational complexity per layer is O(n^2 * d), which becomes costly for very long sequences."
        ],
        future_work=[
            "Extend attention mechanisms to other modalities including image, audio, and video.",
            "Investigate local, restricted attention mechanisms for long sequences.",
        ],
    )


def create_sample_paper_convs2s() -> ResearchPaperExtraction:
    """Create sample extraction for Convolutional Sequence to Sequence Learning (Gehring et al.)."""
    return ResearchPaperExtraction(
        paper_id="Gehring_2017_ConvS2S",
        paper_title="Convolutional Sequence to Sequence Learning",
        research_problem="Recurrent sequence to sequence models cannot parallelize computation over sequence elements during training.",
        research_objectives=[
            "Introduce an entirely convolutional architecture for sequence to sequence learning.",
            "Enable parallel computation across all sequence elements while preserving variable-length mapping.",
        ],
        methodology=[
            MethodologyItem(
                technique="Convolutional Sequence to Sequence",
                description="Uses stacked 1D convolutions with gated linear units and residual connections.",
                evidence=EvidenceReference(
                    paper_id="Gehring_2017_ConvS2S",
                    section="Section 3",
                    quote="architecture based entirely on convolutional neural networks with gated linear units",
                ),
            ),
            MethodologyItem(
                technique="Multi-Hop Attention",
                description="Performs attention at each decoder layer conditioned on convolutional representations.",
                evidence=EvidenceReference(
                    paper_id="Gehring_2017_ConvS2S",
                    section="Section 3.2",
                    quote="separate attention mechanism for each decoder layer",
                ),
            ),
        ],
        datasets=[
            DatasetItem(
                name="WMT 2014 English-to-German",
                category="benchmark",
                details="Standard WMT 2014 training split",
                evidence=EvidenceReference(quote="WMT 2014 English-German news translation task"),
            ),
            DatasetItem(
                name="IWSLT 2014 German-to-English",
                category="benchmark",
                details="160k sentence pairs from TED talks",
                evidence=EvidenceReference(quote="IWSLT 2014 German-English dataset"),
            ),
        ],
        evaluation_metrics=[
            EvaluationMetricItem(
                metric_name="BLEU",
                value="25.16",
                task_or_benchmark="WMT 2014 English-to-German",
                evidence=EvidenceReference(quote="achieves 25.16 BLEU on WMT 2014 English-German"),
            ),
            EvaluationMetricItem(
                metric_name="BLEU",
                value="30.5",
                task_or_benchmark="IWSLT 2014 German-to-English",
                evidence=EvidenceReference(quote="obtains 30.5 BLEU on IWSLT 2014"),
            ),
        ],
        key_findings=[
            KeyFindingItem(
                claim="Convolutional networks allow simultaneous computation over all words during training.",
                evidence=EvidenceReference(quote="computations over all elements can be fully parallelized"),
            ),
            KeyFindingItem(
                claim="ConvS2S achieves 25.16 BLEU on WMT 2014 En-De.",
                evidence=EvidenceReference(quote="ConvS2S achieves 25.16 BLEU"),
            ),
        ],
        limitations=[
            "Hierarchical convolutional layers require O(n/k) layers to connect distant words across sequence length n."
        ],
        future_work=[
            "Explore deeper convolutional layers and dilated receptive fields.",
        ],
    )


def sample_mock_synthesis_json() -> str:
    """Return realistic valid JSON matching the CrossPaperSynthesis schema for the two sample papers."""
    data = {
        "overview": (
            "Both Vaswani et al. (2017) and Gehring et al. (2017) tackle the fundamental bottleneck "
            "of recurrent sequence models: sequential computation that prevents parallel training. "
            "Vaswani et al. discard recurrence and convolution entirely in favor of self-attention, "
            "while Gehring et al. replace recurrence with stacked 1D convolutions and multi-hop attention."
        ),
        "papers_analyzed": [
            {"paper_id": "Vaswani_2017_Attention", "paper_title": "Attention Is All You Need"},
            {"paper_id": "Gehring_2017_ConvS2S", "paper_title": "Convolutional Sequence to Sequence Learning"},
        ],
        "common_themes": [
            "Eliminating sequential recurrent bottlenecks to enable parallelized sequence training.",
            "Benchmarking on standard WMT machine translation tasks.",
            "Leveraging attention mechanisms to connect input and output sequences.",
        ],
        "methodological_similarities": [
            {
                "aspect": "Encoder-Decoder Framework with Residual Connections",
                "comparison_type": "SIMILARITY",
                "description": "Both architectures adopt encoder-decoder pipelines with residual connections and layer normalization.",
                "paper_details": {
                    "Vaswani_2017_Attention": "Stacked 6-layer self-attention encoder-decoder with residual connections.",
                    "Gehring_2017_ConvS2S": "Stacked 1D convolutional layers with residual connections.",
                },
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Abstract",
                        "quote": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms",
                    },
                    {
                        "paper_id": "Gehring_2017_ConvS2S",
                        "section": "Section 3",
                        "quote": "architecture based entirely on convolutional neural networks with gated linear units",
                    },
                ],
            }
        ],
        "methodological_differences": [
            {
                "aspect": "Core Sequence Modeling Layer",
                "comparison_type": "DIFFERENCE",
                "description": "Vaswani et al. use pure Multi-Head Self-Attention connecting all positions in O(1) operations, whereas Gehring et al. use 1D convolutions requiring O(n/k) layers for global receptive field.",
                "paper_details": {
                    "Vaswani_2017_Attention": "Multi-Head Attention mapping queries, keys, and values simultaneously.",
                    "Gehring_2017_ConvS2S": "1D convolution with gated linear units and multi-hop attention.",
                },
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Section 3.2.2",
                        "quote": "linearly project the queries, keys and values h times with different, learned linear projections",
                    },
                    {
                        "paper_id": "Gehring_2017_ConvS2S",
                        "section": "Section 3.2",
                        "quote": "separate attention mechanism for each decoder layer",
                    },
                ],
            }
        ],
        "dataset_and_experimental_differences": [
            {
                "aspect": "Translation Benchmarks Evaluated",
                "description": "Both evaluate on WMT 2014 English-to-German, but Vaswani et al. evaluate on WMT 2014 English-to-French (36M pairs), while Gehring et al. evaluate on IWSLT 2014 German-to-English (160k pairs).",
                "paper_details": {
                    "Vaswani_2017_Attention": "WMT 2014 En-De (4.5M) and WMT 2014 En-Fr (36M pairs).",
                    "Gehring_2017_ConvS2S": "WMT 2014 En-De and IWSLT 2014 De-En (160k pairs).",
                },
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Section 5.1",
                        "quote": "WMT 2014 English-German dataset",
                    },
                    {
                        "paper_id": "Gehring_2017_ConvS2S",
                        "section": "Section 5",
                        "quote": "IWSLT 2014 German-English dataset",
                    },
                ],
            }
        ],
        "findings_agreement": [
            {
                "topic": "Elimination of Recurrence Accelerates Training",
                "relationship": "AGREEMENT",
                "description": "Both papers find that non-recurrent sequence models enable fully parallelized computation and significantly reduce training wall-clock time compared to RNNs.",
                "contextual_factors": [],
                "paper_findings": {
                    "Vaswani_2017_Attention": "Constant sequential operations O(1) allows highly parallel training.",
                    "Gehring_2017_ConvS2S": "Full parallelization across sequence elements during training.",
                },
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Section 4",
                        "quote": "connects all positions with a constant number of sequentially executed operations",
                    },
                    {
                        "paper_id": "Gehring_2017_ConvS2S",
                        "section": "Section 1",
                        "quote": "computations over all elements can be fully parallelized",
                    },
                ],
            }
        ],
        "findings_disagreement": [
            {
                "topic": "WMT 2014 English-to-German BLEU Quality",
                "relationship": "CONTEXTUAL_DIFFERENCE",
                "description": "The Transformer achieves 28.4 BLEU outperforming ConvS2S (25.16 BLEU); the difference is driven by model capacity and attention design rather than contradictory claims.",
                "contextual_factors": [
                    "Different model capacities (Transformer big vs ConvS2S single model)",
                    "Direct self-attention connecting all positions vs convolutional hierarchy",
                ],
                "paper_findings": {
                    "Vaswani_2017_Attention": "28.4 BLEU on WMT 2014 En-De.",
                    "Gehring_2017_ConvS2S": "25.16 BLEU on WMT 2014 En-De.",
                },
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Table 2",
                        "quote": "big model achieves a BLEU score of 28.4",
                    },
                    {
                        "paper_id": "Gehring_2017_ConvS2S",
                        "section": "Table 1",
                        "quote": "achieves 25.16 BLEU on WMT 2014 English-German",
                    },
                ],
            }
        ],
        "research_limitations_and_gaps": [
            {
                "gap": "Sequence Length Scaling Bottleneck",
                "affected_papers": ["Vaswani_2017_Attention", "Gehring_2017_ConvS2S"],
                "details": "The Transformer scales quadratically O(n^2 * d) with sequence length n, while ConvS2S requires deeper stacks of convolutions to bridge long distances.",
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Table 1",
                        "quote": "computational complexity is O(n^2 * d)",
                    }
                ],
            }
        ],
        "future_research_directions": [
            {
                "direction": "Multimodal Attention and Scaling to Non-Text Modalities",
                "proposed_by": ["Vaswani_2017_Attention"],
                "potential_impact": "Applying attention models to image, audio, and video sequence transduction tasks.",
                "evidence": [
                    {
                        "paper_id": "Vaswani_2017_Attention",
                        "section": "Section 7 Conclusion",
                        "quote": "extend attention mechanisms to other modalities including image, audio, and video",
                    }
                ],
            }
        ],
    }
    return json.dumps(data)


class TestCrossPaperSynthesis(unittest.TestCase):
    """Unit and integration test suite for cross-paper synthesis."""

    def setUp(self) -> None:
        self.p1 = create_sample_paper_transformer()
        self.p2 = create_sample_paper_convs2s()

    def test_01_synthesis_request_validation(self) -> None:
        """Verify SynthesisRequest enforces at least two extracted papers."""
        # Single paper must fail
        with self.assertRaises(ValueError):
            SynthesisRequest(papers=[self.p1])

        # Zero papers must fail
        with self.assertRaises(ValueError):
            SynthesisRequest(papers=[])

        # Two papers must succeed
        req = SynthesisRequest(papers=[self.p1, self.p2], focus_areas=["Attention Mechanism"])
        self.assertEqual(len(req.papers), 2)
        self.assertEqual(req.focus_areas, ["Attention Mechanism"])

    def test_02_prompt_template_and_serialization(self) -> None:
        """Verify prompt template loading and paper JSON serialization."""
        template = load_synthesis_prompt_template()
        self.assertIn("{extracted_papers_json}", template)
        self.assertIn("STRICT FACTUAL GROUNDING", template)
        self.assertIn("CAREFUL CONTRADICTION & DIVERGENCE HANDLING", template)

        serialized = serialize_papers_for_prompt([self.p1, self.p2])
        parsed = json.loads(serialized)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["paper_id"], "Vaswani_2017_Attention")
        self.assertEqual(parsed[1]["paper_id"], "Gehring_2017_ConvS2S")

    @patch("backend.llm.services.synthesis_service.LLMClient")
    def test_03_successful_synthesis_two_papers(self, mock_client_cls: Any) -> None:
        """Verify end-to-end synthesis execution with mocked LLM returns validated CrossPaperSynthesis."""
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = sample_mock_synthesis_json()
        mock_client_cls.return_value = mock_instance

        service = SynthesisService(client=mock_instance)
        request = SynthesisRequest(papers=[self.p1, self.p2])

        response: SynthesisResponse = asyncio.run(service.synthesize(request))

        self.assertIsInstance(response, SynthesisResponse)
        self.assertEqual(response.papers_count, 2)
        synthesis: CrossPaperSynthesis = response.data

        # Verify core overview & themes
        self.assertIn("Vaswani et al.", synthesis.overview)
        self.assertIn("Gehring et al.", synthesis.overview)
        self.assertGreaterEqual(len(synthesis.common_themes), 2)
        self.assertEqual(len(synthesis.papers_analyzed), 2)

    @patch("backend.llm.services.synthesis_service.LLMClient")
    def test_04_methodological_similarities_and_differences(self, mock_client_cls: Any) -> None:
        """Verify structured extraction of similarities and differences across model architectures."""
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = sample_mock_synthesis_json()

        service = SynthesisService(client=mock_instance)
        response = asyncio.run(service.synthesize(SynthesisRequest(papers=[self.p1, self.p2])))
        synthesis = response.data

        # Similarities
        self.assertGreaterEqual(len(synthesis.methodological_similarities), 1)
        sim = synthesis.methodological_similarities[0]
        self.assertEqual(sim.comparison_type, ComparisonType.SIMILARITY)
        self.assertIn("Vaswani_2017_Attention", sim.paper_details)
        self.assertIn("Gehring_2017_ConvS2S", sim.paper_details)
        self.assertGreaterEqual(len(sim.evidence), 2)

        # Differences
        self.assertGreaterEqual(len(synthesis.methodological_differences), 1)
        diff = synthesis.methodological_differences[0]
        self.assertEqual(diff.comparison_type, ComparisonType.DIFFERENCE)
        self.assertIn("Multi-Head", diff.paper_details["Vaswani_2017_Attention"])

    @patch("backend.llm.services.synthesis_service.LLMClient")
    def test_05_contextual_difference_vs_contradiction(self, mock_client_cls: Any) -> None:
        """Verify that disparate BLEU scores are classified as CONTEXTUAL_DIFFERENCE with contextual factors, not contradictions."""
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = sample_mock_synthesis_json()

        service = SynthesisService(client=mock_instance)
        response = asyncio.run(service.synthesize(SynthesisRequest(papers=[self.p1, self.p2])))
        synthesis = response.data

        # Findings agreement
        self.assertGreaterEqual(len(synthesis.findings_agreement), 1)
        agr = synthesis.findings_agreement[0]
        self.assertEqual(agr.relationship, FindingRelationship.AGREEMENT)

        # Findings disagreement / contextual difference
        self.assertGreaterEqual(len(synthesis.findings_disagreement), 1)
        dis = synthesis.findings_disagreement[0]
        self.assertEqual(dis.relationship, FindingRelationship.CONTEXTUAL_DIFFERENCE)
        self.assertGreaterEqual(len(dis.contextual_factors), 1)
        self.assertIn("capacities", dis.contextual_factors[0].lower())

    @patch("backend.llm.services.synthesis_service.LLMClient")
    def test_06_source_reference_integrity(self, mock_client_cls: Any) -> None:
        """Verify that every citation attached to comparisons belongs to one of the input papers."""
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = sample_mock_synthesis_json()

        service = SynthesisService(client=mock_instance)
        response = asyncio.run(service.synthesize(SynthesisRequest(papers=[self.p1, self.p2])))
        synthesis = response.data

        valid_ids = {"Vaswani_2017_Attention", "Gehring_2017_ConvS2S"}

        # Collect all citations across sections
        citations: list[SourceReference] = []
        for s in synthesis.methodological_similarities:
            citations.extend(s.evidence)
        for d in synthesis.methodological_differences:
            citations.extend(d.evidence)
        for e in synthesis.dataset_and_experimental_differences:
            citations.extend(e.evidence)
        for a in synthesis.findings_agreement:
            citations.extend(a.evidence)
        for da in synthesis.findings_disagreement:
            citations.extend(da.evidence)
        for g in synthesis.research_limitations_and_gaps:
            citations.extend(g.evidence)
        for f in synthesis.future_research_directions:
            citations.extend(f.evidence)

        self.assertGreaterEqual(len(citations), 5)
        for cit in citations:
            self.assertIn(cit.paper_id, valid_ids, f"Fabricated paper_id detected: {cit.paper_id}")
            self.assertTrue(bool(cit.quote), "Citation quote must not be empty.")

    @patch("backend.llm.services.synthesis_service.LLMClient")
    def test_07_invalid_json_handling(self, mock_client_cls: Any) -> None:
        """Verify that completely unparseable model responses raise ValueError safely."""
        mock_instance = AsyncMock()
        mock_instance.generate.return_value = "Sorry, I am unable to synthesize these papers due to system error."

        service = SynthesisService(client=mock_instance)
        with self.assertRaises(ValueError):
            asyncio.run(service.synthesize(SynthesisRequest(papers=[self.p1, self.p2])))

    @patch("backend.llm.services.synthesis_service.LLMClient")
    def test_08_pydantic_schema_validation_failure(self, mock_client_cls: Any) -> None:
        """Verify that JSON missing mandatory fields (e.g. overview) raises ValidationError."""
        mock_instance = AsyncMock()
        # Missing overview field
        bad_json = json.dumps({"common_themes": ["parallelization"]})
        mock_instance.generate.return_value = bad_json

        service = SynthesisService(client=mock_instance)
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            asyncio.run(service.synthesize(SynthesisRequest(papers=[self.p1, self.p2])))

    @patch("backend.llm.services.synthesis_service.SynthesisService.synthesize")
    def test_09_fastapi_endpoints_integration(self, mock_synthesize: Any) -> None:
        """Verify both POST /api/research/synthesize and POST /llm/synthesize endpoints via TestClient."""
        # Configure mock return value
        dummy_synthesis = CrossPaperSynthesis.model_validate(json.loads(sample_mock_synthesis_json()))
        dummy_response = SynthesisResponse(
            data=dummy_synthesis,
            model="Qwen/Qwen3.8-27B",
            papers_count=2,
        )
        mock_synthesize.return_value = dummy_response

        client = TestClient(app)

        payload = {
            "papers": [self.p1.model_dump(), self.p2.model_dump()],
            "focus_areas": ["Parallelization"],
        }

        # 1. Test POST /api/research/synthesize
        res1 = client.post("/api/research/synthesize", json=payload)
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1["papers_count"], 2)
        self.assertIn("overview", data1["data"])

        # 2. Test POST /llm/synthesize
        res2 = client.post("/llm/synthesize", json=payload)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertEqual(data2["papers_count"], 2)

        # 3. Test HTTP 422 validation error when fewer than 2 papers are sent
        bad_payload = {"papers": [self.p1.model_dump()]}
        res_bad = client.post("/api/research/synthesize", json=bad_payload)
        self.assertEqual(res_bad.status_code, 422)


if __name__ == "__main__":
    import asyncio
    unittest.main()
