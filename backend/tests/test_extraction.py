"""
test_extraction.py — Automated integration and unit tests for the GenAI extraction pipeline.

Covers:
    - Configuration and client initialization
    - Schema serialization and default validation
    - Extraction prompt loading
    - Markdown / JSON parsing resilience
    - Long text chunking (no silent truncation)
    - Multi-chunk extraction merging
    - End-to-end ExtractionService execution with Mock LLM
    - Citation / evidence grounding validation
"""

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any

from backend.llm.config import LLMSettings, get_settings
from backend.llm.schemas.extraction import (
    ComputationalCostItem,
    DatasetItem,
    EvaluationMetricItem,
    EvidenceReference,
    ExtractionRequest,
    ExtractionResponse,
    KeyFindingItem,
    MethodologyItem,
    ResearchPaperExtraction,
    TokenizationItem,
    TrainingConfigItem,
)
from backend.llm.services.extraction_service import (
    ChunkCoverageReport,
    ExtractionService,
    clean_and_parse_json,
    load_extraction_prompt_template,
    merge_extractions,
    split_text_into_chunks,
    verify_chunk_coverage,
)
from backend.llm.validators.grounding import (
    GroundingReport,
    ItemSemanticEvaluation,
    QuoteMatchStatus,
    SemanticValidationReport,
    SupportClassification,
    classify_claim_support,
    detect_fabricated_metric_value,
    detect_forward_reference,
    detect_non_verbatim_notation,
    detect_prior_art_misattribution,
    extract_table_context_for_quote,
    is_mechanism_marked_unavailable,
    normalize_pdf_artifacts,
    verify_evidence_grounding,
    verify_extraction_semantic_support,
    verify_quote_grounding,
)


class MockLLMClient:
    """Mock LLM client to simulate Featherless responses without network calls."""

    def __init__(self, response_text: str | None = None) -> None:
        self.call_count = 0
        self.last_prompt = ""
        self.response_text = response_text or """{
            "paper_title": "Attention Is All You Need",
            "paper_id": null,
            "research_problem": "Computational bottleneck in recurrent language models",
            "research_objectives": ["Eliminate recurrence in favor of self-attention mechanisms"],
            "methodology": [
                {
                    "technique": "Transformer Architecture",
                    "description": "Sequence transduction based entirely on multi-head self-attention",
                    "evidence": {
                        "section": "Section 3",
                        "quote": "The Transformer is the first transduction model relying entirely on self-attention"
                    }
                }
            ],
            "datasets": [
                {
                    "name": "WMT 2014 English-German",
                    "details": "4.5 million sentence pairs",
                    "evidence": {
                        "quote": "trained on the standard WMT 2014 English-German dataset"
                    }
                }
            ],
            "evaluation_metrics": [
                {
                    "metric_name": "BLEU",
                    "value": "28.4",
                    "task_or_benchmark": "WMT 2014 English-to-German",
                    "evidence": {
                        "quote": "improves over the existing best results by over 2 BLEU"
                    }
                }
            ],
            "key_findings": [
                {
                    "claim": "Self-attention can replace recurrence while achieving state-of-the-art results.",
                    "evidence": {
                        "quote": "The Transformer allows for significantly more parallelization and can reach a new state of the art"
                    }
                }
            ],
            "limitations": [
                "Memory consumption scales quadratically with sequence length."
            ],
            "future_work": [
                "Apply attention-based models to images, audio, and video."
            ]
        }"""

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response_text


class TestExtractionPipeline(unittest.TestCase):
    """Test suite for the research paper extraction pipeline."""

    def test_01_config_and_settings(self) -> None:
        """Verify LLM settings configuration and Featherless defaults."""
        settings = get_settings()
        self.assertTrue(settings.llm_base_url.startswith("https://api.featherless.ai"))
        self.assertIn("Qwen", settings.llm_model_name)
        self.assertGreater(settings.llm_max_tokens, 0)
        self.assertGreaterEqual(settings.llm_temperature, 0.0)

    def test_02_prompt_template_loading(self) -> None:
        """Verify extraction prompt template loads and has required anchors."""
        prompt = load_extraction_prompt_template()
        self.assertIn("research_objectives", prompt)
        self.assertIn("methodology", prompt)
        self.assertIn("datasets", prompt)
        self.assertIn("evaluation_metrics", prompt)
        self.assertIn("key_findings", prompt)
        self.assertIn("limitations", prompt)
        self.assertIn("future_work", prompt)
        self.assertIn("{source_text}", prompt)

    def test_03_schema_defaults_and_zero_hallucination(self) -> None:
        """Verify empty schemas do not hallucinate synthetic default values."""
        empty = ResearchPaperExtraction()
        self.assertIsNone(empty.paper_title)
        self.assertIsNone(empty.paper_id)
        self.assertIsNone(empty.research_problem)
        self.assertEqual(empty.research_objectives, [])
        self.assertEqual(empty.methodology, [])
        self.assertEqual(empty.datasets, [])
        self.assertEqual(empty.evaluation_metrics, [])
        self.assertEqual(empty.key_findings, [])
        self.assertEqual(empty.limitations, [])
        self.assertEqual(empty.future_work, [])

    def test_04_json_parsing_resilience(self) -> None:
        """Verify JSON cleaner handles raw JSON, markdown fences, and text wrapping."""
        # 1. Standard raw JSON
        raw = '{"paper_title": "Test Paper", "research_objectives": []}'
        self.assertEqual(clean_and_parse_json(raw)["paper_title"], "Test Paper")

        # 2. Markdown fenced JSON
        fenced = '```json\n{"paper_title": "Test Paper 2", "research_objectives": []}\n```'
        self.assertEqual(clean_and_parse_json(fenced)["paper_title"], "Test Paper 2")

        # 3. Commentary before and after
        wrapped = 'Here is your result:\n```json\n{"paper_title": "Nested {Block}"}\n```\nHope it helps.'
        self.assertEqual(clean_and_parse_json(wrapped)["paper_title"], "Nested {Block}")

        # 4. Malformed JSON raises ValueError
        with self.assertRaises(ValueError):
            clean_and_parse_json("This is not JSON at all.")

    def test_05_long_paper_chunking_no_silent_truncation(self) -> None:
        """Verify text is partitioned across paragraph boundaries without dropping characters."""
        paras = [f"Paragraph {i}: Content for research evaluation block." for i in range(50)]
        full_text = "\n\n".join(paras)

        chunks = split_text_into_chunks(full_text, max_chars=300, overlap_chars=50)
        self.assertGreater(len(chunks), 1)

        # Confirm first paragraph is in first chunk and last paragraph is in last chunk
        self.assertIn("Paragraph 0", chunks[0])
        self.assertIn("Paragraph 49", chunks[-1])

        # Confirm all paragraphs are represented across chunks
        for i in range(50):
            target = f"Paragraph {i}:"
            found = any(target in c for c in chunks)
            self.assertTrue(found, f"Missing paragraph {i} in chunked output")

    def test_06_multi_chunk_merging_and_deduplication(self) -> None:
        """Verify merging preserves unique claims and deduplicates repeated titles/methods."""
        ext1 = ResearchPaperExtraction(
            paper_title="Multi-Modal Transformers",
            paper_id="PAPER_001",
            research_problem="Unimodal representations lack cross-modal synergy.",
            research_objectives=["Unify vision and text"],
            methodology=[MethodologyItem(technique="Cross-Attention")],
            datasets=[DatasetItem(name="MS-COCO")],
            evaluation_metrics=[EvaluationMetricItem(metric_name="CIDEr", value="135.2")],
            key_findings=[KeyFindingItem(claim="Cross-attention improves retrieval by 5%")],
            limitations=["Higher FLOPs"],
            future_work=["Model quantization"],
        )

        ext2 = ResearchPaperExtraction(
            paper_title=None,
            paper_id=None,
            research_objectives=["Unify vision and text", "Optimize latency"],
            methodology=[
                MethodologyItem(technique="Cross-Attention"),
                MethodologyItem(technique="Linear Projections"),
            ],
            datasets=[DatasetItem(name="Flickr30k")],
            evaluation_metrics=[EvaluationMetricItem(metric_name="BLEU-4", value="38.1")],
            key_findings=[KeyFindingItem(claim="Linear projections preserve 98% accuracy")],
            limitations=["GPU memory requirements", "Higher FLOPs"],
            future_work=["Model quantization", "Audio integration"],
        )

        merged = merge_extractions([ext1, ext2], fallback_paper_id="FALLBACK_ID")

        self.assertEqual(merged.paper_title, "Multi-Modal Transformers")
        self.assertEqual(merged.paper_id, "PAPER_001")
        self.assertEqual(merged.research_problem, "Unimodal representations lack cross-modal synergy.")
        self.assertEqual(merged.research_objectives, ["Unify vision and text", "Optimize latency"])
        self.assertEqual(len(merged.methodology), 2)
        self.assertEqual(len(merged.datasets), 2)
        self.assertEqual(len(merged.evaluation_metrics), 2)
        self.assertEqual(len(merged.key_findings), 2)
        self.assertEqual(merged.limitations, ["Higher FLOPs", "GPU memory requirements"])
        self.assertEqual(merged.future_work, ["Model quantization", "Audio integration"])

    def test_07_extraction_service_end_to_end_mock(self) -> None:
        """Verify ExtractionService processes input, injects metadata, and validates schema."""
        mock_client = MockLLMClient()
        settings = LLMSettings(llm_api_key="mock_key", llm_model_name="Qwen/Qwen3.8-27B")
        service = ExtractionService(client=mock_client, settings=settings)

        async def run_test():
            req = ExtractionRequest(
                text="The Transformer is the first transduction model relying entirely on self-attention...",
                paper_id="ARXIV_1706_03762",
                chunk_id="chunk_intro",
            )
            response = await service.extract(req)

            self.assertIsInstance(response, ExtractionResponse)
            self.assertEqual(response.model, "Qwen/Qwen3.8-27B")
            self.assertEqual(response.data.paper_title, "Attention Is All You Need")
            # Metadata injection verification
            self.assertEqual(response.data.paper_id, "ARXIV_1706_03762")
            self.assertEqual(response.data.methodology[0].evidence.paper_id, "ARXIV_1706_03762")
            self.assertEqual(response.data.methodology[0].evidence.chunk_id, "chunk_intro")
            self.assertEqual(mock_client.call_count, 1)

        asyncio.run(run_test())

    def test_08_evidence_grounding_validator(self) -> None:
        """Verify citation/grounding validator checks quotes against original text."""
        source_text = (
            "In this work we propose the Transformer. "
            "The Transformer is the first transduction model relying entirely on self-attention. "
            "We trained on the standard WMT 2014 English-German dataset."
        )

        # 1. Fully grounded extraction
        grounded_ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Transformer",
                    evidence=EvidenceReference(
                        quote="The Transformer is the first transduction model relying entirely on self-attention"
                    ),
                )
            ],
            datasets=[
                DatasetItem(
                    name="WMT",
                    evidence=EvidenceReference(
                        quote="trained on the standard WMT 2014 English-German dataset"
                    ),
                )
            ],
        )

        report = verify_evidence_grounding(grounded_ext, source_text)
        self.assertTrue(report.is_grounded)
        self.assertEqual(report.total_quotes_checked, 2)
        self.assertEqual(report.grounded_quotes_count, 2)
        self.assertEqual(report.ungrounded_quotes, [])

        # 2. Fabricated / hallucinated quote
        hallucinated_ext = ResearchPaperExtraction(
            key_findings=[
                KeyFindingItem(
                    claim="Achieves 99% accuracy on ImageNet",
                    evidence=EvidenceReference(
                        quote="Our model achieved revolutionary 99% top-1 accuracy on ImageNet"
                    ),
                )
            ]
        )

        hallucinated_report = verify_evidence_grounding(hallucinated_ext, source_text)
        self.assertFalse(hallucinated_report.is_grounded)
        self.assertEqual(len(hallucinated_report.ungrounded_quotes), 1)
        self.assertIn("ImageNet", hallucinated_report.ungrounded_quotes[0])

    def test_09_quote_exists_but_does_not_support_claim(self) -> None:
        """Verify that a verbatim quote that is semantically irrelevant to the claim is flagged as UNSUPPORTED."""
        source_text = "The Transformer was trained for twelve hours on eight P100 GPUs."
        unrelated_quote = "The Transformer was trained for twelve hours on eight P100 GPUs."
        unsupported_claim = "Eliminates quadratic memory complexity across long sequences."

        status, reason = classify_claim_support(unsupported_claim, unrelated_quote, source_text)
        self.assertEqual(status, SupportClassification.UNSUPPORTED)
        self.assertIn("does not semantically support", reason)

        # Confirm exact quote match passes, proving exact matching and semantic support are kept strictly separate
        exact_report = verify_evidence_grounding(
            ResearchPaperExtraction(
                key_findings=[
                    KeyFindingItem(claim=unsupported_claim, evidence=EvidenceReference(quote=unrelated_quote))
                ]
            ),
            source_text,
        )
        self.assertTrue(exact_report.is_grounded)

    def test_10_forward_reference_used_as_evidence(self) -> None:
        """Verify forward-looking section references are rejected as evidence for technique explanation."""
        source_text = "Multi-head attention will be described in detail in Section 3.2."
        forward_ref_quote = "Multi-head attention will be described in detail in Section 3.2."

        self.assertTrue(detect_forward_reference(forward_ref_quote))

        status, reason = classify_claim_support(
            "Multi-Head Attention projects queries, keys, and values h times.",
            forward_ref_quote,
            source_text,
        )
        self.assertEqual(status, SupportClassification.UNSUPPORTED)
        self.assertIn("forward reference", reason)

        ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Multi-Head Attention",
                    description="Jointly attends to information from different representation subspaces.",
                    evidence=EvidenceReference(quote=forward_ref_quote),
                )
            ]
        )
        report = verify_extraction_semantic_support(ext, source_text)
        self.assertTrue(report.is_grounded_exact)
        self.assertFalse(report.is_semantically_supported)
        self.assertEqual(len(report.forward_references), 1)

    def test_11_limitation_of_prior_work_misattributed(self) -> None:
        """Verify limitations of prior recurrent models are flagged when misattributed to the proposed Transformer."""
        source_text = (
            "Recurrent models preclude parallelization within training examples due to sequential computation. "
            "We propose the Transformer which dispenses with recurrence entirely."
        )
        misattributed_limitation = (
            "Sequential nature of recurrent computation precludes parallelization within training examples."
        )

        self.assertTrue(detect_prior_art_misattribution(misattributed_limitation))

        ext = ResearchPaperExtraction(
            limitations=[misattributed_limitation],
        )
        report = verify_extraction_semantic_support(ext, source_text)
        self.assertFalse(report.is_semantically_supported)
        self.assertEqual(len(report.misattributed_limitations), 1)
        self.assertIn("prior art", report.evaluations[0].reason)

    def test_12_fabricated_numerical_metric(self) -> None:
        """Verify numerical scores not found in the source text (e.g. 27.5 vs 28.4) are flagged as fabricated."""
        source_text = (
            "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, "
            "and establishes a new state-of-the-art score of 41.0 on English-to-French."
        )

        # 28.4 is authentic, 27.5 is fabricated / preprint draft
        self.assertFalse(detect_fabricated_metric_value("28.4 BLEU", source_text))
        self.assertTrue(detect_fabricated_metric_value("27.5 BLEU", source_text))
        self.assertTrue(detect_fabricated_metric_value("99.4% accuracy", source_text))

        ext = ResearchPaperExtraction(
            evaluation_metrics=[
                EvaluationMetricItem(
                    metric_name="BLEU (English-to-German)",
                    value="27.5 BLEU",
                    task_or_benchmark="WMT 2014 English-German",
                    evidence=EvidenceReference(quote="Our model achieves 28.4 BLEU"),
                )
            ]
        )
        report = verify_extraction_semantic_support(ext, source_text)
        self.assertFalse(report.is_semantically_supported)
        self.assertEqual(len(report.fabricated_metrics), 1)
        self.assertIn("27.5 BLEU", report.fabricated_metrics[0])

    def test_13_non_verbatim_quote_containing_substituted_notation(self) -> None:
        """Verify quotes substituting mathematical notation (e.g. O(1)) instead of verbatim text are detected."""
        source_text = "Self-attention connects all pairs of positions in a constant number of operations."
        non_verbatim_quote = "Self-attention connects all pairs of positions in O(1) operations."

        self.assertTrue(detect_non_verbatim_notation(non_verbatim_quote, source_text))

        status, reason = classify_claim_support(
            "Self-attention achieves constant sequential operations.",
            non_verbatim_quote,
            source_text,
        )
        self.assertEqual(status, SupportClassification.UNSUPPORTED)
        self.assertIn("notation substitution", reason)

    def test_14_unknown_field_should_remain_empty(self) -> None:
        """Verify that when a section/topic is absent from the chunk, populating it with synthetic data is rejected."""
        source_text = (
            "In this work we propose the Transformer. "
            "The model achieves 28.4 BLEU on the WMT 2014 English-German dataset."
        )

        # Case A: Future work hallucinated when absent from text
        hallucinated_ext = ResearchPaperExtraction(
            future_work=["Extend attention mechanisms to audio and video inputs."],
        )
        report_fail = verify_extraction_semantic_support(hallucinated_ext, source_text)
        self.assertFalse(report_fail.is_semantically_supported)
        self.assertEqual(len(report_fail.unexpected_nonempty_fields), 1)
        self.assertIn("future_work", report_fail.unexpected_nonempty_fields[0])

        # Case B: Future work properly left empty []
        valid_ext = ResearchPaperExtraction(future_work=[])
        report_pass = verify_extraction_semantic_support(valid_ext, source_text)
        self.assertTrue(report_pass.is_semantically_supported)
        self.assertEqual(len(report_pass.unexpected_nonempty_fields), 0)

    def test_15_valid_contiguous_quote(self) -> None:
        """Verify that a genuine contiguous verbatim quote is classified as EXACT_CONTIGUOUS and grounded."""
        source_text = (
            "We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        )
        quote = "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms"

        status, reason = verify_quote_grounding(quote, source_text)
        self.assertEqual(status, QuoteMatchStatus.EXACT_CONTIGUOUS)
        self.assertIn("Exact contiguous match", reason)

        ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Transformer",
                    description="Architecture based solely on attention mechanisms.",
                    evidence=EvidenceReference(quote=quote),
                )
            ]
        )
        grounding = verify_evidence_grounding(ext, source_text)
        self.assertTrue(grounding.is_grounded)
        self.assertEqual(grounding.exact_contiguous_count, 1)
        self.assertEqual(grounding.normalized_artifact_count, 0)

        sem_report = verify_extraction_semantic_support(ext, source_text)
        self.assertTrue(sem_report.is_semantically_supported)
        self.assertEqual(sem_report.evaluations[0].support_status, SupportClassification.SUPPORTED)
        self.assertEqual(sem_report.evaluations[0].quote_match_status, QuoteMatchStatus.EXACT_CONTIGUOUS)

    def test_16_quote_formed_by_joining_two_passages(self) -> None:
        """Verify that combining separate passages using ellipsis is classified as NON_CONTIGUOUS_SYNTHESIZED and rejected."""
        source_text = (
            "The Transformer follows this overall architecture using stacked self-attention. "
            "In Section 3, we detail the pipeline. "
            "The encoder is composed of a stack of N = 6 identical layers."
        )
        # Synthesized quote joining first sentence and third sentence with ellipsis
        stitched_quote = (
            "The Transformer follows this overall architecture using stacked self-attention... "
            "The encoder is composed of a stack of N = 6 identical layers."
        )

        status, reason = verify_quote_grounding(stitched_quote, source_text)
        self.assertEqual(status, QuoteMatchStatus.NON_CONTIGUOUS_SYNTHESIZED)
        self.assertIn("non-contiguous", reason)
        self.assertIn("ellipsis", reason)

        ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Encoder Stack",
                    description="Stacked self-attention encoder layers.",
                    evidence=EvidenceReference(quote=stitched_quote),
                )
            ]
        )
        grounding = verify_evidence_grounding(ext, source_text)
        self.assertFalse(grounding.is_grounded)
        self.assertEqual(grounding.non_contiguous_count, 1)
        self.assertIn(stitched_quote, grounding.ungrounded_quotes)

        sem_report = verify_extraction_semantic_support(ext, source_text)
        self.assertFalse(sem_report.is_semantically_supported)
        self.assertEqual(sem_report.evaluations[0].support_status, SupportClassification.UNSUPPORTED)
        self.assertEqual(sem_report.evaluations[0].quote_match_status, QuoteMatchStatus.NON_CONTIGUOUS_SYNTHESIZED)

    def test_17_quote_containing_pdf_page_break_artifact(self) -> None:
        """Verify that a contiguous quote containing PDF extraction artifacts (e.g. \\ufffe) is grounded via normalization."""
        source_text = "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task."
        # PDF extraction output often contains soft hyphen \ufffe across line/page breaks
        artifact_quote = "Our model achieves 28.4 BLEU on the WMT 2014 English\ufffeto-German translation task."

        status, reason = verify_quote_grounding(artifact_quote, source_text)
        self.assertEqual(status, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        self.assertIn("documented PDF artifact normalization", reason)

        ext = ResearchPaperExtraction(
            evaluation_metrics=[
                EvaluationMetricItem(
                    metric_name="BLEU",
                    value="28.4",
                    task_or_benchmark="WMT 2014 English-to-German",
                    evidence=EvidenceReference(quote=artifact_quote),
                )
            ]
        )
        grounding = verify_evidence_grounding(ext, source_text)
        self.assertTrue(grounding.is_grounded)
        self.assertEqual(grounding.normalized_artifact_count, 1)
        self.assertEqual(len(grounding.ungrounded_quotes), 0)

        sem_report = verify_extraction_semantic_support(ext, source_text)
        self.assertTrue(sem_report.is_semantically_supported)
        self.assertEqual(sem_report.evaluations[0].support_status, SupportClassification.SUPPORTED)
        self.assertEqual(sem_report.evaluations[0].quote_match_status, QuoteMatchStatus.NORMALIZED_ARTIFACT)

    def test_18_forward_reference_used_to_explain_mechanism(self) -> None:
        """Verify that citing a forward reference to explain a technique's mechanism is flagged as UNSUPPORTED."""
        source_text = (
            "Relating signals is reduced to a constant number of operations, "
            "an effect we counteract with Multi-Head Attention as described in section 3.2."
        )
        forward_quote = "an effect we counteract with Multi-Head Attention as described in section 3.2"

        self.assertTrue(detect_forward_reference(forward_quote))

        ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Multi-Head Attention",
                    description="Mechanism that counteracts reduced effective resolution by averaging attention positions.",
                    evidence=EvidenceReference(quote=forward_quote),
                )
            ]
        )
        sem_report = verify_extraction_semantic_support(ext, source_text)
        self.assertFalse(sem_report.is_semantically_supported)
        self.assertEqual(len(sem_report.forward_references), 1)
        self.assertEqual(sem_report.evaluations[0].support_status, SupportClassification.UNSUPPORTED)
        self.assertIn("forward reference", sem_report.evaluations[0].reason)

    def test_19_simple_mention_without_claiming_mechanism_explained(self) -> None:
        """Verify that a technique mentioned in text without claiming its mechanism is explained is correctly validated."""
        source_text = (
            "Relating signals is reduced to a constant number of operations, "
            "an effect we counteract with Multi-Head Attention as described in section 3.2."
        )

        # Case A: Technique listed with explicit note that mechanism explanation is unavailable and evidence=None
        mention_ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Multi-Head Attention",
                    description="Mechanism explanation unavailable in supplied text (forward reference only)",
                    evidence=None,
                )
            ]
        )
        sem_report = verify_extraction_semantic_support(mention_ext, source_text)
        self.assertTrue(sem_report.is_semantically_supported)
        self.assertEqual(len(sem_report.forward_references), 0)
        self.assertEqual(sem_report.evaluations[0].support_status, SupportClassification.MENTIONED_ONLY)
        self.assertEqual(sem_report.mentioned_only_count, 1)
        self.assertIn("explanation correctly marked as unavailable", sem_report.evaluations[0].reason)

        # Case B: Technique listed with description=None and evidence=None
        minimal_mention_ext = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Multi-Head Attention",
                    description=None,
                    evidence=None,
                )
            ]
        )
        sem_report_b = verify_extraction_semantic_support(minimal_mention_ext, source_text)
        self.assertTrue(sem_report_b.is_semantically_supported)
        self.assertEqual(sem_report_b.evaluations[0].support_status, SupportClassification.MENTIONED_ONLY)
        self.assertEqual(sem_report_b.mentioned_only_count, 1)

    def test_20_cross_chunk_methodology_upgrade_and_evidence_preservation(self) -> None:
        """Verify that when Chunk 1 has a mention/forward-ref and Chunk 2 has substantive Section 3.2 evidence,

        merge_extractions upgrades the item with the substantive evidence, and preserves distinct evidence.
        """
        # Chunk 1 extraction: mention/forward-reference
        ext_chunk1 = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Multi-Head Attention",
                    description="Mechanism explanation unavailable in supplied text (forward reference only)",
                    evidence=EvidenceReference(
                        section="Section 2",
                        quote="an effect we counteract with Multi-Head Attention as described in section 3.2",
                    ),
                )
            ]
        )

        # Chunk 2 extraction: substantive Section 3.2 explanation and quote
        ext_chunk2 = ResearchPaperExtraction(
            methodology=[
                MethodologyItem(
                    technique="Multi-Head Attention",
                    description="Linearly projects queries, keys, and values h times with learned projections.",
                    evidence=EvidenceReference(
                        section="Section 3.2.2",
                        quote="Instead of performing a single attention function with dmodel-dimensional keys, values and queries",
                    ),
                )
            ]
        )

        merged = merge_extractions([ext_chunk1, ext_chunk2])
        self.assertEqual(len(merged.methodology), 1)
        upgraded_item = merged.methodology[0]
        self.assertEqual(upgraded_item.technique, "Multi-Head Attention")
        self.assertIn("Linearly projects", upgraded_item.description)
        self.assertEqual(upgraded_item.evidence.section, "Section 3.2.2")

    def test_21_table_aware_evidence_validation(self) -> None:
        """Verify that table rows contextualized by column headers validate correctly without accepting fabricated values."""
        table_source = (
            "Table 2: The Transformer achieves better BLEU scores than previous models at lower training cost.\n"
            "Model BLEU Training Cost (FLOPs)\n"
            "EN-DE EN-FR EN-DE EN-FR\n"
            "ByteNet [15] 23.75\n"
            "Transformer (base model) 27.3 38.1 3.3 · 1018\n"
            "Transformer (big) 28.4 41.0 2.3 · 1019\n"
        )
        row_quote = "Transformer (big) 28.4 41.0 2.3 · 1019"

        # 1. Automatic table context resolution: row quote + header in table block
        status, reason = classify_claim_support("Training Cost (FLOPs) 2.3 · 10^19", row_quote, table_source)
        self.assertEqual(status, SupportClassification.SUPPORTED)
        self.assertIn("table context supports claim", reason)

        # 2. Unsupported / fabricated numerical metric in same row must be rejected
        status_fake, reason_fake = classify_claim_support("Training Cost (FLOPs) 99.9 · 10^19", row_quote, table_source)
        self.assertEqual(status_fake, SupportClassification.UNSUPPORTED)

        # 3. Explicit table_header supplied in EvidenceReference
        ext = ResearchPaperExtraction(
            evaluation_metrics=[
                EvaluationMetricItem(
                    metric_name="Training Cost (FLOPs)",
                    value="2.3 · 10^19",
                    task_or_benchmark="WMT 2014 English-to-French",
                    evidence=EvidenceReference(
                        quote=row_quote,
                        table_header="Training Cost (FLOPs)",
                        section="Table 2",
                    ),
                )
            ]
        )
        sem_rep = verify_extraction_semantic_support(ext, table_source)
        self.assertTrue(sem_rep.is_semantically_supported)
        self.assertEqual(sem_rep.evaluations[0].support_status, SupportClassification.SUPPORTED)

        # 4. Reject when table_header does not exist in source text
        ext_bad_header = ResearchPaperExtraction(
            evaluation_metrics=[
                EvaluationMetricItem(
                    metric_name="Carbon Emissions",
                    value="2.3 · 10^19",
                    evidence=EvidenceReference(
                        quote=row_quote,
                        table_header="Carbon Emissions Metric Header",
                    ),
                )
            ]
        )
        sem_bad = verify_extraction_semantic_support(ext_bad_header, table_source)
        self.assertFalse(sem_bad.is_semantically_supported)
        self.assertEqual(sem_bad.evaluations[0].support_status, SupportClassification.UNSUPPORTED)

    def test_22_generalized_cross_reference_resolution_table3_row_e(self) -> None:
        """Verify that a finding citing Table 3 row (E) forward reference is upgraded when later chunk provides substantive evidence.

        Also verify that if no substantive evidence is provided in later chunks, it is not marked supported merely because Table 3 exists.
        """
        # Chunk 2 finding: points forward to Table 3 row (E)
        ext_chunk2 = ResearchPaperExtraction(
            key_findings=[
                KeyFindingItem(
                    claim="Sinusoidal positional encodings and learned positional embeddings produced nearly identical results",
                    evidence=EvidenceReference(
                        section="Section 3.5 Positional Encoding",
                        quote="We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)).",
                    ),
                )
            ]
        )

        # Chunk 3 finding: substantive empirical result from Section 6.2 / Table 3
        ext_chunk3 = ResearchPaperExtraction(
            key_findings=[
                KeyFindingItem(
                    claim="Replacing sinusoidal positional encoding with learned positional embeddings produces nearly identical results",
                    evidence=EvidenceReference(
                        section="Section 6.2 Model Variations",
                        quote="In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [8], and observe nearly identical results to the base model.",
                    ),
                )
            ]
        )

        # Before merge: chunk 2 finding contains forward reference and is UNSUPPORTED
        source_text = (
            "Section 3.5 Positional Encoding\n"
            "We also experimented with using learned positional embeddings [8] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)).\n"
            "Section 6.2 Model Variations\n"
            "In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [8], and observe nearly identical results to the base model.\n"
        )
        sem_c2 = verify_extraction_semantic_support(ext_chunk2, source_text)
        self.assertFalse(sem_c2.is_semantically_supported)
        self.assertEqual(sem_c2.evaluations[0].support_status, SupportClassification.UNSUPPORTED)

        # After merge: generalized cross-reference resolution upgrades the finding to the substantive evidence
        merged = merge_extractions([ext_chunk2, ext_chunk3])
        self.assertEqual(len(merged.key_findings), 1)
        upgraded_finding = merged.key_findings[0]
        self.assertFalse(detect_forward_reference(upgraded_finding.evidence.quote))
        self.assertIn("In row (E)", upgraded_finding.evidence.quote)
        self.assertEqual(upgraded_finding.evidence.section, "Section 6.2 Model Variations")

        # Merged extraction is now fully supported
        sem_merged = verify_extraction_semantic_support(merged, source_text)
        self.assertTrue(sem_merged.is_semantically_supported)
        self.assertEqual(sem_merged.evaluations[0].support_status, SupportClassification.SUPPORTED)

        # Negative test: if chunk 3 only has forward reference or placeholder, no upgrade occurs
        ext_chunk3_empty = ResearchPaperExtraction(
            key_findings=[
                KeyFindingItem(
                    claim="Positional embeddings results",
                    evidence=EvidenceReference(
                        quote="as discussed in section 6",
                    ),
                )
            ]
        )
        merged_no_evidence = merge_extractions([ext_chunk2, ext_chunk3_empty])
        self.assertTrue(detect_forward_reference(merged_no_evidence.key_findings[0].evidence.quote))

    def test_23_improved_extraction_taxonomy_and_backward_compatibility(self) -> None:
        """Verify new taxonomy models (tokenization, training_config, computational_cost) and backward compatibility."""
        ext = ResearchPaperExtraction(
            paper_title="Attention Is All You Need",
            tokenization=[
                TokenizationItem(
                    method="Byte-Pair Encoding (BPE)",
                    vocabulary_size="37000 tokens",
                    details="Shared source-target vocabulary for English-German.",
                    evidence=EvidenceReference(quote="byte-pair encoding"),
                )
            ],
            training_config=[
                TrainingConfigItem(
                    parameter="Adam Optimizer",
                    value="beta1=0.9, beta2=0.98, eps=10^-9",
                    details="Warmup steps = 4000",
                    evidence=EvidenceReference(quote="Adam optimizer [17] with β1 = 0.9"),
                )
            ],
            computational_cost=[
                ComputationalCostItem(
                    hardware="8 NVIDIA P100 GPUs",
                    training_time="3.5 days",
                    flops="2.3 · 10^19",
                    evidence=EvidenceReference(quote="Training took 3.5 days on 8 P100 GPUs"),
                )
            ],
            datasets=[
                DatasetItem(
                    name="WMT 2014 English-German",
                    category="benchmark",
                    details="4.5 million sentence pairs",
                )
            ],
        )

        dumped = ext.model_dump()
        self.assertIn("tokenization", dumped)
        self.assertIn("training_config", dumped)
        self.assertIn("computational_cost", dumped)
        self.assertEqual(dumped["tokenization"][0]["method"], "Byte-Pair Encoding (BPE)")
        self.assertEqual(dumped["training_config"][0]["parameter"], "Adam Optimizer")
        self.assertEqual(dumped["computational_cost"][0]["flops"], "2.3 · 10^19")

        # Backward compatibility: load an older JSON without new fields
        old_json_dict = {
            "paper_title": "Old Paper",
            "methodology": [{"technique": "RNN"}],
            "datasets": [{"name": "MNIST"}],
            "evaluation_metrics": [],
            "key_findings": [],
            "limitations": [],
            "future_work": [],
        }
        loaded = ResearchPaperExtraction.model_validate(old_json_dict)
        self.assertEqual(loaded.tokenization, [])
        self.assertEqual(loaded.training_config, [])
        self.assertEqual(loaded.computational_cost, [])

    def test_24_strengthened_chunk_coverage_and_provenance_integrity(self) -> None:
        """Verify programmatic chunk coverage: no gaps, intentional overlap, page marker preservation, and failure detection."""
        full_text = (
            "--- PAGE 1 ---\n"
            "Attention Is All You Need. Abstract: The dominant sequence transduction models are based on recurrent networks. "
            "We propose the Transformer, based solely on attention mechanisms.\n\n"
            "--- PAGE 2 ---\n"
            "Section 2: Background. The goal of reducing sequential computation also forms the foundation of ByteNet. "
            "Section 3: Model Architecture. The Transformer follows this overall architecture using stacked self-attention.\n\n"
            "--- PAGE 3 ---\n"
            "Section 3.2: Attention. An attention function can be described as mapping a query and a set of key-value pairs to an output. "
            "Section 4: Why Self-Attention. We compare self-attention layers to recurrent layers."
        )

        # Partition text with small chunk size to force multiple overlapping chunks
        chunks = split_text_into_chunks(full_text, max_chars=300, overlap_chars=80)
        self.assertGreaterEqual(len(chunks), 2)

        # Verify continuous coverage
        report = verify_chunk_coverage(full_text, chunks)
        self.assertTrue(report.is_fully_covered, f"Coverage failed: {report.uncovered_ranges}")
        self.assertTrue(report.overlap_integrity)
        self.assertTrue(report.page_markers_preserved)
        self.assertEqual(len(report.missing_page_markers), 0)
        self.assertEqual(len(report.uncovered_ranges), 0)

        # Negative test: drop a chunk to simulate a gap/data loss
        broken_chunks = [chunks[0], chunks[-1]] if len(chunks) >= 3 else [chunks[0][:100]]
        broken_report = verify_chunk_coverage(full_text, broken_chunks)
        self.assertFalse(broken_report.is_fully_covered)
        self.assertGreater(len(broken_report.uncovered_ranges), 0)


if __name__ == "__main__":
    unittest.main()

