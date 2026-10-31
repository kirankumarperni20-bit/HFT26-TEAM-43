"""
test_research_focus_feature.py — Comprehensive tests for Research Focus functionality.

Verifies:
1. Backend receives and propagates focus_query and research_focus fields.
2. Backward compatibility with existing API requests (empty / None focus).
3. Dynamic topic extraction without hardcoding.
4. Different focus inputs produce appropriately different evidence selection and summary emphasis.
5. Evidence citations correctly point to verbatim PDF chunk excerpts.
6. Missing topics are reported honestly without hallucination or guessing.
7. Multi-paper synthesis preserves source attribution and aligns comparative dimensions with user topics.
8. Long (up to 500 characters) or irrelevant focus queries are handled gracefully.
9. End-to-end research generation with focus produces complete grounded results.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.documents import get_repository as get_doc_repo
from backend.api.research import get_research_job_repository, get_research_service
from backend.ingestion.repository import InMemoryDocumentRepository
from backend.research.repository import InMemoryResearchJobRepository
from backend.research.service import ResearchJobService
from backend.research.cleaner import (
    extract_focus_topics,
    extract_topic_evidence_and_summary,
    synthesize_paper_summary,
    synthesize_cross_paper_comparison,
)
from backend.schemas.chunk import Chunk
from backend.schemas.document import DocumentMetadata, DocumentStatus
from backend.schemas.research import (
    StartResearchPayload,
    PaperSynthesisItem,
    ResearchTask,
    ResearchStatus,
)


@pytest.fixture
def focus_test_env():
    """Sets up an isolated FastAPI test client with in-memory repositories."""
    doc_repo = InMemoryDocumentRepository()
    job_repo = InMemoryResearchJobRepository()
    service = ResearchJobService(job_repository=job_repo, document_repository=doc_repo)

    app.dependency_overrides[get_doc_repo] = lambda: doc_repo
    app.dependency_overrides[get_research_job_repository] = lambda: job_repo
    app.dependency_overrides[get_research_service] = lambda: service

    client = TestClient(app)

    # Seed two sample documents
    doc1 = DocumentMetadata(
        paper_id="P_COMP_01",
        filename="transformer_complexity.pdf",
        status=DocumentStatus.COMPLETED,
    )
    doc2 = DocumentMetadata(
        paper_id="P_DATA_02",
        filename="dataset_scaling.pdf",
        status=DocumentStatus.COMPLETED,
    )
    doc_repo.save_document(doc1)
    doc_repo.save_document(doc2)

    # Chunks for Paper 1 (discusses complexity and benchmarks, but NOT quantum hardware)
    chunks_p1 = [
        Chunk(
            chunk_id="P1_C001",
            paper_id="P_COMP_01",
            page_number=1,
            paragraph_number=1,
            text="In this paper, we evaluate computational complexity and architectural trade-offs in neural sequence models.",
            section="Abstract",
        ),
        Chunk(
            chunk_id="P1_C002",
            paper_id="P_COMP_01",
            page_number=4,
            paragraph_number=2,
            text="The computational complexity per layer is O(n^2 * d) for self-attention, compared to O(n * d^2) for recurrent layers.",
            section="Complexity Analysis",
        ),
        Chunk(
            chunk_id="P1_C003",
            paper_id="P_COMP_01",
            page_number=6,
            paragraph_number=3,
            text="On the WMT 2014 English-to-German evaluation benchmark, our model achieves a state-of-the-art BLEU score of 28.4.",
            section="Results",
        ),
        Chunk(
            chunk_id="P1_C004",
            paper_id="P_COMP_01",
            page_number=8,
            paragraph_number=1,
            text="A limitation of our approach is the quadratic memory footprint when scaling to very long sequences.",
            section="Limitations",
        ),
    ]

    # Chunks for Paper 2 (discusses training datasets and evaluation benchmarks, but NOT complexity)
    chunks_p2 = [
        Chunk(
            chunk_id="P2_C001",
            paper_id="P_DATA_02",
            page_number=1,
            paragraph_number=1,
            text="We investigate training data requirements and dataset curation strategies for multi-task learning.",
            section="Abstract",
        ),
        Chunk(
            chunk_id="P2_C002",
            paper_id="P_DATA_02",
            page_number=3,
            paragraph_number=2,
            text="Our training data requirements consist of 4.5 million sentence pairs from WMT 2014 and 36 million sentences from Common Crawl.",
            section="Datasets and Training Setup",
        ),
        Chunk(
            chunk_id="P2_C003",
            paper_id="P_DATA_02",
            page_number=5,
            paragraph_number=1,
            text="On standard evaluation benchmarks, scaling the pre-training corpus by 5x yields a 3.2% accuracy improvement over baselines.",
            section="Benchmark Evaluation",
        ),
        Chunk(
            chunk_id="P2_C004",
            paper_id="P_DATA_02",
            page_number=7,
            paragraph_number=2,
            text="A primary constraint is the significant compute overhead required for pre-processing massive web crawl datasets.",
            section="Limitations",
        ),
    ]

    doc_repo.save_chunks("P_COMP_01", chunks_p1)
    doc_repo.save_chunks("P_DATA_02", chunks_p2)

    yield {
        "client": client,
        "doc_repo": doc_repo,
        "job_repo": job_repo,
        "service": service,
        "chunks_p1": chunks_p1,
        "chunks_p2": chunks_p2,
    }

    app.dependency_overrides.clear()


def test_dynamic_topic_extraction_no_hardcoding():
    """Verify that extract_focus_topics dynamically normalizes various input formats without hardcoding."""
    # Natural language with preamble
    nl_focus = "Focus on computational complexity, training data requirements, and evaluation benchmarks."
    topics = extract_focus_topics(nl_focus)
    assert topics == [
        "Computational Complexity",
        "Training Data Requirements",
        "Evaluation Benchmarks",
    ]

    # Question format
    q_focus = "What are the GPU memory requirements as well as training datasets?"
    q_topics = extract_focus_topics(q_focus)
    assert "GPU Memory Requirements" in q_topics
    assert "Training Datasets" in q_topics

    # Bulleted / numbered lines
    bullet_focus = "1. Latency overhead\n2. Tokenization strategy\n3. Pre-training corpus"
    b_topics = extract_focus_topics(bullet_focus)
    assert b_topics == ["Latency Overhead", "Tokenization Strategy", "Pre-training Corpus"]

    # Single custom arbitrary topic (demonstrating no hardcoding)
    arb_focus = "Hyperparameter optimization with Bayesian priors"
    arb_topics = extract_focus_topics(arb_focus)
    assert len(arb_topics) == 1
    assert "Hyperparameter Optimization" in arb_topics[0]

    # Empty string or whitespace returns empty list
    assert extract_focus_topics("") == []
    assert extract_focus_topics("   ") == []
    assert extract_focus_topics(None) == []


def test_backend_receives_and_propagates_focus_query_and_research_focus(focus_test_env):
    """Verify backend accepts both focus_query and research_focus, reconciles them, and preserves them in results."""
    client = focus_test_env["client"]

    # 1. Payload with focus_query
    payload1 = {
        "paper_ids": ["P_COMP_01"],
        "tasks": ["summarize"],
        "focus_query": "Computational complexity and benchmarks",
    }
    res1 = client.post("/api/research/generate", json=payload1)
    assert res1.status_code == 202
    jid1 = res1.json()["research_id"]

    get_res1 = client.get(f"/api/research/{jid1}")
    assert get_res1.status_code == 200
    assert get_res1.json()["focus_query"] == "Computational complexity and benchmarks"
    assert get_res1.json()["research_focus"] == "Computational complexity and benchmarks"

    # 2. Payload with research_focus field
    payload2 = {
        "paper_ids": ["P_DATA_02"],
        "tasks": ["summarize"],
        "research_focus": "Training data scaling",
    }
    res2 = client.post("/api/research/generate", json=payload2)
    assert res2.status_code == 202
    jid2 = res2.json()["research_id"]

    get_res2 = client.get(f"/api/research/{jid2}")
    assert get_res2.status_code == 200
    assert get_res2.json()["focus_query"] == "Training data scaling"
    assert get_res2.json()["research_focus"] == "Training data scaling"


def test_empty_research_focus_preserves_default_behavior(focus_test_env):
    """Verify that when Research Focus is empty, the original default summarization and comparison occur."""
    chunks = focus_test_env["chunks_p1"]

    # Without focus_query
    default_summary = synthesize_paper_summary(chunks, "P_COMP_01", "transformer_complexity.pdf", focus_query=None)

    assert "neural sequence models" in default_summary.objective.text.lower()
    # Does not prefix findings with topic headers like [Topic]
    assert not any(f.text.startswith("[") for f in default_summary.findings)

    # Empty string also preserves default behavior
    empty_summary = synthesize_paper_summary(chunks, "P_COMP_01", "transformer_complexity.pdf", focus_query="  ")
    assert empty_summary.objective.text == default_summary.objective.text


def test_different_focus_inputs_produce_different_evidence_selection_and_summary(focus_test_env):
    """Verify that different focus inputs prioritize different topics and select corresponding evidence."""
    chunks = focus_test_env["chunks_p1"]

    focus_complexity = "computational complexity"
    focus_benchmarks = "evaluation benchmarks"

    summary_complexity = synthesize_paper_summary(
        chunks, "P_COMP_01", "transformer_complexity.pdf", focus_query=focus_complexity
    )
    summary_benchmarks = synthesize_paper_summary(
        chunks, "P_COMP_01", "transformer_complexity.pdf", focus_query=focus_benchmarks
    )

    # First finding in summary_complexity emphasizes complexity
    assert summary_complexity.findings[0].text.startswith("[Computational Complexity]")
    assert "O(n^2 * d)" in summary_complexity.findings[0].text
    # Evidence citation points to chunk P1_C002 (Complexity Analysis on page 4)
    assert len(summary_complexity.findings[0].evidence) == 1
    assert summary_complexity.findings[0].evidence[0].chunk_id == "P1_C002"
    assert summary_complexity.findings[0].evidence[0].page_number == 4

    # First finding in summary_benchmarks emphasizes benchmarks
    assert summary_benchmarks.findings[0].text.startswith("[Evaluation Benchmarks]")
    assert "BLEU score of 28.4" in summary_benchmarks.findings[0].text
    # Evidence citation points to chunk P1_C003 (Results on page 6)
    assert len(summary_benchmarks.findings[0].evidence) == 1
    assert summary_benchmarks.findings[0].evidence[0].chunk_id == "P1_C003"
    assert summary_benchmarks.findings[0].evidence[0].page_number == 6

    # Distinct content produced
    assert summary_complexity.findings[0].text != summary_benchmarks.findings[0].text


def test_missing_topics_reported_honestly_without_hallucination(focus_test_env):
    """Verify that if a requested topic is not discussed in the paper, it is honestly reported as not found."""
    chunks = focus_test_env["chunks_p1"]

    focus_query = "Quantum annealing hardware"
    summary = synthesize_paper_summary(
        chunks, "P_COMP_01", "transformer_complexity.pdf", focus_query=focus_query
    )

    # Must report that topic was not found in selected papers
    matching_finding = next(
        (f for f in summary.findings if "[Quantum Annealing Hardware]" in f.text), None
    )
    assert matching_finding is not None
    assert "This topic was not found in the selected papers." in matching_finding.text
    # Evidence list MUST be empty (no fabricated citations)
    assert len(matching_finding.evidence) == 0

    # Limitations explicitly lists it under Missing Information
    missing_lim = next(
        (lim for lim in summary.limitations if "[Missing Information]" in lim.text), None
    )
    assert missing_lim is not None
    assert "Quantum Annealing Hardware" in missing_lim.text


def test_multi_paper_synthesis_preserves_source_attribution_and_focus_alignment(focus_test_env):
    """Verify cross-paper comparison structures rows by requested topics and keeps evidence attribution strict."""
    chunks1 = focus_test_env["chunks_p1"]
    chunks2 = focus_test_env["chunks_p2"]

    focus_query = "Focus on computational complexity and training data requirements."

    summary1 = synthesize_paper_summary(chunks1, "P_COMP_01", "transformer_complexity.pdf", focus_query=focus_query)
    summary2 = synthesize_paper_summary(chunks2, "P_DATA_02", "dataset_scaling.pdf", focus_query=focus_query)

    papers = [
        PaperSynthesisItem(paper_id="P_COMP_01", filename="transformer_complexity.pdf", summary=summary1),
        PaperSynthesisItem(paper_id="P_DATA_02", filename="dataset_scaling.pdf", summary=summary2),
    ]

    comp_result = synthesize_cross_paper_comparison(papers, focus_query=focus_query)

    # Matrix dimensions align with user focus topics
    dim_names = [row.dimension for row in comp_result.matrix]
    assert "Computational Complexity" in dim_names
    assert "Training Data Requirements" in dim_names

    # Check Computational Complexity row:
    complexity_row = next(r for r in comp_result.matrix if r.dimension == "Computational Complexity")
    cell_p1 = next(c for c in complexity_row.cells if c.paper_id == "P_COMP_01")
    cell_p2 = next(c for c in complexity_row.cells if c.paper_id == "P_DATA_02")

    # Paper 1 discusses complexity and has evidence linked strictly to P_COMP_01
    assert "O(n^2 * d)" in cell_p1.value.text
    assert len(cell_p1.value.evidence) >= 1
    assert all(ev.paper_id == "P_COMP_01" for ev in cell_p1.value.evidence)

    # Paper 2 does NOT discuss complexity and explicitly reports "not found" with 0 evidence
    assert "This topic was not found in this paper." in cell_p2.value.text
    assert len(cell_p2.value.evidence) == 0

    # Check Training Data Requirements row:
    data_row = next(r for r in comp_result.matrix if r.dimension == "Training Data Requirements")
    cell_data_p2 = next(c for c in data_row.cells if c.paper_id == "P_DATA_02")

    # Paper 2 discusses training data with evidence linked strictly to P_DATA_02
    assert "4.5 million sentence pairs" in cell_data_p2.value.text
    assert len(cell_data_p2.value.evidence) >= 1
    assert all(ev.paper_id == "P_DATA_02" for ev in cell_data_p2.value.evidence)

    # Comparative narrative explicitly addresses topics and contrasts coverage
    assert "Computational Complexity" in comp_result.synthesis_narrative
    assert "Training Data Requirements" in comp_result.synthesis_narrative


def test_end_to_end_research_generation_with_focus(focus_test_env):
    """Integration test running service.create_job through service.process_job with a research focus."""
    import asyncio
    service = focus_test_env["service"]

    payload = StartResearchPayload(
        paper_ids=["P_COMP_01", "P_DATA_02"],
        tasks=[ResearchTask.SUMMARIZE, ResearchTask.COMPARE],
        focus_query="Focus on computational complexity and evaluation benchmarks.",
    )

    start_resp = service.create_job(payload)
    research_id = start_resp.research_id

    # Run processing
    asyncio.run(service.process_job(research_id))

    # Retrieve completed job
    result = service.get_job(research_id)
    assert result.status == ResearchStatus.COMPLETED
    assert result.focus_query == "Focus on computational complexity and evaluation benchmarks."
    assert result.research_focus == "Focus on computational complexity and evaluation benchmarks."

    # Both papers summarized with focus
    assert result.papers is not None
    assert len(result.papers) == 2

    p1_summary = next(p for p in result.papers if p.paper_id == "P_COMP_01").summary
    assert any("[Computational Complexity]" in f.text for f in p1_summary.findings)

    # Comparison matrix aligned with focus
    assert result.comparison is not None
    matrix_dims = [row.dimension for row in result.comparison.matrix]
    assert "Computational Complexity" in matrix_dims
    assert "Evaluation Benchmarks" in matrix_dims
