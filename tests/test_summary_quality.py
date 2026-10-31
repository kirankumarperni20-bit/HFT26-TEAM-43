"""
test_summary_quality.py — Unit and integration tests for summary relevance,
metadata contamination prevention, citation mapping integrity, and missing evidence handling.
"""

import asyncio
import pytest
from backend.schemas.chunk import Chunk
from backend.schemas.research import (
    GroundedField,
    PaperSynthesisItem,
    ResearchTask,
    StartResearchPayload,
)
from backend.research.cleaner import (
    is_metadata_or_boilerplate,
    clean_chunk_text,
    extract_grounded_objective,
    extract_grounded_methodology,
    extract_grounded_findings,
    extract_grounded_limitations,
    synthesize_paper_summary,
    synthesize_cross_paper_comparison,
)
from backend.research.service import ResearchJobService
from backend.ingestion.repository import InMemoryDocumentRepository
from backend.research.repository import InMemoryResearchJobRepository
from backend.schemas.document import DocumentMetadata, DocumentStatus


def test_metadata_and_boilerplate_detection():
    """Verify detection of various publication metadata, journal banners, and author lines."""
    contaminated_lines = [
        "IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 32, NO. 4, APRIL 2021",
        "ISSN: 1045-9227, DOI: 10.1109/TNNLS.2020.1234567",
        "Manuscript received October 15, 2020; revised February 8, 2021; accepted March 1, 2021.",
        "John Doe, Member, IEEE, and Jane Smith, Fellow, IEEE",
        "Department of Computer Science, Stanford University (email: {jdoe, jsmith}@stanford.edu)",
        "Provided proper attribution is provided, Google hereby grants permission to reproduce tables",
        "All rights reserved. Copyright © 2022 Elsevier B.V.",
        "Preprint. Under review at ICLR 2024.",
        "Index Terms—Deep learning, natural language processing, transformers.",
    ]
    for line in contaminated_lines:
        assert is_metadata_or_boilerplate(line) is True, f"Failed to flag metadata: {line}"

    substantive_lines = [
        "We propose a novel self-attention mechanism that eliminates recurrence entirely.",
        "Our architecture uses stacked multi-head attention layers with residual connections.",
        "On the WMT 2014 English-to-German task, the model achieves a BLEU score of 28.4.",
        "A major limitation of our approach is the quadratic memory complexity with sequence length.",
    ]
    for line in substantive_lines:
        assert is_metadata_or_boilerplate(line) is False, f"Falsely flagged substantive text: {line}"


def test_rejection_of_metadata_in_research_objective():
    """Verify that journal names, ISSNs, and copyright do not contaminate the research objective."""
    contaminated_chunks = [
        Chunk(
            chunk_id="P1_C001",
            paper_id="P1",
            page_number=1,
            paragraph_number=1,
            text=(
                "ACM Computing Surveys, Vol. 55, No. 2, Article 35. Publication date: March 2023.\n"
                "ISSN: 0360-0300, DOI: 10.1145/1234567.8901234\n"
                "Copyright © 2023 by the Association for Computing Machinery, Inc. All rights reserved.\n"
                "Author affiliations: University of Oxford, UK. Email: contact@ox.ac.uk"
            ),
            section=None,
        ),
        Chunk(
            chunk_id="P1_C002",
            paper_id="P1",
            page_number=1,
            paragraph_number=2,
            text=(
                "Abstract—In this paper, we propose a scalable sparse attention architecture "
                "designed to reduce the quadratic memory footprint of self-attention networks."
            ),
            section="Abstract",
        ),
    ]

    objective = extract_grounded_objective(contaminated_chunks, "P1")

    # Objective MUST be the research proposal, not the ACM journal banner or ISSN
    assert "ACM Computing Surveys" not in objective.text
    assert "ISSN" not in objective.text
    assert "DOI" not in objective.text
    assert "Copyright" not in objective.text
    assert "we propose a scalable sparse attention architecture" in objective.text

    # Citation must point to C002, NOT C001
    assert len(objective.evidence) == 1
    assert objective.evidence[0].chunk_id == "P1_C002"
    assert objective.evidence[0].page_number == 1
    assert "we propose a scalable sparse attention" in objective.evidence[0].excerpt


def test_rejection_of_dates_in_methodology():
    """Verify methodology extracts architectural methods rather than publication dates or metadata."""
    chunks = [
        Chunk(
            chunk_id="P2_C001",
            paper_id="P2",
            page_number=1,
            paragraph_number=1,
            text=(
                "Manuscript received January 12, 2022; revised June 18, 2022; accepted August 4, 2022.\n"
                "Date of publication September 1, 2022; date of current version October 15, 2022."
            ),
            section=None,
        ),
        Chunk(
            chunk_id="P2_C002",
            paper_id="P2",
            page_number=2,
            paragraph_number=1,
            text=(
                "Our architecture employs a 12-layer bidirectional Transformer encoder "
                "with 768 hidden dimensions and 12 attention heads trained using masked language modeling."
            ),
            section="Model Architecture",
        ),
    ]

    methodology = extract_grounded_methodology(chunks, "P2")

    assert "Manuscript received" not in methodology.text
    assert "Date of publication" not in methodology.text
    assert "12-layer bidirectional Transformer encoder" in methodology.text
    assert len(methodology.evidence) == 1
    assert methodology.evidence[0].chunk_id == "P2_C002"


def test_rejection_of_authors_and_affiliations_in_key_findings():
    """Verify key findings extracts quantitative metrics rather than author names or bios."""
    chunks = [
        Chunk(
            chunk_id="P3_C001",
            paper_id="P3",
            page_number=1,
            paragraph_number=2,
            text=(
                "Alice Johnson is an Assistant Professor in the Department of Computer Science "
                "at Princeton University. Her research interests include deep learning and NLP. "
                "Email: alice@princeton.edu"
            ),
            section="Author Information",
        ),
        Chunk(
            chunk_id="P3_C002",
            paper_id="P3",
            page_number=6,
            paragraph_number=3,
            text=(
                "Experimental results show our model achieves 88.5% top-1 accuracy on ImageNet, "
                "outperforming the ResNet-50 baseline by 4.2% while reducing latency by 35%."
            ),
            section="Experimental Results",
        ),
    ]

    findings = extract_grounded_findings(chunks, "P3")

    assert len(findings) >= 1
    top_finding = findings[0]
    assert "Alice Johnson" not in top_finding.text
    assert "Princeton University" not in top_finding.text
    assert "88.5% top-1 accuracy on ImageNet" in top_finding.text
    assert "outperforming the ResNet-50 baseline" in top_finding.text
    assert top_finding.evidence[0].chunk_id == "P3_C002"


def test_citation_mapping_verbatim_provenance():
    """Verify that every citation contains an excerpt that is an exact verbatim substring of the source chunk."""
    chunk_text = (
        "We evaluate our system on the GLUE benchmark. The proposed model achieves an average score of 82.1, "
        "which improves by 3.5 points over previous BERT models. Furthermore, inference speed is increased by 2.1x."
    )
    chunk = Chunk(
        chunk_id="P4_C005",
        paper_id="P4",
        page_number=5,
        paragraph_number=2,
        text=chunk_text,
        section="Results",
    )

    findings = extract_grounded_findings([chunk], "P4")
    assert len(findings) > 0
    for f in findings:
        for ev in f.evidence:
            assert ev.chunk_id == "P4_C005"
            assert ev.paper_id == "P4"
            assert ev.page_number == 5
            assert ev.paragraph_number == 2
            # The excerpt MUST be a verbatim substring of chunk_text
            assert ev.excerpt in chunk_text


def test_missing_evidence_returns_honest_unavailability():
    """Verify that when evidence is missing, honest placeholders are returned with empty evidence arrays."""
    # Chunk containing generic background but no explicit methodology, findings, or limitations
    minimal_chunk = Chunk(
        chunk_id="P5_C001",
        paper_id="P5",
        page_number=1,
        paragraph_number=1,
        text="Information retrieval has undergone rapid evolution over recent decades.",
        section="Background",
    )

    methodology = extract_grounded_methodology([minimal_chunk], "P5")
    assert "Explicit methodological details are not detailed" in methodology.text
    assert methodology.evidence == []

    findings = extract_grounded_findings([minimal_chunk], "P5")
    assert len(findings) == 1
    assert "No empirical findings or experimental metrics are reported" in findings[0].text
    assert findings[0].evidence == []

    limitations = extract_grounded_limitations([minimal_chunk], "P5")
    assert len(limitations) == 1
    assert "No explicit limitations or scope constraints are reported" in limitations[0].text
    assert limitations[0].evidence == []


def test_cross_paper_comparison_evidence_isolation():
    """Verify that comparison matrix cells strictly preserve evidence provenance per paper."""
    p1_chunks = [
        Chunk(
            chunk_id="PA_C001",
            paper_id="PA",
            page_number=1,
            paragraph_number=1,
            text="In this paper, we propose Paper A for dense vector search.",
            section="Abstract",
        ),
        Chunk(
            chunk_id="PA_C002",
            paper_id="PA",
            page_number=3,
            paragraph_number=1,
            text="Our architecture employs hierarchical HNSW indexing over 768-dim embeddings.",
            section="Methods",
        ),
        Chunk(
            chunk_id="PA_C003",
            paper_id="PA",
            page_number=5,
            paragraph_number=1,
            text="Results show Paper A achieves 95.2% recall@10 on MS MARCO.",
            section="Results",
        ),
    ]

    p2_chunks = [
        Chunk(
            chunk_id="PB_C001",
            paper_id="PB",
            page_number=1,
            paragraph_number=1,
            text="We introduce Paper B, a sparse lexical inverted index optimization.",
            section="Abstract",
        ),
        Chunk(
            chunk_id="PB_C002",
            paper_id="PB",
            page_number=3,
            paragraph_number=1,
            text="The framework uses BM25 with dynamic block-max WAND pruning.",
            section="Methods",
        ),
        Chunk(
            chunk_id="PB_C003",
            paper_id="PB",
            page_number=5,
            paragraph_number=1,
            text="Paper B achieves 15ms latency per query, outperforming Lucene by 40%.",
            section="Results",
        ),
    ]

    summary_a = synthesize_paper_summary(p1_chunks, "PA", "paper_a.pdf")
    summary_b = synthesize_paper_summary(p2_chunks, "PB", "paper_b.pdf")

    item_a = PaperSynthesisItem(paper_id="PA", filename="paper_a.pdf", summary=summary_a)
    item_b = PaperSynthesisItem(paper_id="PB", filename="paper_b.pdf", summary=summary_b)

    comparison = synthesize_cross_paper_comparison([item_a, item_b])

    assert len(comparison.matrix) == 4
    for row in comparison.matrix:
        assert len(row.cells) == 2
        for cell in row.cells:
            # Evidence in cell PA must ONLY reference PA
            if cell.paper_id == "PA":
                for ev in cell.value.evidence:
                    assert ev.paper_id == "PA"
                    assert ev.chunk_id.startswith("PA_")
            # Evidence in cell PB must ONLY reference PB
            elif cell.paper_id == "PB":
                for ev in cell.value.evidence:
                    assert ev.paper_id == "PB"
                    assert ev.chunk_id.startswith("PB_")

    assert "paper_a.pdf" in comparison.synthesis_narrative
    assert "paper_b.pdf" in comparison.synthesis_narrative


def test_end_to_end_research_service_job_processing():
    """Verify that ResearchJobService.process_job completes with clean, non-contaminated results and is_claim_verified=False."""
    doc_repo = InMemoryDocumentRepository()
    job_repo = InMemoryResearchJobRepository()
    service = ResearchJobService(job_repository=job_repo, document_repository=doc_repo)

    doc = DocumentMetadata(paper_id="P_E2E", filename="real_paper.pdf", status=DocumentStatus.COMPLETED)
    doc_repo.save_document(doc)

    chunks = [
        Chunk(
            chunk_id="P_E2E_C001",
            paper_id="P_E2E",
            page_number=1,
            paragraph_number=1,
            text="IEEE TRANSACTIONS ON COMPUTING. ISSN: 1234-5678, DOI: 10.1109/TC.2023.1\nAuthor: Dr. Test. Email: test@ieee.org",
            section=None,
        ),
        Chunk(
            chunk_id="P_E2E_C002",
            paper_id="P_E2E",
            page_number=1,
            paragraph_number=2,
            text="In this paper, we propose an energy-efficient neuromorphic processor.",
            section="Abstract",
        ),
        Chunk(
            chunk_id="P_E2E_C003",
            paper_id="P_E2E",
            page_number=2,
            paragraph_number=1,
            text="Our architecture incorporates 256 spiking neurons with asynchronous event-driven routing.",
            section="Hardware Architecture",
        ),
        Chunk(
            chunk_id="P_E2E_C004",
            paper_id="P_E2E",
            page_number=4,
            paragraph_number=1,
            text="Measurements show the processor achieves 98.2% classification accuracy consuming only 1.4mW.",
            section="Experimental Results",
        ),
    ]
    doc_repo.save_chunks("P_E2E", chunks)

    payload = StartResearchPayload(
        paper_ids=["P_E2E"],
        tasks=[ResearchTask.SUMMARIZE],
    )
    start_resp = service.create_job(payload)

    # Process job synchronously via asyncio.run
    asyncio.run(service.process_job(start_resp.research_id))

    result = service.get_job(start_resp.research_id)
    assert result.status.value == "completed"
    # Claim verification is not claimed true unless specifically verified
    assert result.is_claim_verified is False

    assert result.papers is not None
    assert len(result.papers) == 1
    paper_summary = result.papers[0].summary

    # Ensure no IEEE metadata leaked into summary fields
    assert "IEEE TRANSACTIONS" not in paper_summary.objective.text
    assert "ISSN" not in paper_summary.objective.text
    assert "energy-efficient neuromorphic processor" in paper_summary.objective.text
    assert paper_summary.objective.evidence[0].chunk_id == "P_E2E_C002"

    assert "spiking neurons" in paper_summary.methodology.text
    assert paper_summary.methodology.evidence[0].chunk_id == "P_E2E_C003"

    assert "98.2% classification accuracy" in paper_summary.findings[0].text
    assert paper_summary.findings[0].evidence[0].chunk_id == "P_E2E_C004"
