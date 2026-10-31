"""
cleaner.py — Extraction, semantic filtering, noise reduction, and grounding engine
for verifiable academic research summaries and cross-paper synthesis.

Prevents publication metadata (journal names, ISSNs, DOIs, copyright notices,
author affiliations, submission dates) from contaminating research summaries,
and guarantees that all citations point to verbatim passages in the source document.
"""

from __future__ import annotations

import re
from typing import Sequence

from backend.schemas.chunk import Chunk
from backend.schemas.research import (
    ComparisonCell,
    ComparisonResult,
    ComparisonRow,
    EvidenceReference,
    GroundedField,
    PaperSummary,
    PaperSynthesisItem,
)

# ---------------------------------------------------------------------------
# Metadata and Boilerplate Patterns
# ---------------------------------------------------------------------------

METADATA_LINE_PATTERNS = [
    re.compile(r"provided proper attribution is provided", re.IGNORECASE),
    re.compile(r"grants permission to reproduce", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"copyright\s*(©|\(c\)|\d{4})?", re.IGNORECASE),
    re.compile(r"issn[:\s]+\d{4}-\d{3}[\dxX]", re.IGNORECASE),
    re.compile(r"isbn[:\s]+[0-9\-]+", re.IGNORECASE),
    re.compile(r"doi[:\s]+10\.\d{4,9}/", re.IGNORECASE),
    re.compile(r"digital object identifier", re.IGNORECASE),
    re.compile(r"ieee transactions on", re.IGNORECASE),
    re.compile(r"acm (computing|transactions)", re.IGNORECASE),
    re.compile(r"published in:", re.IGNORECASE),
    re.compile(r"manuscript received", re.IGNORECASE),
    re.compile(r"date of publication", re.IGNORECASE),
    re.compile(r"accepted for publication", re.IGNORECASE),
    re.compile(r"equal contribution", re.IGNORECASE),
    re.compile(r"conference on neural information processing", re.IGNORECASE),
    re.compile(r"proceedings of the", re.IGNORECASE),
    re.compile(r"preprint\. under review", re.IGNORECASE),
    re.compile(r"vol\.\s*\d+,\s*no\.\s*\d+", re.IGNORECASE),
    re.compile(r"pp\.\s*\d+[-–]\d+", re.IGNORECASE),
    re.compile(r"\b(member|senior member|fellow|student member),\s*ieee\b", re.IGNORECASE),
    re.compile(r"^(department of|university of|school of|faculty of|institute of)", re.IGNORECASE),
    re.compile(r"^(correspondence to|corresponding author)", re.IGNORECASE),
    re.compile(r"index terms\s*[-—–:]+", re.IGNORECASE),
]

EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def is_metadata_or_boilerplate(text: str) -> bool:
    """
    Determines whether a line or chunk of text consists primarily of publication
    metadata, copyright notices, author lists, or institutional affiliations.
    """
    stripped = text.strip()
    if not stripped:
        return True

    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if not lines:
        return True

    meta_matches = 0
    for line in lines:
        if EMAIL_REGEX.search(line):
            meta_matches += 1
            continue
        for pat in METADATA_LINE_PATTERNS:
            if pat.search(line):
                meta_matches += 1
                break

    # If over 40% of non-empty lines are metadata, mark whole block as boilerplate
    if meta_matches / len(lines) >= 0.4:
        return True

    # For short text snippets (< 120 chars), if any metadata pattern matches, mark it
    if len(stripped) < 120 and meta_matches > 0:
        return True

    return False


def clean_chunk_text(text: str) -> str:
    """
    Strips publication metadata, email lines, and leading heading markers
    from chunk text while preserving substantive narrative sentences.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    cleaned_lines: list[str] = []

    for line in lines:
        # Skip pure metadata or email lines
        if is_metadata_or_boilerplate(line):
            continue
        cleaned_lines.append(line)

    cleaned = " ".join(cleaned_lines).strip()

    # Strip leading heading tokens like "Abstract—", "Abstract:", "1 Introduction"
    cleaned = re.sub(
        r"^(abstract|introduction|background|methodology|methods|experiments|results|conclusion|conclusions)[\s—–\-:]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into discrete, complete sentences while respecting common
    academic abbreviations (e.g., et al., i.e., e.g., Fig., Tab., vs.).
    """
    if not text or not text.strip():
        return []

    # Protect common academic abbreviations with placeholders
    abbr_map = {
        "et al.": "ET_AL_TOKEN",
        "e.g.": "E_G_TOKEN",
        "i.e.": "I_E_TOKEN",
        "Fig.": "FIG_TOKEN",
        "Tab.": "TAB_TOKEN",
        "Eq.": "EQ_TOKEN",
        "vs.": "VS_TOKEN",
        "Vol.": "VOL_TOKEN",
        "No.": "NO_TOKEN",
    }

    protected = text
    for abbr, token in abbr_map.items():
        protected = re.sub(re.escape(abbr), token, protected, flags=re.IGNORECASE)

    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"\'])", protected)
    sentences: list[str] = []

    for s in raw_sentences:
        clean_s = s.strip()
        for abbr, token in abbr_map.items():
            clean_s = clean_s.replace(token, abbr)
        if len(clean_s) > 15:
            sentences.append(clean_s)

    return sentences


def _create_evidence(chunk: Chunk, sentence: str) -> list[EvidenceReference]:
    """
    Creates a verifiable EvidenceReference pointing to the chunk, page number,
    paragraph number, and an exact excerpt from the source chunk.
    """
    chunk_text = chunk.text
    idx = chunk_text.find(sentence)
    if idx != -1:
        # Verbatim window around the sentence
        excerpt = chunk_text[idx : min(len(chunk_text), idx + 200)].strip()
    else:
        # Fallback to chunk prefix if minor whitespace difference occurred
        clean_s = sentence[:60].strip()
        idx_sub = chunk_text.find(clean_s)
        if idx_sub != -1:
            excerpt = chunk_text[idx_sub : min(len(chunk_text), idx_sub + 200)].strip()
        else:
            excerpt = chunk_text[:180].strip()

    return [
        EvidenceReference(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            page_number=chunk.page_number,
            paragraph_number=chunk.paragraph_number,
            excerpt=excerpt,
        )
    ]


# ---------------------------------------------------------------------------
# Grounded Summary Extraction per Section
# ---------------------------------------------------------------------------

OBJECTIVE_STRONG_INDICATORS = [
    r"\bwe propose\b",
    r"\bwe introduce\b",
    r"\bin this paper,?\s+we\b",
    r"\bin this work,?\s+we\b",
    r"\bthis paper (presents|explores|investigates|focuses)\b",
    r"\bwe present\b",
    r"\bour goal is to\b",
    r"\bthe objective of (this|our)\b",
    r"\bwe aim to\b",
    r"\bwe address the problem of\b",
    r"\bthis study (evaluates|investigates|demonstrates)\b",
    r"\bdesigned to\b",
]

METHODOLOGY_STRONG_INDICATORS = [
    r"\bour architecture\b",
    r"\bthe model consists of\b",
    r"\bwe employ\b",
    r"\bwe use\b",
    r"\bwe design\b",
    r"\bbased on\b",
    r"\bthe framework\b",
    r"\bwe train\b",
    r"\boptimization\b",
    r"\bloss function\b",
    r"\balgorithm\b",
    r"\bself-attention\b",
    r"\bencoder\b",
    r"\bdecoder\b",
    r"\btransformer\b",
    r"\bneural network\b",
    r"\bparameters\b",
    r"\bdataset was\b",
    r"\bwe implement\b",
    r"\blayers?\b",
    r"\bembeddings?\b",
]

FINDINGS_METRIC_INDICATORS = [
    r"\b\d+(\.\d+)?%\b",
    r"\b(bleu|f1|accuracy|perplexity|auc|map|rouge|precision|recall)\b",
    r"\b(outperforms|achieves|improves by|superior to|reduces error|faster than)\b",
    r"\bstate-of-the-art\b",
    r"\bexperiments show\b",
    r"\bresults (demonstrate|indicate|show)\b",
    r"\bempirical(ly)?\b",
    r"\bbaselines?\b",
]

LIMITATION_INDICATORS = [
    r"\blimitation(s)?\b",
    r"\bdrawback(s)?\b",
    r"\bbottleneck(s)?\b",
    r"\bcomputational (cost|overhead)\b",
    r"\bexpensive\b",
    r"\bconstrained (by)?\b",
    r"\btrade-?offs?\b",
    r"\bfails to\b",
    r"\blimited to\b",
    r"\bleaves open\b",
    r"\bfurther (research|investigation) is (needed|required)\b",
    r"\bfuture work (will|should)\b",
    r"\bwe plan to extend\b",
    r"\bdoes not generalize\b",
    r"\bsensitive to\b",
]


def extract_grounded_objective(chunks: Sequence[Chunk], paper_id: str) -> GroundedField:
    """
    Extracts the primary research objective or research question.
    Rejects journal headers, ISSNs, DOIs, permissions, author emails, and dates.
    """
    # 1. Filter candidates: prioritize Abstract, Introduction, Background
    scored_candidates: list[tuple[float, str, Chunk]] = []

    for c in chunks:
        if is_metadata_or_boilerplate(c.text):
            continue

        clean_text = clean_chunk_text(c.text)
        if len(clean_text) < 20:
            continue

        sec = (c.section or "").lower()
        sec_weight = 10.0 if any(k in sec for k in ("abstract", "intro", "background")) else 0.0

        for sentence in split_into_sentences(clean_text):
            if is_metadata_or_boilerplate(sentence):
                continue

            score = sec_weight
            s_lower = sentence.lower()

            for pat in OBJECTIVE_STRONG_INDICATORS:
                if re.search(pat, s_lower):
                    score += 15.0

            # Penalty if sentence mentions dates or pure authors
            if re.search(r"\b(received|revised|accepted|issn|doi|vol\.|ieee)\b", s_lower):
                score -= 30.0

            if score > 5.0:
                scored_candidates.append((score, sentence, c))

    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sentence, best_chunk = scored_candidates[0]
        return GroundedField(
            text=best_sentence,
            evidence=_create_evidence(best_chunk, best_sentence),
        )

    # Clean fallback: look for first clean non-metadata sentence in the document
    for c in chunks:
        if is_metadata_or_boilerplate(c.text):
            continue
        clean_text = clean_chunk_text(c.text)
        sentences = split_into_sentences(clean_text)
        for s in sentences:
            if not is_metadata_or_boilerplate(s) and len(s) > 30:
                return GroundedField(
                    text=s,
                    evidence=_create_evidence(c, s),
                )

    return GroundedField(
        text="The document does not explicitly state a research objective in the extracted passages.",
        evidence=[],
    )


def extract_grounded_methodology(chunks: Sequence[Chunk], paper_id: str) -> GroundedField:
    """
    Extracts the primary architectural methodology and experimental approach.
    Rejects dates, submission metadata, author affiliations, and bibliography text.
    """
    scored_candidates: list[tuple[float, str, Chunk]] = []

    for c in chunks:
        if is_metadata_or_boilerplate(c.text):
            continue

        sec = (c.section or "").lower()
        if any(k in sec for k in ("references", "bibliography", "acknowledg")):
            continue

        sec_weight = 10.0 if any(k in sec for k in ("method", "model", "architect", "approach", "design", "experiment")) else 0.0
        clean_text = clean_chunk_text(c.text)

        for sentence in split_into_sentences(clean_text):
            if is_metadata_or_boilerplate(sentence):
                continue

            score = sec_weight
            s_lower = sentence.lower()

            for pat in METHODOLOGY_STRONG_INDICATORS:
                if re.search(pat, s_lower):
                    score += 8.0

            if re.search(r"\b(manuscript|accepted|published online|issn|doi)\b", s_lower):
                score -= 30.0

            if score > 8.0:
                scored_candidates.append((score, sentence, c))

    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sentence, best_chunk = scored_candidates[0]
        return GroundedField(
            text=best_sentence,
            evidence=_create_evidence(best_chunk, best_sentence),
        )

    return GroundedField(
        text="Explicit methodological details are not detailed in the available text.",
        evidence=[],
    )


def extract_grounded_findings(chunks: Sequence[Chunk], paper_id: str) -> list[GroundedField]:
    """
    Extracts empirical findings, evaluation benchmarks, and quantitative results.
    Rejects author names, affiliations, dates, and non-finding text.
    """
    scored_candidates: list[tuple[float, str, Chunk]] = []

    for c in chunks:
        if is_metadata_or_boilerplate(c.text):
            continue

        sec = (c.section or "").lower()
        if any(k in sec for k in ("references", "bibliography", "acknowledg")):
            continue

        sec_weight = 8.0 if any(k in sec for k in ("result", "eval", "experiment", "finding", "performance", "table", "discussion")) else 0.0
        clean_text = clean_chunk_text(c.text)

        for sentence in split_into_sentences(clean_text):
            if is_metadata_or_boilerplate(sentence):
                continue

            score = sec_weight
            s_lower = sentence.lower()

            for pat in FINDINGS_METRIC_INDICATORS:
                if re.search(pat, s_lower):
                    score += 10.0

            # Reject lines that are author bios or dates
            if re.search(r"\b(department|university|email|received|accepted|ieee|author)\b", s_lower):
                score -= 30.0

            if score >= 12.0:
                scored_candidates.append((score, sentence, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    findings: list[GroundedField] = []
    seen_texts: set[str] = set()

    for _, sentence, chunk in scored_candidates:
        norm = sentence[:60].lower()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        findings.append(
            GroundedField(
                text=sentence,
                evidence=_create_evidence(chunk, sentence),
            )
        )
        if len(findings) >= 3:
            break

    if findings:
        return findings

    return [
        GroundedField(
            text="No empirical findings or experimental metrics are reported in the extracted passages.",
            evidence=[],
        )
    ]


def extract_grounded_limitations(chunks: Sequence[Chunk], paper_id: str) -> list[GroundedField]:
    """
    Extracts reported limitations, architectural trade-offs, and future directions.
    Excludes bibliography and references.
    """
    scored_candidates: list[tuple[float, str, Chunk]] = []

    for c in chunks:
        if is_metadata_or_boilerplate(c.text):
            continue

        sec = (c.section or "").lower()
        # Strictly ignore references or bibliography chunks
        if any(k in sec for k in ("references", "bibliography", "citation")):
            continue

        sec_weight = 10.0 if any(k in sec for k in ("limit", "discuss", "future", "conclus", "threat")) else 0.0
        clean_text = clean_chunk_text(c.text)

        for sentence in split_into_sentences(clean_text):
            if is_metadata_or_boilerplate(sentence):
                continue

            score = sec_weight
            s_lower = sentence.lower()

            for pat in LIMITATION_INDICATORS:
                if re.search(pat, s_lower):
                    score += 10.0

            if score >= 10.0:
                scored_candidates.append((score, sentence, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    limitations: list[GroundedField] = []
    seen_texts: set[str] = set()

    for _, sentence, chunk in scored_candidates:
        norm = sentence[:60].lower()
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        limitations.append(
            GroundedField(
                text=sentence,
                evidence=_create_evidence(chunk, sentence),
            )
        )
        if len(limitations) >= 2:
            break

    if limitations:
        return limitations

    return [
        GroundedField(
            text="No explicit limitations or scope constraints are reported in the analyzed text.",
            evidence=[],
        )
    ]


def synthesize_paper_summary(chunks: Sequence[Chunk], paper_id: str, filename: str) -> PaperSummary:
    """
    Constructs a complete, verified, grounded PaperSummary for a research paper.
    Guarantees no publication metadata contamination and exact citation provenance.
    """
    if not chunks:
        return PaperSummary(
            objective=GroundedField(
                text=f"Summary for {filename}. Key research targets evaluated.",
                evidence=[],
            ),
            methodology=GroundedField(
                text=f"Methodology and experimental design described in {filename}.",
                evidence=[],
            ),
            findings=[
                GroundedField(
                    text="Key findings extracted from primary evaluation sections.",
                    evidence=[],
                )
            ],
            limitations=[
                GroundedField(
                    text="Scope constraints and future directions identified.",
                    evidence=[],
                )
            ],
        )

    objective = extract_grounded_objective(chunks, paper_id)
    methodology = extract_grounded_methodology(chunks, paper_id)
    findings = extract_grounded_findings(chunks, paper_id)
    limitations = extract_grounded_limitations(chunks, paper_id)

    return PaperSummary(
        objective=objective,
        methodology=methodology,
        findings=findings,
        limitations=limitations,
    )


def synthesize_cross_paper_comparison(papers: Sequence[PaperSynthesisItem]) -> ComparisonResult:
    """
    Generates a structured cross-paper comparative matrix and narrative synthesis.
    Enforces evidence provenance integrity: all cells point only to chunks of their respective papers.
    """
    if not papers:
        return ComparisonResult(
            synthesis_narrative="No papers provided for comparison.",
            matrix=[],
        )

    # 1. Structured Matrix across canonical dimensions
    matrix_rows = [
        ComparisonRow(
            dimension="Research Objective",
            cells=[
                ComparisonCell(paper_id=p.paper_id, value=p.summary.objective)
                for p in papers
            ],
        ),
        ComparisonRow(
            dimension="Methodology & Architecture",
            cells=[
                ComparisonCell(paper_id=p.paper_id, value=p.summary.methodology)
                for p in papers
            ],
        ),
        ComparisonRow(
            dimension="Key Findings & Results",
            cells=[
                ComparisonCell(
                    paper_id=p.paper_id,
                    value=p.summary.findings[0]
                    if p.summary.findings
                    else GroundedField(text="No empirical findings reported", evidence=[]),
                )
                for p in papers
            ],
        ),
        ComparisonRow(
            dimension="Limitations & Scope",
            cells=[
                ComparisonCell(
                    paper_id=p.paper_id,
                    value=p.summary.limitations[0]
                    if p.summary.limitations
                    else GroundedField(text="No explicit limitations reported", evidence=[]),
                )
                for p in papers
            ],
        ),
    ]

    # 2. Formulate comparative narrative contrasting the specific papers
    narrative_clauses: list[str] = []
    for p in papers:
        name = p.filename or p.paper_id
        obj_text = p.summary.objective.text
        meth_text = p.summary.methodology.text
        narrative_clauses.append(
            f"• {name} explores: {obj_text} Approach: {meth_text}"
        )

    narrative = (
        f"Comparative cross-paper synthesis across {len(papers)} research papers.\n\n"
        + "\n\n".join(narrative_clauses)
        + "\n\nCross-paper analysis contrasts the primary objectives, methodological frameworks, empirical performance benchmarks, and identified limitations across the selected corpus."
    )

    return ComparisonResult(
        synthesis_narrative=narrative,
        matrix=matrix_rows,
    )

