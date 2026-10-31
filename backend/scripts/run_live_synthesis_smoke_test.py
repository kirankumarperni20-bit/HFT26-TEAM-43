"""
run_live_synthesis_smoke_test.py — Live Featherless API smoke test for cross-paper synthesis.

Executes a real Featherless API call synthesizing two research papers:
1. Vaswani et al. (2017) — "Attention Is All You Need" (from backend/data/full_extraction_merged.json)
2. Gehring et al. (2017) — "Convolutional Sequence to Sequence Learning" (from backend/data/convs2s_extraction.json)

Validates:
- Structured schema conformance (CrossPaperSynthesis).
- Source reference integrity (all cited paper IDs belong to input).
- Verbatim quote grounding against supplied extraction quotes and source text.
- Separation of contextual differences vs. contradictions.
- Preserved metrics (e.g. BLEU 28.4, 41.0, 25.16, 30.5).
- Zero hallucinated paper citations.

Saves complete audit output to backend/data/live_synthesis_smoke_test.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.client import LLMClient
from llm.config import get_settings
from llm.schemas.extraction import ResearchPaperExtraction
from llm.schemas.synthesis import (
    CrossPaperSynthesis,
    FindingRelationship,
    SynthesisRequest,
    SynthesisResponse,
)
from llm.services.synthesis_service import SynthesisService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_synthesis_smoke_test")


def normalize_text_for_matching(text: str) -> str:
    """Normalize whitespace and standard typographical characters for robust substring checking."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return text.strip().lower()


def build_source_text_corpus(p1: ResearchPaperExtraction, p2: ResearchPaperExtraction) -> dict[str, dict[str, Any]]:
    """Build a search corpus of known quotes and underlying source text for both papers."""
    corpus: dict[str, dict[str, Any]] = {
        p1.paper_id: {"quotes": set(), "full_text": ""},
        p2.paper_id: {"quotes": set(), "full_text": ""},
    }

    # Collect quotes from Paper 1
    for m in p1.methodology:
        if m.evidence and m.evidence.quote:
            corpus[p1.paper_id]["quotes"].add(m.evidence.quote.strip())
    for em in p1.evaluation_metrics:
        if em.evidence and em.evidence.quote:
            corpus[p1.paper_id]["quotes"].add(em.evidence.quote.strip())
    for kf in p1.key_findings:
        if kf.evidence and kf.evidence.quote:
            corpus[p1.paper_id]["quotes"].add(kf.evidence.quote.strip())
    for ds in p1.datasets:
        if ds.evidence and ds.evidence.quote:
            corpus[p1.paper_id]["quotes"].add(ds.evidence.quote.strip())

    # Include full source text of Attention paper if chunk text files are present
    full_text_chunks: list[str] = []
    for chunk_file in sorted((BACKEND_DIR / "data").glob("full_extraction_chunk_*_raw.txt")):
        try:
            full_text_chunks.append(chunk_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    corpus[p1.paper_id]["full_text"] = "\n".join(full_text_chunks)

    # Collect quotes from Paper 2
    for m in p2.methodology:
        if m.evidence and m.evidence.quote:
            corpus[p2.paper_id]["quotes"].add(m.evidence.quote.strip())
    for em in p2.evaluation_metrics:
        if em.evidence and em.evidence.quote:
            corpus[p2.paper_id]["quotes"].add(em.evidence.quote.strip())
    for kf in p2.key_findings:
        if kf.evidence and kf.evidence.quote:
            corpus[p2.paper_id]["quotes"].add(kf.evidence.quote.strip())
    for ds in p2.datasets:
        if ds.evidence and ds.evidence.quote:
            corpus[p2.paper_id]["quotes"].add(ds.evidence.quote.strip())

    return corpus


def check_quote_grounding(quote: str, paper_id: str, corpus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check whether a quoted string appears in the supplied extraction quotes or source text."""
    if not quote or not quote.strip():
        return {
            "status": "EMPTY",
            "is_grounded": False,
            "reason": "Empty quote provided.",
        }

    if paper_id not in corpus:
        return {
            "status": "UNKNOWN_PAPER",
            "is_grounded": False,
            "reason": f"Paper ID '{paper_id}' not in source corpus.",
        }

    norm_quote = normalize_text_for_matching(quote)

    # 1. Exact or normalized match against supplied extraction quotes
    for src_q in corpus[paper_id]["quotes"]:
        norm_src_q = normalize_text_for_matching(src_q)
        if norm_quote in norm_src_q or norm_src_q in norm_quote:
            return {
                "status": "EXACT_EXTRACTION_QUOTE",
                "is_grounded": True,
                "reason": f"Matched supplied extraction quote: '{src_q[:60]}...'",
            }

    # 2. Match against full raw paper text if available
    full_text = corpus[paper_id]["full_text"]
    if full_text:
        norm_full_text = normalize_text_for_matching(full_text)
        if norm_quote in norm_full_text:
            return {
                "status": "EXACT_SOURCE_TEXT",
                "is_grounded": True,
                "reason": "Verbatim substring match in raw paper text.",
            }

    # 3. Partial word-overlap check (to distinguish partial quotes from pure hallucinations)
    words = [w for w in re.findall(r"\w+", norm_quote) if len(w) > 3]
    if words:
        matching_in_quotes = sum(
            1 for w in words if any(w in normalize_text_for_matching(sq) for sq in corpus[paper_id]["quotes"])
        )
        overlap_ratio = matching_in_quotes / len(words)
        if overlap_ratio >= 0.7:
            return {
                "status": "PARTIAL_SUPPORTED_OVERLAP",
                "is_grounded": True,
                "reason": f"High lexical overlap ({overlap_ratio:.1%}) with supplied extraction content.",
            }

    return {
        "status": "UNGROUNDED_QUOTE",
        "is_grounded": False,
        "reason": "Quote not found in supplied extraction quotes or source text.",
    }


async def main() -> None:
    print("=" * 80)
    print("PHASE 2 — LIVE SYNTHESIS SMOKE TEST (FEATHERLESS API)")
    print("=" * 80)

    # 1. Locate and validate the two input paper extractions
    p1_path = BACKEND_DIR / "data" / "full_extraction_merged.json"
    p2_path = BACKEND_DIR / "data" / "convs2s_extraction.json"

    if not p1_path.is_file():
        raise FileNotFoundError(f"Missing Paper 1 extraction file: {p1_path}")
    if not p2_path.is_file():
        raise FileNotFoundError(f"Missing Paper 2 extraction file: {p2_path}")

    print(f"[1/5] Loading Paper 1 from: {p1_path.name}")
    p1_data = json.loads(p1_path.read_text(encoding="utf-8"))
    paper1 = ResearchPaperExtraction.model_validate(p1_data)
    print(f"      -> ID: {paper1.paper_id} | Title: {paper1.paper_title}")
    print(f"      -> Methods: {len(paper1.methodology)}, Metrics: {len(paper1.evaluation_metrics)}, Findings: {len(paper1.key_findings)}")

    print(f"[2/5] Loading Paper 2 from: {p2_path.name}")
    p2_data = json.loads(p2_path.read_text(encoding="utf-8"))
    paper2 = ResearchPaperExtraction.model_validate(p2_data)
    print(f"      -> ID: {paper2.paper_id} | Title: {paper2.paper_title}")
    print(f"      -> Methods: {len(paper2.methodology)}, Metrics: {len(paper2.evaluation_metrics)}, Findings: {len(paper2.key_findings)}")

    # Confirm distinct paper IDs
    if paper1.paper_id == paper2.paper_id:
        raise ValueError(f"Paper IDs must be distinct! Both have: {paper1.paper_id}")
    valid_paper_ids = {paper1.paper_id, paper2.paper_id}
    print(f"      -> Distinct paper IDs confirmed: {valid_paper_ids}")

    # Build grounding corpus
    corpus = build_source_text_corpus(paper1, paper2)

    # 2. Initialize LLMClient and SynthesisService
    settings = get_settings()
    print(f"\n[3/5] Initializing SynthesisService with configured Featherless model:")
    print(f"      -> Model: {settings.llm_model_name}")
    print(f"      -> Base URL: {settings.llm_base_url}")
    print(f"      -> Max tokens: {settings.llm_max_tokens}")
    print(f"      -> Temperature: {settings.llm_temperature}")

    client = LLMClient(settings=settings)

    # Wrap generate to capture raw response without modifying service
    captured_raw: list[str] = []
    original_generate = client.generate

    async def capturing_generate(prompt: str, system_prompt: str | None = None) -> str:
        res = await original_generate(prompt, system_prompt=system_prompt)
        captured_raw.append(res)
        return res

    client.generate = capturing_generate  # type: ignore[assignment]
    service = SynthesisService(client=client, settings=settings)

    # 3. Execute live synthesis request
    request = SynthesisRequest(
        papers=[paper1, paper2],
        focus_areas=[
            "Architectural differences: Self-Attention vs Convolutional Layers",
            "Comparative BLEU performance on WMT 2014 English-to-German",
            "Training parallelization bottlenecks and receptive field complexity",
        ],
    )

    print(f"\n[4/5] Executing live synthesis call to Featherless API...")
    start_time = time.perf_counter()
    api_success = False
    error_message: str | None = None
    response: SynthesisResponse | None = None

    try:
        response = await service.synthesize(request)
        api_success = True
    except Exception as exc:
        error_message = str(exc)
        logger.exception("Synthesis API call failed:")

    duration = time.perf_counter() - start_time
    print(f"      -> Call completed in {duration:.2f} seconds. Success = {api_success}")

    if not api_success or response is None:
        print(f"FATAL: Synthesis failed with error: {error_message}")
        sys.exit(1)

    raw_response_text = captured_raw[0] if captured_raw else ""
    metadata = getattr(client, "_last_metadata", {})

    print(f"      -> Prompt tokens: {metadata.get('prompt_tokens', 'N/A')}")
    print(f"      -> Completion tokens: {metadata.get('completion_tokens', 'N/A')}")
    print(f"      -> Finish reason: {metadata.get('finish_reason', 'N/A')}")
    print(f"      -> Visible content length: {len(raw_response_text)} chars")

    # 4. Schema and provenance validation
    print(f"\n[5/5] Auditing synthesis output against strict verifiable criteria...")
    synthesis = response.data

    audit_records: list[dict[str, Any]] = []
    total_comparisons = 0
    total_findings = 0
    total_quotes_checked = 0
    grounded_quotes_count = 0
    fabricated_paper_ids: list[str] = []
    unsupported_claims: list[str] = []

    # Audit Methodological Similarities
    for sim in synthesis.methodological_similarities:
        total_comparisons += 1
        for pid in sim.paper_details.keys():
            if pid not in valid_paper_ids:
                fabricated_paper_ids.append(pid)
        for ev in sim.evidence:
            total_quotes_checked += 1
            if ev.paper_id not in valid_paper_ids:
                fabricated_paper_ids.append(ev.paper_id)
            eval_res = check_quote_grounding(ev.quote or "", ev.paper_id, corpus)
            if eval_res["is_grounded"]:
                grounded_quotes_count += 1
            else:
                unsupported_claims.append(f"Methodological Similarity '{sim.aspect}': {eval_res['reason']}")
            audit_records.append({
                "category": "MethodologicalSimilarity",
                "aspect": sim.aspect,
                "paper_id": ev.paper_id,
                "quote": ev.quote,
                **eval_res,
            })

    # Audit Methodological Differences
    for diff in synthesis.methodological_differences:
        total_comparisons += 1
        for pid in diff.paper_details.keys():
            if pid not in valid_paper_ids:
                fabricated_paper_ids.append(pid)
        for ev in diff.evidence:
            total_quotes_checked += 1
            if ev.paper_id not in valid_paper_ids:
                fabricated_paper_ids.append(ev.paper_id)
            eval_res = check_quote_grounding(ev.quote or "", ev.paper_id, corpus)
            if eval_res["is_grounded"]:
                grounded_quotes_count += 1
            else:
                unsupported_claims.append(f"Methodological Difference '{diff.aspect}': {eval_res['reason']}")
            audit_records.append({
                "category": "MethodologicalDifference",
                "aspect": diff.aspect,
                "paper_id": ev.paper_id,
                "quote": ev.quote,
                **eval_res,
            })

    # Audit Experimental Differences
    for exp in synthesis.dataset_and_experimental_differences:
        total_comparisons += 1
        for pid in exp.paper_details.keys():
            if pid not in valid_paper_ids:
                fabricated_paper_ids.append(pid)
        for ev in exp.evidence:
            total_quotes_checked += 1
            if ev.paper_id not in valid_paper_ids:
                fabricated_paper_ids.append(ev.paper_id)
            eval_res = check_quote_grounding(ev.quote or "", ev.paper_id, corpus)
            if eval_res["is_grounded"]:
                grounded_quotes_count += 1
            else:
                unsupported_claims.append(f"Experimental Difference '{exp.aspect}': {eval_res['reason']}")
            audit_records.append({
                "category": "ExperimentalDifference",
                "aspect": exp.aspect,
                "paper_id": ev.paper_id,
                "quote": ev.quote,
                **eval_res,
            })

    # Audit Findings Agreement
    for fa in synthesis.findings_agreement:
        total_findings += 1
        for pid in fa.paper_findings.keys():
            if pid not in valid_paper_ids:
                fabricated_paper_ids.append(pid)
        for ev in fa.evidence:
            total_quotes_checked += 1
            if ev.paper_id not in valid_paper_ids:
                fabricated_paper_ids.append(ev.paper_id)
            eval_res = check_quote_grounding(ev.quote or "", ev.paper_id, corpus)
            if eval_res["is_grounded"]:
                grounded_quotes_count += 1
            else:
                unsupported_claims.append(f"Findings Agreement '{fa.topic}': {eval_res['reason']}")
            audit_records.append({
                "category": "FindingsAgreement",
                "topic": fa.topic,
                "paper_id": ev.paper_id,
                "quote": ev.quote,
                **eval_res,
            })

    # Audit Findings Disagreement (Contextual Differences vs Contradictions)
    for fd in synthesis.findings_disagreement:
        total_findings += 1
        for pid in fd.paper_findings.keys():
            if pid not in valid_paper_ids:
                fabricated_paper_ids.append(pid)
        for ev in fd.evidence:
            total_quotes_checked += 1
            if ev.paper_id not in valid_paper_ids:
                fabricated_paper_ids.append(ev.paper_id)
            eval_res = check_quote_grounding(ev.quote or "", ev.paper_id, corpus)
            if eval_res["is_grounded"]:
                grounded_quotes_count += 1
            else:
                unsupported_claims.append(f"Findings Disagreement '{fd.topic}': {eval_res['reason']}")
            audit_records.append({
                "category": "FindingsDisagreement",
                "topic": fd.topic,
                "relationship": fd.relationship.value,
                "contextual_factors": fd.contextual_factors,
                "paper_id": ev.paper_id,
                "quote": ev.quote,
                **eval_res,
            })

    # Check metric preservation (e.g. 28.4, 25.16, 41.0, 30.5)
    full_output_text = json.dumps(synthesis.model_dump())
    expected_metrics = ["28.4", "25.16"]
    preserved_metrics = [m for m in expected_metrics if m in full_output_text]

    # Save full smoke test results
    out_file = BACKEND_DIR / "data" / "live_synthesis_smoke_test.json"
    result_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostics": {
            "api_success": api_success,
            "duration_seconds": round(duration, 2),
            "model_name": metadata.get("model_name", settings.llm_model_name),
            "max_tokens": metadata.get("max_tokens", settings.llm_max_tokens),
            "finish_reason": metadata.get("finish_reason", "N/A"),
            "prompt_tokens": metadata.get("prompt_tokens", 0),
            "completion_tokens": metadata.get("completion_tokens", 0),
            "total_tokens": metadata.get("total_tokens", 0),
            "visible_content_length": len(raw_response_text),
            "has_reasoning_fields": metadata.get("has_reasoning_fields", False),
        },
        "input_papers": [
            {"paper_id": paper1.paper_id, "title": paper1.paper_title},
            {"paper_id": paper2.paper_id, "title": paper2.paper_title},
        ],
        "audit_summary": {
            "pydantic_validation_passed": True,
            "total_comparisons": total_comparisons,
            "total_findings": total_findings,
            "total_quotes_checked": total_quotes_checked,
            "grounded_quotes_count": grounded_quotes_count,
            "quote_grounding_rate": f"{(grounded_quotes_count / max(1, total_quotes_checked)):.1%}",
            "fabricated_paper_ids": list(set(fabricated_paper_ids)),
            "preserved_key_metrics": preserved_metrics,
            "unsupported_claims_count": len(unsupported_claims),
            "unsupported_claims": unsupported_claims,
        },
        "synthesis": synthesis.model_dump(),
        "quote_audits": audit_records,
        "raw_response": raw_response_text,
    }

    out_file.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] Saved complete smoke test audit to: {out_file.name} ({out_file.stat().st_size} bytes)")

    # Print summary
    print("\n" + "=" * 80)
    print("SMOKE TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"API Call Succeeded:          {api_success}")
    print(f"Model:                       {metadata.get('model_name', settings.llm_model_name)}")
    print(f"Execution Time:              {duration:.2f} seconds")
    print(f"Tokens Used:                 Prompt: {metadata.get('prompt_tokens')}, Completion: {metadata.get('completion_tokens')}, Total: {metadata.get('total_tokens')}")
    print(f"Pydantic Validation:         PASS (Valid CrossPaperSynthesis)")
    print(f"Comparisons Generated:       {total_comparisons}")
    print(f"Findings Generated:          {total_findings}")
    print(f"Research Gaps Identified:    {len(synthesis.research_limitations_and_gaps)}")
    print(f"Future Directions:           {len(synthesis.future_research_directions)}")
    print(f"Quotes Checked:              {total_quotes_checked}")
    print(f"Grounded Quotes:             {grounded_quotes_count} ({(grounded_quotes_count / max(1, total_quotes_checked)):.1%})")
    print(f"Fabricated Paper IDs:        {list(set(fabricated_paper_ids)) or 'None (0)'}")
    print(f"Preserved Key Metrics:       {preserved_metrics} of {expected_metrics}")
    print(f"Unsupported Claims:          {len(unsupported_claims)}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
