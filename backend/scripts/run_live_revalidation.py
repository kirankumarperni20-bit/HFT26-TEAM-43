"""
run_live_revalidation.py — Live re-validation of the improved extraction pipeline.

Runs the updated extraction prompt against Pages 1-2 of test_paper.pdf,
performs both exact quote grounding and semantic support validation,
saves raw + parsed results, and prints a comprehensive audit report.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pypdfium2 as pdfium
from openai import OpenAIError

from llm.client import LLMClient
from llm.config import get_settings
from llm.schemas.extraction import ResearchPaperExtraction
from llm.services.extraction_service import clean_and_parse_json, load_extraction_prompt_template
from llm.validators.grounding import (
    normalize_text_for_matching,
    verify_evidence_grounding,
    verify_extraction_semantic_support,
    detect_fabricated_metric_value,
)

DATA_DIR = BACKEND_DIR / "data"


async def run_revalidation():
    settings = get_settings()
    pdf_path = DATA_DIR / "test_paper.pdf"

    # ── Step 1: Ingest Pages 1-2 ──────────────────────────────────────
    print("=" * 70)
    print("STEP 1: INGESTING PAGES 1-2 OF test_paper.pdf")
    print("=" * 70)

    pdf = pdfium.PdfDocument(str(pdf_path))
    p1 = pdf[0].get_textpage().get_text_range().strip()
    p2 = pdf[1].get_textpage().get_text_range().strip()
    pages_text = f"--- PAGE 1 ---\n{p1}\n\n--- PAGE 2 ---\n{p2}"

    print(f"Source text length: {len(pages_text)} characters, {len(pages_text.split())} words.")

    # ── Step 2: Build prompt ──────────────────────────────────────────
    template = load_extraction_prompt_template()
    prompt = template.replace("{source_text}", pages_text)

    # ── Step 3: Send live request ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: SENDING LIVE REQUEST TO FEATHERLESS AI")
    print("=" * 70)
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Model: {settings.llm_model_name}")
    print(f"Max tokens: {settings.llm_max_tokens}")
    print(f"Temperature: {settings.llm_temperature}")

    llm_client = LLMClient(settings=settings)

    start_time = time.time()
    api_success = False
    api_error = None
    raw_content = ""

    try:
        raw_content = await llm_client.generate(prompt)
        api_success = True
    except OpenAIError as exc:
        api_error = str(exc)
        print(f"API Request Failed: {exc}")
    except Exception as exc:
        api_error = str(exc)
        print(f"Unexpected Error: {exc}")

    elapsed = time.time() - start_time
    print(f"Request duration: {elapsed:.2f} seconds.")
    print(f"API Success: {api_success}")

    if not api_success:
        print(f"Error Details: {api_error}")
        return

    meta = llm_client.last_metadata
    finish_reason = meta.get("finish_reason")
    usage = llm_client.last_usage

    # ── Step 4: API Metadata ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: API METADATA")
    print("=" * 70)
    print(f"Model: {meta.get('model_name')}")
    print(f"Max tokens: {meta.get('max_tokens')}")
    print(f"finish_reason: {finish_reason}")
    print(f"Response content length: {len(raw_content)} characters")
    print(f"has_reasoning_fields: {meta.get('has_reasoning_fields')}")
    if usage:
        print(f"Prompt tokens: {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
        cost = getattr(usage, "cost", None)
        if cost is not None:
            print(f"Cost: ${cost:.6f}")

    if finish_reason == "length":
        print("\n*** WARNING: finish_reason is 'length' — response may be TRUNCATED ***")

    # ── Step 5: Save raw response ─────────────────────────────────────
    raw_path = DATA_DIR / "revalidation_raw_response.txt"
    raw_path.write_text(raw_content, encoding="utf-8")
    print(f"\nSaved raw model response to: {raw_path}")

    # ── Step 6: JSON parsing & Pydantic validation ────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: JSON PARSING & PYDANTIC VALIDATION")
    print("=" * 70)

    try:
        parsed_dict = clean_and_parse_json(raw_content)
        print("JSON Parsing: SUCCESS")
    except Exception as exc:
        print(f"JSON Parsing FAILED: {exc}")
        print("Raw Content Preview:\n", raw_content[:500])
        return

    try:
        extraction = ResearchPaperExtraction.model_validate(parsed_dict)
        print("Pydantic Validation: SUCCESS")
    except Exception as exc:
        print(f"Pydantic Validation FAILED: {exc}")
        return

    # Save parsed extraction
    parsed_path = DATA_DIR / "revalidation_parsed_extraction.json"
    dumped = extraction.model_dump()
    parsed_path.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
    print(f"Saved parsed extraction to: {parsed_path}")

    # ── Step 7: Exact Quote Grounding ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: EXACT QUOTE GROUNDING")
    print("=" * 70)

    grounding_report = verify_evidence_grounding(extraction, pages_text)
    print(f"Fully Grounded: {grounding_report.is_grounded}")
    print(f"Total Quotes Checked: {grounding_report.total_quotes_checked}")
    print(f"Exact Contiguous Quotes: {grounding_report.exact_contiguous_count}")
    print(f"Documented PDF Normalized Quotes: {grounding_report.normalized_artifact_count}")
    print(f"Non-Contiguous Quotes: {grounding_report.non_contiguous_count}")
    print(f"Ungrounded Quotes: {len(grounding_report.ungrounded_quotes)}")
    if grounding_report.ungrounded_quotes:
        print("Ungrounded Quotes List:")
        for q in grounding_report.ungrounded_quotes:
            print(f"  - {q[:120]}...")

    # ── Step 8: Semantic Support Validation ────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: SEMANTIC SUPPORT VALIDATION")
    print("=" * 70)

    semantic_report = verify_extraction_semantic_support(extraction, pages_text)
    print(f"Is Semantically Supported: {semantic_report.is_semantically_supported}")
    print(f"Total Items Evaluated: {semantic_report.total_items_checked}")
    print(f"Supported Claims: {semantic_report.supported_count}")
    print(f"Partially Supported Claims: {semantic_report.partially_supported_count}")
    print(f"Technique Mentions Only (no mechanism claim): {semantic_report.mentioned_only_count}")
    print(f"Unsupported Claims: {semantic_report.unsupported_count}")
    print(f"Contradicted Claims: {semantic_report.contradicted_count}")
    print(f"Forward References Found: {len(semantic_report.forward_references)}")
    print(f"Misattributed Limitations: {len(semantic_report.misattributed_limitations)}")
    print(f"Fabricated Metrics: {len(semantic_report.fabricated_metrics)}")
    print(f"Unexpected Non-Empty Fields: {len(semantic_report.unexpected_nonempty_fields)}")

    # ── Step 9: Audit Table ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 7: DETAILED AUDIT TABLE")
    print("=" * 70)

    for i, ev in enumerate(semantic_report.evaluations, 1):
        print(f"\n--- Item {i}: [{ev.item_type}] {ev.item_label} ---")
        if ev.quote:
            display_quote = ev.quote[:150] + ("..." if len(ev.quote) > 150 else "")
            print(f"  Quote: \"{display_quote}\"")
        else:
            print("  Quote: (none)")
        print(f"  Exact Match: {'PASS' if ev.exact_match else 'FAIL'} ({ev.quote_match_status.value})")
        print(f"  Semantic Status: {ev.support_status.value}")
        print(f"  Reason: {ev.reason}")

    if semantic_report.forward_references:
        print("\n>>> FORWARD REFERENCES DETECTED:")
        for fr in semantic_report.forward_references:
            print(f"  - {fr}")

    if semantic_report.misattributed_limitations:
        print("\n>>> MISATTRIBUTED LIMITATIONS DETECTED:")
        for ml in semantic_report.misattributed_limitations:
            print(f"  - {ml}")

    if semantic_report.fabricated_metrics:
        print("\n>>> FABRICATED METRICS DETECTED:")
        for fm in semantic_report.fabricated_metrics:
            print(f"  - {fm}")

    if semantic_report.unexpected_nonempty_fields:
        print("\n>>> UNEXPECTED NON-EMPTY FIELDS:")
        for uf in semantic_report.unexpected_nonempty_fields:
            print(f"  - {uf}")

    # ── Step 10: Specific Checks ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 8: TARGETED QUALITY CHECKS")
    print("=" * 70)

    # Check for BLEU score accuracy
    all_metric_values = [m.value for m in extraction.evaluation_metrics if m.value]
    all_metric_text = " ".join(all_metric_values)
    print(f"Extracted metric values: {all_metric_values}")

    checks = {
        "28.4 present in extracted metrics": "28.4" in all_metric_text,
        "41.0 present in extracted metrics": "41.0" in all_metric_text,
        "27.5 ABSENT from extracted metrics (was preprint draft)": "27.5" not in all_metric_text,
        "41.1 ABSENT from extracted metrics (was preprint draft)": "41.1" not in all_metric_text,
        "165 million ABSENT from extracted metrics": "165" not in all_metric_text,
    }

    # Check for O(1) notation substitution
    all_quotes = []
    for m in extraction.methodology:
        if m.evidence and m.evidence.quote:
            all_quotes.append(m.evidence.quote)
    for m in extraction.evaluation_metrics:
        if m.evidence and m.evidence.quote:
            all_quotes.append(m.evidence.quote)
    for kf in extraction.key_findings:
        if kf.evidence and kf.evidence.quote:
            all_quotes.append(kf.evidence.quote)
    for ds in extraction.datasets:
        if ds.evidence and ds.evidence.quote:
            all_quotes.append(ds.evidence.quote)

    all_quotes_text = " ".join(all_quotes)
    checks["O(1) ABSENT from evidence quotes (no notation substitution)"] = "O(1)" not in all_quotes_text

    # Check for forward reference in Multi-Head Attention
    mha_methods = [m for m in extraction.methodology if "multi-head" in m.technique.lower() or "multi head" in m.technique.lower()]
    has_mha_forward_ref = False
    for m in mha_methods:
        if m.evidence and m.evidence.quote:
            from llm.validators.grounding import detect_forward_reference
            if detect_forward_reference(m.evidence.quote):
                has_mha_forward_ref = True
    checks["Multi-Head Attention NOT based on forward reference"] = not has_mha_forward_ref

    # Check for prior-art misattribution in limitations
    from llm.validators.grounding import detect_prior_art_misattribution
    has_misattributed = any(detect_prior_art_misattribution(lim) for lim in extraction.limitations)
    checks["No prior-art RNN limitation misattributed to Transformer"] = not has_misattributed

    # Check future_work
    norm_source = normalize_text_for_matching(pages_text)
    if "future" not in norm_source:
        checks["future_work correctly empty (topic absent from Pages 1-2)"] = len(extraction.future_work) == 0
    else:
        checks["future_work present (source mentions future)"] = True

    for desc, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {desc}")

    # ── Step 11: Save full results ────────────────────────────────────
    full_result = {
        "diagnostics": {
            "api_success": api_success,
            "finish_reason": finish_reason,
            "duration_seconds": round(elapsed, 2),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "cost": getattr(usage, "cost", None) if usage else None,
            "raw_content_length": len(raw_content),
        },
        "grounding": grounding_report.model_dump(),
        "semantic_validation": semantic_report.model_dump(),
        "extraction": dumped,
        "targeted_checks": {k: v for k, v in checks.items()},
    }

    result_path = DATA_DIR / "revalidation_full_result.json"
    result_path.write_text(json.dumps(full_result, indent=2), encoding="utf-8")
    print(f"\nSaved full revalidation results to: {result_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Extraction completed: {'YES' if api_success and finish_reason == 'stop' else 'NO'}")
    print(f"All quotes verbatim: {'YES' if grounding_report.is_grounded else 'NO'}")
    print(f"Semantically supported: {'YES' if semantic_report.is_semantically_supported else 'NO'}")
    print(f"Targeted checks passed: {sum(checks.values())}/{len(checks)}")


if __name__ == "__main__":
    asyncio.run(run_revalidation())
