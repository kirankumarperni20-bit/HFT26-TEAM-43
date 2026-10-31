"""
run_focused_live_test.py — Focused live extraction test on Pages 1-2 of test_paper.pdf.

Captures all diagnostic criteria required by Step 2:
- API request success/failure
- HTTP/API error details
- Response content length
- finish_reason
- usage (prompt_tokens, completion_tokens, cost)
- JSON validity
- Pydantic schema validation
- Evidence grounding check against source text
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pypdfium2 as pdfium
from openai import AsyncOpenAI, OpenAIError

from llm.config import get_settings
from llm.schemas.extraction import ResearchPaperExtraction
from llm.services.extraction_service import clean_and_parse_json, load_extraction_prompt_template
from llm.validators.grounding import verify_evidence_grounding


async def run_focused_test():
    settings = get_settings()
    pdf_path = BACKEND_DIR / "data" / "test_paper.pdf"

    print("==================================================")
    print("STEP 1: INGESTING PAGES 1-2 OF test_paper.pdf")
    print("==================================================")

    pdf = pdfium.PdfDocument(str(pdf_path))
    p1 = pdf[0].get_textpage().get_text_range().strip()
    p2 = pdf[1].get_textpage().get_text_range().strip()
    pages_text = f"--- PAGE 1 ---\n{p1}\n\n--- PAGE 2 ---\n{p2}"

    print(f"Source text length: {len(pages_text)} characters, {len(pages_text.split())} words.")

    template = load_extraction_prompt_template()
    prompt = template.replace("{source_text}", pages_text)

    print("\n==================================================")
    print("STEP 2: SENDING LIVE REQUEST TO FEATHERLESS AI")
    print("==================================================")
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Model: {settings.llm_model_name}")
    print(f"Max tokens: {settings.llm_max_tokens}")
    print(f"Temperature: {settings.llm_temperature}")

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    start_time = time.time()
    api_success = False
    api_error = None
    response = None

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise scientific research information extraction engine. "
                        "DO NOT write extensive internal reasoning or chain-of-thought. "
                        "Keep internal reasoning concise (under 100 words) and immediately output "
                        "the final complete, valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
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

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason
    usage = response.usage

    print("\n==================================================")
    print("STEP 3: DIAGNOSTIC API METADATA")
    print("==================================================")
    print(f"finish_reason: {finish_reason}")
    print(f"Response content length: {len(content)} characters")
    if usage:
        print(f"Prompt tokens: {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
        cost = getattr(usage, "cost", None)
        if cost is not None:
            print(f"Cost: ${cost:.6f}")

    print("\n==================================================")
    print("STEP 4: JSON PARSING & PYDANTIC VALIDATION")
    print("==================================================")

    is_valid_json = False
    parsed_dict = None
    try:
        parsed_dict = clean_and_parse_json(content)
        is_valid_json = True
        print("JSON Parsing: SUCCESS")
    except Exception as exc:
        print(f"JSON Parsing FAILED: {exc}")
        print("Raw Content Preview:\n", content[:500])
        return

    pydantic_success = False
    extraction = None
    try:
        extraction = ResearchPaperExtraction.model_validate(parsed_dict)
        pydantic_success = True
        print("Pydantic Validation: SUCCESS")
    except Exception as exc:
        print(f"Pydantic Validation FAILED: {exc}")
        return

    print("\n==================================================")
    print("STEP 5: STRUCTURED EXTRACTION OUTPUT")
    print("==================================================")
    dumped = extraction.model_dump()
    print(json.dumps(dumped, indent=2))

    print("\n==================================================")
    print("STEP 6: EVIDENCE GROUNDING CHECK AGAINST SOURCE TEXT")
    print("==================================================")
    report = verify_evidence_grounding(extraction, pages_text)
    print(f"Fully Grounded: {report.is_grounded}")
    print(f"Total Quotes Checked: {report.total_quotes_checked}")
    print(f"Grounded Quotes Count: {report.grounded_quotes_count}")
    print(f"Ungrounded Quotes Count: {len(report.ungrounded_quotes)}")
    if report.ungrounded_quotes:
        print("Ungrounded Quotes List:")
        for q in report.ungrounded_quotes:
            print(f"  - {q}")

    # Save results to data folder
    result_path = BACKEND_DIR / "data" / "focused_extraction_result.json"
    result_path.write_text(json.dumps({
        "diagnostics": {
            "api_success": api_success,
            "finish_reason": finish_reason,
            "duration_seconds": elapsed,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cost": getattr(usage, "cost", None),
            "is_valid_json": is_valid_json,
            "pydantic_success": pydantic_success,
        },
        "grounding": report.model_dump(),
        "extraction": dumped,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved focused test results to: {result_path}")


if __name__ == "__main__":
    asyncio.run(run_focused_test())
