"""
run_real_extraction.py — Live execution script for real-world research paper extraction.

Performs:
    1. PDF text extraction from backend/data/test_paper.pdf
    2. Live Featherless AI extraction via ExtractionService & LLMClient
    3. Grounding verification against original text using verify_evidence_grounding
    4. Execution metrics, token usage, and accuracy reporting
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

from llm.client import LLMClient
from llm.schemas.extraction import ExtractionRequest
from llm.services.extraction_service import ExtractionService
from llm.validators.grounding import verify_evidence_grounding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("real_extraction")


def ingest_pdf(pdf_path: Path) -> tuple[str, dict]:
    """Ingest a PDF file and return its full text and metadata."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pdf = pdfium.PdfDocument(str(pdf_path))
    num_pages = len(pdf)
    pages_text = []

    for i in range(num_pages):
        page_text = pdf[i].get_textpage().get_text_range().strip()
        if page_text:
            pages_text.append(f"--- PAGE {i + 1} ---\n" + page_text)

    full_text = "\n\n".join(pages_text)
    metadata = {
        "filename": pdf_path.name,
        "page_count": num_pages,
        "char_count": len(full_text),
    }
    return full_text, metadata


async def main():
    pdf_path = BACKEND_DIR / "data" / "test_paper.pdf"
    print(f"\n=======================================================")
    print(f"1. INGESTING PDF: {pdf_path}")
    print(f"=======================================================")

    full_text, meta = ingest_pdf(pdf_path)
    print(f"Pages ingested: {meta['page_count']}")
    print(f"Total characters: {meta['char_count']}")
    print(f"Preview (first 400 chars):\n{full_text[:400]}...\n")

    print(f"=======================================================")
    print(f"2. INVOKING FEATHERLESS AI VIA ExtractionService")
    print(f"=======================================================")

    client = LLMClient()
    service = ExtractionService(client=client)

    request = ExtractionRequest(
        text=full_text,
        paper_id="Vaswani_2017_Attention",
    )

    start_time = time.time()
    response = await service.extract(request)
    elapsed = time.time() - start_time

    print(f"\nExtraction completed in {elapsed:.2f} seconds.")
    print(f"Model used: {response.model}")

    usage = getattr(client, "last_usage", None)
    if usage:
        print(f"Tokens consumed — Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

    extracted_data = response.data
    json_output = extracted_data.model_dump()

    print(f"\n=======================================================")
    print(f"3. RESULTING STRUCTURED EXTRACTION")
    print(f"=======================================================")
    print(json.dumps(json_output, indent=2))

    print(f"\n=======================================================")
    print(f"4. VERIFYING CITATION GROUNDING AGAINST SOURCE TEXT")
    print(f"=======================================================")
    report = verify_evidence_grounding(extracted_data, full_text)

    print(f"Is Fully Grounded: {report.is_grounded}")
    print(f"Total Quotes Checked: {report.total_quotes_checked}")
    print(f"Grounded Quotes: {report.grounded_quotes_count}")
    print(f"Ungrounded Quotes: {len(report.ungrounded_quotes)}")
    if report.ungrounded_quotes:
        print(f"List of ungrounded quotes:")
        for q in report.ungrounded_quotes:
            print(f"  - {q}")

    # Save output to artifacts/data
    output_file = BACKEND_DIR / "data" / "extraction_result.json"
    output_file.write_text(json.dumps({
        "metadata": meta,
        "elapsed_seconds": elapsed,
        "model": response.model,
        "tokens": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        },
        "grounding_report": report.model_dump(),
        "extraction": json_output,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved complete extraction result to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
