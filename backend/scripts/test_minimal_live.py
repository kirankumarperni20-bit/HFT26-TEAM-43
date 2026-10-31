"""
test_minimal_live.py — Minimal live test using LLMClient.

Sends a short prompt asking the model to return a small JSON object.
Confirms visible JSON output, token usage, and diagnostic logging.
"""

import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.client import LLMClient
from llm.config import get_settings


async def main():
    settings = get_settings()
    client = LLMClient(settings=settings)

    print("=== Minimal Live LLMClient Test ===")
    print(f"Target Model: {settings.llm_model_name}")
    print(f"Base URL: {settings.llm_base_url}")
    print(f"Configured Max Tokens: {settings.llm_max_tokens}")

    prompt = (
        "Return a JSON object with two fields:\n"
        '- "status": "ok"\n'
        '- "task": "minimal_test"\n'
        "Output ONLY the JSON object."
    )

    content = await client.generate(
        prompt=prompt,
        system_prompt="You are a precise JSON generator. Output ONLY a valid JSON object.",
    )

    meta = client.last_metadata

    print("\n=== Live Test Result ===")
    print(f"Model Name: {meta.get('model_name')}")
    print(f"Max Tokens: {meta.get('max_tokens')}")
    print(f"Finish Reason: {meta.get('finish_reason')}")
    print(f"Prompt Tokens: {meta.get('prompt_tokens')}")
    print(f"Completion Tokens: {meta.get('completion_tokens')}")
    print(f"Total Tokens: {meta.get('total_tokens')}")
    print(f"Visible Content Length: {meta.get('visible_content_length')}")
    print(f"Has Separate Reasoning Fields: {meta.get('has_reasoning_fields')}")
    print(f"Visible Content:\n{content}")

    from llm.services.extraction_service import clean_and_parse_json

    # Validate JSON parsing
    parsed = clean_and_parse_json(content)
    print(f"\nParsed JSON successfully: {parsed}")


if __name__ == "__main__":
    asyncio.run(main())
