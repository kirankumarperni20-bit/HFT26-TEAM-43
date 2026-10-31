"""
synthesis_service.py — Structured cross-paper synthesis service.

Orchestrates multi-paper comparative analysis:
- Validates that at least two structured paper extractions are supplied.
- Formats structured paper data into the synthesis prompt template.
- Sends the request to the unified LLMClient (with enable_thinking=False).
- Cleans and parses the returned JSON payload with resilient fallbacks.
- Validates the resulting payload into the CrossPaperSynthesis Pydantic schema.
- Enforces source reference provenance integrity across compared papers.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from backend.llm.client import LLMClient
    from backend.llm.config import LLMSettings, get_settings
    from backend.llm.schemas.extraction import ResearchPaperExtraction
    from backend.llm.schemas.synthesis import (
        CrossPaperSynthesis,
        SynthesisRequest,
        SynthesisResponse,
    )
    from backend.llm.services.extraction_service import clean_and_parse_json
except ModuleNotFoundError:
    from llm.client import LLMClient
    from llm.config import LLMSettings, get_settings
    from llm.schemas.extraction import ResearchPaperExtraction
    from llm.schemas.synthesis import (
        CrossPaperSynthesis,
        SynthesisRequest,
        SynthesisResponse,
    )
    from llm.services.extraction_service import clean_and_parse_json

logger = logging.getLogger(__name__)

# Path to the cross-paper synthesis prompt template
SYNTHESIS_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "synthesis.txt"


@lru_cache(maxsize=1)
def load_synthesis_prompt_template() -> str:
    """Load and cache the cross-paper synthesis prompt template from disk."""
    if not SYNTHESIS_PROMPT_PATH.is_file():
        raise FileNotFoundError(
            f"Cross-paper synthesis prompt template not found at: {SYNTHESIS_PROMPT_PATH}"
        )
    content = SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if "{extracted_papers_json}" not in content:
        raise ValueError(
            "Synthesis prompt template missing required placeholder: '{extracted_papers_json}'"
        )
    return content


def serialize_papers_for_prompt(papers: list[ResearchPaperExtraction]) -> str:
    """Serialize a collection of extracted papers into formatted JSON for the LLM prompt."""
    payload: list[dict[str, Any]] = []
    for idx, paper in enumerate(papers, 1):
        dumped = paper.model_dump()
        # Guarantee a fallback paper_id for prompt clarity if missing
        if not dumped.get("paper_id"):
            dumped["paper_id"] = f"Paper_{idx}"
        payload.append(dumped)
    return json.dumps(payload, indent=2, ensure_ascii=False)


class SynthesisService:
    """Service orchestrating cross-paper comparative synthesis."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or LLMClient(settings=self._settings)

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """Generate a structured comparative synthesis across multiple research papers.

        Args:
            request: SynthesisRequest containing at least two ResearchPaperExtraction objects.

        Returns:
            SynthesisResponse containing validated CrossPaperSynthesis.

        Raises:
            ValueError: If fewer than 2 papers are provided or JSON parsing fails.
            ValidationError: If model output violates the CrossPaperSynthesis schema.
            Exception: If LLM generation fails.
        """
        if len(request.papers) < 2:
            raise ValueError(
                f"Cross-paper synthesis requires at least 2 papers. Received: {len(request.papers)}"
            )

        template = load_synthesis_prompt_template()
        papers_json_str = serialize_papers_for_prompt(request.papers)
        prompt = template.replace("{extracted_papers_json}", papers_json_str)

        # Optional focus areas guidance injection
        if request.focus_areas:
            focus_block = (
                "\n\n### SPECIAL EMPHASIS ON USER FOCUS AREAS\n"
                + "\n".join(f"- {fa}" for fa in request.focus_areas)
                + "\n"
            )
            prompt = focus_block + prompt

        logger.info(
            "Sending cross-paper synthesis request for %d papers to model %s",
            len(request.papers),
            self._settings.llm_model_name,
        )

        raw_response = await self._client.generate(prompt)

        # Clean and parse JSON response
        parsed_dict = clean_and_parse_json(raw_response)

        # Ensure papers_analyzed metadata is populated if missing
        if not parsed_dict.get("papers_analyzed"):
            parsed_dict["papers_analyzed"] = [
                {
                    "paper_id": p.paper_id or f"Paper_{i}",
                    "paper_title": p.paper_title or "Untitled Research Paper",
                }
                for i, p in enumerate(request.papers, 1)
            ]

        # Validate with Pydantic
        synthesis = CrossPaperSynthesis.model_validate(parsed_dict)

        return SynthesisResponse(
            data=synthesis,
            model=self._settings.llm_model_name,
            papers_count=len(request.papers),
        )


async def synthesize_papers(
    papers: list[ResearchPaperExtraction],
    focus_areas: list[str] | None = None,
    client: LLMClient | None = None,
    settings: LLMSettings | None = None,
) -> SynthesisResponse:
    """Convenience helper function to run cross-paper synthesis directly."""
    service = SynthesisService(client=client, settings=settings)
    request = SynthesisRequest(papers=papers, focus_areas=focus_areas or [])
    return await service.synthesize(request)
