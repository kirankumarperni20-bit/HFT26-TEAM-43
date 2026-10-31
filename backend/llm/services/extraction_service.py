"""
extraction_service.py — Structured information extraction service.

Orchestrates paper text ingestion, prompt compilation, LLM communication
via Featherless AI (OpenAI-compatible), robust JSON parsing, and validation
against the Pydantic extraction schema without silent truncation.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

try:
    from backend.llm.client import LLMClient
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
    from backend.llm.validators.grounding import (
        detect_forward_reference,
        is_mechanism_marked_unavailable,
    )
except ModuleNotFoundError:
    from llm.client import LLMClient
    from llm.config import LLMSettings, get_settings
    from llm.schemas.extraction import (
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
    from llm.validators.grounding import (
        detect_forward_reference,
        is_mechanism_marked_unavailable,
    )

logger = logging.getLogger(__name__)

# Path to the extraction prompt template
PROMPT_FILE_PATH = Path(__file__).parent.parent / "prompts" / "extraction.txt"

# Safe default chunk limits (approx ~3,500 tokens of English text)
DEFAULT_MAX_CHUNK_CHARS = 14000
DEFAULT_CHUNK_OVERLAP_CHARS = 1000


@lru_cache(maxsize=1)
def load_extraction_prompt_template() -> str:
    """Load and cache the extraction prompt from prompts/extraction.txt."""
    if not PROMPT_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Extraction prompt template not found at {PROMPT_FILE_PATH}"
        )
    return PROMPT_FILE_PATH.read_text(encoding="utf-8")


def clean_and_parse_json(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from model output.

    Handles markdown fences (```json ... ```) and leading/trailing commentary.

    Args:
        raw_text: Raw string returned by the LLM.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted or parsed.
    """
    text = raw_text.strip()

    # 1. Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # 2. Direct JSON decode attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. Fallback: extract the outermost JSON object { ... }
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            logger.debug("Slice-extracted JSON block failed to parse: %s", exc)

    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            logger.debug("Regex-extracted JSON block failed to parse: %s", exc)

    # 4. Fallback: repair truncated or malformed JSON using json_repair
    try:
        import json_repair

        repaired = json_repair.loads(text)
        if isinstance(repaired, dict) and repaired:
            logger.info("Successfully recovered JSON using json_repair fallback.")
            return repaired
    except Exception as exc:
        logger.debug("json_repair fallback failed: %s", exc)

    snippet = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text
    raise ValueError(
        f"Failed to parse LLM response into valid JSON dictionary. Model output snippet:\n{snippet}"
    )


def split_text_into_chunks(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split long paper text into paragraph-aware chunks without silent truncation.

    Args:
        text: Full source text to split.
        max_chars: Maximum character budget per chunk.
        overlap_chars: Character overlap between consecutive chunks.

    Returns:
        List of non-empty text chunks covering the entire text.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    total_len = len(text)

    while start < total_len:
        end = start + max_chars
        if end >= total_len:
            chunks.append(text[start:].strip())
            break

        # Attempt to break at paragraph boundary
        split_point = text.rfind("\n\n", start, end)
        if split_point == -1 or split_point <= start:
            # Fallback to single newline
            split_point = text.rfind("\n", start, end)
        if split_point == -1 or split_point <= start:
            # Fallback to sentence ending
            split_point = text.rfind(". ", start, end)
            if split_point != -1:
                split_point += 1
        if split_point == -1 or split_point <= start:
            # Hard cutoff at word boundary
            split_point = text.rfind(" ", start, end)
        if split_point == -1 or split_point <= start:
            split_point = end

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, respecting overlap and snapping to clean sentence/line boundary
        raw_overlap_start = max(split_point - overlap_chars, start + 1)
        snapped_start = text.find("\n\n", raw_overlap_start, split_point)
        if snapped_start == -1:
            snapped_start = text.find("\n", raw_overlap_start, split_point)
        if snapped_start == -1:
            snapped_start = text.find(". ", raw_overlap_start, split_point)
            if snapped_start != -1:
                snapped_start += 2
        start = (
            snapped_start
            if (snapped_start != -1 and snapped_start < split_point)
            else raw_overlap_start
        )

    return chunks


class ChunkCoverageReport(BaseModel):
    """Detailed report on text coverage, overlap integrity, and provenance preservation across chunks."""

    is_fully_covered: bool = Field(
        description="True if every character index in the source text is represented in at least one chunk without gaps.",
    )
    total_text_length: int = Field(
        description="Total character count of the source text.",
    )
    total_chunks: int = Field(
        description="Total number of chunks partitioned.",
    )
    uncovered_ranges: list[tuple[int, int]] = Field(
        default_factory=list,
        description="List of (start, end) text index ranges missing from all chunks.",
    )
    overlap_integrity: bool = Field(
        default=True,
        description="True if all adjacent chunks exhibit intentional non-negative overlap without gaps.",
    )
    page_markers_preserved: bool = Field(
        default=True,
        description="True if all page markers in source text appear intact in chunks.",
    )
    missing_page_markers: list[str] = Field(
        default_factory=list,
        description="Page markers from source text not found in any chunk.",
    )
    diagnostics: list[str] = Field(
        default_factory=list,
        description="Diagnostic notes on chunk transitions and boundaries.",
    )


def verify_chunk_coverage(text: str, chunks: list[str]) -> ChunkCoverageReport:
    """Programmatically verify continuous chunk coverage of source text.

    Verifies:
    1. Every source text region is represented in at least one chunk without gaps.
    2. Overlap between adjacent chunks is intentional and does not drop characters.
    3. Page markers (--- PAGE X ---) remain valid and intact.
    """
    if not text:
        return ChunkCoverageReport(
            is_fully_covered=True,
            total_text_length=0,
            total_chunks=len(chunks),
        )
    if not chunks:
        return ChunkCoverageReport(
            is_fully_covered=False,
            total_text_length=len(text),
            total_chunks=0,
            uncovered_ranges=[(0, len(text))],
            overlap_integrity=False,
        )

    # Track intervals for each chunk
    intervals: list[tuple[int, int]] = []
    search_pos = 0
    diagnostics: list[str] = []

    for i, c in enumerate(chunks):
        c_clean = c.strip()
        if not c_clean:
            continue
        probe_head = c_clean[:min(100, len(c_clean))]
        probe_tail = c_clean[-min(100, len(c_clean)):]
        start_idx = text.find(probe_head, max(0, search_pos - 2000))
        if start_idx == -1:
            start_idx = text.find(probe_head)
        end_idx = text.find(probe_tail, start_idx if start_idx != -1 else 0)
        if end_idx != -1:
            end_idx += len(probe_tail)
        else:
            end_idx = start_idx + len(c_clean) if start_idx != -1 else len(text)

        intervals.append((max(0, start_idx), min(len(text), end_idx)))
        search_pos = max(0, start_idx)

    # Check for gaps between adjacent chunks
    overlap_ok = True
    for i in range(len(intervals) - 1):
        prev_end = intervals[i][1]
        next_start = intervals[i + 1][0]
        overlap = prev_end - next_start
        diagnostics.append(f"Transition Chunk {i+1} -> Chunk {i+2}: overlap={overlap} chars")
        if next_start > prev_end:
            overlap_ok = False
            diagnostics.append(f"GAP DETECTED between index {prev_end} and {next_start}!")

    # Check union coverage of [0, len(text)]
    uncovered: list[tuple[int, int]] = []
    covered_mask = [False] * len(text)
    for st, en in intervals:
        for idx in range(st, min(en, len(text))):
            covered_mask[idx] = True

    in_gap = False
    gap_start = 0
    for idx, is_cov in enumerate(covered_mask):
        if not is_cov and not in_gap:
            in_gap = True
            gap_start = idx
        elif is_cov and in_gap:
            in_gap = False
            uncovered.append((gap_start, idx))
    if in_gap:
        uncovered.append((gap_start, len(text)))

    # Check page markers
    page_markers = re.findall(r"--- PAGE \d+ ---", text)
    missing_markers = []
    chunks_combined = " ".join(chunks)
    for pm in page_markers:
        if pm not in chunks_combined:
            missing_markers.append(pm)

    return ChunkCoverageReport(
        is_fully_covered=len(uncovered) == 0,
        total_text_length=len(text),
        total_chunks=len(chunks),
        uncovered_ranges=uncovered,
        overlap_integrity=overlap_ok,
        page_markers_preserved=len(missing_markers) == 0,
        missing_page_markers=missing_markers,
        diagnostics=diagnostics,
    )


def merge_extractions(
    extractions: list[ResearchPaperExtraction],
    fallback_paper_id: str | None = None,
) -> ResearchPaperExtraction:
    """Merge multiple chunk-level extractions into a unified paper extraction.

    Preserves all unique findings, datasets, metrics, and methodology items
    while deduplicating repeated items.
    """
    if not extractions:
        return ResearchPaperExtraction(paper_id=fallback_paper_id)

    if len(extractions) == 1:
        single = extractions[0]
        if not single.paper_id and fallback_paper_id:
            single.paper_id = fallback_paper_id
        return single

    # Select title: first non-empty
    paper_title = next(
        (e.paper_title for e in extractions if e.paper_title and e.paper_title.strip()),
        None,
    )

    # Select paper_id: first non-empty or fallback
    paper_id = next(
        (e.paper_id for e in extractions if e.paper_id and e.paper_id.strip()),
        fallback_paper_id,
    )

    # Select research problem: longest descriptive statement
    problems = [
        e.research_problem
        for e in extractions
        if e.research_problem and e.research_problem.strip()
    ]
    research_problem = max(problems, key=len) if problems else None

    # Merge research objectives (deduplicated, order preserved)
    objectives_seen: set[str] = set()
    merged_objectives: list[str] = []
    for ext in extractions:
        for obj in ext.research_objectives:
            norm = obj.strip()
            if norm and norm.lower() not in objectives_seen:
                objectives_seen.add(norm.lower())
                merged_objectives.append(norm)

    # Merge methodology (deduplicated by technique name, upgrading placeholders with substantive evidence)
    merged_methodology: list[MethodologyItem] = []
    methodology_indices: dict[str, int] = {}
    for ext in extractions:
        for item in ext.methodology:
            key = item.technique.strip().lower()
            if not key:
                continue

            if key not in methodology_indices:
                methodology_indices[key] = len(merged_methodology)
                merged_methodology.append(item)
            else:
                existing_idx = methodology_indices[key]
                existing = merged_methodology[existing_idx]

                existing_quote = existing.evidence.quote if existing.evidence else ""
                new_quote = item.evidence.quote if item.evidence else ""

                existing_is_substantive = bool(
                    existing.description
                    and not is_mechanism_marked_unavailable(existing.description)
                    and existing_quote
                    and not detect_forward_reference(existing_quote)
                )
                new_is_substantive = bool(
                    item.description
                    and not is_mechanism_marked_unavailable(item.description)
                    and new_quote
                    and not detect_forward_reference(new_quote)
                )

                if not existing_is_substantive and new_is_substantive:
                    # Upgrade placeholder/mention to fully supported substantive item from later chunk
                    merged_methodology[existing_idx] = item
                elif not existing.evidence and item.evidence:
                    existing.evidence = item.evidence
                    if item.description and (
                        not existing.description or len(item.description) > len(existing.description)
                    ):
                        existing.description = item.description
                elif (
                    existing_is_substantive
                    and new_is_substantive
                    and existing_quote.strip().lower() != new_quote.strip().lower()
                ):
                    # Both chunks provide distinct substantive evidence quotes; preserve both
                    merged_methodology.append(item)

    # Merge datasets (deduplicated by dataset name, enriching missing details/evidence)
    merged_datasets: list[DatasetItem] = []
    datasets_indices: dict[str, int] = {}
    for ext in extractions:
        for ds in ext.datasets:
            key = ds.name.strip().lower()
            if not key:
                continue
            if key not in datasets_indices:
                datasets_indices[key] = len(merged_datasets)
                merged_datasets.append(ds)
            else:
                existing_idx = datasets_indices[key]
                existing = merged_datasets[existing_idx]
                if not existing.details and ds.details:
                    existing.details = ds.details
                if not existing.evidence and ds.evidence:
                    existing.evidence = ds.evidence

    # Merge evaluation metrics (deduplicated by metric name + benchmark, preserving distinct evidence)
    merged_metrics: list[EvaluationMetricItem] = []
    metrics_indices: dict[tuple[str, str | None], int] = {}
    for ext in extractions:
        for m in ext.evaluation_metrics:
            key = (
                m.metric_name.strip().lower(),
                m.task_or_benchmark.strip().lower() if m.task_or_benchmark else None,
            )
            if key not in metrics_indices:
                metrics_indices[key] = len(merged_metrics)
                merged_metrics.append(m)
            else:
                existing_idx = metrics_indices[key]
                existing = merged_metrics[existing_idx]
                if (not existing.value and m.value) or (not existing.evidence and m.evidence):
                    merged_metrics[existing_idx] = m
                elif (
                    existing.evidence
                    and m.evidence
                    and existing.evidence.quote
                    and m.evidence.quote
                    and existing.evidence.quote.strip().lower() != m.evidence.quote.strip().lower()
                ):
                    merged_metrics.append(m)

    # Merge tokenization (deduplicated by method name, preserving details)
    merged_tokenization: list[TokenizationItem] = []
    seen_tokenization: set[str] = set()
    for ext in extractions:
        for tok in ext.tokenization:
            key = tok.method.strip().lower()
            if key and key not in seen_tokenization:
                seen_tokenization.add(key)
                merged_tokenization.append(tok)

    # Merge training config (deduplicated by parameter name)
    merged_training_config: list[TrainingConfigItem] = []
    seen_training_params: set[str] = set()
    for ext in extractions:
        for tc in ext.training_config:
            key = tc.parameter.strip().lower()
            if key and key not in seen_training_params:
                seen_training_params.add(key)
                merged_training_config.append(tc)

    # Merge computational cost (deduplicated by hardware/flops)
    merged_computational_cost: list[ComputationalCostItem] = []
    seen_cost_keys: set[str] = set()
    for ext in extractions:
        for cc in ext.computational_cost:
            key = f"{cc.hardware or ''} {cc.flops or ''}".strip().lower()
            if key and key not in seen_cost_keys:
                seen_cost_keys.add(key)
                merged_computational_cost.append(cc)

    # Merge key findings (upgrading forward references to substantive experimental findings)
    merged_findings: list[KeyFindingItem] = []
    seen_finding_quotes: set[str] = set()
    stop_words = {"this", "that", "with", "from", "have", "were", "been", "their", "they", "will", "also"}

    for ext in extractions:
        for f in ext.key_findings:
            quote_key = f.evidence.quote.strip().lower() if f.evidence and f.evidence.quote else ""
            if quote_key and quote_key in seen_finding_quotes:
                continue

            f_claim_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", f.claim.lower())} - stop_words
            upgraded = False

            for idx, existing in enumerate(merged_findings):
                ex_claim_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", existing.claim.lower())} - stop_words
                overlap_words = f_claim_words & ex_claim_words
                union_words = f_claim_words | ex_claim_words
                jaccard = len(overlap_words) / len(union_words) if union_words else 0.0

                if jaccard >= 0.45 or len(overlap_words) >= 4:
                    ex_quote = existing.evidence.quote if existing.evidence and existing.evidence.quote else ""
                    new_quote = f.evidence.quote if f.evidence and f.evidence.quote else ""

                    existing_is_fwd = detect_forward_reference(ex_quote) if ex_quote else True
                    new_is_fwd = detect_forward_reference(new_quote) if new_quote else True

                    # Upgrade forward reference to substantive experimental evidence from later chunk
                    if existing_is_fwd and (not new_is_fwd and new_quote):
                        merged_findings[idx] = f
                        if ex_quote and ex_quote in seen_finding_quotes:
                            seen_finding_quotes.remove(ex_quote)
                        if new_quote:
                            seen_finding_quotes.add(new_quote)
                        upgraded = True
                        break

            if not upgraded:
                if quote_key:
                    seen_finding_quotes.add(quote_key)
                merged_findings.append(f)

    # Merge limitations (deduplicated)
    limits_seen: set[str] = set()
    merged_limitations: list[str] = []
    for ext in extractions:
        for lim in ext.limitations:
            norm = lim.strip()
            if norm and norm.lower() not in limits_seen:
                limits_seen.add(norm.lower())
                merged_limitations.append(norm)

    # Merge future work (deduplicated)
    future_seen: set[str] = set()
    merged_future_work: list[str] = []
    for ext in extractions:
        for fw in ext.future_work:
            norm = fw.strip()
            if norm and norm.lower() not in future_seen:
                future_seen.add(norm.lower())
                merged_future_work.append(norm)

    return ResearchPaperExtraction(
        paper_title=paper_title,
        paper_id=paper_id,
        research_problem=research_problem,
        research_objectives=merged_objectives,
        methodology=merged_methodology,
        datasets=merged_datasets,
        tokenization=merged_tokenization,
        training_config=merged_training_config,
        computational_cost=merged_computational_cost,
        evaluation_metrics=merged_metrics,
        key_findings=merged_findings,
        limitations=merged_limitations,
        future_work=merged_future_work,
    )


class ExtractionService:
    """Service orchestrating research paper information extraction."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or LLMClient(settings=self._settings)

    async def _extract_chunk(
        self,
        chunk_text: str,
        paper_id: str | None = None,
        chunk_id: str | None = None,
    ) -> ResearchPaperExtraction:
        """Process a single text chunk through the extraction prompt and validate."""
        template = load_extraction_prompt_template()

        # Build prompt: replace {source_text} placeholder
        prompt = template.replace("{source_text}", chunk_text)

        logger.info(
            "Sending extraction request to Featherless AI (paper_id=%s, chunk_id=%s, len=%d chars)",
            paper_id,
            chunk_id,
            len(chunk_text),
        )

        raw_response = await self._client.generate(prompt)

        # Parse JSON output
        parsed_dict = clean_and_parse_json(raw_response)

        # Populate provenance if not present in model output
        if paper_id and not parsed_dict.get("paper_id"):
            parsed_dict["paper_id"] = paper_id

        # Attach chunk_id / paper_id to evidence references if model left them null
        for category in ("methodology", "datasets", "evaluation_metrics", "key_findings"):
            items = parsed_dict.get(category)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        ev = item.get("evidence")
                        if isinstance(ev, dict):
                            if paper_id and not ev.get("paper_id"):
                                ev["paper_id"] = paper_id
                            if chunk_id and not ev.get("chunk_id"):
                                ev["chunk_id"] = chunk_id

        # Validate with Pydantic
        try:
            return ResearchPaperExtraction.model_validate(parsed_dict)
        except ValidationError as exc:
            logger.error("Pydantic validation error during extraction: %s", exc)
            raise ValueError(
                f"LLM output failed extraction schema validation: {exc}"
            ) from exc

    async def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """Extract structured information from an ExtractionRequest.

        Automatically splits long texts into manageable chunks without
        silent truncation and merges results.
        """
        chunks = split_text_into_chunks(request.text)

        if not chunks:
            raise ValueError("Input text cannot be empty.")

        chunk_extractions: list[ResearchPaperExtraction] = []
        for idx, chunk in enumerate(chunks):
            # Derive chunk ID for provenance if not provided
            active_chunk_id = (
                request.chunk_id
                if len(chunks) == 1 and request.chunk_id
                else f"{request.chunk_id or 'chunk'}_{idx + 1}"
            )
            extracted = await self._extract_chunk(
                chunk_text=chunk,
                paper_id=request.paper_id,
                chunk_id=active_chunk_id,
            )
            chunk_extractions.append(extracted)

        # Merge results across chunks
        unified = merge_extractions(
            chunk_extractions,
            fallback_paper_id=request.paper_id,
        )

        return ExtractionResponse(
            data=unified,
            model=self._settings.llm_model_name,
        )

    async def extract_from_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractionResponse:
        """Convenience method for ingestion modules supplying raw text and metadata dict.

        Args:
            text: Raw extracted text of the paper.
            metadata: Ingestion metadata containing optional 'paper_id', 'chunk_id', etc.
        """
        meta = metadata or {}
        req = ExtractionRequest(
            text=text,
            paper_id=meta.get("paper_id"),
            chunk_id=meta.get("chunk_id"),
        )
        return await self.extract(req)


# Module-level convenience function
async def extract_paper(
    request: ExtractionRequest,
    client: LLMClient | None = None,
) -> ExtractionResponse:
    """Convenience function to extract structured data using default service."""
    service = ExtractionService(client=client)
    return await service.extract(request)
