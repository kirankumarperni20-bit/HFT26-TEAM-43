"""
claim_service.py — Service for decomposing synthesis statements into atomic claims.

Orchestrates:
- Statement extraction from CrossPaperSynthesis or explicit statement inputs.
- LLM prompt rendering and generation via shared LLMClient.
- Robust JSON parsing with clean_and_parse_json.
- Deterministic Python-side character offset calculation (Unicode character offsets).
- Strict verification that source_text[start_offset:end_offset] == claim_text.
- Identification and flagging of paraphrased or unaligned claims.
- Preservation of paper IDs and inherited source references.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    from backend.llm.client import LLMClient
    from backend.llm.config import LLMSettings, get_settings
    from backend.llm.schemas.claim import (
        AlignmentStatus,
        AtomicClaim,
        ClaimDecompositionRequest,
        ClaimDecompositionResponse,
        ClaimType,
        StatementInputItem,
    )
    from backend.llm.schemas.synthesis import CrossPaperSynthesis, SourceReference
    from backend.llm.services.extraction_service import clean_and_parse_json
except ModuleNotFoundError:
    from llm.client import LLMClient
    from llm.config import LLMSettings, get_settings
    from llm.schemas.claim import (
        AlignmentStatus,
        AtomicClaim,
        ClaimDecompositionRequest,
        ClaimDecompositionResponse,
        ClaimType,
        StatementInputItem,
    )
    from llm.schemas.synthesis import CrossPaperSynthesis, SourceReference
    from llm.services.extraction_service import clean_and_parse_json

logger = logging.getLogger(__name__)

# Path to the claim decomposition prompt template
PROMPT_FILE_PATH = Path(__file__).parent.parent / "prompts" / "claim_decomposition.txt"


@lru_cache(maxsize=1)
def load_decomposition_prompt_template() -> str:
    """Load and cache the claim decomposition prompt template from disk."""
    if not PROMPT_FILE_PATH.is_file():
        raise FileNotFoundError(
            f"Claim decomposition prompt template not found at: {PROMPT_FILE_PATH}"
        )
    content = PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
    if "{statements_json}" not in content:
        raise ValueError(
            "Claim decomposition prompt template missing required placeholder: '{statements_json}'"
        )
    return content


def extract_statements_from_synthesis(synthesis: CrossPaperSynthesis) -> list[StatementInputItem]:
    """Extract substantive comparative statements from a CrossPaperSynthesis object."""
    statements: list[StatementInputItem] = []
    stmt_idx = 1

    all_paper_ids = [
        p.get("paper_id") for p in synthesis.papers_analyzed if p.get("paper_id")
    ]

    # 1. Overview
    if synthesis.overview and synthesis.overview.strip():
        statements.append(
            StatementInputItem(
                statement_id=f"stmt_{stmt_idx:03d}",
                text=synthesis.overview.strip(),
                involved_papers=all_paper_ids,
                source_references=[],
            )
        )
        stmt_idx += 1

    # 2. Methodological Similarities
    for sim in synthesis.methodological_similarities:
        if sim.description and sim.description.strip():
            pids = list(sim.paper_details.keys()) if sim.paper_details else all_paper_ids
            statements.append(
                StatementInputItem(
                    statement_id=f"stmt_{stmt_idx:03d}",
                    text=sim.description.strip(),
                    involved_papers=pids,
                    source_references=sim.evidence,
                )
            )
            stmt_idx += 1

    # 3. Methodological Differences
    for diff in synthesis.methodological_differences:
        if diff.description and diff.description.strip():
            pids = list(diff.paper_details.keys()) if diff.paper_details else all_paper_ids
            statements.append(
                StatementInputItem(
                    statement_id=f"stmt_{stmt_idx:03d}",
                    text=diff.description.strip(),
                    involved_papers=pids,
                    source_references=diff.evidence,
                )
            )
            stmt_idx += 1

    # 4. Dataset and Experimental Differences
    for exp in synthesis.dataset_and_experimental_differences:
        if exp.description and exp.description.strip():
            pids = list(exp.paper_details.keys()) if exp.paper_details else all_paper_ids
            statements.append(
                StatementInputItem(
                    statement_id=f"stmt_{stmt_idx:03d}",
                    text=exp.description.strip(),
                    involved_papers=pids,
                    source_references=exp.evidence,
                )
            )
            stmt_idx += 1

    # 5. Findings Agreement
    for fa in synthesis.findings_agreement:
        if fa.description and fa.description.strip():
            pids = list(fa.paper_findings.keys()) if fa.paper_findings else all_paper_ids
            statements.append(
                StatementInputItem(
                    statement_id=f"stmt_{stmt_idx:03d}",
                    text=fa.description.strip(),
                    involved_papers=pids,
                    source_references=fa.evidence,
                )
            )
            stmt_idx += 1

    # 6. Findings Disagreement
    for fd in synthesis.findings_disagreement:
        if fd.description and fd.description.strip():
            pids = list(fd.paper_findings.keys()) if fd.paper_findings else all_paper_ids
            statements.append(
                StatementInputItem(
                    statement_id=f"stmt_{stmt_idx:03d}",
                    text=fd.description.strip(),
                    involved_papers=pids,
                    source_references=fd.evidence,
                )
            )
            stmt_idx += 1

    # 7. Research Limitations and Gaps
    for gap in synthesis.research_limitations_and_gaps:
        text = gap.gap.strip()
        if gap.details and gap.details.strip():
            text = f"{text}. {gap.details.strip()}"
        statements.append(
            StatementInputItem(
                statement_id=f"stmt_{stmt_idx:03d}",
                text=text,
                involved_papers=gap.affected_papers or all_paper_ids,
                source_references=gap.evidence,
            )
        )
        stmt_idx += 1

    # 8. Future Directions
    for fut in synthesis.future_research_directions:
        text = fut.direction.strip()
        if fut.potential_impact and fut.potential_impact.strip():
            text = f"{text}. {fut.potential_impact.strip()}"
        statements.append(
            StatementInputItem(
                statement_id=f"stmt_{stmt_idx:03d}",
                text=text,
                involved_papers=fut.proposed_by or all_paper_ids,
                source_references=fut.evidence,
            )
        )
        stmt_idx += 1

    return statements


def compute_deterministic_offsets(
    claim_text: str,
    source_text: str,
    last_end_offset: int = 0,
) -> tuple[int | None, int | None, AlignmentStatus, str | None]:
    """Deterministically locate claim_text inside source_text using Python Unicode indexing.

    Returns:
        (start_offset, end_offset, alignment_status, validation_notes)
    """
    if not claim_text or not claim_text.strip():
        return None, None, AlignmentStatus.UNALIGNED, "Claim text is empty."

    # Search for exact substring
    pattern = re.escape(claim_text)
    matches = [m.start() for m in re.finditer(pattern, source_text)]

    if not matches:
        return (
            None,
            None,
            AlignmentStatus.PARAPHRASED,
            "Claim text is not an exact contiguous substring of the source statement (paraphrased or altered).",
        )

    if len(matches) == 1:
        start = matches[0]
        end = start + len(claim_text)
        assert source_text[start:end] == claim_text, "Internal offset calculation verification failed."
        return start, end, AlignmentStatus.EXACT, None

    # Multiple matches exist: attempt monotonic resolution from last_end_offset
    monotonic_matches = [m for m in matches if m >= last_end_offset]
    if monotonic_matches:
        start = monotonic_matches[0]
        end = start + len(claim_text)
        assert source_text[start:end] == claim_text, "Internal offset calculation verification failed."
        return start, end, AlignmentStatus.EXACT, "Resolved sequential occurrence monotonically."

    # Cannot resolve unambiguously
    start = matches[0]
    end = start + len(claim_text)
    return (
        start,
        end,
        AlignmentStatus.AMBIGUOUS,
        f"Multiple occurrences ({len(matches)}) found in source text without monotonic alignment.",
    )


class ClaimDecompositionService:
    """Service orchestrating atomic claim decomposition and verification."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: LLMSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or LLMClient(settings=self._settings)

    async def decompose(self, request: ClaimDecompositionRequest) -> ClaimDecompositionResponse:
        """Decompose synthesis statements into atomic claims with verifiable Unicode offsets."""
        # 1. Resolve statements to process
        if request.statements is not None:
            statements = request.statements
        elif request.synthesis is not None:
            statements = extract_statements_from_synthesis(request.synthesis)
        else:
            raise ValueError("No synthesis or statements provided for claim decomposition.")

        if not statements:
            return ClaimDecompositionResponse(
                claims=[],
                total_claims=0,
                source_statement_count=0,
                exact_alignment_count=0,
                flagged_claims_count=0,
                model=self._settings.llm_model_name,
            )

        # Index statements by statement_id for verification
        stmt_map: dict[str, StatementInputItem] = {s.statement_id: s for s in statements}
        all_known_paper_ids: set[str] = {
            pid for s in statements for pid in s.involved_papers if pid
        }

        # 2. Render prompt
        template = load_decomposition_prompt_template()
        statements_payload = [
            {
                "statement_id": s.statement_id,
                "text": s.text,
                "involved_papers": s.involved_papers,
            }
            for s in statements
        ]
        prompt = template.replace(
            "{statements_json}",
            json.dumps(statements_payload, indent=2, ensure_ascii=False),
        )

        logger.info(
            "Sending claim decomposition request for %d statements to model %s",
            len(statements),
            self._settings.llm_model_name,
        )

        raw_response = await self._client.generate(prompt)

        # 3. Clean and parse JSON response
        parsed_dict = clean_and_parse_json(raw_response)
        decompositions = parsed_dict.get("decompositions", [])
        if not isinstance(decompositions, list):
            raise ValueError(
                f"Model response 'decompositions' field must be a list. Received: {type(decompositions)}"
            )

        # 4. Deterministic claim creation and offset validation
        decomposed_claims: list[AtomicClaim] = []
        exact_align_count = 0
        flagged_count = 0
        global_claim_idx = 1

        for decomp_group in decompositions:
            stmt_id = decomp_group.get("statement_id")
            if stmt_id not in stmt_map:
                logger.warning("Model returned unknown statement_id: %s. Skipping.", stmt_id)
                continue

            source_stmt = stmt_map[stmt_id]
            source_text = source_stmt.text
            raw_claims = decomp_group.get("atomic_claims", [])

            last_offset = 0
            for item in raw_claims:
                claim_text = item.get("claim_text", "").strip()
                if not claim_text:
                    continue

                # Parse claim_type safely
                raw_type = item.get("claim_type", "FACTUAL").upper()
                try:
                    claim_type = ClaimType(raw_type)
                except ValueError:
                    claim_type = ClaimType.FACTUAL

                # Calculate deterministic offsets in Python
                start_off, end_off, align_status, notes = compute_deterministic_offsets(
                    claim_text=claim_text,
                    source_text=source_text,
                    last_end_offset=last_offset,
                )

                if align_status == AlignmentStatus.EXACT:
                    exact_align_count += 1
                    is_exact = True
                    if end_off is not None:
                        last_offset = end_off
                else:
                    flagged_count += 1
                    is_exact = False

                # Validate and filter paper IDs against input statement
                raw_pids = item.get("involved_papers", [])
                validated_pids: list[str] = []
                for pid in raw_pids:
                    if pid in source_stmt.involved_papers or pid in all_known_paper_ids:
                        validated_pids.append(pid)
                    else:
                        logger.warning(
                            "Rejected hallucinated paper_id '%s' for claim in statement %s",
                            pid,
                            stmt_id,
                        )

                if not validated_pids:
                    # Inherit from parent statement if omitted
                    validated_pids = list(source_stmt.involved_papers)

                # Inherit source references matching involved papers
                relevant_refs: list[SourceReference] = []
                for ref in source_stmt.source_references:
                    if ref.paper_id in validated_pids or not validated_pids:
                        relevant_refs.append(ref)

                claim_id = f"claim_{global_claim_idx:03d}"
                global_claim_idx += 1

                claim_obj = AtomicClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    source_text=source_text,
                    start_offset=start_off,
                    end_offset=end_off,
                    claim_type=claim_type,
                    involved_papers=validated_pids,
                    source_references=relevant_refs,
                    alignment_status=align_status,
                    is_exact_substring=is_exact,
                    statement_id=stmt_id,
                    validation_notes=notes,
                )
                decomposed_claims.append(claim_obj)

        return ClaimDecompositionResponse(
            claims=decomposed_claims,
            total_claims=len(decomposed_claims),
            source_statement_count=len(statements),
            exact_alignment_count=exact_align_count,
            flagged_claims_count=flagged_count,
            model=self._settings.llm_model_name,
        )


async def decompose_claims(
    request: ClaimDecompositionRequest,
    client: LLMClient | None = None,
    settings: LLMSettings | None = None,
) -> ClaimDecompositionResponse:
    """Convenience helper to decompose claims directly."""
    service = ClaimDecompositionService(client=client, settings=settings)
    return await service.decompose(request)
