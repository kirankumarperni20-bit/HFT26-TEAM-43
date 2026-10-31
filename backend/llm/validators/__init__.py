"""
validators — Output validation utilities for LLM responses.

This module houses functions that verify LLM-generated content:
    - Citation accuracy checking
    - Hallucination detection heuristics
    - Structured-output format validation
    - Length / safety guardrails
"""

from __future__ import annotations

try:
    from backend.llm.validators.grounding import GroundingReport, verify_evidence_grounding
except ModuleNotFoundError:
    from llm.validators.grounding import GroundingReport, verify_evidence_grounding

__all__ = [
    "GroundingReport",
    "verify_evidence_grounding",
]
