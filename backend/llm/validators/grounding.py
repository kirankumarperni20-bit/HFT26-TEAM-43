"""
grounding.py — Citation accuracy, exact source matching, and semantic support verification.

Separates and distinguishes:
1. Exact contiguous quote matching (character-for-character contiguous substring in source text).
2. Documented PDF artifact normalization (resolving discretionary hyphens \ufffe / \xad, line-break hyphens).
3. Non-contiguous or synthesized quotes (combining discontinuous passages with ellipsis '...' or notation substitution).
4. Forward reference detection (rejecting forward references used as mechanism explanations).
5. Explicit ungrounded claims vs. legitimate mentions without mechanism explanation.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

try:
    from backend.llm.schemas.extraction import EvaluationMetricItem, ResearchPaperExtraction
except ModuleNotFoundError:
    from llm.schemas.extraction import EvaluationMetricItem, ResearchPaperExtraction


class SupportClassification(str, Enum):
    """Semantic support classification levels."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MENTIONED_ONLY = "MENTIONED_ONLY"


class QuoteMatchStatus(str, Enum):
    """Classification of quote grounding fidelity against source text."""

    EXACT_CONTIGUOUS = "EXACT_CONTIGUOUS"
    NORMALIZED_ARTIFACT = "NORMALIZED_ARTIFACT"
    NON_CONTIGUOUS_SYNTHESIZED = "NON_CONTIGUOUS_SYNTHESIZED"
    UNGROUNDED = "UNGROUNDED"


# ---------------------------------------------------------------------------
# Documented PDF Artifact Normalization & Helpers
# ---------------------------------------------------------------------------


def normalize_pdf_artifacts(text: str) -> str:
    """Apply documented PDF extraction artifact normalization:
    1. Replaces soft/discretionary hyphen characters (\ufffe, \xad) with standard hyphens.
    2. Normalizes non-breaking spaces (\xa0) to standard spaces.
    3. Strips page-break / form-feed control characters (\x0c, \f).
    4. Resolves soft hyphen line wraps across whitespace (e.g. 'word-\\n next' -> 'word-next').
    5. Resolves line breaks immediately preceding punctuation (e.g. 'word\\n.' -> 'word.').
    6. Resolves math formula kerning and exponent line breaks (e.g. 'P E(' -> 'PE(', 'd -0.5' -> 'd-0.5').
    """
    if not text:
        return ""
    cleaned = text.replace("\ufffe", "-").replace("\xad", "-")
    cleaned = cleaned.replace("\xa0", " ").replace("\x0c", " ").replace("\f", " ")
    cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1-\2", cleaned)
    cleaned = re.sub(r"\s+([.,;:?!])", r"\1", cleaned)
    cleaned = re.sub(r"\bP\s+E\(", "PE(", cleaned)
    cleaned = re.sub(r"(\b[a-zA-Z])\s*([−-])\s*(\d)", r"\1\2\3", cleaned)
    return cleaned


def normalize_whitespace_only(text: str) -> str:
    """Normalize only whitespace (spaces, tabs, newlines) without altering characters or artifacts."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def normalize_text_for_matching(text: str) -> str:
    """Normalize text by applying documented PDF artifact normalization and collapsing whitespace."""
    if not text:
        return ""
    cleaned = normalize_pdf_artifacts(text)
    return re.sub(r"\s+", " ", cleaned.lower()).strip()


# ---------------------------------------------------------------------------
# Exact Quote Grounding Models & Verification
# ---------------------------------------------------------------------------


class GroundingReport(BaseModel):
    """Detailed report on quote veracity and source grounding."""

    is_grounded: bool = Field(
        ...,
        description="True if all evidence quotes were verified verbatim (exact contiguous or documented PDF artifact normalized) in the source text.",
    )
    total_quotes_checked: int = Field(
        default=0,
        description="Total number of evidence quotes provided by the model.",
    )
    grounded_quotes_count: int = Field(
        default=0,
        description="Number of quotes verified to exist as contiguous passages in the source text.",
    )
    ungrounded_quotes: list[str] = Field(
        default_factory=list,
        description="Quotes claimed by the model that could not be verified in the source or were non-contiguous.",
    )
    exact_contiguous_count: int = Field(
        default=0,
        description="Number of quotes matching as exact contiguous substrings without artifact normalization.",
    )
    normalized_artifact_count: int = Field(
        default=0,
        description="Number of quotes matching as contiguous substrings after documented PDF artifact normalization.",
    )
    non_contiguous_count: int = Field(
        default=0,
        description="Number of quotes formed by joining discontinuous passages or using notation substitutions.",
    )
    quote_evaluations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed per-quote match status and rationale.",
    )


def verify_quote_grounding(quote: str | None, source_text: str) -> tuple[QuoteMatchStatus, str]:
    """Verify and classify the grounding fidelity of an individual evidence quote.

    Distinguishes:
    - EXACT_CONTIGUOUS: Quote is a single, uninterrupted verbatim substring of the source text.
    - NORMALIZED_ARTIFACT: Quote matches as a contiguous substring only after documented PDF
      artifact normalization (e.g. resolving soft hyphens \ufffe / \xad or line-break hyphens).
    - NON_CONTIGUOUS_SYNTHESIZED: Quote joins separate discontinuous passages (e.g. using ellipsis
      '...' or '…') or introduces ungrounded notation substitutions (e.g. O(1)).
    - UNGROUNDED: Quote cannot be matched in the source text.
    """
    if not quote or not quote.strip():
        return (QuoteMatchStatus.UNGROUNDED, "Evidence quote is empty or missing.")

    raw_quote = quote.strip()

    # 1. Non-contiguous check: explicit ellipsis joining separate passages
    if "..." in raw_quote or "…" in raw_quote or re.search(r"\.\s*\.\s*\.", raw_quote):
        return (
            QuoteMatchStatus.NON_CONTIGUOUS_SYNTHESIZED,
            "Quote is non-contiguous: combines separate passages using ellipsis ('...'). Every quote must be a single contiguous substring.",
        )

    # 2. Check for notation substitution (e.g. writing O(1) instead of source text words)
    if detect_non_verbatim_notation(raw_quote, source_text):
        return (
            QuoteMatchStatus.NON_CONTIGUOUS_SYNTHESIZED,
            "Quote contains non-verbatim notation substitution (e.g., O(1)) not found in source text.",
        )

    # 3. Exact contiguous check (whitespace-collapsed only, preserving all characters and artifacts)
    ws_quote = normalize_whitespace_only(raw_quote).lower()
    ws_source = normalize_whitespace_only(source_text).lower()

    if ws_quote in ws_source:
        return (QuoteMatchStatus.EXACT_CONTIGUOUS, "Exact contiguous match in source text.")

    # 4. Documented PDF artifact normalization check
    norm_quote = normalize_text_for_matching(raw_quote)
    norm_source = normalize_text_for_matching(source_text)

    if norm_quote in norm_source:
        return (
            QuoteMatchStatus.NORMALIZED_ARTIFACT,
            "Matches source text after documented PDF artifact normalization (soft hyphens / page-break encoding).",
        )

    return (
        QuoteMatchStatus.UNGROUNDED,
        "Quote does not exist in source text (ungrounded or fabricated).",
    )


def verify_evidence_grounding(
    extraction: ResearchPaperExtraction,
    source_text: str,
) -> GroundingReport:
    """Verify that all evidence quotes in an extraction exist verbatim in the source text.

    Distinguishes exact contiguous matches, quotes requiring documented PDF artifact
    normalization, and non-contiguous / synthesized quotes.
    """
    quotes_to_check: list[str] = []

    for item in extraction.methodology:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    for item in extraction.datasets:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    for item in extraction.tokenization:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    for item in extraction.training_config:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    for item in extraction.computational_cost:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    for item in extraction.evaluation_metrics:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    for item in extraction.key_findings:
        if item.evidence and item.evidence.quote:
            quotes_to_check.append(item.evidence.quote)

    total_quotes = len(quotes_to_check)
    if total_quotes == 0:
        return GroundingReport(
            is_grounded=True,
            total_quotes_checked=0,
            grounded_quotes_count=0,
            ungrounded_quotes=[],
            exact_contiguous_count=0,
            normalized_artifact_count=0,
            non_contiguous_count=0,
            quote_evaluations=[],
        )

    ungrounded: list[str] = []
    grounded_count = 0
    exact_count = 0
    norm_count = 0
    non_contiguous_count = 0
    quote_evals: list[dict[str, Any]] = []

    for quote in quotes_to_check:
        status, reason = verify_quote_grounding(quote, source_text)
        quote_evals.append({"quote": quote, "status": status.value, "reason": reason})

        if status == QuoteMatchStatus.EXACT_CONTIGUOUS:
            exact_count += 1
            grounded_count += 1
        elif status == QuoteMatchStatus.NORMALIZED_ARTIFACT:
            norm_count += 1
            grounded_count += 1
        elif status == QuoteMatchStatus.NON_CONTIGUOUS_SYNTHESIZED:
            non_contiguous_count += 1
            ungrounded.append(quote)
        else:
            ungrounded.append(quote)

    return GroundingReport(
        is_grounded=len(ungrounded) == 0,
        total_quotes_checked=total_quotes,
        grounded_quotes_count=grounded_count,
        ungrounded_quotes=ungrounded,
        exact_contiguous_count=exact_count,
        normalized_artifact_count=norm_count,
        non_contiguous_count=non_contiguous_count,
        quote_evaluations=quote_evals,
    )


# ---------------------------------------------------------------------------
# Semantic Support & Advanced Validation Models & Classifiers
# ---------------------------------------------------------------------------

# Patterns indicating forward references rather than substantive explanations
FORWARD_REFERENCE_PATTERNS = [
    re.compile(r"(?:described|discussed|detailed|presented|introduced)\s+in\s+(?:section|appendix|chapter)", re.I),
    re.compile(r"will\s+be\s+(?:described|discussed|detailed|presented)\s+in", re.I),
    re.compile(r"see\s+(?:section|appendix|table|figure)\s+\d+", re.I),
    re.compile(r"in\s+the\s+following\s+sections", re.I),
    re.compile(r"as\s+described\s+in\s+section", re.I),
    re.compile(r"described\s+in\s+more\s+detail\s+in\s+section", re.I),
]

# Patterns representing mathematical notation substitutions (e.g. O(1))
NOTATION_SUBSTITUTION_PATTERNS = [
    re.compile(r"\bO\([1nkd\^]+\)", re.I),
    re.compile(r"\bO\(n\^2\)", re.I),
]

# Patterns indicating limitations of prior art / recurrent baselines
PRIOR_ART_LIMITATION_PATTERNS = [
    re.compile(r"(?:recurrent|rnn|lstm|convolutional|inherently sequential)\b.*precludes\s+parallelization", re.I),
    re.compile(r"sequential\s+nature\s+of\s+(?:recurrent|prior|traditional|existing)", re.I),
]

# Patterns indicating that a mechanism explanation is explicitly marked as unavailable
UNAVAILABLE_EXPLANATION_PATTERNS = [
    re.compile(r"unavailable\s+in\s+supplied\s+text", re.I),
    re.compile(r"not\s+explained\s+in\s+(?:supplied|provided)\s+text", re.I),
    re.compile(r"forward\s+reference\s+only", re.I),
    re.compile(r"mechanism\s+(?:is\s+)?unavailable", re.I),
    re.compile(r"mentioned\s+only", re.I),
    re.compile(r"explanation\s+not\s+(?:available|provided)", re.I),
]


class ItemSemanticEvaluation(BaseModel):
    """Detailed semantic evaluation for an individual extracted element."""

    item_type: str = Field(..., description="Category: methodology, dataset, metric, finding, limitation, etc.")
    item_label: str = Field(..., description="Identifier or primary label of the item.")
    quote: str | None = Field(default=None, description="Evidence quote supplied.")
    exact_match: bool = Field(..., description="Whether the quote exists verbatim in source text.")
    quote_match_status: QuoteMatchStatus = Field(
        default=QuoteMatchStatus.EXACT_CONTIGUOUS,
        description="EXACT_CONTIGUOUS, NORMALIZED_ARTIFACT, NON_CONTIGUOUS_SYNTHESIZED, or UNGROUNDED.",
    )
    support_status: SupportClassification = Field(..., description="SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED.")
    reason: str = Field(..., description="Detailed rationale for the classification.")


class SemanticValidationReport(BaseModel):
    """Comprehensive semantic validation report keeping exact matching and semantics distinct."""

    is_grounded_exact: bool = Field(..., description="True if all quotes matched exact source text.")
    is_semantically_supported: bool = Field(..., description="True if all items have valid semantic support.")
    total_items_checked: int = Field(default=0)
    supported_count: int = Field(default=0)
    partially_supported_count: int = Field(default=0)
    unsupported_count: int = Field(default=0)
    contradicted_count: int = Field(default=0)
    mentioned_only_count: int = Field(default=0)
    evaluations: list[ItemSemanticEvaluation] = Field(default_factory=list)
    forward_references: list[str] = Field(default_factory=list)
    misattributed_limitations: list[str] = Field(default_factory=list)
    fabricated_metrics: list[str] = Field(default_factory=list)
    unexpected_nonempty_fields: list[str] = Field(default_factory=list)


def detect_forward_reference(text: str) -> bool:
    """Check if the text merely points forward to a future section without explaining."""
    if not text:
        return False
    return any(p.search(text) for p in FORWARD_REFERENCE_PATTERNS)


def detect_non_verbatim_notation(quote: str, source_text: str) -> bool:
    """Detect if the quote substituted mathematical notation (e.g., O(1)) instead of verbatim text."""
    if not quote:
        return False
    has_notation = any(p.search(quote) for p in NOTATION_SUBSTITUTION_PATTERNS)
    if has_notation:
        normalized_quote = normalize_text_for_matching(quote)
        normalized_source = normalize_text_for_matching(source_text)
        if normalized_quote not in normalized_source:
            return True
    return False


def detect_fabricated_metric_value(metric_val: str | None, source_text: str) -> bool:
    """Verify whether a numerical metric value actually appears in the source text."""
    if not metric_val:
        return False
    tokens = re.findall(r"\b\d+(?:\.\d+)?%?", metric_val)
    if not tokens:
        return False
    normalized_source = normalize_text_for_matching(source_text)
    for token in tokens:
        norm_token = normalize_text_for_matching(token)
        if norm_token not in normalized_source:
            return True
    return False


def detect_prior_art_misattribution(limitation_text: str) -> bool:
    """Check if a limitation belonging to prior architectures is misattributed to the proposed model."""
    if not limitation_text:
        return False
    return any(p.search(limitation_text) for p in PRIOR_ART_LIMITATION_PATTERNS)


def is_mechanism_marked_unavailable(description: str | None) -> bool:
    """Check if description explicitly states the mechanism explanation is unavailable."""
    if not description or not description.strip():
        return True
    return any(p.search(description) for p in UNAVAILABLE_EXPLANATION_PATTERNS)


def extract_table_context_for_quote(quote: str, source_text: str) -> str | None:
    """If quote is located within a table block in source_text, extract the enclosing table caption and column headers."""
    if not quote or not source_text:
        return None
    norm_q = normalize_whitespace_only(quote).lower()
    norm_s = normalize_whitespace_only(source_text).lower()
    pos = norm_s.find(norm_q)
    if pos == -1:
        norm_q = normalize_text_for_matching(quote)
        norm_s = normalize_text_for_matching(source_text)
        pos = norm_s.find(norm_q)
    if pos == -1:
        return None

    window_start = max(0, pos - 1500)
    preceding = norm_s[window_start:pos]
    table_match = list(re.finditer(r"\btable\s+(\d+)\s*:\s*([^\n\r]+)", preceding, re.I))
    if not table_match:
        table_match = list(re.finditer(r"\btable\s+(\d+)\b", preceding, re.I))
    if table_match:
        last_table = table_match[-1]
        return preceding[last_table.start():]
    return None


def classify_claim_support(
    claim: str,
    quote: str | None,
    source_text: str,
    table_header: str | None = None,
) -> tuple[SupportClassification, str]:
    """Classify semantic support of a claim given an evidence quote, source text, and optional table header.

    Distinguishes:
    - Exact contiguous quote match
    - Table-aware quote grounding (row quote contextualized by table header/caption)
    - Quote requiring documented PDF artifact normalization
    - Non-contiguous or synthesized quote
    - Forward reference used as mechanism evidence
    - Mention without mechanism explanation
    """
    # Check for simple mention where mechanism explanation was correctly withheld
    if is_mechanism_marked_unavailable(claim) and (not quote or not quote.strip()):
        return (
            SupportClassification.MENTIONED_ONLY,
            "Technique mentioned in source text; explanation correctly marked as unavailable in supplied excerpt.",
        )

    if not quote or not quote.strip():
        return (
            SupportClassification.UNSUPPORTED,
            "No evidence quote provided for claim.",
        )

    # 1. Exact quote existence & continuity check
    match_status, match_reason = verify_quote_grounding(quote, source_text)

    if match_status == QuoteMatchStatus.NON_CONTIGUOUS_SYNTHESIZED:
        return (SupportClassification.UNSUPPORTED, match_reason)

    if match_status == QuoteMatchStatus.UNGROUNDED:
        return (SupportClassification.UNSUPPORTED, match_reason)

    # 2. Check for forward reference as mechanism evidence
    if detect_forward_reference(quote):
        return (
            SupportClassification.UNSUPPORTED,
            "Evidence is a forward reference pointing to a future section; cannot serve as mechanism evidence.",
        )

    # 3. Check table header if explicitly provided
    if table_header and table_header.strip():
        th_status, _ = verify_quote_grounding(table_header, source_text)
        if th_status == QuoteMatchStatus.UNGROUNDED:
            return (
                SupportClassification.UNSUPPORTED,
                f"Table header '{table_header}' not found in source text.",
            )

    # 4. Semantic relevance check (ensure quote addresses key concept in claim)
    claim_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", claim.lower())}
    quote_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", quote.lower())}
    if table_header:
        quote_words |= {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", table_header.lower())}

    stop_words = {"this", "that", "with", "from", "have", "were", "been", "their", "they", "will", "also"}
    substantive_claim_words = claim_words - stop_words
    overlap = substantive_claim_words & quote_words

    if substantive_claim_words and len(overlap) == 0:
        # Table-aware fallback: check if quote is a row within a table whose header/caption provides context
        table_context = extract_table_context_for_quote(quote, source_text)
        if table_context:
            table_words = {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", table_context.lower())}
            table_overlap = substantive_claim_words & table_words
            if table_overlap:
                # 1. Reject if claim contains any fabricated numerical value
                if detect_fabricated_metric_value(claim, source_text):
                    return (
                        SupportClassification.UNSUPPORTED,
                        f"Claim contains fabricated numerical value not found in source text: '{claim}'",
                    )

                # 2. Ensure that significant numbers in the claim actually appear in this specific row quote
                claim_nums = re.findall(r"\b\d+(?:\.\d+)?", claim)
                quote_nums = re.findall(r"\b\d+(?:\.\d+)?", quote)
                sig_claim_nums = [n for n in claim_nums if float(n) > 2 or "." in n] or claim_nums

                if not sig_claim_nums or all(
                    any(cn == qn or (cn in qn and len(qn) >= 3) for qn in quote_nums)
                    for cn in sig_claim_nums
                ):
                    return (
                        SupportClassification.SUPPORTED,
                        f"Quote matches table row and table context supports claim '{', '.join(sorted(table_overlap))}' with grounded numeric value.",
                    )
        return (
            SupportClassification.UNSUPPORTED,
            "Quote exists in text but does not semantically support the claim (zero topical word overlap).",
        )

    if len(overlap) < min(2, len(substantive_claim_words)):
        return (
            SupportClassification.PARTIALLY_SUPPORTED,
            f"Quote partially supports claim with minimal overlap ({', '.join(overlap)}).",
        )

    if match_status == QuoteMatchStatus.NORMALIZED_ARTIFACT:
        return (
            SupportClassification.SUPPORTED,
            "Quote matches source text after documented PDF artifact normalization and directly supports the claim.",
        )

    return (
        SupportClassification.SUPPORTED,
        "Quote exists verbatim and directly supports the claim semantically.",
    )


def verify_extraction_semantic_support(
    extraction: ResearchPaperExtraction,
    source_text: str,
) -> SemanticValidationReport:
    """Perform comprehensive validation keeping exact quote matching separate from semantic support."""
    exact_report = verify_evidence_grounding(extraction, source_text)

    evaluations: list[ItemSemanticEvaluation] = []
    forward_refs: list[str] = []
    misattributed_limits: list[str] = []
    fabricated_metrics: list[str] = []
    unexpected_fields: list[str] = []

    # Check Methodology
    for m in extraction.methodology:
        quote = m.evidence.quote if m.evidence else None
        if quote:
            quote_status, quote_reason = verify_quote_grounding(quote, source_text)
            exact_match = quote_status in (QuoteMatchStatus.EXACT_CONTIGUOUS, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        else:
            quote_status = QuoteMatchStatus.UNGROUNDED
            exact_match = False

        # Check for simple mention without mechanism explanation
        if is_mechanism_marked_unavailable(m.description) and not quote:
            status = SupportClassification.MENTIONED_ONLY
            reason = "Technique mentioned in source text; explanation correctly marked as unavailable in supplied excerpt."
        elif is_mechanism_marked_unavailable(m.description) and quote and not detect_forward_reference(quote):
            status = SupportClassification.MENTIONED_ONLY
            reason = f"Technique mentioned in source text with grounded mention quote."
        else:
            status, reason = classify_claim_support(m.description or m.technique, quote, source_text)

        if quote and detect_forward_reference(quote):
            forward_refs.append(f"Method '{m.technique}': {quote}")

        evaluations.append(
            ItemSemanticEvaluation(
                item_type="methodology",
                item_label=m.technique,
                quote=quote,
                exact_match=exact_match,
                quote_match_status=quote_status,
                support_status=status,
                reason=reason,
            )
        )

    # Check Evaluation Metrics
    for metric in extraction.evaluation_metrics:
        quote = metric.evidence.quote if metric.evidence else None
        if quote:
            quote_status, quote_reason = verify_quote_grounding(quote, source_text)
            exact_match = quote_status in (QuoteMatchStatus.EXACT_CONTIGUOUS, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        else:
            quote_status = QuoteMatchStatus.UNGROUNDED
            exact_match = False

        if detect_fabricated_metric_value(metric.value, source_text):
            fabricated_metrics.append(f"{metric.metric_name}: {metric.value}")
            evaluations.append(
                ItemSemanticEvaluation(
                    item_type="evaluation_metric",
                    item_label=metric.metric_name,
                    quote=quote,
                    exact_match=exact_match,
                    quote_match_status=quote_status,
                    support_status=SupportClassification.UNSUPPORTED,
                    reason=f"Fabricated metric value '{metric.value}' not found in source text.",
                )
            )
        else:
            table_hdr = metric.evidence.table_header if metric.evidence else None
            status, reason = classify_claim_support(
                f"{metric.metric_name} {metric.value}", quote, source_text, table_header=table_hdr
            )
            evaluations.append(
                ItemSemanticEvaluation(
                    item_type="evaluation_metric",
                    item_label=metric.metric_name,
                    quote=quote,
                    exact_match=exact_match,
                    quote_match_status=quote_status,
                    support_status=status,
                    reason=reason,
                )
            )

    # Check Key Findings
    for kf in extraction.key_findings:
        quote = kf.evidence.quote if kf.evidence else None
        if quote:
            quote_status, quote_reason = verify_quote_grounding(quote, source_text)
            exact_match = quote_status in (QuoteMatchStatus.EXACT_CONTIGUOUS, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        else:
            quote_status = QuoteMatchStatus.UNGROUNDED
            exact_match = False

        table_hdr = kf.evidence.table_header if kf.evidence else None
        status, reason = classify_claim_support(kf.claim, quote, source_text, table_header=table_hdr)
        if quote and detect_forward_reference(quote):
            forward_refs.append(f"Finding '{kf.claim[:40]}...': {quote}")
        evaluations.append(
            ItemSemanticEvaluation(
                item_type="key_finding",
                item_label=kf.claim[:50],
                quote=quote,
                exact_match=exact_match,
                quote_match_status=quote_status,
                support_status=status,
                reason=reason,
            )
        )

    # Check Tokenization
    for tok in extraction.tokenization:
        quote = tok.evidence.quote if tok.evidence else None
        if quote:
            quote_status, quote_reason = verify_quote_grounding(quote, source_text)
            exact_match = quote_status in (QuoteMatchStatus.EXACT_CONTIGUOUS, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        else:
            quote_status = QuoteMatchStatus.UNGROUNDED
            exact_match = False
        claim_str = f"{tok.method} {tok.vocabulary_size or ''}".strip()
        status, reason = classify_claim_support(claim_str, quote, source_text)
        evaluations.append(
            ItemSemanticEvaluation(
                item_type="tokenization",
                item_label=tok.method[:50],
                quote=quote,
                exact_match=exact_match,
                quote_match_status=quote_status,
                support_status=status,
                reason=reason,
            )
        )

    # Check Training Config
    for tc in extraction.training_config:
        quote = tc.evidence.quote if tc.evidence else None
        if quote:
            quote_status, quote_reason = verify_quote_grounding(quote, source_text)
            exact_match = quote_status in (QuoteMatchStatus.EXACT_CONTIGUOUS, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        else:
            quote_status = QuoteMatchStatus.UNGROUNDED
            exact_match = False
        table_hdr = tc.evidence.table_header if tc.evidence else None
        claim_str = f"{tc.parameter} {tc.value or ''}".strip()
        status, reason = classify_claim_support(claim_str, quote, source_text, table_header=table_hdr)
        evaluations.append(
            ItemSemanticEvaluation(
                item_type="training_config",
                item_label=tc.parameter[:50],
                quote=quote,
                exact_match=exact_match,
                quote_match_status=quote_status,
                support_status=status,
                reason=reason,
            )
        )

    # Check Computational Cost
    for cc in extraction.computational_cost:
        quote = cc.evidence.quote if cc.evidence else None
        if quote:
            quote_status, quote_reason = verify_quote_grounding(quote, source_text)
            exact_match = quote_status in (QuoteMatchStatus.EXACT_CONTIGUOUS, QuoteMatchStatus.NORMALIZED_ARTIFACT)
        else:
            quote_status = QuoteMatchStatus.UNGROUNDED
            exact_match = False
        table_hdr = cc.evidence.table_header if cc.evidence else None
        claim_str = f"{cc.hardware or ''} {cc.training_time or ''} {cc.flops or ''}".strip()
        status, reason = classify_claim_support(claim_str, quote, source_text, table_header=table_hdr)
        evaluations.append(
            ItemSemanticEvaluation(
                item_type="computational_cost",
                item_label=(cc.flops or cc.hardware or cc.training_time or "cost")[:50],
                quote=quote,
                exact_match=exact_match,
                quote_match_status=quote_status,
                support_status=status,
                reason=reason,
            )
        )

    # Check Limitations (verify attribution)
    for lim in extraction.limitations:
        if detect_prior_art_misattribution(lim):
            misattributed_limits.append(lim)
            evaluations.append(
                ItemSemanticEvaluation(
                    item_type="limitation",
                    item_label=lim[:50],
                    quote=None,
                    exact_match=False,
                    quote_match_status=QuoteMatchStatus.UNGROUNDED,
                    support_status=SupportClassification.UNSUPPORTED,
                    reason="Limitation of prior art (e.g. RNN/recurrence bottleneck) misattributed to proposed method.",
                )
            )
        else:
            evaluations.append(
                ItemSemanticEvaluation(
                    item_type="limitation",
                    item_label=lim[:50],
                    quote=None,
                    exact_match=True,
                    quote_match_status=QuoteMatchStatus.EXACT_CONTIGUOUS,
                    support_status=SupportClassification.SUPPORTED,
                    reason="Proposed method limitation properly attributed.",
                )
            )

    # Check Unknown / Absent fields (e.g., future_work when absent from source chunk)
    norm_source = normalize_text_for_matching(source_text)
    if "future" not in norm_source and extraction.future_work:
        for fw in extraction.future_work:
            unexpected_fields.append(f"future_work: {fw}")
            evaluations.append(
                ItemSemanticEvaluation(
                    item_type="unknown_field",
                    item_label="future_work",
                    quote=None,
                    exact_match=False,
                    quote_match_status=QuoteMatchStatus.UNGROUNDED,
                    support_status=SupportClassification.UNSUPPORTED,
                    reason="Field 'future_work' populated when topic is absent from supplied source text.",
                )
            )

    supported_count = sum(1 for e in evaluations if e.support_status == SupportClassification.SUPPORTED)
    partially_supported_count = sum(
        1 for e in evaluations if e.support_status == SupportClassification.PARTIALLY_SUPPORTED
    )
    mentioned_only_count = sum(
        1 for e in evaluations if e.support_status == SupportClassification.MENTIONED_ONLY
    )
    unsupported_count = sum(1 for e in evaluations if e.support_status == SupportClassification.UNSUPPORTED)
    contradicted_count = sum(1 for e in evaluations if e.support_status == SupportClassification.CONTRADICTED)

    is_semantically_supported = (
        unsupported_count == 0
        and contradicted_count == 0
        and len(forward_refs) == 0
        and len(misattributed_limits) == 0
        and len(fabricated_metrics) == 0
        and len(unexpected_fields) == 0
    )

    return SemanticValidationReport(
        is_grounded_exact=exact_report.is_grounded,
        is_semantically_supported=is_semantically_supported,
        total_items_checked=len(evaluations),
        supported_count=supported_count,
        partially_supported_count=partially_supported_count,
        unsupported_count=unsupported_count,
        contradicted_count=contradicted_count,
        mentioned_only_count=mentioned_only_count,
        evaluations=evaluations,
        forward_references=forward_refs,
        misattributed_limitations=misattributed_limits,
        fabricated_metrics=fabricated_metrics,
        unexpected_nonempty_fields=unexpected_fields,
    )
