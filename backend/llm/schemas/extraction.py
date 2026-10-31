"""
extraction.py — Pydantic schemas for structured research paper extraction.

Provides data contracts for parsing, validating, and structuring key
scientific components extracted from research paper text.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Grounding / Provenance Models
# ---------------------------------------------------------------------------


class EvidenceReference(BaseModel):
    """Grounding reference locating where extracted information originates."""

    paper_id: str | None = Field(
        default=None,
        description="Source paper identifier if present in the input text or metadata.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Source chunk identifier if present in the input text or metadata.",
    )
    section: str | None = Field(
        default=None,
        description="Section heading or title where this item appeared, if explicitly stated.",
    )
    quote: str | None = Field(
        default=None,
        description="Verbatim excerpt from the source text grounding this item.",
    )
    table_header: str | None = Field(
        default=None,
        description="Table header or column name grounding this value when extracted from a table.",
    )
    table_row: str | None = Field(
        default=None,
        description="Specific table row text when extracted from a table.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_evidence(cls, data: Any) -> Any:
        """Coerce raw strings or lists returned by LLM into a valid EvidenceReference dictionary."""
        if data is None:
            return None
        if isinstance(data, str):
            return {"quote": data.strip()}
        if isinstance(data, list):
            if not data:
                return None
            first = data[0]
            if isinstance(first, dict):
                return first
            return {"quote": " ".join(str(x) for x in data if x).strip()}
        return data


# ---------------------------------------------------------------------------
# Domain Item Schemas
# ---------------------------------------------------------------------------


class MethodologyItem(BaseModel):
    """Specific algorithmic, experimental, or architectural technique."""

    technique: str = Field(
        ...,
        description="Name or title of the method, architecture, or algorithmic approach.",
    )
    description: str | None = Field(
        default=None,
        description="Concise description of the method's design, formulation, or execution.",
    )
    category: str | None = Field(
        default=None,
        description="Methodological category: 'model_architecture', 'attention_mechanism', 'regularization', 'positional_encoding', etc.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source evidence or quote grounding this methodology.",
    )


class DatasetItem(BaseModel):
    """Dataset used, evaluated against, or introduced in the research paper."""

    name: str = Field(
        ...,
        description="Name or identifier of the dataset (e.g. 'ImageNet', 'SQuAD 2.0').",
    )
    details: str | None = Field(
        default=None,
        description="Size, split details, domain, or modality if explicitly mentioned.",
    )
    category: str | None = Field(
        default=None,
        description="Taxonomy category: 'benchmark', 'pretraining_corpus', 'evaluation_set', 'tokenization_vocabulary', etc.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source evidence or quote grounding this dataset reference.",
    )


class TokenizationItem(BaseModel):
    """Subword segmentation method, byte-pair encoding, or vocabulary specification."""

    method: str = Field(
        ...,
        description="Tokenization method name (e.g. 'Byte-Pair Encoding (BPE)', 'WordPiece', 'SentencePiece').",
    )
    vocabulary_size: str | None = Field(
        default=None,
        description="Reported vocabulary size (e.g. '37000 tokens', '32000 wordpieces').",
    )
    details: str | None = Field(
        default=None,
        description="Shared vocabulary details, source languages, or preprocessing rules.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source quote grounding this tokenization method.",
    )


class TrainingConfigItem(BaseModel):
    """Training hyperparameter, optimizer, regularization, or schedule specification."""

    parameter: str = Field(
        ...,
        description="Name of training setting (e.g. 'Optimizer', 'Learning Rate Schedule', 'Warmup Steps', 'Dropout Rate', 'Label Smoothing').",
    )
    value: str | None = Field(
        default=None,
        description="Configured value or formula (e.g. 'Adam with beta1=0.9, beta2=0.98', '4000 steps', '0.1').",
    )
    details: str | None = Field(
        default=None,
        description="Additional hyperparameter context or formulas.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source quote grounding this training configuration.",
    )


class ComputationalCostItem(BaseModel):
    """Hardware infrastructure, computational FLOPs, or training duration."""

    hardware: str | None = Field(
        default=None,
        description="Hardware used (e.g. '8 NVIDIA P100 GPUs').",
    )
    training_time: str | None = Field(
        default=None,
        description="Wall-clock training duration (e.g. '3.5 days', '12 hours').",
    )
    flops: str | None = Field(
        default=None,
        description="Estimated floating-point operations (e.g. '3.3 · 10^18', '2.3 · 10^19').",
    )
    details: str | None = Field(
        default=None,
        description="Model variant or training step count associated with this cost.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source quote grounding this computational cost claim.",
    )


class EvaluationMetricItem(BaseModel):
    """Quantitative or qualitative evaluation metric and reported value."""

    metric_name: str = Field(
        ...,
        description="Name of the metric (e.g., 'BLEU-4', 'Accuracy', 'Latency', 'F1').",
    )
    value: str | None = Field(
        default=None,
        description="Exact quantitative or qualitative score reported (e.g., '92.4%', '3.2 ms').",
    )
    task_or_benchmark: str | None = Field(
        default=None,
        description="Task or benchmark on which this metric was evaluated.",
    )
    metric_category: str | None = Field(
        default=None,
        description="Evaluation category: 'quality_bleu', 'computational_cost_flops', 'training_time', 'parsing_f1', etc.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source evidence or quote grounding this metric.",
    )


class KeyFindingItem(BaseModel):
    """Empirical or theoretical result/claim with direct evidence grounding."""

    claim: str = Field(
        ...,
        description="Concise summary of the empirical finding, observation, or conclusion.",
    )
    evidence: EvidenceReference | None = Field(
        default=None,
        description="Source quote or section grounding this specific claim.",
    )


# ---------------------------------------------------------------------------
# Core Extraction Models
# ---------------------------------------------------------------------------


class ResearchPaperExtraction(BaseModel):
    """Structured information extracted from a research paper."""

    paper_title: str | None = Field(
        default=None,
        description="Title of the research paper if explicitly identified in the text.",
    )
    paper_id: str | None = Field(
        default=None,
        description="Optional paper identifier from input metadata.",
    )
    research_problem: str | None = Field(
        default=None,
        description="The primary scientific or engineering problem addressed by the paper.",
    )
    research_objectives: list[str] = Field(
        default_factory=list,
        description="Explicit research goals, hypotheses, or research questions addressed.",
    )
    methodology: list[MethodologyItem] = Field(
        default_factory=list,
        description="Methodological approaches, architectures, or algorithmic techniques used.",
    )
    datasets: list[DatasetItem] = Field(
        default_factory=list,
        description="Datasets used for training, evaluation, or benchmarking.",
    )
    tokenization: list[TokenizationItem] = Field(
        default_factory=list,
        description="Tokenization methods, byte-pair encodings, and vocabulary specifications.",
    )
    training_config: list[TrainingConfigItem] = Field(
        default_factory=list,
        description="Training configurations, optimizers, learning rate schedules, and hyperparameters.",
    )
    computational_cost: list[ComputationalCostItem] = Field(
        default_factory=list,
        description="Hardware infrastructure, training duration, and FLOPs.",
    )
    evaluation_metrics: list[EvaluationMetricItem] = Field(
        default_factory=list,
        description="Evaluation metrics, benchmarks, and experimental results.",
    )
    key_findings: list[KeyFindingItem] = Field(
        default_factory=list,
        description="Core empirical findings or discoveries with grounded evidence.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Explicitly mentioned constraints, caveats, weaknesses, or failure modes.",
    )
    future_work: list[str] = Field(
        default_factory=list,
        description="Explicit suggestions, open problems, or directions for future research.",
    )


# ---------------------------------------------------------------------------
# API Request / Response Schemas
# ---------------------------------------------------------------------------


class ExtractionRequest(BaseModel):
    """Request payload for extracting structured information from paper text."""

    text: str = Field(
        ...,
        min_length=1,
        description="Source text (paper or paper chunk) to extract information from.",
    )
    paper_id: str | None = Field(
        default=None,
        description="Optional paper identifier for provenance tracking.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="Optional chunk identifier for provenance tracking.",
    )


class ExtractionResponse(BaseModel):
    """Response payload returned after successful extraction."""

    data: ResearchPaperExtraction = Field(
        ...,
        description="Structured extraction payload.",
    )
    model: str = Field(
        ...,
        description="Identifier of the model that produced the extraction.",
    )
