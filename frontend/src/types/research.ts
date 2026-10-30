// ─────────────────────────────────────────────────────────────────────────────
// Frontend types for Milestone F4: Research Summaries & Cross-Paper Synthesis
// ─────────────────────────────────────────────────────────────────────────────

/** Supported research synthesis tasks. */
export type ResearchTask = 'summarize' | 'compare';

/** Lifecycle status of an asynchronous research job. */
export type ResearchStatus = 'queued' | 'processing' | 'completed' | 'failed';

/**
 * Citation reference linking a statement back to an exact ingested source chunk.
 * Compatible with backend Chunks schema (paper_id, page_number, paragraph_number).
 */
export interface EvidenceReference {
  chunk_id: string;
  paper_id: string;
  page_number: number;
  paragraph_number: number;
  excerpt: string;
}

/**
 * A piece of synthesized text accompanied by optional supporting evidence references.
 * Enables verifiable claims without breaking plain text rendering.
 */
export interface GroundedField {
  text: string;
  evidence?: EvidenceReference[];
}

/**
 * Plain-language summary structure for a single paper.
 * Findings and limitations are represented as lists of grounded fields.
 */
export interface PaperSummary {
  /** Paper's primary research objective or thesis */
  objective: GroundedField;
  /** Methodology and experimental approach */
  methodology: GroundedField;
  /** List of extracted key findings with supporting evidence */
  findings: GroundedField[];
  /** List of identified research limitations with supporting evidence */
  limitations: GroundedField[];
}

/**
 * Synthesis item wrapping an individual paper's metadata and structured summary.
 */
export interface PaperSynthesisItem {
  paper_id: string;
  filename?: string;
  summary: PaperSummary;
}

/**
 * A single cell in the cross-paper comparison matrix, mapping a paper ID to its grounded value.
 */
export interface ComparisonCell {
  paper_id: string;
  value: GroundedField;
}

/**
 * A row in the comparison matrix representing a specific dimension
 * (e.g., 'Objective', 'Methodology', 'Datasets', 'Metrics', 'Results', 'Limitations').
 */
export interface ComparisonRow {
  dimension: string;
  cells: ComparisonCell[];
}

/**
 * Comparison result across two or more papers.
 * Includes optional high-level narrative synthesis and structured comparison matrix.
 */
export interface ComparisonResult {
  synthesis_narrative?: string;
  matrix: ComparisonRow[];
}

/**
 * Structured error details for failed research generation jobs.
 * Supports safe user-facing display, error categorization, and retry logic.
 */
export interface ResearchJobError {
  /** Machine-readable error code (e.g., 'RATE_LIMIT_EXCEEDED', 'LLM_TIMEOUT', 'CONTEXT_TOO_LARGE') */
  code: string;
  /** Safe, sanitized human-readable error message */
  message: string;
  /** Whether the user or client can safely retry the request */
  retryable: boolean;
  /** Optional recommended cooldown duration in seconds before retrying */
  retry_after_seconds?: number;
}

/**
 * Complete response from GET /api/research/{research_id}.
 * Task-specific results ('papers', 'comparison') are optional to support
 * summary-only or comparison-only request configurations.
 */
export interface ResearchResult {
  research_id: string;
  status: ResearchStatus;
  created_at?: string;
  completed_at?: string;
  paper_ids: string[];
  tasks: ResearchTask[];
  focus_query?: string | null;
  poll_url?: string;
  /** Populated when 'summarize' task is requested and status is 'completed' */
  papers?: PaperSynthesisItem[];
  /** Populated when 'compare' task is requested and status is 'completed' */
  comparison?: ComparisonResult;
  /** Explicitly false in Milestone F4; claim-level verification is deferred to F5 */
  is_claim_verified: boolean;
  /** Populated when status is 'failed' */
  error?: ResearchJobError | null;
}

/**
 * Request payload for POST /api/research/generate.
 */
export interface StartResearchPayload {
  /** 1–4 distinct paper IDs to analyze */
  paper_ids: string[];
  /** One or both of 'summarize' and 'compare'. Comparison requires at least 2 papers. */
  tasks: ResearchTask[];
  /** Optional guiding focus question (maximum 500 characters) */
  focus_query?: string | null;
}

/**
 * Initial response from POST /api/research/generate (typically HTTP 202 Accepted).
 */
export interface StartResearchResponse {
  research_id: string;
  status: ResearchStatus;
  paper_ids: string[];
  created_at: string;
  poll_url?: string;
  message?: string;
}
