/**
 * API service — connects the frontend to the FastAPI backend.
 *
 * Base URL is read from VITE_API_BASE_URL (set in .env.local).
 * Falls back to http://localhost:8000/api for local development.
 */

import type {
  ApiDocument,
  ApiChunk,
  DocumentStatusResponse,
  PageTextResponse,
  UploadResponse,
} from '@/types/document';
import type {
  StartResearchPayload,
  StartResearchResponse,
  ResearchResult,
} from '@/types/research';

const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  'http://localhost:8000/api';

// ─────────────────────────────────────────────────────────────────────────────
// Internal fetch wrapper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generic JSON fetch wrapper.
 * IMPORTANT: Do NOT pass a Content-Type header for FormData bodies —
 * the browser must set it with the multipart boundary automatically.
 */
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const isFormData = options?.body instanceof FormData;

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      // Only set Content-Type for JSON requests; let the browser handle FormData
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let message = `API error ${response.status}: ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        message =
          typeof body.detail === 'string'
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      // Keep the status-based message if body cannot be parsed
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Health
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Health check — returns true when the backend responds with { status: "healthy" }.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const result = await apiFetch<{ status: string }>('/health');
    return result.status === 'healthy';
  } catch {
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Documents
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GET /api/documents
 * Returns all stored documents with their current status.
 */
export async function listDocuments(): Promise<ApiDocument[]> {
  return apiFetch<ApiDocument[]>('/documents');
}

/**
 * POST /api/documents/upload
 * Uploads a single PDF file using multipart/form-data.
 * The field name must be "file" as required by the backend.
 *
 * Throws a human-readable Error on HTTP 400, 413, or 500.
 */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);

  return apiFetch<UploadResponse>('/documents/upload', {
    method: 'POST',
    body: form,
    // Do NOT set Content-Type — the browser adds the multipart boundary
  });
}

/**
 * GET /api/documents/{paper_id}
 * Retrieves metadata for a specific document.
 */
export async function getDocument(paperId: string): Promise<ApiDocument> {
  return apiFetch<ApiDocument>(`/documents/${encodeURIComponent(paperId)}`);
}

/**
 * GET /api/documents/{paper_id}/status
 * Fetches the current processing status of a specific document.
 */
export async function getDocumentStatus(paperId: string): Promise<DocumentStatusResponse> {
  return apiFetch<DocumentStatusResponse>(`/documents/${encodeURIComponent(paperId)}/status`);
}

/**
 * GET /api/documents/{paper_id}/chunks
 * Retrieves all text chunks for a specific document.
 */
export async function getDocumentChunks(paperId: string): Promise<ApiChunk[]> {
  return apiFetch<ApiChunk[]>(`/documents/${encodeURIComponent(paperId)}/chunks`);
}

/**
 * Returns the URL for streaming or viewing the raw PDF of a document.
 */
export function getDocumentPdfUrl(paperId: string): string {
  return `${BASE_URL}/documents/${encodeURIComponent(paperId)}/pdf`;
}

/**
 * GET /api/documents/{paper_id}/pages/{page_number}
 * Retrieves the exact, verbatim extracted text for a specific page.
 */
export async function getPageText(
  paperId: string,
  pageNumber: number,
): Promise<PageTextResponse> {
  return apiFetch<PageTextResponse>(
    `/documents/${encodeURIComponent(paperId)}/pages/${encodeURIComponent(pageNumber)}`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Research & Synthesis (Milestones F4 & F5)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST /api/research/generate
 * Initiates an asynchronous research synthesis (summarize and/or compare) job.
 */
export async function startResearchSynthesis(
  payload: StartResearchPayload,
): Promise<StartResearchResponse> {
  return apiFetch<StartResearchResponse>('/research/generate', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/**
 * GET /api/research/{research_id}
 * Retrieves the status and results of a research synthesis job.
 */
export async function getResearchResult(
  researchId: string,
): Promise<ResearchResult> {
  return apiFetch<ResearchResult>(`/research/${encodeURIComponent(researchId)}`);
}

/**
 * GET /api/research
 * Retrieves all stored research synthesis jobs from the backend.
 */
export async function listResearchJobs(): Promise<ResearchResult[]> {
  return apiFetch<ResearchResult[]>('/research');
}

/**
 * POST /api/research/{research_id}/verify
 * Evaluates claims from a completed research summary against document evidence (Milestone F5).
 */
export async function verifyResearchClaims(
  researchId: string,
): Promise<ResearchResult> {
  return apiFetch<ResearchResult>(`/research/${encodeURIComponent(researchId)}/verify`, {
    method: 'POST',
  });
}

// Re-export apiFetch for callers that need custom requests
export { apiFetch };
