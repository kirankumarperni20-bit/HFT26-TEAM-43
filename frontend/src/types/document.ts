// ─────────────────────────────────────────────────────────────────────────────
// Backend API types — match the FastAPI schemas exactly.
// Do NOT add fields the backend does not return.
// ─────────────────────────────────────────────────────────────────────────────

/** Status values returned by the backend (DocumentStatus enum). */
export type ApiDocumentStatus = 'uploaded' | 'processing' | 'completed' | 'failed';

/** DocumentMetadata as returned by GET /api/documents and POST /api/documents/upload */
export interface ApiDocument {
  paper_id: string;
  filename: string;
  status: ApiDocumentStatus;
}

/** Response envelope from POST /api/documents/upload */
export interface UploadResponse {
  documents: ApiDocument[];
}

/** Chunk as returned by GET /api/documents/{paper_id}/chunks */
export interface ApiChunk {
  chunk_id: string;
  paper_id: string;
  page_number: number;
  paragraph_number: number;
  text: string;
  section: string | null;
}

/** Status response from GET /api/documents/{paper_id}/status */
export interface DocumentStatusResponse {
  paper_id: string;
  status: ApiDocumentStatus;
}

/** Page text response from GET /api/documents/{paper_id}/pages/{page_number} */
export interface PageTextResponse {
  paper_id: string;
  page_number: number;
  text: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy display type kept for components that need richer optional metadata.
// Fields that the backend does not provide remain optional.
// ─────────────────────────────────────────────────────────────────────────────

/** @deprecated Prefer ApiDocument. Only used for display-layer optional fields. */
export type DocumentStatus = ApiDocumentStatus;

/** Display-layer document — wraps ApiDocument with optional UI-only fields. */
export interface Document {
  id: string;       // maps to paper_id
  filename: string;
  status: DocumentStatus;
  uploadedAt?: string; // not provided by the backend; undefined when using real data
  pageCount?: number;  // not provided by the backend
  sizeKb?: number;     // not provided by the backend
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard stats — derived client-side from the document list.
// ─────────────────────────────────────────────────────────────────────────────
export interface DashboardStats {
  totalPapers: number;
  processedPapers: number;
  processingPapers: number;
  failedPapers: number;
}

/** Convert an ApiDocument to the display Document shape. */
export function apiDocToDocument(d: ApiDocument): Document {
  return {
    id: d.paper_id,
    filename: d.filename,
    status: d.status,
  };
}

/** Derive dashboard stats from a live document list. */
export function deriveDashboardStats(docs: ApiDocument[]): DashboardStats {
  return {
    totalPapers: docs.length,
    processedPapers: docs.filter((d) => d.status === 'completed').length,
    processingPapers: docs.filter(
      (d) => d.status === 'processing' || d.status === 'uploaded',
    ).length,
    failedPapers: docs.filter((d) => d.status === 'failed').length,
  };
}
