import { useState, useCallback } from 'react';
import { listDocuments, uploadDocument } from '@/services/api';
import type { ApiDocument } from '@/types/document';

export interface UseDocumentsResult {
  documents: ApiDocument[];
  loading: boolean;
  error: string | null;
  uploading: boolean;
  uploadError: string | null;
  fetch: () => Promise<void>;
  upload: (file: File) => Promise<ApiDocument | null>;
  clearUploadError: () => void;
}

/**
 * Custom hook that manages the document list and PDF upload state.
 *
 * - fetch()  — loads the document list from GET /api/documents
 * - upload() — posts a PDF, refreshes the list on success, returns the new document
 *
 * Error states are separate for list-fetch and upload so they can be surfaced
 * independently in the UI.
 */
export function useDocuments(): UseDocumentsResult {
  const [documents, setDocuments] = useState<ApiDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents.');
    } finally {
      setLoading(false);
    }
  }, []);

  const upload = useCallback(async (file: File): Promise<ApiDocument | null> => {
    setUploading(true);
    setUploadError(null);
    try {
      const response = await uploadDocument(file);
      // Refresh the full list after a successful upload
      const docs = await listDocuments();
      setDocuments(docs);
      return response.documents[0] ?? null;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed.';
      setUploadError(message);
      return null;
    } finally {
      setUploading(false);
    }
  }, []);

  const clearUploadError = useCallback(() => setUploadError(null), []);

  return {
    documents,
    loading,
    error,
    uploading,
    uploadError,
    fetch,
    upload,
    clearUploadError,
  };
}
