import { useEffect, useState } from 'react';
import { Upload, FileText, Loader, AlertCircle, RefreshCw } from 'lucide-react';
import DocumentRow from '@/components/documents/DocumentRow';
import EmptyState from '@/components/common/EmptyState';
import UploadModal from '@/components/documents/UploadModal';
import { useDocuments } from '@/hooks/useDocuments';
import type { ApiDocumentStatus } from '@/types/document';

type FilterValue = 'all' | ApiDocumentStatus;

const filters: { value: FilterValue; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'completed', label: 'Completed' },
  { value: 'processing', label: 'Processing' },
  { value: 'uploaded', label: 'Uploaded' },
  { value: 'failed', label: 'Failed' },
];

export default function Documents() {
  const { documents, loading, error, uploading, uploadError, fetch, upload, clearUploadError } =
    useDocuments();
  const [activeFilter, setActiveFilter] = useState<FilterValue>('all');
  const [showUpload, setShowUpload] = useState(false);

  // Fetch the live document list on mount
  useEffect(() => {
    void fetch();
  }, [fetch]);

  const filtered =
    activeFilter === 'all'
      ? documents
      : documents.filter((d) => d.status === activeFilter);

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Page header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-gray-900">Documents</h1>
          <p className="mt-1 text-sm text-gray-500">
            Upload and manage your research papers.
          </p>
        </div>
        <button
          id="documents-upload-btn"
          onClick={() => setShowUpload(true)}
          className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 transition-colors"
          aria-label="Upload a research paper"
        >
          <Upload size={16} />
          Upload Research Paper
        </button>
      </div>

      {/* Document list card */}
      <section
        aria-labelledby="documents-list-heading"
        className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden"
      >
        {/* Toolbar */}
        <div className="flex flex-col gap-3 border-b border-gray-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
          <h2 id="documents-list-heading" className="text-base font-semibold text-gray-900">
            All Documents
            {!loading && !error && (
              <span className="ml-2 text-sm font-normal text-gray-400">
                ({filtered.length})
              </span>
            )}
          </h2>

          {/* Status filters */}
          <div
            className="flex items-center gap-1 rounded-lg bg-gray-100 p-1"
            role="group"
            aria-label="Filter by status"
          >
            {filters.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setActiveFilter(value)}
                aria-pressed={activeFilter === value}
                className={[
                  'rounded-md px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-blue-500',
                  activeFilter === value
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-500 hover:text-gray-700',
                ].join(' ')}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center gap-2 px-5 py-14 text-gray-400">
            <Loader size={20} className="animate-spin" />
            <span className="text-sm">Loading documents…</span>
          </div>
        )}

        {/* Error state */}
        {!loading && error && (
          <div className="flex flex-col items-center gap-3 px-5 py-12 text-center">
            <AlertCircle size={28} className="text-red-400" />
            <div>
              <p className="text-sm font-semibold text-gray-900">Could not load documents</p>
              <p className="mt-1 text-sm text-gray-500">{error}</p>
            </div>
            <button
              onClick={() => void fetch()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus-visible:outline-2 focus-visible:outline-blue-500"
            >
              <RefreshCw size={14} />
              Retry
            </button>
          </div>
        )}

        {/* Empty state */}
        {!loading && !error && filtered.length === 0 && (
          <EmptyState
            icon={<FileText size={32} />}
            title={
              activeFilter === 'all'
                ? 'No documents yet'
                : `No ${activeFilter} documents`
            }
            description={
              activeFilter === 'all'
                ? 'Upload your first research paper to get started.'
                : 'Try selecting a different status filter.'
            }
            action={
              activeFilter === 'all' ? (
                <button
                  onClick={() => setShowUpload(true)}
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-blue-500"
                >
                  <Upload size={14} />
                  Upload Research Paper
                </button>
              ) : undefined
            }
          />
        )}

        {/* Document rows */}
        {!loading && !error && filtered.length > 0 && (
          <div className="divide-y divide-gray-100">
            {filtered.map((doc) => (
              <DocumentRow key={doc.paper_id} document={doc} />
            ))}
          </div>
        )}
      </section>

      {/* Upload modal */}
      {showUpload && (
        <UploadModal
          uploading={uploading}
          uploadError={uploadError}
          onUpload={upload}
          onClose={() => setShowUpload(false)}
          onClearError={clearUploadError}
        />
      )}
    </div>
  );
}
