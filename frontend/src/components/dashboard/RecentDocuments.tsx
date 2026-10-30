import { Link } from 'react-router-dom';
import { FileText, Loader, AlertCircle, RefreshCw } from 'lucide-react';
import DocumentRow from '@/components/documents/DocumentRow';
import EmptyState from '@/components/common/EmptyState';
import type { ApiDocument } from '@/types/document';

interface RecentDocumentsProps {
  documents: ApiDocument[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

const RECENT_LIMIT = 5;

export default function RecentDocuments({
  documents,
  loading,
  error,
  onRetry,
}: RecentDocumentsProps) {
  const recent = documents.slice(0, RECENT_LIMIT);

  return (
    <section
      aria-labelledby="recent-docs-heading"
      className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden"
    >
      {/* Card header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <h2 id="recent-docs-heading" className="text-base font-semibold text-gray-900">
          Recent Documents
        </h2>
        <Link
          to="/documents"
          className="text-sm font-medium text-blue-600 hover:text-blue-700 hover:underline focus-visible:outline-2 focus-visible:outline-blue-500 rounded"
        >
          View all
        </Link>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center gap-2 px-5 py-10 text-gray-400">
          <Loader size={18} className="animate-spin" />
          <span className="text-sm">Loading documents…</span>
        </div>
      )}

      {/* Error */}
      {!loading && error && (
        <div className="flex flex-col items-center gap-3 px-5 py-8 text-center">
          <AlertCircle size={24} className="text-red-400" />
          <p className="text-sm text-gray-600">{error}</p>
          <button
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 focus-visible:outline-2 focus-visible:outline-blue-500"
          >
            <RefreshCw size={13} />
            Retry
          </button>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && recent.length === 0 && (
        <EmptyState
          icon={<FileText size={32} />}
          title="No documents yet"
          description="Upload your first research paper to get started."
        />
      )}

      {/* Document list */}
      {!loading && !error && recent.length > 0 && (
        <div className="divide-y divide-gray-100">
          {recent.map((doc) => (
            <DocumentRow key={doc.paper_id} document={doc} />
          ))}
        </div>
      )}
    </section>
  );
}
