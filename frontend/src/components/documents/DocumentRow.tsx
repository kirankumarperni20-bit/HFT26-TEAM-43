import { Link } from 'react-router-dom';
import { FileText, ChevronRight, MoreHorizontal } from 'lucide-react';
import type { ApiDocument } from '@/types/document';
import DocumentStatusBadge from './DocumentStatusBadge';

interface DocumentRowProps {
  document: ApiDocument;
}

export default function DocumentRow({ document }: DocumentRowProps) {
  const { paper_id, filename, status } = document;
  const encodedId = encodeURIComponent(paper_id);

  return (
    <div className="flex items-center gap-4 px-4 py-3 hover:bg-gray-50 transition-colors group">
      {/* Icon */}
      <div
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600"
        aria-hidden="true"
      >
        <FileText size={18} />
      </div>

      {/* Filename + paper ID — accessible link */}
      <Link
        to={`/documents/${encodedId}`}
        className="min-w-0 flex-1 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded-sm"
        title={`View details for ${filename}`}
      >
        <p className="truncate text-sm font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
          {filename}
        </p>
        <p className="mt-0.5 text-xs text-gray-400 font-mono">
          {paper_id}
        </p>
      </Link>

      {/* Status badge */}
      <DocumentStatusBadge status={status} />

      {/* Explicit View document action */}
      <Link
        to={`/documents/${encodedId}`}
        className="hidden sm:inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity px-2 py-1 rounded hover:bg-blue-50 focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-blue-500"
        aria-label={`View details for ${filename}`}
      >
        <span>View</span>
        <ChevronRight size={14} />
      </Link>

      {/* Action button — stops propagation */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
        className="ml-1 rounded p-1 text-gray-400 opacity-0 group-hover:opacity-100 hover:bg-gray-100 hover:text-gray-600 transition focus-visible:opacity-100 focus-visible:outline-2 focus-visible:outline-blue-500"
        aria-label={`More options for ${filename}`}
      >
        <MoreHorizontal size={16} />
      </button>
    </div>
  );
}
