import { useState, useId } from 'react';
import { Search, X, AlertCircle, RefreshCw, Loader, FileText, Check } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { ApiDocument } from '@/types/document';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import EmptyState from '@/components/common/EmptyState';

interface PaperSelectorProps {
  documents: ApiDocument[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  selectedPaperIds: string[];
  onTogglePaper: (paperId: string) => void;
  onDeselectPaper: (paperId: string) => void;
  onClearSelection: () => void;
}

export default function PaperSelector({
  documents,
  loading,
  error,
  onRetry,
  selectedPaperIds,
  onTogglePaper,
  onDeselectPaper,
  onClearSelection,
}: PaperSelectorProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCompletedOnly, setFilterCompletedOnly] = useState(false);
  const searchInputId = useId();

  // Selected document objects (persists across filtering and search)
  const selectedDocsMap = new Map(documents.map((d) => [d.paper_id, d]));

  // Filtered documents for display
  const filteredDocuments = documents.filter((doc) => {
    const matchesSearch = doc.filename
      .toLowerCase()
      .includes(searchQuery.trim().toLowerCase());
    const matchesFilter = filterCompletedOnly ? doc.status === 'completed' : true;
    return matchesSearch && matchesFilter;
  });

  const completedCount = documents.filter((d) => d.status === 'completed').length;
  const isMaxReached = selectedPaperIds.length >= 4;

  return (
    <section
      aria-labelledby="paper-selector-heading"
      className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden flex flex-col"
    >
      {/* Header and selection counters */}
      <div className="border-b border-gray-100 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 id="paper-selector-heading" className="text-base font-semibold text-gray-900">
              1. Select Research Papers
            </h2>
            <p className="mt-0.5 text-xs text-gray-500">
              Choose 1 to 4 ingested papers with completed status for synthesis.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-inset ${
                selectedPaperIds.length === 0
                  ? 'bg-gray-50 text-gray-600 ring-gray-200'
                  : selectedPaperIds.length === 4
                  ? 'bg-emerald-50 text-emerald-700 ring-emerald-300'
                  : 'bg-blue-50 text-blue-700 ring-blue-300'
              }`}
              aria-live="polite"
              aria-atomic="true"
            >
              {selectedPaperIds.length === 4 && <Check size={12} className="stroke-[3]" />}
              {selectedPaperIds.length} / 4 Selected
            </span>

            {selectedPaperIds.length > 0 && (
              <button
                type="button"
                onClick={onClearSelection}
                className="text-xs font-medium text-gray-500 hover:text-red-600 px-2 py-1 rounded hover:bg-gray-50 transition-colors focus-visible:outline-2 focus-visible:outline-red-500"
                aria-label="Deselect all selected papers"
              >
                Clear all
              </button>
            )}
          </div>
        </div>

        {/* Selected papers chips bar (visible when 1+ selected) */}
        {selectedPaperIds.length > 0 && (
          <div className="mt-4 pt-3 border-t border-gray-100" aria-label="Selected papers list">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-xs font-medium text-gray-700">Currently Selected:</span>
              <span className="text-[11px] text-gray-400">Click &times; to remove</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {selectedPaperIds.map((id) => {
                const doc = selectedDocsMap.get(id);
                const displayName = doc ? doc.filename : id;
                return (
                  <span
                    key={id}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-blue-50 border border-blue-200 pl-2.5 pr-1.5 py-1 text-xs font-medium text-blue-900 group"
                  >
                    <span className="max-w-[200px] truncate" title={displayName}>
                      {displayName}
                    </span>
                    <button
                      type="button"
                      onClick={() => onDeselectPaper(id)}
                      className="rounded p-0.5 text-blue-400 hover:bg-blue-200/60 hover:text-blue-800 transition-colors focus-visible:outline-2 focus-visible:outline-blue-500"
                      aria-label={`Deselect ${displayName}`}
                    >
                      <X size={14} />
                    </button>
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Search and filter toolbar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2 border-b border-gray-100 bg-gray-50/50 px-5 py-3">
        {/* Search bar */}
        <div className="relative flex-1">
          <label htmlFor={searchInputId} className="sr-only">
            Search papers by filename
          </label>
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
            aria-hidden="true"
          />
          <input
            id={searchInputId}
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search papers by filename..."
            className="w-full rounded-lg border border-gray-200 bg-white pl-9 pr-8 py-1.5 text-xs text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 transition"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 rounded p-0.5"
              aria-label="Clear search query"
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Filter completed only */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setFilterCompletedOnly(!filterCompletedOnly)}
            className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium border transition-colors ${
              filterCompletedOnly
                ? 'bg-blue-50 border-blue-200 text-blue-700'
                : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
            }`}
            aria-pressed={filterCompletedOnly}
          >
            <span>Completed only</span>
            <span className="rounded-full bg-gray-100 px-1.5 py-0.2 text-[10px] text-gray-600">
              {completedCount}
            </span>
          </button>
        </div>
      </div>

      {/* Document List Content */}
      <div className="flex-1 divide-y divide-gray-100 max-h-[460px] overflow-y-auto">
        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center gap-2 py-16 text-gray-400">
            <Loader size={20} className="animate-spin text-blue-600" />
            <span className="text-sm">Loading ingested documents…</span>
          </div>
        )}

        {/* Error state */}
        {!loading && error && (
          <div className="flex flex-col items-center gap-3 px-5 py-12 text-center">
            <AlertCircle size={28} className="text-red-500" />
            <div>
              <p className="text-sm font-semibold text-gray-900">Unable to load documents</p>
              <p className="mt-1 text-xs text-gray-500 max-w-sm">{error}</p>
            </div>
            <button
              type="button"
              onClick={onRetry}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 px-3.5 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 transition focus-visible:outline-2 focus-visible:outline-blue-500"
            >
              <RefreshCw size={14} />
              Retry
            </button>
          </div>
        )}

        {/* Empty state: No documents in workspace */}
        {!loading && !error && documents.length === 0 && (
          <EmptyState
            icon={<FileText size={32} />}
            title="No documents ingested yet"
            description="You need at least one processed document to generate research summaries or comparisons."
            action={
              <Link
                to="/documents"
                className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-blue-700 transition focus-visible:outline-2 focus-visible:outline-blue-500"
              >
                Go to Documents to Upload
              </Link>
            }
          />
        )}

        {/* Empty state: No search match */}
        {!loading && !error && documents.length > 0 && filteredDocuments.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
            <p className="text-sm font-medium text-gray-800">No documents match your filter</p>
            <p className="mt-1 text-xs text-gray-400">
              Try adjusting your search query or toggling filters.
            </p>
            <button
              type="button"
              onClick={() => {
                setSearchQuery('');
                setFilterCompletedOnly(false);
              }}
              className="mt-3 text-xs text-blue-600 font-medium hover:underline"
            >
              Reset search & filters
            </button>
          </div>
        )}

        {/* Documents list */}
        {!loading &&
          !error &&
          filteredDocuments.map((doc) => {
            const isSelected = selectedPaperIds.includes(doc.paper_id);
            const isCompleted = doc.status === 'completed';
            const isDisabled = !isCompleted || (!isSelected && isMaxReached);

            let disabledReason: string | undefined;
            if (!isCompleted) {
              disabledReason = `Status '${doc.status}' is not eligible. Only completed papers can be selected.`;
            } else if (!isSelected && isMaxReached) {
              disabledReason = 'Maximum of 4 papers already selected.';
            }

            const inputId = `paper-select-${doc.paper_id}`;

            return (
              <div
                key={doc.paper_id}
                className={`relative flex items-center gap-3.5 px-5 py-3 transition-colors ${
                  isSelected
                    ? 'bg-blue-50/40'
                    : isDisabled
                    ? 'bg-gray-50/40 opacity-60'
                    : 'hover:bg-gray-50/80'
                }`}
              >
                {/* Accessible Checkbox */}
                <div className="flex h-5 items-center">
                  <input
                    id={inputId}
                    type="checkbox"
                    checked={isSelected}
                    disabled={isDisabled}
                    onChange={() => onTogglePaper(doc.paper_id)}
                    aria-describedby={disabledReason ? `desc-${doc.paper_id}` : undefined}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 disabled:cursor-not-allowed cursor-pointer"
                  />
                </div>

                {/* Document Information Label */}
                <label
                  htmlFor={inputId}
                  className={`min-w-0 flex-1 select-none ${
                    isDisabled ? 'cursor-not-allowed' : 'cursor-pointer'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`truncate text-sm font-medium ${
                        isSelected ? 'text-blue-900 font-semibold' : 'text-gray-900'
                      }`}
                    >
                      {doc.filename}
                    </span>
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-gray-400">
                    <span className="font-mono text-[11px]">{doc.paper_id}</span>
                    {disabledReason && (
                      <span id={`desc-${doc.paper_id}`} className="text-[11px] text-amber-700">
                        • {disabledReason}
                      </span>
                    )}
                  </div>
                </label>

                {/* Status Badge */}
                <div className="shrink-0">
                  <DocumentStatusBadge status={doc.status} />
                </div>
              </div>
            );
          })}
      </div>
    </section>
  );
}

