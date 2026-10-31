import { useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import type { EvidenceReference } from '@/types/research';

interface EvidenceReferenceListProps {
  references?: EvidenceReference[];
  paperFilenameMap?: Record<string, string>;
}

export default function EvidenceReferenceList({
  references,
  paperFilenameMap = {},
}: EvidenceReferenceListProps) {
  const [expanded, setExpanded] = useState(false);

  if (!references || references.length === 0) {
    return null;
  }

  return (
    <div className="mt-2.5 pt-2 border-t border-gray-100">
      <div className="flex items-center justify-between">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:text-blue-800 transition-colors focus-visible:outline-2 focus-visible:outline-blue-500 rounded px-1 -ml-1 py-0.5"
          aria-expanded={expanded}
          aria-label={`View supporting evidence (${references.length})`}
        >
          <BookOpen size={13} className="text-blue-500 shrink-0" />
          <span>
            View Supporting Evidence ({references.length})
          </span>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        <span className="text-[10px] text-gray-400 italic">
          Evidence from the Paper
        </span>
      </div>

      {expanded && (
        <div className="mt-2 space-y-2" role="region" aria-label="Evidence from the paper">
          {references.map((ref, index) => {
            const displayName =
              paperFilenameMap[ref.paper_id] || ref.paper_id;
            const docUrl = `/documents/${encodeURIComponent(ref.paper_id)}?page=${ref.page_number}&chunk_id=${encodeURIComponent(ref.chunk_id)}&excerpt=${encodeURIComponent(ref.excerpt)}`;

            return (
              <div
                key={`${ref.chunk_id}-${index}`}
                className="rounded-lg bg-gray-50 border border-gray-200/80 p-2.5 text-xs text-gray-700 space-y-1.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-1.5 text-[11px] font-medium text-gray-500">
                  <Link
                    to={docUrl}
                    className="text-gray-900 font-semibold truncate max-w-[280px] hover:text-blue-600 hover:underline inline-flex items-center gap-1"
                    title={`Open ${displayName} in document viewer`}
                  >
                    <span>{displayName}</span>
                    <ExternalLink size={10} className="text-gray-400 shrink-0" />
                  </Link>
                  <div className="flex items-center gap-2 text-gray-500">
                    <Link
                      to={docUrl}
                      className="text-blue-600 font-semibold hover:text-blue-800 hover:underline transition-colors"
                      title={`Jump to Page ${ref.page_number} in PDF viewer`}
                    >
                      Page {ref.page_number}
                    </Link>
                    <span>•</span>
                    <span>Para {ref.paragraph_number}</span>
                    <span>•</span>
                    <Link
                      to={`${docUrl}&tab=chunks`}
                      className="font-mono text-[10px] text-blue-600 hover:text-blue-800 hover:underline bg-white px-1 py-0.5 rounded border border-gray-200"
                      title={`View chunk ${ref.chunk_id}`}
                    >
                      Chunk {ref.chunk_id.slice(0, 8)}…
                    </Link>
                  </div>
                </div>

                {ref.excerpt && (
                  <blockquote className="border-l-2 border-blue-400 pl-2.5 py-0.5 text-gray-600 italic bg-white/70 rounded-r text-[11px] leading-relaxed">
                    “{ref.excerpt}”
                  </blockquote>
                )}

                <div className="pt-1 flex items-center justify-end">
                  <Link
                    to={docUrl}
                    className="text-[11px] text-blue-600 hover:text-blue-800 font-medium inline-flex items-center gap-1 hover:underline"
                  >
                    <span>Open in Document Viewer</span>
                    <ExternalLink size={10} />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
