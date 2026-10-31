import { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react';
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

            return (
              <div
                key={`${ref.chunk_id}-${index}`}
                className="rounded-lg bg-gray-50 border border-gray-200/80 p-2.5 text-xs text-gray-700 space-y-1.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-1.5 text-[11px] font-medium text-gray-500">
                  <span className="text-gray-900 font-semibold truncate max-w-[280px]" title={displayName}>
                    {displayName}
                  </span>
                  <div className="flex items-center gap-2 text-gray-500">
                    <span>Page {ref.page_number}</span>
                    <span>•</span>
                    <span>Para {ref.paragraph_number}</span>
                    <span>•</span>
                    <span className="font-mono text-[10px] text-gray-400">
                      Chunk {ref.chunk_id.slice(0, 8)}…
                    </span>
                  </div>
                </div>

                {ref.excerpt && (
                  <blockquote className="border-l-2 border-blue-400 pl-2.5 py-0.5 text-gray-600 italic bg-white/70 rounded-r text-[11px] leading-relaxed">
                    “{ref.excerpt}”
                  </blockquote>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
