import { FileText, Target, Cpu, CheckCircle, AlertOctagon } from 'lucide-react';
import type { PaperSynthesisItem } from '@/types/research';
import GroundedFieldView from './GroundedFieldView';

interface PaperSummaryCardProps {
  item: PaperSynthesisItem;
  paperFilenameMap?: Record<string, string>;
}

export default function PaperSummaryCard({
  item,
  paperFilenameMap = {},
}: PaperSummaryCardProps) {
  const { paper_id, filename, summary } = item;
  const displayName = filename || paperFilenameMap[paper_id] || paper_id;

  return (
    <article
      aria-label={`Summary for ${displayName}`}
      className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm space-y-5"
    >
      {/* Header */}
      <div className="border-b border-gray-100 pb-3 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
            <FileText size={18} />
          </div>
          <div className="min-w-0">
            <h3 className="text-sm font-bold text-gray-900 truncate" title={displayName}>
              {displayName}
            </h3>
            <p className="font-mono text-[11px] text-gray-400">{paper_id}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Research Question */}
        <div className="rounded-lg bg-gray-50/70 border border-gray-100 p-3.5 space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-600 flex items-center gap-1.5">
            <Target size={14} className="text-blue-600" />
            Research Question
          </h4>
          <GroundedFieldView
            field={summary?.objective}
            paperFilenameMap={paperFilenameMap}
            fallbackText="No explicit research question extracted."
          />
        </div>

        {/* How the Study Was Conducted */}
        <div className="rounded-lg bg-gray-50/70 border border-gray-100 p-3.5 space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-600 flex items-center gap-1.5">
            <Cpu size={14} className="text-purple-600" />
            How the Study Was Conducted
          </h4>
          <GroundedFieldView
            field={summary?.methodology}
            paperFilenameMap={paperFilenameMap}
            fallbackText="No explicit methodology extracted."
          />
        </div>
      </div>

      {/* Main Findings */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-600 flex items-center gap-1.5">
          <CheckCircle size={14} className="text-emerald-600" />
          Main Findings
        </h4>
        {summary?.findings && summary.findings.length > 0 ? (
          <ul className="space-y-2.5" role="list">
            {summary.findings.map((finding, idx) => (
              <li
                key={idx}
                className="rounded-lg border border-gray-100 bg-white p-3 shadow-xs space-y-1.5"
              >
                <GroundedFieldView
                  field={finding}
                  paperFilenameMap={paperFilenameMap}
                />
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-gray-400 italic bg-gray-50 p-3 rounded-lg border border-gray-100">
            No specific findings recorded.
          </p>
        )}
      </div>

      {/* Limitations */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-600 flex items-center gap-1.5">
          <AlertOctagon size={14} className="text-amber-600" />
          Limitations
        </h4>
        {summary?.limitations && summary.limitations.length > 0 ? (
          <ul className="space-y-2.5" role="list">
            {summary.limitations.map((limitation, idx) => (
              <li
                key={idx}
                className="rounded-lg border border-gray-100 bg-white p-3 shadow-xs space-y-1.5"
              >
                <GroundedFieldView
                  field={limitation}
                  paperFilenameMap={paperFilenameMap}
                />
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-gray-400 italic bg-gray-50 p-3 rounded-lg border border-gray-100">
            No explicit limitations identified.
          </p>
        )}
      </div>
    </article>
  );
}
