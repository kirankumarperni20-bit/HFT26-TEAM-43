import { CheckCircle2, FlaskConical, X } from 'lucide-react';
import type { StartResearchPayload } from '@/types/research';
import type { ApiDocument } from '@/types/document';

interface ReadyNoticeProps {
  payload: StartResearchPayload;
  selectedDocuments: ApiDocument[];
  onDismiss: () => void;
}

export default function ReadyNotice({
  payload,
  selectedDocuments,
  onDismiss,
}: ReadyNoticeProps) {
  return (
    <div
      role="region"
      aria-label="Research generation status"
      className="relative rounded-xl border border-blue-200 bg-blue-50/70 p-5 text-blue-900 shadow-sm"
    >
      <button
        onClick={onDismiss}
        className="absolute top-4 right-4 rounded-lg p-1 text-blue-400 hover:bg-blue-100 hover:text-blue-600 transition-colors focus-visible:outline-2 focus-visible:outline-blue-500"
        aria-label="Dismiss ready notice"
      >
        <X size={18} />
      </button>

      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white shadow-sm">
          <CheckCircle2 size={22} />
        </div>

        <div className="flex-1 space-y-3 pr-6">
          <div>
            <h2 className="text-base font-semibold text-blue-950 flex items-center gap-2">
              Configuration Ready for Generation
              <span className="inline-flex items-center rounded-full bg-blue-200/70 px-2 py-0.5 text-xs font-semibold text-blue-800">
                Milestone F4.2
              </span>
            </h2>
            <p className="mt-1 text-sm text-blue-800">
              Your paper selection and research parameters are verified and stored in the workspace state.
              The backend research synthesis endpoint (<code className="font-mono text-xs font-semibold text-blue-950 bg-blue-100 px-1 py-0.5 rounded">POST /api/research/generate</code>)
              is currently being implemented. Live asynchronous execution and result polling will be connected in Milestone F4.3.
            </p>
          </div>

          <div className="rounded-lg bg-white/80 p-4 border border-blue-100 text-xs space-y-2">
            <div>
              <span className="font-semibold text-gray-700">Selected Papers ({payload.paper_ids.length}):</span>
              <ul className="mt-1 list-disc list-inside text-gray-600 space-y-0.5">
                {selectedDocuments.map((doc) => (
                  <li key={doc.paper_id} className="truncate">
                    <span className="font-medium text-gray-900">{doc.filename}</span>
                    <span className="ml-1 text-gray-400 font-mono text-[11px]">({doc.paper_id})</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-blue-100/60">
              <span className="font-semibold text-gray-700">Configured Tasks:</span>
              {payload.tasks.map((task) => (
                <span
                  key={task}
                  className="capitalize inline-flex items-center rounded bg-blue-100 px-2 py-0.5 font-medium text-blue-800 text-[11px]"
                >
                  {task}
                </span>
              ))}
            </div>

            {payload.focus_query && (
              <div className="pt-1 border-t border-blue-100/60">
                <span className="font-semibold text-gray-700">Research Focus:</span>
                <p className="mt-0.5 text-gray-600 italic">“{payload.focus_query}”</p>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 text-xs text-blue-700">
            <FlaskConical size={14} className="shrink-0 text-blue-600" />
            <span>No backend requests were dispatched for this milestone.</span>
          </div>
        </div>
      </div>
    </div>
  );
}

