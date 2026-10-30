import { Loader, AlertTriangle, RefreshCw, XCircle, Clock } from 'lucide-react';
import type { ResearchStatus, ResearchJobError } from '@/types/research';

interface JobStatusBannerProps {
  jobId: string;
  status: ResearchStatus;
  pollingError: string | null;
  jobError?: ResearchJobError | null;
  onRetryPolling: () => void;
  onReset: () => void;
}

export default function JobStatusBanner({
  jobId,
  status,
  pollingError,
  jobError,
  onRetryPolling,
  onReset,
}: JobStatusBannerProps) {
  return (
    <div className="space-y-3">
      {/* Active Job Status Banner */}
      <div
        role="status"
        aria-live="polite"
        className={`rounded-xl border p-4 shadow-sm transition ${
          status === 'completed'
            ? 'border-emerald-200 bg-emerald-50/60 text-emerald-950'
            : status === 'failed'
            ? 'border-red-200 bg-red-50/70 text-red-950'
            : status === 'processing'
            ? 'border-blue-200 bg-blue-50/60 text-blue-950'
            : 'border-amber-200 bg-amber-50/60 text-amber-950'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            {status === 'queued' ? (
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-600">
                <Clock size={20} />
              </div>
            ) : status === 'processing' ? (
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-100 text-blue-600">
                <Loader size={20} className="animate-spin" />
              </div>
            ) : status === 'failed' ? (
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-red-100 text-red-600">
                <XCircle size={20} />
              </div>
            ) : (
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-100 text-emerald-600">
                <RefreshCw size={20} />
              </div>
            )}

            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold capitalize">
                  {status === 'queued'
                    ? 'Job Queued'
                    : status === 'processing'
                    ? 'Synthesizing Research…'
                    : status === 'failed'
                    ? 'Research Generation Failed'
                    : 'Synthesis Complete'}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                    status === 'completed'
                      ? 'bg-emerald-200/80 text-emerald-800'
                      : status === 'failed'
                      ? 'bg-red-200/80 text-red-800'
                      : status === 'processing'
                      ? 'bg-blue-200/80 text-blue-800'
                      : 'bg-amber-200/80 text-amber-800'
                  }`}
                >
                  {status}
                </span>
              </div>
              <p className="mt-0.5 text-xs text-gray-600">
                Job ID: <code className="font-mono text-gray-800">{jobId}</code>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-center">
            {(status === 'queued' || status === 'processing') && (
              <span className="flex items-center gap-1.5 text-xs text-gray-500 font-medium">
                <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                Polling every 2.5s
              </span>
            )}
            {(status === 'completed' || status === 'failed') && (
              <button
                type="button"
                onClick={onReset}
                className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-semibold text-gray-700 hover:bg-gray-50 shadow-xs transition focus-visible:outline-2 focus-visible:outline-blue-500"
              >
                Configure New Research
              </button>
            )}
          </div>
        </div>

        {/* Failed Job Details */}
        {status === 'failed' && (
          <div className="mt-3.5 pt-3 border-t border-red-200/60 text-xs text-red-900 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="font-semibold">Error Code:</span>
              <code className="bg-red-100 px-1.5 py-0.5 rounded font-mono text-red-800">
                {jobError?.code || 'RESEARCH_EXECUTION_ERROR'}
              </code>
              {jobError?.retryable && (
                <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800">
                  Retryable
                </span>
              )}
            </div>

            <p className="text-red-800">
              {jobError?.message ||
                'The backend reported an error while processing the research synthesis request.'}
            </p>

            {jobError?.retry_after_seconds && (
              <p className="text-[11px] text-red-600">
                Recommended cooldown: {jobError.retry_after_seconds}s before retrying.
              </p>
            )}
          </div>
        )}
      </div>

      {/* Temporary Polling / Connectivity Warning (Non-Fatal) */}
      {pollingError && (status === 'queued' || status === 'processing') && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800 shadow-xs"
        >
          <div className="flex items-center gap-2">
            <AlertTriangle size={16} className="text-amber-600 shrink-0" />
            <span>
              Temporary network issue checking status: {pollingError}. Job is still active on backend.
            </span>
          </div>
          <button
            type="button"
            onClick={onRetryPolling}
            className="shrink-0 inline-flex items-center gap-1 font-semibold text-amber-900 hover:underline"
          >
            <RefreshCw size={12} />
            Retry Now
          </button>
        </div>
      )}
    </div>
  );
}
