import { useState, useEffect, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { listDocuments } from '@/services/api';
import type { ApiDocument } from '@/types/document';
import type { ResearchTask, StartResearchPayload } from '@/types/research';
import {
  useResearchJob,
  STORAGE_KEY_LAST_PAPERS,
  STORAGE_KEY_LAST_FOCUS,
  STORAGE_KEY_LAST_TASKS,
} from '@/hooks/useResearchJob';
import PaperSelector from '@/components/research/PaperSelector';
import TaskConfigForm from '@/components/research/TaskConfigForm';
import JobStatusBanner from '@/components/research/JobStatusBanner';
import ResearchResultsView from '@/components/research/ResearchResultsView';
import { AlertCircle, X, Layers, History } from 'lucide-react';

export default function ResearchWorkspace() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [documents, setDocuments] = useState<ApiDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Paper & Task selection state (pre-filled from localStorage if available)
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_LAST_PAPERS);
      if (saved) return JSON.parse(saved);
    } catch {}
    return [];
  });

  const [selectedTasks, setSelectedTasks] = useState<ResearchTask[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_LAST_TASKS);
      if (saved) return JSON.parse(saved);
    } catch {}
    return ['summarize'];
  });

  const [focusQuery, setFocusQuery] = useState<string>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_LAST_FOCUS);
      if (saved) return saved;
    } catch {}
    return '';
  });

  // Research job hook connecting to real backend API with persistence
  const {
    jobId,
    status,
    result,
    isSubmitting,
    submitError,
    pollingError,
    isVerifying,
    verificationError,
    isRestoring,
    restoreError,
    previousJobs,
    startJob,
    verifyClaims,
    retryPolling,
    resetJob,
    clearSubmitError,
    loadJob,
  } = useResearchJob();

  // Sync URL query params with active jobId
  useEffect(() => {
    const currentParam = searchParams.get('job_id');
    if (jobId && currentParam !== jobId) {
      setSearchParams({ job_id: jobId }, { replace: true });
    } else if (!jobId && currentParam) {
      const next = new URLSearchParams(searchParams);
      next.delete('job_id');
      setSearchParams(next, { replace: true });
    }
  }, [jobId, searchParams, setSearchParams]);

  // Synchronize form values when an existing job result is active
  useEffect(() => {
    if (result) {
      if (result.paper_ids && result.paper_ids.length > 0) {
        setSelectedPaperIds(result.paper_ids);
      }
      if (result.tasks && result.tasks.length > 0) {
        setSelectedTasks(result.tasks);
      }
      if (result.research_focus || result.focus_query) {
        setFocusQuery(result.research_focus || result.focus_query || '');
      }
    }
  }, [result]);

  // Fetch live documents from backend
  const loadDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to retrieve documents from server.',
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  // Paper selection handlers
  const handleTogglePaper = (paperId: string) => {
    setSelectedPaperIds((prev) => {
      if (prev.includes(paperId)) {
        return prev.filter((id) => id !== paperId);
      }
      // Safety check: max 4 papers
      if (prev.length >= 4) {
        return prev;
      }
      // Safety check: ensure paper is completed
      const doc = documents.find((d) => d.paper_id === paperId);
      if (doc && doc.status !== 'completed') {
        return prev;
      }
      return [...prev, paperId];
    });
  };

  const handleDeselectPaper = (paperId: string) => {
    setSelectedPaperIds((prev) => prev.filter((id) => id !== paperId));
  };

  const handleClearSelection = () => {
    setSelectedPaperIds([]);
  };

  // Task selection handlers
  const handleToggleTask = (task: ResearchTask) => {
    setSelectedTasks((prev) =>
      prev.includes(task) ? prev.filter((t) => t !== task) : [...prev, task],
    );
  };

  // Dispatch real research synthesis job to backend
  const handleGenerate = async () => {
    const isCompareSelected = selectedTasks.includes('compare');
    if (selectedPaperIds.length < 1 || selectedPaperIds.length > 4) return;
    if (selectedTasks.length === 0) return;
    if (isCompareSelected && selectedPaperIds.length < 2) return;

    const cleanFocus = focusQuery.trim() ? focusQuery.trim() : null;
    const payload: StartResearchPayload = {
      paper_ids: [...selectedPaperIds],
      tasks: [...selectedTasks],
      focus_query: cleanFocus,
      research_focus: cleanFocus,
    };

    await startJob(payload);
  };

  const handleResetJob = () => {
    resetJob();
    const next = new URLSearchParams(searchParams);
    next.delete('job_id');
    setSearchParams(next, { replace: true });
  };

  // Map paper IDs to filenames for human-readable display
  const paperFilenameMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const doc of documents) {
      map[doc.paper_id] = doc.filename;
    }
    return map;
  }, [documents]);

  const selectedDocuments = documents.filter((doc) =>
    selectedPaperIds.includes(doc.paper_id),
  );

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-gray-900">
            Research Workspace
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Generate plain-language summaries and comparison of research papers with verifiable evidence.
          </p>
        </div>

        {jobId && (
          <div className="flex items-center gap-2 self-start sm:self-auto flex-wrap">
            {previousJobs.length > 1 && (
              <select
                aria-label="Select previous research session"
                value={jobId}
                onChange={(e) => void loadJob(e.target.value)}
                className="rounded-lg border border-gray-300 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-700 shadow-xs focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
              >
                {previousJobs.map((j) => (
                  <option key={j.research_id} value={j.research_id}>
                    {j.research_id === jobId ? 'Active: ' : ''}
                    {j.research_id.slice(0, 8)}... ({j.tasks.join(', ')})
                  </option>
                ))}
              </select>
            )}

            <button
              type="button"
              onClick={handleResetJob}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-xs font-semibold text-gray-700 hover:bg-gray-50 shadow-xs transition"
            >
              <Layers size={14} />
              Configure New Job
            </button>
          </div>
        )}
      </div>

      {/* Restoration Error Banner */}
      {restoreError && (
        <div
          role="alert"
          className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900 shadow-sm flex items-start justify-between gap-3"
        >
          <div className="flex items-start gap-2.5">
            <AlertCircle size={18} className="text-amber-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-950">
                Could not restore previous research session
              </p>
              <p className="text-amber-800 mt-0.5">{restoreError}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleResetJob}
            className="shrink-0 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900 hover:bg-amber-100/50 shadow-xs transition"
          >
            Start New Job
          </button>
        </div>
      )}

      {/* Submission Error Alert */}
      {submitError && (
        <div
          role="alert"
          className="relative rounded-xl border border-red-200 bg-red-50 p-4 text-xs text-red-900 shadow-sm"
        >
          <button
            onClick={clearSubmitError}
            className="absolute top-3 right-3 text-red-400 hover:text-red-700 p-1"
            aria-label="Dismiss error"
          >
            <X size={16} />
          </button>
          <div className="flex items-start gap-2.5">
            <AlertCircle size={18} className="text-red-500 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-semibold text-red-950">
                Could not start research synthesis job
              </p>
              <p className="text-red-800">{submitError}</p>
            </div>
          </div>
        </div>
      )}

      {/* Restoring State (Prevents UI empty-state flash on refresh / navigation) */}
      {isRestoring && (
        <div className="rounded-xl border border-gray-200 bg-white p-8 shadow-sm text-center space-y-3">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
            <span className="relative flex h-4 w-4">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-4 w-4 bg-blue-600"></span>
            </span>
          </div>
          <h3 className="text-base font-semibold text-gray-900">
            Restoring Research Workspace...
          </h3>
          <p className="text-xs text-gray-500 max-w-md mx-auto leading-relaxed">
            Retrieving your research job and verified synthesis results from the server.
          </p>
        </div>
      )}

      {/* Active Research Job Tracking */}
      {!isRestoring && jobId && status && (
        <div className="space-y-6">
          <JobStatusBanner
            jobId={jobId}
            status={status}
            pollingError={pollingError}
            jobError={result?.error}
            onRetryPolling={retryPolling}
            onReset={handleResetJob}
          />

          {/* If Job is still in-progress ('queued' | 'processing') */}
          {(status === 'queued' || status === 'processing') && (
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm text-center space-y-3">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
                <span className="relative flex h-4 w-4">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-4 w-4 bg-blue-600"></span>
                </span>
              </div>
              <h3 className="text-base font-semibold text-gray-900">
                {status === 'queued' ? 'Queued on Backend' : 'Processing Synthesis'}
              </h3>
              <p className="text-xs text-gray-500 max-w-md mx-auto leading-relaxed">
                The backend is extracting grounded fields, analyzing objectives, and building comparative dimensions. Results will update automatically.
              </p>

              <div className="pt-3 border-t border-gray-100 max-w-sm mx-auto text-left text-xs space-y-1.5 text-gray-600">
                <span className="font-semibold text-gray-700">Target Papers:</span>
                <ul className="list-disc list-inside space-y-0.5">
                  {selectedDocuments.map((d) => (
                    <li key={d.paper_id} className="truncate">
                      {d.filename}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Completed Job Results */}
          {status === 'completed' && result && (
            <ResearchResultsView
              result={result}
              paperFilenameMap={paperFilenameMap}
              onVerify={verifyClaims}
              isVerifying={isVerifying}
              verificationError={verificationError}
            />
          )}
        </div>
      )}

      {/* Configuration View: only show if no active research job and not currently restoring */}
      {!isRestoring && !jobId && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
            {/* Left Column: Document Selection (7 cols) */}
            <div className="lg:col-span-7">
              <PaperSelector
                documents={documents}
                loading={loading}
                error={error}
                onRetry={loadDocuments}
                selectedPaperIds={selectedPaperIds}
                onTogglePaper={handleTogglePaper}
                onDeselectPaper={handleDeselectPaper}
                onClearSelection={handleClearSelection}
              />
            </div>

            {/* Right Column: Task Configuration & Trigger (5 cols) */}
            <div className="lg:col-span-5">
              <TaskConfigForm
                selectedTasks={selectedTasks}
                onToggleTask={handleToggleTask}
                focusQuery={focusQuery}
                onChangeFocusQuery={setFocusQuery}
                selectedPaperCount={selectedPaperIds.length}
                onGenerate={handleGenerate}
                isSubmitting={isSubmitting}
              />
            </div>
          </div>

          {/* Past Research Summaries List */}
          {previousJobs.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-2 border-b border-gray-100 pb-3">
                <History size={16} className="text-gray-500" />
                <h2 className="text-xs font-bold uppercase tracking-wider text-gray-600">
                  Past Research Summaries ({previousJobs.length})
                </h2>
              </div>
              <div className="divide-y divide-gray-100">
                {previousJobs.slice(0, 6).map((pj) => (
                  <div
                    key={pj.research_id}
                    className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                  >
                    <div className="min-w-0 space-y-0.5">
                      <div className="font-semibold text-gray-800 flex items-center gap-2">
                        <span>Research Job {pj.research_id.slice(0, 8)}</span>
                        <span
                          className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            pj.status === 'completed'
                              ? 'bg-green-50 text-green-700 border border-green-200'
                              : pj.status === 'failed'
                              ? 'bg-red-50 text-red-700 border border-red-200'
                              : 'bg-blue-50 text-blue-700 border border-blue-200'
                          }`}
                        >
                          {pj.status}
                        </span>
                      </div>
                      <p className="text-gray-500 text-[11px]">
                        {pj.paper_ids.length} paper{pj.paper_ids.length !== 1 ? 's' : ''} &bull;{' '}
                        {pj.tasks.join(', ')}
                        {pj.research_focus || pj.focus_query ? ` &bull; Focus: "${pj.research_focus || pj.focus_query}"` : ''}
                        {pj.created_at ? ` &bull; ${new Date(pj.created_at).toLocaleString()}` : ''}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => void loadJob(pj.research_id)}
                      className="shrink-0 inline-flex items-center justify-center px-3 py-1.5 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 text-xs font-semibold shadow-xs transition"
                    >
                      View Summary
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
