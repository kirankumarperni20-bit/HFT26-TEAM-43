import { useState, useEffect, useCallback, useMemo } from 'react';
import { listDocuments } from '@/services/api';
import type { ApiDocument } from '@/types/document';
import type { ResearchTask, StartResearchPayload } from '@/types/research';
import { useResearchJob } from '@/hooks/useResearchJob';
import PaperSelector from '@/components/research/PaperSelector';
import TaskConfigForm from '@/components/research/TaskConfigForm';
import JobStatusBanner from '@/components/research/JobStatusBanner';
import ResearchResultsView from '@/components/research/ResearchResultsView';
import { AlertCircle, X, Layers } from 'lucide-react';

export default function ResearchWorkspace() {
  const [documents, setDocuments] = useState<ApiDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Paper & Task selection state
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [selectedTasks, setSelectedTasks] = useState<ResearchTask[]>(['summarize']);
  const [focusQuery, setFocusQuery] = useState('');

  // Research job hook connecting to real backend API
  const {
    jobId,
    status,
    result,
    isSubmitting,
    submitError,
    pollingError,
    startJob,
    retryPolling,
    resetJob,
    clearSubmitError,
  } = useResearchJob();

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

    const payload: StartResearchPayload = {
      paper_ids: [...selectedPaperIds],
      tasks: [...selectedTasks],
      focus_query: focusQuery.trim() ? focusQuery.trim() : null,
    };

    await startJob(payload);
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
            Generate plain-language summaries and cross-paper comparative synthesis with verifiable citations.
          </p>
        </div>

        {jobId && (
          <button
            type="button"
            onClick={resetJob}
            className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3.5 py-2 text-xs font-semibold text-gray-700 hover:bg-gray-50 shadow-xs transition self-start"
          >
            <Layers size={14} />
            Configure New Job
          </button>
        )}
      </div>

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

      {/* Active Research Job Tracking */}
      {jobId && status && (
        <div className="space-y-6">
          <JobStatusBanner
            jobId={jobId}
            status={status}
            pollingError={pollingError}
            jobError={result?.error}
            onRetryPolling={retryPolling}
            onReset={resetJob}
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
            />
          )}
        </div>
      )}

      {/* Configuration View: only show if no active research job */}
      {!jobId && (
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
      )}
    </div>
  );
}
