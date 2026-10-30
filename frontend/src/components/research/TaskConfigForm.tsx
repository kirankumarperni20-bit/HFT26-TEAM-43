import { useId } from 'react';
import { FileText, GitCompare, CheckCircle2, AlertTriangle, Sparkles, HelpCircle, Loader } from 'lucide-react';
import type { ResearchTask } from '@/types/research';

interface TaskConfigFormProps {
  selectedTasks: ResearchTask[];
  onToggleTask: (task: ResearchTask) => void;
  focusQuery: string;
  onChangeFocusQuery: (value: string) => void;
  selectedPaperCount: number;
  onGenerate: () => void;
  isSubmitting?: boolean;
}

export default function TaskConfigForm({
  selectedTasks,
  onToggleTask,
  focusQuery,
  onChangeFocusQuery,
  selectedPaperCount,
  onGenerate,
  isSubmitting = false,
}: TaskConfigFormProps) {
  const summarizeId = useId();
  const compareId = useId();
  const focusInputId = useId();

  const isSummarizeChecked = selectedTasks.includes('summarize');
  const isCompareChecked = selectedTasks.includes('compare');

  // Validation rules
  const hasValidPaperCount = selectedPaperCount >= 1 && selectedPaperCount <= 4;
  const hasSelectedTask = selectedTasks.length > 0;
  const isCompareValid = !isCompareChecked || selectedPaperCount >= 2;

  const isValid = hasValidPaperCount && hasSelectedTask && isCompareValid;

  return (
    <div className="space-y-6">
      {/* Task Selection Card */}
      <section
        aria-labelledby="tasks-heading"
        className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm space-y-4"
      >
        <div>
          <h2 id="tasks-heading" className="text-base font-semibold text-gray-900">
            2. Choose Synthesis Tasks
          </h2>
          <p className="mt-0.5 text-xs text-gray-500">
            Select one or both analysis tasks.
          </p>
        </div>

        <div className="space-y-3">
          {/* Summarize Option */}
          <div
            onClick={() => onToggleTask('summarize')}
            className={`flex items-start gap-3 rounded-lg border p-3.5 transition cursor-pointer select-none ${
              isSummarizeChecked
                ? 'border-blue-400 bg-blue-50/40 ring-1 ring-blue-300'
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50/50'
            }`}
          >
            <div className="flex h-5 items-center">
              <input
                id={summarizeId}
                type="checkbox"
                checked={isSummarizeChecked}
                onChange={() => onToggleTask('summarize')}
                onClick={(e) => e.stopPropagation()}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
              />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between gap-2">
                <label
                  htmlFor={summarizeId}
                  className="text-sm font-semibold text-gray-900 flex items-center gap-1.5 cursor-pointer"
                  onClick={(e) => e.stopPropagation()}
                >
                  <FileText size={16} className="text-blue-600" />
                  Paper Summaries
                </label>
                <span className="rounded bg-blue-100/70 px-2 py-0.5 text-[10px] font-semibold text-blue-800">
                  1–4 papers
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-500 leading-relaxed">
                Extracts structured objectives, methodology, key findings, and limitations backed by chunk-level citations.
              </p>
            </div>
          </div>

          {/* Compare Option */}
          <div
            onClick={() => onToggleTask('compare')}
            className={`flex items-start gap-3 rounded-lg border p-3.5 transition cursor-pointer select-none ${
              isCompareChecked
                ? 'border-purple-400 bg-purple-50/40 ring-1 ring-purple-300'
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50/50'
            }`}
          >
            <div className="flex h-5 items-center">
              <input
                id={compareId}
                type="checkbox"
                checked={isCompareChecked}
                onChange={() => onToggleTask('compare')}
                onClick={(e) => e.stopPropagation()}
                className="h-4 w-4 rounded border-gray-300 text-purple-600 focus:ring-purple-500 cursor-pointer"
              />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between gap-2">
                <label
                  htmlFor={compareId}
                  className="text-sm font-semibold text-gray-900 flex items-center gap-1.5 cursor-pointer"
                  onClick={(e) => e.stopPropagation()}
                >
                  <GitCompare size={16} className="text-purple-600" />
                  Cross-Paper Comparison
                </label>
                <span className="rounded bg-purple-100/70 px-2 py-0.5 text-[10px] font-semibold text-purple-800">
                  2–4 papers
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-500 leading-relaxed">
                Synthesizes a comparative matrix and qualitative narrative aligning methods, datasets, and outcomes.
              </p>

              {/* Invalidation helper for Compare when < 2 papers */}
              {isCompareChecked && selectedPaperCount < 2 && (
                <div className="mt-2.5 flex items-center gap-1.5 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                  <AlertTriangle size={14} className="shrink-0" />
                  <span>
                    Comparison requires at least 2 papers. Currently selected: {selectedPaperCount}.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Research Focus (Optional) */}
      <section
        aria-labelledby="focus-heading"
        className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm space-y-3"
      >
        <div className="flex items-center justify-between gap-2">
          <label
            htmlFor={focusInputId}
            id="focus-heading"
            className="text-base font-semibold text-gray-900 flex items-center gap-1.5"
          >
            <Sparkles size={16} className="text-amber-500" />
            3. Research Focus
            <span className="text-xs font-normal text-gray-400">(Optional)</span>
          </label>
          <span
            className={`text-xs font-mono ${
              focusQuery.length >= 480 ? 'text-amber-600 font-semibold' : 'text-gray-400'
            }`}
          >
            {focusQuery.length} / 500
          </span>
        </div>

        <p className="text-xs text-gray-500">
          Provide specific questions, evaluation benchmarks, or angles to guide the synthesis.
        </p>

        <textarea
          id={focusInputId}
          rows={3}
          maxLength={500}
          value={focusQuery}
          onChange={(e) => onChangeFocusQuery(e.target.value)}
          placeholder="e.g. Focus on computational complexity, training data requirements, and evaluation benchmarks..."
          className="w-full rounded-lg border border-gray-200 p-3 text-xs text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 transition resize-none"
        />
      </section>

      {/* Validation Checklist & Generation Trigger */}
      <section
        aria-labelledby="generation-heading"
        className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm space-y-4"
      >
        <h3 id="generation-heading" className="text-xs font-semibold uppercase tracking-wider text-gray-500">
          Generation Readiness
        </h3>

        <ul className="space-y-2 text-xs text-gray-600">
          <li className="flex items-center gap-2">
            {hasValidPaperCount ? (
              <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
            ) : (
              <AlertTriangle size={16} className="text-amber-500 shrink-0" />
            )}
            <span>
              1 to 4 completed papers selected ({selectedPaperCount} currently selected)
            </span>
          </li>

          <li className="flex items-center gap-2">
            {hasSelectedTask ? (
              <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
            ) : (
              <AlertTriangle size={16} className="text-amber-500 shrink-0" />
            )}
            <span>At least one task selected (Summarize and/or Compare)</span>
          </li>

          {isCompareChecked && (
            <li className="flex items-center gap-2">
              {isCompareValid ? (
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
              ) : (
                <AlertTriangle size={16} className="text-amber-500 shrink-0" />
              )}
              <span>Cross-paper comparison has &ge; 2 papers</span>
            </li>
          )}
        </ul>

        <button
          type="button"
          id="generate-research-btn"
          disabled={!isValid || isSubmitting}
          onClick={onGenerate}
          className={`w-full inline-flex items-center justify-center gap-2 rounded-lg py-2.5 px-4 text-sm font-semibold shadow-sm transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 ${
            isValid && !isSubmitting
              ? 'bg-blue-600 text-white hover:bg-blue-700 cursor-pointer'
              : 'bg-gray-100 text-gray-400 border border-gray-200 cursor-not-allowed opacity-75'
          }`}
          aria-label={
            isSubmitting
              ? 'Starting research synthesis…'
              : isValid
              ? 'Generate Research'
              : 'Generate Research (disabled until requirements are met)'
          }
        >
          {isSubmitting ? (
            <>
              <Loader size={16} className="animate-spin" />
              Starting Research Job…
            </>
          ) : (
            <>
              <Sparkles size={16} />
              Generate Research
            </>
          )}
        </button>

        <div className="flex items-start gap-1.5 text-[11px] text-gray-400">
          <HelpCircle size={14} className="shrink-0 mt-0.5" />
          <span>
            Initiates an asynchronous research job on the backend and polls every 2.5s until completion.
          </span>
        </div>
      </section>
    </div>
  );
}

