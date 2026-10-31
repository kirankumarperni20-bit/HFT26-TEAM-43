import { useState } from 'react';
import {
  FileText,
  GitCompare,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import type { ResearchResult } from '@/types/research';
import PaperSummaryCard from './PaperSummaryCard';
import ComparisonMatrixView from './ComparisonMatrixView';
import ClaimVerificationView from './ClaimVerificationView';

interface ResearchResultsViewProps {
  result: ResearchResult;
  paperFilenameMap?: Record<string, string>;
  onVerify?: () => void;
  isVerifying?: boolean;
  verificationError?: string | null;
}

export default function ResearchResultsView({
  result,
  paperFilenameMap = {},
  onVerify = () => {},
  isVerifying = false,
  verificationError = null,
}: ResearchResultsViewProps) {
  const hasPapers = Boolean(result.papers && result.papers.length > 0);
  const hasComparison = Boolean(result.comparison);

  // Active tab state
  const [activeTab, setActiveTab] = useState<'summaries' | 'comparison' | 'verification'>(
    hasPapers ? 'summaries' : hasComparison ? 'comparison' : 'verification',
  );

  const activeFocus = result.focus_query || result.research_focus;

  return (
    <div className="space-y-6">
      {/* Active Research Focus Notice */}
      {activeFocus && (
        <div
          role="region"
          aria-label="Active research focus"
          className="rounded-xl border border-amber-200 bg-amber-50/80 p-4 text-xs text-amber-950 flex items-start justify-between gap-3 shadow-xs"
        >
          <div className="flex items-start gap-2.5">
            <Sparkles size={16} className="text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-semibold uppercase tracking-wider text-[11px] text-amber-900">
                Applied Research Focus
              </span>
              <p className="text-gray-900 font-medium leading-relaxed">
                "{activeFocus}"
              </p>
              <p className="text-[11px] text-amber-800">
                Summary findings, comparative matrix dimensions, and evidence retrieval are prioritized for this focus.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Verification Status Banner */}
      <div
        role="region"
        aria-label="Claim verification status"
        className={`rounded-xl border p-4 text-xs transition-colors flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 ${
          result.is_claim_verified
            ? 'border-emerald-200 bg-emerald-50/70 text-emerald-950'
            : 'border-blue-200 bg-blue-50/70 text-blue-950'
        }`}
      >
        <div className="flex items-start gap-2.5">
          <ShieldCheck
            size={18}
            className={`shrink-0 mt-0.5 ${
              result.is_claim_verified ? 'text-emerald-600' : 'text-blue-600'
            }`}
          />
          <div>
            <p className="font-semibold text-gray-900">
              {result.is_claim_verified
                ? 'Evidence-Based Claim Verification Completed'
                : 'Claim Verification Available'}
            </p>
            <p className="text-gray-600 mt-0.5 leading-relaxed">
              {result.is_claim_verified
                ? `Evaluated ${result.verifications?.length ?? 0} claims directly against source document evidence.`
                : 'Evaluate each research statement against verbatim excerpts in the source papers.'}
            </p>
          </div>
        </div>

        <div>
          <button
            type="button"
            onClick={onVerify}
            disabled={isVerifying}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60 shadow-xs transition-colors shrink-0"
          >
            <ShieldCheck size={13} />
            <span>
              {isVerifying
                ? 'Checking claim…'
                : result.is_claim_verified
                ? 'Re-verify Claims'
                : 'Verify Claims'}
            </span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-gray-200 pb-2">
        {hasPapers && (
          <button
            type="button"
            onClick={() => setActiveTab('summaries')}
            className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'summaries'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <FileText size={14} />
            Paper Summaries ({result.papers?.length})
          </button>
        )}

        {hasComparison && (
          <button
            type="button"
            onClick={() => setActiveTab('comparison')}
            className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === 'comparison'
                ? 'bg-purple-600 text-white shadow-xs'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <GitCompare size={14} />
            Comparison of Research Papers
          </button>
        )}

        <button
          type="button"
          onClick={() => setActiveTab('verification')}
          className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
            activeTab === 'verification'
              ? 'bg-emerald-600 text-white shadow-xs'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          <ShieldCheck size={14} />
          Claim Verification
          {result.is_claim_verified && (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-300"></span>
          )}
        </button>
      </div>

      {/* Tab 1: Summaries View */}
      {activeTab === 'summaries' && hasPapers && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-gray-900 flex items-center gap-2">
              <FileText size={18} className="text-blue-600" />
              Paper Summaries
            </h3>
            <span className="text-xs text-gray-400">
              {result.papers?.length} paper{result.papers?.length !== 1 ? 's' : ''} summarized
            </span>
          </div>

          <div className="grid grid-cols-1 gap-6">
            {result.papers?.map((item) => (
              <PaperSummaryCard
                key={item.paper_id}
                item={item}
                paperFilenameMap={paperFilenameMap}
              />
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: Comparison View */}
      {activeTab === 'comparison' && hasComparison && result.comparison && (
        <ComparisonMatrixView
          comparison={result.comparison}
          paperIds={result.paper_ids}
          paperFilenameMap={paperFilenameMap}
        />
      )}

      {/* Tab 3: Claim Verification View */}
      {activeTab === 'verification' && (
        <ClaimVerificationView
          verifications={result.verifications}
          isVerified={result.is_claim_verified}
          isVerifying={isVerifying}
          onVerify={onVerify}
          error={verificationError}
          paperFilenameMap={paperFilenameMap}
        />
      )}

      {/* Empty State: If neither section has content */}
      {!hasPapers && !hasComparison && activeTab !== 'verification' && (
        <div className="rounded-xl border border-gray-200 bg-white p-8 text-center text-xs text-gray-500">
          <Sparkles size={28} className="mx-auto text-gray-400 mb-2" />
          <p className="font-semibold text-gray-700">No synthesis content returned</p>
          <p className="mt-1">
            The job completed successfully, but returned neither summaries nor comparison rows.
          </p>
        </div>
      )}
    </div>
  );
}
