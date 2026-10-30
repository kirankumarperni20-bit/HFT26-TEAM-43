import { useState } from 'react';
import { FileText, GitCompare, ShieldAlert, Sparkles } from 'lucide-react';
import type { ResearchResult } from '@/types/research';
import PaperSummaryCard from './PaperSummaryCard';
import ComparisonMatrixView from './ComparisonMatrixView';

interface ResearchResultsViewProps {
  result: ResearchResult;
  paperFilenameMap?: Record<string, string>;
}

export default function ResearchResultsView({
  result,
  paperFilenameMap = {},
}: ResearchResultsViewProps) {
  const hasPapers = result.papers && result.papers.length > 0;
  const hasComparison = Boolean(result.comparison);

  // Tab state if both are present
  const [activeTab, setActiveTab] = useState<'summaries' | 'comparison'>(
    hasPapers ? 'summaries' : 'comparison',
  );

  return (
    <div className="space-y-6">
      {/* Verification Disclaimer Banner */}
      <div
        role="note"
        className="rounded-xl border border-blue-200 bg-blue-50/60 p-4 text-xs text-blue-900 flex items-start gap-3"
      >
        <ShieldAlert size={18} className="text-blue-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-semibold text-blue-950">
            Unverified Research Output (Milestone F4)
          </p>
          <p className="text-blue-800 leading-relaxed">
            In Milestone F4, claims are synthesized with grounded chunk citations but are not yet verified by automated claim checking (<code className="font-mono text-[11px] bg-blue-100 px-1 py-0.5 rounded">is_claim_verified: false</code>). Automated verification will be introduced in Milestone F5. Always inspect primary sources.
          </p>
        </div>
      </div>

      {/* Navigation Tabs (if both summaries and comparison are present) */}
      {hasPapers && hasComparison && (
        <div className="flex items-center gap-2 border-b border-gray-200 pb-2">
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
            Cross-Paper Comparison
          </button>
        </div>
      )}

      {/* Summaries View */}
      {hasPapers && (activeTab === 'summaries' || !hasComparison) && (
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

      {/* Comparison View */}
      {hasComparison && result.comparison && (activeTab === 'comparison' || !hasPapers) && (
        <ComparisonMatrixView
          comparison={result.comparison}
          paperIds={result.paper_ids}
          paperFilenameMap={paperFilenameMap}
        />
      )}

      {/* Empty State: If neither section has content */}
      {!hasPapers && !hasComparison && (
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

