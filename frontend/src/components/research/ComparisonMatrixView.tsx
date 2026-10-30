import { GitCompare, Table, AlignLeft } from 'lucide-react';
import type { ComparisonResult } from '@/types/research';
import GroundedFieldView from './GroundedFieldView';

interface ComparisonMatrixViewProps {
  comparison: ComparisonResult;
  paperIds: string[];
  paperFilenameMap?: Record<string, string>;
}

export default function ComparisonMatrixView({
  comparison,
  paperIds,
  paperFilenameMap = {},
}: ComparisonMatrixViewProps) {
  const { synthesis_narrative, matrix } = comparison;

  return (
    <section
      aria-labelledby="comparison-results-heading"
      className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm space-y-6"
    >
      <div className="flex items-center gap-2.5 border-b border-gray-100 pb-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-purple-50 text-purple-600">
          <GitCompare size={18} />
        </div>
        <div>
          <h3 id="comparison-results-heading" className="text-base font-bold text-gray-900">
            Cross-Paper Synthesis & Comparison
          </h3>
          <p className="text-xs text-gray-500">
            Comparative analysis and structured multi-dimension matrix.
          </p>
        </div>
      </div>

      {/* Synthesis Narrative */}
      {synthesis_narrative && (
        <div className="space-y-2 rounded-lg bg-purple-50/40 border border-purple-100 p-4">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-purple-900 flex items-center gap-1.5">
            <AlignLeft size={14} className="text-purple-600" />
            Comparative Synthesis Narrative
          </h4>
          <p className="text-xs text-gray-800 leading-relaxed whitespace-pre-line">
            {synthesis_narrative}
          </p>
        </div>
      )}

      {/* Comparison Matrix Table */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-600 flex items-center gap-1.5">
          <Table size={14} className="text-gray-600" />
          Comparative Matrix
        </h4>

        {matrix && matrix.length > 0 ? (
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200 text-left text-xs">
              <thead className="bg-gray-50">
                <tr>
                  <th
                    scope="col"
                    className="py-3 px-4 font-semibold text-gray-900 w-1/4 min-w-[160px]"
                  >
                    Criterion / Dimension
                  </th>
                  {paperIds.map((pid) => {
                    const name = paperFilenameMap[pid] || pid;
                    return (
                      <th
                        key={pid}
                        scope="col"
                        className="py-3 px-4 font-semibold text-gray-900 min-w-[220px]"
                      >
                        <div className="truncate max-w-[240px]" title={name}>
                          {name}
                        </div>
                        <div className="font-mono text-[10px] text-gray-400 font-normal">
                          {pid}
                        </div>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 bg-white">
                {matrix.map((row, rowIdx) => {
                  const cellsByPaper = new Map(
                    row.cells.map((c) => [c.paper_id, c.value]),
                  );

                  return (
                    <tr key={rowIdx} className="hover:bg-gray-50/60 transition-colors">
                      <td className="py-3 px-4 font-semibold text-gray-800 align-top bg-gray-50/30">
                        {row.dimension}
                      </td>
                      {paperIds.map((pid) => {
                        const cellValue = cellsByPaper.get(pid);
                        return (
                          <td key={pid} className="py-3 px-4 align-top">
                            {cellValue ? (
                              <GroundedFieldView
                                field={cellValue}
                                paperFilenameMap={paperFilenameMap}
                                fallbackText="Not mentioned"
                              />
                            ) : (
                              <span className="text-gray-300 italic text-xs">
                                —
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-gray-400 italic bg-gray-50 p-4 rounded-lg border border-gray-100">
            No structured comparison matrix rows returned for this job.
          </p>
        )}
      </div>
    </section>
  );
}

