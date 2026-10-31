import { useState } from 'react';
import { Copy, Check, Hash, Bookmark } from 'lucide-react';
import type { ApiChunk } from '@/types/document';

interface ChunkCardProps {
  chunk: ApiChunk;
  index: number;
  isHighlighted?: boolean;
}

export default function ChunkCard({ chunk, index, isHighlighted = false }: ChunkCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(chunk.chunk_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback or ignore clipboard error
    }
  };

  return (
    <article
      id={`chunk-${chunk.chunk_id}`}
      className={`rounded-xl border p-5 shadow-sm transition ${
        isHighlighted
          ? 'border-blue-500 bg-blue-50/40 ring-2 ring-blue-500/20'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
      aria-labelledby={`chunk-${chunk.chunk_id}-heading`}
    >
      {/* Header / Metadata bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-100 pb-3">
        <div className="flex flex-wrap items-center gap-2">
          {/* Order index */}
          <span className="text-xs font-semibold text-gray-400">
            #{index + 1}
          </span>

          {isHighlighted && (
            <span className="inline-flex items-center rounded-md bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 border border-amber-300">
              Cited Evidence
            </span>
          )}

          {/* Chunk ID badge with copy action */}
          <div className="inline-flex items-center gap-1.5 rounded-md bg-gray-100 px-2 py-0.5 text-xs font-mono font-medium text-gray-800">
            <Hash size={12} className="text-gray-400" />
            <span id={`chunk-${chunk.chunk_id}-heading`}>{chunk.chunk_id}</span>
            <button
              onClick={handleCopyId}
              type="button"
              className="ml-1 rounded p-0.5 text-gray-400 hover:text-gray-700 hover:bg-gray-200 transition focus-visible:outline-2 focus-visible:outline-blue-500"
              title="Copy Chunk ID"
              aria-label={`Copy chunk ID ${chunk.chunk_id}`}
            >
              {copied ? (
                <Check size={12} className="text-emerald-600" />
              ) : (
                <Copy size={12} />
              )}
            </button>
          </div>

          {/* Page badge */}
          <span className="inline-flex items-center rounded-md bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10">
            Page {chunk.page_number}
          </span>

          {/* Paragraph badge */}
          <span className="inline-flex items-center rounded-md bg-purple-50 px-2 py-0.5 text-xs font-medium text-purple-700 ring-1 ring-inset ring-purple-700/10">
            Paragraph {chunk.paragraph_number}
          </span>

          {/* Section badge */}
          {chunk.section && (
            <span className="inline-flex items-center gap-1 rounded-md bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-800 ring-1 ring-inset ring-amber-800/10">
              <Bookmark size={11} className="text-amber-600" />
              <span className="truncate max-w-[200px]" title={chunk.section}>
                {chunk.section}
              </span>
            </span>
          )}
        </div>

        {/* Copy confirmation label */}
        {copied && (
          <span className="text-xs font-medium text-emerald-600 animate-fade-in">
            ID copied!
          </span>
        )}
      </div>

      {/* Chunk Content */}
      <div className="mt-3">
        <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-gray-800 selection:bg-blue-100 selection:text-blue-900">
          {chunk.text}
        </pre>
      </div>
    </article>
  );
}
