import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  ChevronRight,
  FileText,
  Layers,
  FileCode2,
  Copy,
  Check,
  Loader,
  AlertCircle,
  RefreshCw,
  Info,
  ChevronLeft,
} from 'lucide-react';
import { getDocument, getDocumentChunks, getPageText } from '@/services/api';
import type { ApiDocument, ApiChunk } from '@/types/document';
import DocumentStatusBadge from '@/components/documents/DocumentStatusBadge';
import ChunkCard from '@/components/documents/ChunkCard';
import EmptyState from '@/components/common/EmptyState';

type TabType = 'overview' | 'text' | 'chunks';

export default function DocumentDetails() {
  const { paperId } = useParams<{ paperId: string }>();

  // Document metadata state
  const [document, setDocument] = useState<ApiDocument | null>(null);
  const [docLoading, setDocLoading] = useState(true);
  const [docError, setDocError] = useState<string | null>(null);

  // Chunks state
  const [chunks, setChunks] = useState<ApiChunk[]>([]);
  const [chunksLoading, setChunksLoading] = useState(true);
  const [chunksError, setChunksError] = useState<string | null>(null);

  // Active tab state
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  // Extracted Page Text state
  const [selectedPage, setSelectedPage] = useState<number | null>(null);
  const [pageTextMap, setPageTextMap] = useState<Record<number, string>>({});
  const [pageLoading, setPageLoading] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);

  // Copy paper ID feedback
  const [copiedId, setCopiedId] = useState(false);

  // Search/filter within chunks tab
  const [chunkPageFilter, setChunkPageFilter] = useState<number | 'all'>('all');
  const [chunkQuery, setChunkQuery] = useState('');

  // ───────────────────────────────────────────────────────────────────────────
  // Fetch document metadata and chunks
  // ───────────────────────────────────────────────────────────────────────────
  const fetchDocAndChunks = useCallback(async () => {
    if (!paperId) return;

    setDocLoading(true);
    setDocError(null);
    setChunksLoading(true);
    setChunksError(null);

    // 1. Fetch document metadata
    try {
      const doc = await getDocument(paperId);
      setDocument(doc);
    } catch (err) {
      setDocError(
        err instanceof Error
          ? err.message
          : `Failed to load document with ID ${paperId}`,
      );
    } finally {
      setDocLoading(false);
    }

    // 2. Fetch document chunks
    try {
      const chunkList = await getDocumentChunks(paperId);
      setChunks(chunkList);
    } catch (err) {
      setChunksError(
        err instanceof Error ? err.message : 'Failed to load document chunks',
      );
    } finally {
      setChunksLoading(false);
    }
  }, [paperId]);

  useEffect(() => {
    void fetchDocAndChunks();
  }, [fetchDocAndChunks]);

  // ───────────────────────────────────────────────────────────────────────────
  // Derive available pages from chunks (Option A)
  // ───────────────────────────────────────────────────────────────────────────
  const availablePages = useMemo(() => {
    const pages = new Set<number>();
    for (const chunk of chunks) {
      if (typeof chunk.page_number === 'number') {
        pages.add(chunk.page_number);
      }
    }
    return Array.from(pages).sort((a, b) => a - b);
  }, [chunks]);

  // Set default selected page once availablePages are derived
  useEffect(() => {
    if (availablePages.length > 0 && selectedPage === null) {
      setSelectedPage(availablePages[0]);
    }
  }, [availablePages, selectedPage]);

  // ───────────────────────────────────────────────────────────────────────────
  // Fetch text for the selected page
  // ───────────────────────────────────────────────────────────────────────────
  const fetchPage = useCallback(
    async (pageNumber: number) => {
      if (!paperId) return;
      if (pageTextMap[pageNumber] !== undefined) return; // Cached

      setPageLoading(true);
      setPageError(null);

      try {
        const res = await getPageText(paperId, pageNumber);
        setPageTextMap((prev) => ({ ...prev, [pageNumber]: res.text }));
      } catch (err) {
        setPageError(
          err instanceof Error
            ? err.message
            : `Failed to retrieve text for page ${pageNumber}`,
        );
      } finally {
        setPageLoading(false);
      }
    },
    [paperId, pageTextMap],
  );

  // Trigger page text fetch when selected page changes
  useEffect(() => {
    if (activeTab === 'text' && selectedPage !== null) {
      void fetchPage(selectedPage);
    }
  }, [activeTab, selectedPage, fetchPage]);

  // Copy Paper ID handler
  const handleCopyPaperId = async () => {
    if (!paperId) return;
    try {
      await navigator.clipboard.writeText(paperId);
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    } catch {
      // Ignore clipboard error
    }
  };

  // ───────────────────────────────────────────────────────────────────────────
  // Filtered Chunks for Chunks Tab
  // ───────────────────────────────────────────────────────────────────────────
  const filteredChunks = useMemo(() => {
    return chunks.filter((chunk) => {
      if (chunkPageFilter !== 'all' && chunk.page_number !== chunkPageFilter) {
        return false;
      }
      if (chunkQuery.trim()) {
        const query = chunkQuery.toLowerCase();
        const matchesText = chunk.text.toLowerCase().includes(query);
        const matchesSection = chunk.section?.toLowerCase().includes(query) ?? false;
        const matchesId = chunk.chunk_id.toLowerCase().includes(query);
        return matchesText || matchesSection || matchesId;
      }
      return true;
    });
  }, [chunks, chunkPageFilter, chunkQuery]);

  // ───────────────────────────────────────────────────────────────────────────
  // Loading & Global Error States
  // ───────────────────────────────────────────────────────────────────────────
  if (!paperId) {
    return (
      <div className="max-w-5xl space-y-4">
        <Link
          to="/documents"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-blue-600 hover:text-blue-700"
        >
          <ArrowLeft size={16} />
          Back to Documents
        </Link>
        <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-red-700">
          <p className="font-semibold">Invalid paper ID.</p>
        </div>
      </div>
    );
  }

  if (docLoading && !document) {
    return (
      <div className="max-w-5xl space-y-6">
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Link to="/documents" className="hover:text-gray-900 transition-colors">
            Documents
          </Link>
          <ChevronRight size={14} />
          <span className="font-mono text-gray-400">{paperId}</span>
        </div>
        <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-gray-200 bg-white p-16 shadow-sm">
          <Loader size={28} className="animate-spin text-blue-600" />
          <p className="text-sm font-medium text-gray-600">Loading document details…</p>
        </div>
      </div>
    );
  }

  if (docError && !document) {
    return (
      <div className="max-w-5xl space-y-6">
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Link
            to="/documents"
            className="inline-flex items-center gap-1.5 hover:text-gray-900 transition-colors"
          >
            <ArrowLeft size={16} />
            Back to Documents
          </Link>
        </div>
        <div className="flex flex-col items-center gap-4 rounded-xl border border-red-200 bg-white p-12 text-center shadow-sm">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-50 text-red-500">
            <AlertCircle size={24} />
          </div>
          <div>
            <h2 className="text-base font-semibold text-gray-900">Document not found</h2>
            <p className="mt-1 max-w-md text-sm text-gray-500">{docError}</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => void fetchDocAndChunks()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus-visible:outline-2 focus-visible:outline-blue-500"
            >
              <RefreshCw size={14} />
              Retry
            </button>
            <Link
              to="/documents"
              className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-blue-500"
            >
              Back to Documents
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Render Main Page Content
  // ───────────────────────────────────────────────────────────────────────────
  const filename = document?.filename ?? 'Document';
  const docStatus = document?.status ?? 'uploaded';

  return (
    <div className="max-w-5xl space-y-6">
      {/* Breadcrumb & Navigation */}
      <nav aria-label="Breadcrumb" className="flex items-center justify-between">
        <ol className="flex items-center gap-2 text-sm text-gray-500">
          <li>
            <Link
              to="/documents"
              className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-900 transition-colors"
            >
              <ArrowLeft size={14} />
              <span>Documents</span>
            </Link>
          </li>
          <li aria-hidden="true">
            <ChevronRight size={14} className="text-gray-400" />
          </li>
          <li className="truncate max-w-[280px] sm:max-w-md font-medium text-gray-900" title={filename}>
            {filename}
          </li>
        </ol>

        <button
          onClick={() => void fetchDocAndChunks()}
          title="Refresh document data"
          className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition focus-visible:outline-2 focus-visible:outline-blue-500"
        >
          <RefreshCw size={13} className={docLoading || chunksLoading ? 'animate-spin' : ''} />
          <span>Refresh</span>
        </button>
      </nav>

      {/* Header Card */}
      <header className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-start gap-4">
            <div
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-600"
              aria-hidden="true"
            >
              <FileText size={24} />
            </div>
            <div className="min-w-0">
              <h1 className="text-xl font-bold tracking-tight text-gray-900 break-words">
                {filename}
              </h1>
              <div className="mt-2 flex flex-wrap items-center gap-2.5">
                <div className="inline-flex items-center gap-1.5 rounded-md bg-gray-100 px-2 py-0.5 text-xs font-mono font-medium text-gray-700">
                  <span className="text-gray-400">ID:</span>
                  <span>{paperId}</span>
                  <button
                    onClick={handleCopyPaperId}
                    type="button"
                    className="ml-1 rounded p-0.5 text-gray-400 hover:text-gray-700 hover:bg-gray-200 transition focus-visible:outline-2 focus-visible:outline-blue-500"
                    title="Copy Paper ID"
                    aria-label={`Copy Paper ID ${paperId}`}
                  >
                    {copiedId ? (
                      <Check size={12} className="text-emerald-600" />
                    ) : (
                      <Copy size={12} />
                    )}
                  </button>
                </div>
                <DocumentStatusBadge status={docStatus} />
              </div>
            </div>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="mt-6 border-b border-gray-100">
          <div className="flex space-x-8" role="tablist" aria-label="Document view tabs">
            <button
              role="tab"
              id="tab-overview"
              aria-selected={activeTab === 'overview'}
              aria-controls="panel-overview"
              onClick={() => setActiveTab('overview')}
              className={[
                'flex items-center gap-2 border-b-2 pb-3 text-sm font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-blue-500',
                activeTab === 'overview'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
              ].join(' ')}
            >
              <Info size={16} />
              Overview
            </button>

            <button
              role="tab"
              id="tab-text"
              aria-selected={activeTab === 'text'}
              aria-controls="panel-text"
              onClick={() => setActiveTab('text')}
              className={[
                'flex items-center gap-2 border-b-2 pb-3 text-sm font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-blue-500',
                activeTab === 'text'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
              ].join(' ')}
            >
              <FileCode2 size={16} />
              Extracted Text
              {availablePages.length > 0 && (
                <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
                  {availablePages.length} {availablePages.length === 1 ? 'page' : 'pages'}
                </span>
              )}
            </button>

            <button
              role="tab"
              id="tab-chunks"
              aria-selected={activeTab === 'chunks'}
              aria-controls="panel-chunks"
              onClick={() => setActiveTab('chunks')}
              className={[
                'flex items-center gap-2 border-b-2 pb-3 text-sm font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-blue-500',
                activeTab === 'chunks'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700',
              ].join(' ')}
            >
              <Layers size={16} />
              Chunks
              {!chunksLoading && (
                <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
                  {chunks.length}
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Tab Panels */}

      {/* ───────────────────────────────────────────────────────────────────── */}
      {/* TAB 1: OVERVIEW */}
      {/* ───────────────────────────────────────────────────────────────────── */}
      {activeTab === 'overview' && (
        <section
          id="panel-overview"
          role="tabpanel"
          aria-labelledby="tab-overview"
          className="space-y-6"
        >
          {/* Metadata Cards Grid */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* Status card */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">
                Processing Status
              </p>
              <div className="mt-2">
                <DocumentStatusBadge status={docStatus} />
              </div>
              <p className="mt-2 text-xs text-gray-400">
                Document lifecycle status in backend
              </p>
            </div>

            {/* Paper ID Card */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">
                Paper ID
              </p>
              <p className="mt-2 text-sm font-mono font-semibold text-gray-900 break-all">
                {paperId}
              </p>
              <p className="mt-1 text-xs text-gray-400">Deterministic unique identifier</p>
            </div>

            {/* Retrieved Chunks Card */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">
                Retrieved Chunks
              </p>
              <p className="mt-2 text-2xl font-bold tracking-tight text-gray-900">
                {chunksLoading ? '…' : chunks.length}
              </p>
              <p className="mt-1 text-xs text-gray-400">
                Paragraph chunks extracted from document
              </p>
            </div>

            {/* Represented Pages Card */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">
                Pages in Chunks
              </p>
              <p className="mt-2 text-2xl font-bold tracking-tight text-gray-900">
                {chunksLoading ? '…' : availablePages.length}
              </p>
              <p className="mt-1 text-xs text-gray-500 leading-tight">
                Pages represented in extracted chunks
              </p>
            </div>
          </div>

          {/* Details & Disclaimers */}
          <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm space-y-4">
            <h2 className="text-base font-semibold text-gray-900">Document Overview</h2>

            <dl className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2 text-sm border-t border-gray-100 pt-4">
              <div>
                <dt className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Original Filename
                </dt>
                <dd className="mt-1 text-sm font-medium text-gray-900 break-words">
                  {filename}
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  File Format
                </dt>
                <dd className="mt-1 text-sm font-medium text-gray-900">
                  PDF (Portable Document Format)
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Distinct Pages in Chunks
                </dt>
                <dd className="mt-1 text-sm font-medium text-gray-900">
                  {chunksLoading
                    ? 'Loading…'
                    : availablePages.length > 0
                    ? availablePages.join(', ')
                    : 'None detected'}
                </dd>
              </div>
              <div>
                <dt className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Chunk Generation
                </dt>
                <dd className="mt-1 text-sm font-medium text-gray-900">
                  {docStatus === 'completed' ? 'Completed successfully' : docStatus}
                </dd>
              </div>
            </dl>

            {/* Note box regarding page counts */}
            <div className="rounded-lg bg-blue-50 border border-blue-100 p-4 text-xs text-blue-800 flex items-start gap-3">
              <Info size={16} className="text-blue-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-blue-900">Page Count Clarification</p>
                <p className="mt-0.5 leading-relaxed">
                  The page count above represents pages represented in extracted chunks (derived
                  from chunk metadata). Pages without chunks (e.g. blank pages, diagrams without text,
                  or cover pages) may not appear in the extracted chunks list.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ───────────────────────────────────────────────────────────────────── */}
      {/* TAB 2: EXTRACTED TEXT */}
      {/* ───────────────────────────────────────────────────────────────────── */}
      {activeTab === 'text' && (
        <section
          id="panel-text"
          role="tabpanel"
          aria-labelledby="tab-text"
          className="space-y-4"
        >
          {chunksLoading ? (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-200 bg-white px-5 py-16 text-gray-400">
              <Loader size={20} className="animate-spin text-blue-600" />
              <span className="text-sm">Deriving available pages from chunks…</span>
            </div>
          ) : chunksError ? (
            <div className="flex flex-col items-center gap-3 rounded-xl border border-red-200 bg-white p-10 text-center">
              <AlertCircle size={28} className="text-red-400" />
              <p className="text-sm font-semibold text-gray-900">Could not derive page list</p>
              <p className="text-sm text-gray-500">{chunksError}</p>
              <button
                onClick={() => void fetchDocAndChunks()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
              >
                <RefreshCw size={13} />
                Retry
              </button>
            </div>
          ) : availablePages.length === 0 ? (
            <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
              <FileCode2 size={32} className="mx-auto text-gray-300" />
              <h3 className="mt-2 text-sm font-semibold text-gray-900">
                No page numbers available
              </h3>
              <p className="mt-1 text-sm text-gray-500 max-w-md mx-auto">
                No page numbers can be derived from the extracted chunks for this document.
                Pages without chunks may not appear in the selector.
              </p>
            </div>
          ) : (
            <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
              {/* Page Selector Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-100 bg-gray-50/75 px-5 py-3.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-gray-500">
                    Select Page:
                  </span>
                  <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Page selection">
                    {availablePages.map((pg) => {
                      const isSelected = selectedPage === pg;
                      return (
                        <button
                          key={pg}
                          onClick={() => setSelectedPage(pg)}
                          aria-pressed={isSelected}
                          className={[
                            'min-w-[34px] h-[30px] rounded-md px-2.5 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-blue-500',
                            isSelected
                              ? 'bg-blue-600 text-white shadow-sm font-semibold'
                              : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-100',
                          ].join(' ')}
                        >
                          {pg}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Page Prev/Next navigation */}
                <div className="flex items-center gap-2">
                  <button
                    disabled={
                      selectedPage === null ||
                      availablePages.indexOf(selectedPage) <= 0
                    }
                    onClick={() => {
                      if (selectedPage === null) return;
                      const idx = availablePages.indexOf(selectedPage);
                      if (idx > 0) setSelectedPage(availablePages[idx - 1]);
                    }}
                    className="inline-flex items-center gap-1 rounded-md border border-gray-200 bg-white px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition"
                    title="Previous page"
                  >
                    <ChevronLeft size={14} />
                    <span>Prev</span>
                  </button>

                  <span className="text-xs text-gray-500 font-medium">
                    Page {selectedPage ?? '—'} of {availablePages[availablePages.length - 1]}
                  </span>

                  <button
                    disabled={
                      selectedPage === null ||
                      availablePages.indexOf(selectedPage) >= availablePages.length - 1
                    }
                    onClick={() => {
                      if (selectedPage === null) return;
                      const idx = availablePages.indexOf(selectedPage);
                      if (idx >= 0 && idx < availablePages.length - 1) {
                        setSelectedPage(availablePages[idx + 1]);
                      }
                    }}
                    className="inline-flex items-center gap-1 rounded-md border border-gray-200 bg-white px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition"
                    title="Next page"
                  >
                    <span>Next</span>
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>

              {/* Informational banner about chunk-derived pages */}
              <div className="border-b border-gray-100 bg-amber-50/60 px-5 py-2 text-xs text-amber-800 flex items-center justify-between">
                <span>
                  <strong>Note:</strong> Showing page numbers derived from extracted chunks. Pages without text chunks may not appear in the selector.
                </span>
                <span className="text-amber-700 font-mono text-[11px]">
                  GET /api/documents/{paperId}/pages/{selectedPage ?? 1}
                </span>
              </div>

              {/* Page Content Display */}
              <div className="p-6">
                {pageLoading && (
                  <div className="flex items-center justify-center gap-2 py-16 text-gray-400">
                    <Loader size={20} className="animate-spin text-blue-600" />
                    <span className="text-sm">Loading page {selectedPage} text from backend…</span>
                  </div>
                )}

                {!pageLoading && pageError && (
                  <div className="flex flex-col items-center gap-3 py-12 text-center">
                    <AlertCircle size={28} className="text-red-400" />
                    <p className="text-sm font-semibold text-gray-900">
                      Failed to load page text
                    </p>
                    <p className="text-sm text-gray-500 max-w-md">{pageError}</p>
                    {selectedPage !== null && (
                      <button
                        onClick={() => void fetchPage(selectedPage)}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50 focus-visible:outline-2 focus-visible:outline-blue-500"
                      >
                        <RefreshCw size={13} />
                        Retry Page {selectedPage}
                      </button>
                    )}
                  </div>
                )}

                {!pageLoading && !pageError && selectedPage !== null && (
                  <div>
                    {pageTextMap[selectedPage] === undefined ? null : pageTextMap[
                        selectedPage
                      ].trim().length === 0 ? (
                      <div className="py-12 text-center text-sm text-gray-500 italic">
                        This page contains no extractable text.
                      </div>
                    ) : (
                      <div className="relative">
                        <pre className="overflow-x-auto rounded-lg bg-gray-50/70 p-6 font-mono text-xs sm:text-sm leading-relaxed text-gray-800 whitespace-pre-wrap selection:bg-blue-100 selection:text-blue-900 border border-gray-200">
                          {pageTextMap[selectedPage]}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      )}

      {/* ───────────────────────────────────────────────────────────────────── */}
      {/* TAB 3: CHUNKS */}
      {/* ───────────────────────────────────────────────────────────────────── */}
      {activeTab === 'chunks' && (
        <section
          id="panel-chunks"
          role="tabpanel"
          aria-labelledby="tab-chunks"
          className="space-y-4"
        >
          {chunksLoading ? (
            <div className="flex items-center justify-center gap-2 rounded-xl border border-gray-200 bg-white px-5 py-16 text-gray-400">
              <Loader size={20} className="animate-spin text-blue-600" />
              <span className="text-sm">Loading chunks…</span>
            </div>
          ) : chunksError ? (
            <div className="flex flex-col items-center gap-3 rounded-xl border border-red-200 bg-white p-10 text-center">
              <AlertCircle size={28} className="text-red-400" />
              <p className="text-sm font-semibold text-gray-900">Could not retrieve chunks</p>
              <p className="text-sm text-gray-500">{chunksError}</p>
              <button
                onClick={() => void fetchDocAndChunks()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
              >
                <RefreshCw size={13} />
                Retry Chunks
              </button>
            </div>
          ) : chunks.length === 0 ? (
            <EmptyState
              icon={<Layers size={32} />}
              title="No chunks found"
              description={
                docStatus === 'processing' || docStatus === 'uploaded'
                  ? 'This document is still being processed. Chunks will appear once ingestion completes.'
                  : 'No chunks were generated for this document.'
              }
            />
          ) : (
            <div className="space-y-4">
              {/* Filter / Search Bar */}
              <div className="flex flex-col gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-gray-900">
                    Document Chunks
                  </span>
                  <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
                    {filteredChunks.length} of {chunks.length}
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {/* Page Filter */}
                  {availablePages.length > 1 && (
                    <select
                      value={chunkPageFilter}
                      onChange={(e) =>
                        setChunkPageFilter(
                          e.target.value === 'all'
                            ? 'all'
                            : Number(e.target.value),
                        )
                      }
                      className="rounded-lg border border-gray-200 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-700 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="all">All pages</option>
                      {availablePages.map((pg) => (
                        <option key={pg} value={pg}>
                          Page {pg}
                        </option>
                      ))}
                    </select>
                  )}

                  {/* Text search */}
                  <input
                    type="search"
                    placeholder="Search chunk text or ID…"
                    value={chunkQuery}
                    onChange={(e) => setChunkQuery(e.target.value)}
                    className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 sm:w-60"
                  />
                </div>
              </div>

              {/* Chunk list */}
              {filteredChunks.length === 0 ? (
                <div className="rounded-xl border border-gray-200 bg-white p-8 text-center text-sm text-gray-500">
                  No chunks match your current filter or search criteria.
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredChunks.map((chunk, index) => (
                    <ChunkCard key={chunk.chunk_id} chunk={chunk} index={index} />
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
