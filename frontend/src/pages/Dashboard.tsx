import { useEffect, useState } from 'react';
import { Upload, FileText, CheckCircle, Clock, ShieldCheck, XCircle } from 'lucide-react';
import SummaryCard from '@/components/dashboard/SummaryCard';
import RecentDocuments from '@/components/dashboard/RecentDocuments';
import GetStarted from '@/components/dashboard/GetStarted';
import UploadModal from '@/components/documents/UploadModal';
import { useDocuments } from '@/hooks/useDocuments';
import { deriveDashboardStats } from '@/types/document';

export default function Dashboard() {
  const { documents, loading, error, uploading, uploadError, fetch, upload, clearUploadError } =
    useDocuments();
  const [showUpload, setShowUpload] = useState(false);

  // Fetch document list on mount
  useEffect(() => {
    void fetch();
  }, [fetch]);

  const stats = deriveDashboardStats(documents);

  const cards = [
    {
      title: 'Total Papers',
      value: stats.totalPapers,
      subtitle: 'Uploaded documents',
      icon: <FileText size={20} />,
      iconBg: 'bg-blue-50',
      iconColor: 'text-blue-600',
    },
    {
      title: 'Processed Papers',
      value: stats.processedPapers,
      subtitle: 'Ready for research',
      icon: <CheckCircle size={20} />,
      iconBg: 'bg-emerald-50',
      iconColor: 'text-emerald-600',
    },
    {
      title: 'Processing',
      value: stats.processingPapers,
      subtitle: 'Currently processing',
      icon: <Clock size={20} />,
      iconBg: 'bg-amber-50',
      iconColor: 'text-amber-600',
    },
    {
      title: 'Claim Verification',
      value: stats.processedPapers,
      subtitle: 'Evidence-based checking',
      icon: <ShieldCheck size={20} />,
      iconBg: 'bg-purple-50',
      iconColor: 'text-purple-600',
    },
  ];

  // Add a Failed card only when there are failures (keeps layout compact otherwise)
  if (stats.failedPapers > 0) {
    cards.push({
      title: 'Failed',
      value: stats.failedPapers,
      subtitle: 'Ingestion errors',
      icon: <XCircle size={20} />,
      iconBg: 'bg-red-50',
      iconColor: 'text-red-500',
    });
  }

  return (
    <div className="space-y-8 max-w-6xl">
      {/* Page intro */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-gray-900">
            Welcome to Research Assistant
          </h1>
          <p className="mt-1 text-sm text-gray-500 max-w-xl">
            Upload research papers, explore their content, and generate verifiable summaries
            with evidence.
          </p>
        </div>
        <button
          id="upload-paper-btn"
          onClick={() => setShowUpload(true)}
          className="inline-flex shrink-0 items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 transition-colors"
          aria-label="Upload a research paper"
        >
          <Upload size={16} />
          Upload Research Paper
        </button>
      </div>

      {/* Summary cards */}
      <section aria-label="Summary statistics">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {cards.map((card) => (
            <SummaryCard key={card.title} {...card} />
          ))}
        </div>
      </section>

      {/* Main dashboard grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Recent documents (7 cols) */}
        <div className="lg:col-span-7">
          <RecentDocuments
            documents={documents}
            loading={loading}
            error={error}
            onRetry={fetch}
          />
        </div>

        {/* Get started steps (5 cols) */}
        <div className="lg:col-span-5">
          <GetStarted />
        </div>
      </div>

      {/* Upload modal */}
      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUpload={upload}
          uploading={uploading}
          uploadError={uploadError}
          onClearError={clearUploadError}
        />
      )}
    </div>
  );
}
