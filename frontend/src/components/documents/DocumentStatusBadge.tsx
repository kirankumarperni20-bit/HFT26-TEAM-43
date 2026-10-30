import type { ApiDocumentStatus } from '@/types/document';

const config: Record<
  ApiDocumentStatus,
  { label: string; classes: string; indicator: string }
> = {
  uploaded: {
    label: 'Uploaded',
    classes: 'bg-blue-50 text-blue-700 ring-blue-200',
    indicator: 'bg-blue-400',
  },
  completed: {
    label: 'Completed',
    classes: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
    indicator: 'bg-emerald-500',
  },
  processing: {
    label: 'Processing',
    classes: 'bg-amber-50 text-amber-700 ring-amber-200',
    indicator: 'bg-amber-500',
  },
  failed: {
    label: 'Failed',
    classes: 'bg-red-50 text-red-700 ring-red-200',
    indicator: 'bg-red-500',
  },
};

interface DocumentStatusBadgeProps {
  status: ApiDocumentStatus;
}

export default function DocumentStatusBadge({ status }: DocumentStatusBadgeProps) {
  const resolved = config[status] ?? config.failed;
  const { label, classes, indicator } = resolved;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${classes}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${indicator}`} aria-hidden="true" />
      {label}
    </span>
  );
}
