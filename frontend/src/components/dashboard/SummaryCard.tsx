import type { ReactNode } from 'react';

interface SummaryCardProps {
  title: string;
  value: number;
  subtitle: string;
  icon: ReactNode;
  iconBg: string;
  iconColor: string;
}

export default function SummaryCard({
  title,
  value,
  subtitle,
  icon,
  iconBg,
  iconColor,
}: SummaryCardProps) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="mt-1 text-3xl font-bold text-gray-900">{value}</p>
          <p className="mt-1 text-xs text-gray-400">{subtitle}</p>
        </div>
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-lg ${iconBg}`}
          aria-hidden="true"
        >
          <span className={iconColor}>{icon}</span>
        </div>
      </div>
    </div>
  );
}
