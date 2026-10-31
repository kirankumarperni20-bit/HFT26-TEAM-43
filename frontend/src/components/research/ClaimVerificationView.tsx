import { useState } from 'react';
import {
  ShieldCheck,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
  FileText,
} from 'lucide-react';
import type {
  ClaimVerificationItem,
  VerificationDecision,
  VerifiedEvidenceItem,
} from '@/types/research';

interface ClaimVerificationViewProps {
  verifications?: ClaimVerificationItem[];
  isVerified: boolean;
  isVerifying: boolean;
  onVerify: () => void;
  error?: string | null;
  paperFilenameMap?: Record<string, string>;
}

export function getStatusBadgeConfig(status: VerificationDecision) {
  const norm = String(status).toUpperCase().replace('-', '_');

  if (norm === 'SUPPORTED') {
    return {
      label: 'Supported by the paper',
      badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      icon: CheckCircle2,
      iconClass: 'text-emerald-600',
    };
  }
  if (norm === 'PARTIALLY_SUPPORTED') {
    return {
      label: 'Supported by the paper',
      badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      icon: CheckCircle2,
      iconClass: 'text-emerald-600',
    };
  }
  if (norm === 'CONTRADICTED') {
    return {
      label: 'Conflicts with the paper',
      badgeClass: 'bg-rose-50 text-rose-700 border-rose-200',
      icon: XCircle,
      iconClass: 'text-rose-600',
    };
  }
  if (
    norm === 'INSUFFICIENT_EVIDENCE' ||
    norm === 'INSUFFICIENT EVIDENCE' ||
    norm === 'UNRESOLVED'
  ) {
    return {
      label: 'Not enough evidence',
      badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
      icon: AlertTriangle,
      iconClass: 'text-amber-600',
    };
  }
  if (norm === 'PENDING') {
    return {
      label: 'Checking claim…',
      badgeClass: 'bg-blue-50 text-blue-700 border-blue-200',
      icon: Loader2,
      iconClass: 'text-blue-600 animate-spin',
    };
  }
  return {
    label: 'Could not check this claim',
    badgeClass: 'bg-gray-100 text-gray-700 border-gray-300',
    icon: HelpCircle,
    iconClass: 'text-gray-500',
  };
}

function EvidenceSnippet({
  item,
  paperFilenameMap,
  type = 'supporting',
}: {
  item: VerifiedEvidenceItem;
  paperFilenameMap: Record<string, string>;
  type?: 'supporting' | 'conflicting';
}) {
  const paperName = paperFilenameMap[item.paper_id] || item.paper_id;
  const isConflicting = type === 'conflicting';

  return (
    <div
      className={`rounded-lg p-3 border text-xs ${
        isConflicting
          ? 'bg-rose-50/60 border-rose-200/80 text-rose-950'
          : 'bg-gray-50 border-gray-200 text-gray-800'
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5 font-medium">
        <div className="flex items-center gap-1.5 truncate max-w-[280px]">
          <FileText size={12} className={isConflicting ? 'text-rose-500' : 'text-blue-500'} />
          <span className="font-semibold text-gray-900 truncate" title={paperName}>
            {paperName}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-gray-500">
          {item.page && <span>Page {item.page}</span>}
          {item.section && (
            <>
              <span>•</span>
              <span className="italic truncate max-w-[150px]">{item.section}</span>
            </>
          )}
          {item.chunk_id && (
            <>
              <span>•</span>
              <span className="font-mono text-[10px] bg-white px-1.5 py-0.5 rounded border border-gray-200">
                {item.chunk_id}
              </span>
            </>
          )}
        </div>
      </div>

      <blockquote className="italic border-l-2 pl-2.5 py-0.5 border-gray-300 text-gray-700 font-serif text-[13px] leading-relaxed">
        "{item.quote}"
      </blockquote>

      {item.validation_notes && (
        <p className="mt-1.5 text-[10px] text-gray-500">{item.validation_notes}</p>
      )}
    </div>
  );
}

function ClaimCard({
  claim,
  paperFilenameMap,
}: {
  claim: ClaimVerificationItem;
  paperFilenameMap: Record<string, string>;
}) {
  const [expanded, setExpanded] = useState(false);
  const cfg = getStatusBadgeConfig(claim.verification_status);
  const StatusIcon = cfg.icon;

  const hasSupporting = claim.supporting_evidence && claim.supporting_evidence.length > 0;
  const hasContradicting =
    claim.contradicting_evidence && claim.contradicting_evidence.length > 0;
  const hasEvidence = hasSupporting || hasContradicting;

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-xs p-4 transition-all hover:border-gray-300">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${cfg.badgeClass}`}
            >
              <StatusIcon size={13} className={cfg.iconClass} />
              {cfg.label}
            </span>
          </div>

          <p className="text-sm font-medium text-gray-900 leading-snug">
            {claim.claim_text}
          </p>

          <p className="mt-1.5 text-xs text-gray-600 leading-relaxed">
            {claim.explanation}
          </p>

          {claim.unresolved_reason && (
            <p className="mt-1 text-xs text-amber-700 italic">
              Note: {claim.unresolved_reason}
            </p>
          )}
        </div>
      </div>

      {hasEvidence && (
        <div className="mt-3 pt-2.5 border-t border-gray-100">
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:text-blue-800 transition-colors focus-visible:outline-2 focus-visible:outline-blue-500 rounded px-1 -ml-1 py-0.5"
            aria-expanded={expanded}
          >
            <span>
              {hasContradicting
                ? 'View Conflicting Evidence'
                : 'View Supporting Evidence'}
            </span>
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          {expanded && (
            <div className="mt-2.5 space-y-2">
              <h5 className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                Evidence from the Paper
              </h5>

              {hasSupporting &&
                claim.supporting_evidence.map((ev, i) => (
                  <EvidenceSnippet
                    key={`${ev.evidence_id}-${i}`}
                    item={ev}
                    paperFilenameMap={paperFilenameMap}
                    type="supporting"
                  />
                ))}

              {hasContradicting &&
                claim.contradicting_evidence.map((ev, i) => (
                  <EvidenceSnippet
                    key={`${ev.evidence_id}-${i}`}
                    item={ev}
                    paperFilenameMap={paperFilenameMap}
                    type="conflicting"
                  />
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ClaimVerificationView({
  verifications = [],
  isVerified,
  isVerifying,
  onVerify,
  error,
  paperFilenameMap = {},
}: ClaimVerificationViewProps) {
  // Counts by outcome
  const supportedCount = verifications.filter(
    (v) =>
      v.verification_status === 'SUPPORTED' ||
      v.verification_status === 'PARTIALLY_SUPPORTED',
  ).length;

  const contradictedCount = verifications.filter(
    (v) => v.verification_status === 'CONTRADICTED',
  ).length;

  const insufficientCount = verifications.filter(
    (v) =>
      v.verification_status === 'INSUFFICIENT EVIDENCE' ||
      v.verification_status === 'INSUFFICIENT_EVIDENCE' ||
      v.verification_status === 'UNRESOLVED',
  ).length;

  const failedCount = verifications.filter(
    (v) =>
      v.verification_status === 'VERIFICATION FAILED' ||
      v.verification_status === 'VERIFICATION_FAILED',
  ).length;

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-xs">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-4 border-b border-gray-100">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck size={20} className="text-blue-600 shrink-0" />
            <h3 className="text-base font-semibold text-gray-900">
              Claim Verification
            </h3>
            {isVerified && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                Verified
              </span>
            )}
          </div>
          <p className="mt-0.5 text-xs text-gray-500">
            Evaluating whether claims in the generated summary are supported, contradicted, or lack evidence.
          </p>
        </div>

        <div>
          <button
            type="button"
            onClick={onVerify}
            disabled={isVerifying}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60 disabled:cursor-not-allowed shadow-xs transition-colors"
          >
            {isVerifying ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                <span>Checking claim…</span>
              </>
            ) : isVerified ? (
              <>
                <ShieldCheck size={14} />
                <span>Re-verify Claims</span>
              </>
            ) : (
              <>
                <ShieldCheck size={14} />
                <span>Verify Claims</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="mt-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center justify-between">
          <span>{error}</span>
          <button
            type="button"
            onClick={onVerify}
            className="ml-3 font-semibold text-rose-900 underline hover:no-underline"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Verification Summary Counts */}
      {isVerified && verifications.length > 0 && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <div className="rounded-lg bg-emerald-50/70 border border-emerald-200 p-2.5 text-center">
            <span className="block text-lg font-bold text-emerald-800">
              {supportedCount}
            </span>
            <span className="text-[11px] font-medium text-emerald-700">
              Supported by the paper
            </span>
          </div>

          <div className="rounded-lg bg-rose-50/70 border border-rose-200 p-2.5 text-center">
            <span className="block text-lg font-bold text-rose-800">
              {contradictedCount}
            </span>
            <span className="text-[11px] font-medium text-rose-700">
              Conflicts with the paper
            </span>
          </div>

          <div className="rounded-lg bg-amber-50/70 border border-amber-200 p-2.5 text-center">
            <span className="block text-lg font-bold text-amber-800">
              {insufficientCount}
            </span>
            <span className="text-[11px] font-medium text-amber-700">
              Not enough evidence
            </span>
          </div>

          {failedCount > 0 ? (
            <div className="rounded-lg bg-gray-100 border border-gray-300 p-2.5 text-center">
              <span className="block text-lg font-bold text-gray-800">
                {failedCount}
              </span>
              <span className="text-[11px] font-medium text-gray-700">
                Could not check this claim
              </span>
            </div>
          ) : (
            <div className="rounded-lg bg-blue-50/70 border border-blue-200 p-2.5 text-center">
              <span className="block text-lg font-bold text-blue-800">
                {verifications.length}
              </span>
              <span className="text-[11px] font-medium text-blue-700">
                Total Claims Checked
              </span>
            </div>
          )}
        </div>
      )}

      {/* Claims List */}
      <div className="mt-4 space-y-3">
        {isVerifying && verifications.length === 0 ? (
          <div className="py-8 text-center space-y-2">
            <Loader2 size={24} className="animate-spin text-blue-600 mx-auto" />
            <p className="text-xs font-medium text-gray-600">
              Checking claim… Evaluating each proposition against candidate document passages.
            </p>
          </div>
        ) : !isVerified && verifications.length === 0 ? (
          <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50/60 p-6 text-center">
            <ShieldCheck size={28} className="text-gray-400 mx-auto mb-2" />
            <h4 className="text-xs font-semibold text-gray-700">
              Verify Claims in this Research Summary
            </h4>
            <p className="text-xs text-gray-500 max-w-md mx-auto mt-1 mb-3">
              Click &quot;Verify Claims&quot; to cross-reference every objective, methodology step, and finding directly against verbatim passages from the source documents.
            </p>
            <button
              type="button"
              onClick={onVerify}
              disabled={isVerifying}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 shadow-xs transition-colors"
            >
              <ShieldCheck size={13} />
              <span>Verify Claims</span>
            </button>
          </div>
        ) : (
          verifications.map((claim) => (
            <ClaimCard
              key={claim.claim_id}
              claim={claim}
              paperFilenameMap={paperFilenameMap}
            />
          ))
        )}
      </div>
    </section>
  );
}
