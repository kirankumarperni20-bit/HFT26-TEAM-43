import type { GroundedField } from '@/types/research';
import EvidenceReferenceList from './EvidenceReferenceList';

interface GroundedFieldViewProps {
  field?: GroundedField;
  paperFilenameMap?: Record<string, string>;
  fallbackText?: string;
}

export default function GroundedFieldView({
  field,
  paperFilenameMap,
  fallbackText = 'No information provided.',
}: GroundedFieldViewProps) {
  if (!field || !field.text.trim()) {
    return <p className="text-xs text-gray-400 italic">{fallbackText}</p>;
  }

  return (
    <div className="space-y-1">
      <p className="text-xs text-gray-800 leading-relaxed whitespace-pre-line">
        {field.text}
      </p>

      {field.evidence && field.evidence.length > 0 && (
        <EvidenceReferenceList
          references={field.evidence}
          paperFilenameMap={paperFilenameMap}
        />
      )}
    </div>
  );
}

