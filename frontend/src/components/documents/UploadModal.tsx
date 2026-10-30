import { useRef, useState, type DragEvent, type ChangeEvent } from 'react';
import { Upload, X, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import type { ApiDocument } from '@/types/document';

const ACCEPTED_MIME = 'application/pdf';
const ACCEPTED_EXT = '.pdf';

interface UploadModalProps {
  uploading: boolean;
  uploadError: string | null;
  onUpload: (file: File) => Promise<ApiDocument | null>;
  onClose: () => void;
  onClearError: () => void;
}

type ModalState = 'idle' | 'uploading' | 'success' | 'error';

/** Validates a File before submitting to the API. Returns an error string or null. */
function validateFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(ACCEPTED_EXT)) {
    return `"${file.name}" is not a PDF file. Please select a .pdf file.`;
  }
  if (file.type && file.type !== ACCEPTED_MIME) {
    return `"${file.name}" does not appear to be a valid PDF.`;
  }
  if (file.size === 0) {
    return 'The selected file is empty (0 bytes). Please choose a valid PDF.';
  }
  return null;
}

export default function UploadModal({
  uploading,
  uploadError,
  onUpload,
  onClose,
  onClearError,
}: UploadModalProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [successDoc, setSuccessDoc] = useState<ApiDocument | null>(null);

  const state: ModalState = uploading
    ? 'uploading'
    : successDoc
    ? 'success'
    : uploadError || localError
    ? 'error'
    : 'idle';

  function handleFileSelected(file: File) {
    setLocalError(null);
    onClearError();

    const validationError = validateFile(file);
    if (validationError) {
      setLocalError(validationError);
      return;
    }

    void handleUpload(file);
  }

  async function handleUpload(file: File) {
    const result = await onUpload(file);
    if (result) {
      setSuccessDoc(result);
    }
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFileSelected(file);
    // Reset the input so the same file can be re-selected after a failure
    e.target.value = '';
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(true);
  }

  function handleDragLeave() {
    setDragging(false);
  }

  function handleClose() {
    if (uploading) return; // Block dismissal while uploading
    onClearError();
    setLocalError(null);
    setSuccessDoc(null);
    onClose();
  }

  const errorMessage = localError ?? uploadError;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/40"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Dialog */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-modal-title"
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
      >
        <div className="relative w-full max-w-md rounded-xl bg-white shadow-xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-gray-100 px-6 py-4">
            <h2 id="upload-modal-title" className="text-base font-semibold text-gray-900">
              Upload Research Paper
            </h2>
            <button
              onClick={handleClose}
              disabled={uploading}
              className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-blue-500"
              aria-label="Close upload dialog"
            >
              <X size={18} />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 space-y-4">
            {/* Drop zone — hidden when uploading or complete */}
            {state === 'idle' || state === 'error' ? (
              <div
                role="button"
                tabIndex={0}
                aria-label="Drop PDF here or click to browse"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
                className={[
                  'flex flex-col items-center gap-3 rounded-lg border-2 border-dashed px-6 py-10 text-center cursor-pointer transition-colors',
                  dragging
                    ? 'border-blue-400 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50',
                ].join(' ')}
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 text-blue-500">
                  <Upload size={22} />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    Drop a PDF here, or{' '}
                    <span className="text-blue-600 underline underline-offset-2">
                      browse
                    </span>
                  </p>
                  <p className="mt-1 text-xs text-gray-400">PDF only · max 25 MB</p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_EXT}
                  className="sr-only"
                  onChange={handleInputChange}
                  aria-hidden="true"
                  tabIndex={-1}
                />
              </div>
            ) : null}

            {/* Uploading state */}
            {state === 'uploading' && (
              <div className="flex flex-col items-center gap-3 rounded-lg border border-gray-100 bg-gray-50 px-6 py-10">
                <Loader size={28} className="animate-spin text-blue-500" />
                <p className="text-sm font-medium text-gray-700">
                  Uploading and processing…
                </p>
                <p className="text-xs text-gray-400">This may take a few seconds.</p>
              </div>
            )}

            {/* Success state */}
            {state === 'success' && successDoc && (
              <div className="flex flex-col items-center gap-3 rounded-lg border border-emerald-100 bg-emerald-50 px-6 py-8 text-center">
                <CheckCircle size={28} className="text-emerald-500" />
                <div>
                  <p className="text-sm font-semibold text-gray-900">Upload successful</p>
                  <p className="mt-1 text-sm text-gray-600 truncate max-w-xs" title={successDoc.filename}>
                    {successDoc.filename}
                  </p>
                  <p className="mt-1 text-xs text-gray-400">
                    Paper ID: <span className="font-mono">{successDoc.paper_id}</span>
                    {' · '}Status: {successDoc.status}
                  </p>
                </div>
              </div>
            )}

            {/* Error banner */}
            {errorMessage && (
              <div
                role="alert"
                className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4"
              >
                <AlertCircle size={18} className="shrink-0 text-red-500 mt-0.5" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-red-700">Upload failed</p>
                  <p className="mt-0.5 text-sm text-red-600 break-words">{errorMessage}</p>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex justify-end gap-2 border-t border-gray-100 px-6 py-4">
            {state === 'success' ? (
              <>
                <button
                  onClick={() => {
                    setSuccessDoc(null);
                    setLocalError(null);
                    onClearError();
                  }}
                  className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus-visible:outline-2 focus-visible:outline-blue-500"
                >
                  Upload Another
                </button>
                <button
                  onClick={handleClose}
                  className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 focus-visible:outline-2 focus-visible:outline-blue-500"
                >
                  Done
                </button>
              </>
            ) : (
              <button
                onClick={handleClose}
                disabled={uploading}
                className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-blue-500"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
