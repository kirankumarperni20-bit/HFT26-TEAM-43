import { useState, useEffect, useRef, useCallback } from 'react';
import {
  startResearchSynthesis,
  getResearchResult,
  verifyResearchClaims,
} from '@/services/api';
import type {
  StartResearchPayload,
  ResearchResult,
  ResearchStatus,
} from '@/types/research';

export interface UseResearchJobResult {
  jobId: string | null;
  status: ResearchStatus | null;
  result: ResearchResult | null;
  isSubmitting: boolean;
  submitError: string | null;
  pollingError: string | null;
  consecutivePollErrors: number;
  isVerifying: boolean;
  verificationError: string | null;
  startJob: (payload: StartResearchPayload) => Promise<boolean>;
  verifyClaims: () => Promise<boolean>;
  retryPolling: () => void;
  resetJob: () => void;
  clearSubmitError: () => void;
}

const POLLING_INTERVAL_MS = 2500;
const MAX_CONSECUTIVE_POLL_ERRORS = 10;

/**
 * Custom hook to manage the lifecycle of a research synthesis job:
 * 1. Submission: Calls POST /api/research/generate
 * 2. Polling: Polls GET /api/research/{research_id} every 2.5s while 'queued' or 'processing'
 * 3. Terminal state: Halts polling on 'completed' or 'failed'
 * 4. Verification: Calls POST /api/research/{research_id}/verify (Milestone F5)
 * 5. Cleanup: Cleans up timers and in-flight polling on unmount or job reset
 */
export function useResearchJob(): UseResearchJobResult {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<ResearchStatus | null>(null);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [pollingError, setPollingError] = useState<string | null>(null);
  const [consecutivePollErrors, setConsecutivePollErrors] = useState(0);

  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationError, setVerificationError] = useState<string | null>(null);

  const pollTimeoutRef = useRef<number | null>(null);
  const isPollingActiveRef = useRef(false);
  const isMountedRef = useRef(true);

  // Track component mount status
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (pollTimeoutRef.current !== null) {
        window.clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
    };
  }, []);

  // Clear any existing polling timer
  const clearTimer = useCallback(() => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  // Execute a single poll request without overlap
  const executePoll = useCallback(
    async (id: string) => {
      if (!isMountedRef.current || isPollingActiveRef.current) {
        return;
      }

      isPollingActiveRef.current = true;
      try {
        const response = await getResearchResult(id);
        if (!isMountedRef.current) return;

        setResult(response);
        setStatus(response.status);
        setPollingError(null);
        setConsecutivePollErrors(0);

        // Continue polling if job is still in progress
        if (response.status === 'queued' || response.status === 'processing') {
          clearTimer();
          const delay =
            typeof response.error?.retry_after_seconds === 'number' &&
            response.error.retry_after_seconds > 0
              ? response.error.retry_after_seconds * 1000
              : POLLING_INTERVAL_MS;

          pollTimeoutRef.current = window.setTimeout(() => {
            void executePoll(id);
          }, delay);
        } else {
          clearTimer();
        }
      } catch (err) {
        if (!isMountedRef.current) return;

        const message =
          err instanceof Error
            ? err.message
            : 'Temporary network issue while retrieving job status.';

        setPollingError(message);
        setConsecutivePollErrors((prev) => {
          const nextCount = prev + 1;
          clearTimer();
          if (nextCount < MAX_CONSECUTIVE_POLL_ERRORS) {
            pollTimeoutRef.current = window.setTimeout(() => {
              void executePoll(id);
            }, POLLING_INTERVAL_MS);
          }
          return nextCount;
        });
      } finally {
        isPollingActiveRef.current = false;
      }
    },
    [clearTimer],
  );

  // Manage polling loop when jobId or status changes
  useEffect(() => {
    if (!jobId) {
      clearTimer();
      return;
    }

    if (status === 'queued' || status === 'processing') {
      void executePoll(jobId);
    } else {
      clearTimer();
    }

    return () => {
      clearTimer();
    };
  }, [jobId, status, executePoll, clearTimer]);

  // Start research synthesis job
  const startJob = useCallback(
    async (payload: StartResearchPayload): Promise<boolean> => {
      clearTimer();
      setIsSubmitting(true);
      setSubmitError(null);
      setPollingError(null);
      setConsecutivePollErrors(0);
      setResult(null);
      setIsVerifying(false);
      setVerificationError(null);

      try {
        const response = await startResearchSynthesis(payload);
        if (!isMountedRef.current) return false;

        setJobId(response.research_id);
        setStatus(response.status);
        setIsSubmitting(false);
        return true;
      } catch (err) {
        if (!isMountedRef.current) return false;

        const message =
          err instanceof Error
            ? err.message
            : 'Failed to initiate research synthesis job.';
        setSubmitError(message);
        setIsSubmitting(false);
        return false;
      }
    },
    [clearTimer],
  );

  // Claim verification action (Milestone F5)
  const verifyClaims = useCallback(async (): Promise<boolean> => {
    if (!jobId) return false;
    setIsVerifying(true);
    setVerificationError(null);

    try {
      const updated = await verifyResearchClaims(jobId);
      if (!isMountedRef.current) return false;

      setResult(updated);
      setIsVerifying(false);
      return true;
    } catch (err) {
      if (!isMountedRef.current) return false;

      const message =
        err instanceof Error
          ? err.message
          : 'Failed to verify claims against source documents.';
      setVerificationError(message);
      setIsVerifying(false);
      return false;
    }
  }, [jobId]);

  // Manual retry for polling if stopped after consecutive errors
  const retryPolling = useCallback(() => {
    if (jobId && (status === 'queued' || status === 'processing')) {
      setPollingError(null);
      setConsecutivePollErrors(0);
      void executePoll(jobId);
    }
  }, [jobId, status, executePoll]);

  // Reset entire research state to start a new job
  const resetJob = useCallback(() => {
    clearTimer();
    isPollingActiveRef.current = false;
    setJobId(null);
    setStatus(null);
    setResult(null);
    setIsSubmitting(false);
    setSubmitError(null);
    setPollingError(null);
    setConsecutivePollErrors(0);
    setIsVerifying(false);
    setVerificationError(null);
  }, [clearTimer]);

  const clearSubmitError = useCallback(() => {
    setSubmitError(null);
  }, []);

  return {
    jobId,
    status,
    result,
    isSubmitting,
    submitError,
    pollingError,
    consecutivePollErrors,
    isVerifying,
    verificationError,
    startJob,
    verifyClaims,
    retryPolling,
    resetJob,
    clearSubmitError,
  };
}
