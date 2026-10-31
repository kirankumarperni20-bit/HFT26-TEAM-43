import { useState, useEffect, useRef, useCallback } from 'react';
import {
  startResearchSynthesis,
  getResearchResult,
  listResearchJobs,
  verifyResearchClaims,
} from '@/services/api';
import type {
  StartResearchPayload,
  ResearchResult,
  ResearchStatus,
} from '@/types/research';

export const STORAGE_KEY_LAST_JOB_ID = 'researchai_active_job_id';
export const STORAGE_KEY_LAST_PAPERS = 'researchai_selected_papers';
export const STORAGE_KEY_LAST_FOCUS = 'researchai_research_focus';
export const STORAGE_KEY_LAST_TASKS = 'researchai_selected_tasks';

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
  isRestoring: boolean;
  restoreError: string | null;
  previousJobs: ResearchResult[];
  startJob: (payload: StartResearchPayload) => Promise<boolean>;
  verifyClaims: () => Promise<boolean>;
  retryPolling: () => void;
  resetJob: () => void;
  clearSubmitError: () => void;
  loadJob: (id: string) => Promise<boolean>;
  refreshJobsList: () => Promise<void>;
}

const POLLING_INTERVAL_MS = 2500;
const MAX_CONSECUTIVE_POLL_ERRORS = 10;

/**
 * Custom hook to manage the lifecycle and persistence of research synthesis jobs:
 * 1. Submission: Calls POST /api/research/generate and persists active job ID in localStorage.
 * 2. Persistence & Restoration: Automatically restores the active job on mount, page navigation, or browser refresh.
 * 3. Polling: Polls GET /api/research/{research_id} every 2.5s while 'queued' or 'processing'.
 * 4. Terminal state: Halts polling on 'completed' or 'failed'.
 * 5. Multi-Job History: Keeps access to previously completed jobs from the backend.
 * 6. Verification: Calls POST /api/research/{research_id}/verify (Milestone F5).
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

  // Restoration state to prevent UI empty-state flicker
  const [isRestoring, setIsRestoring] = useState<boolean>(() => {
    try {
      if (typeof window !== 'undefined') {
        const urlParams = new URLSearchParams(window.location.search);
        return Boolean(urlParams.get('job_id') || localStorage.getItem(STORAGE_KEY_LAST_JOB_ID));
      }
    } catch {}
    return false;
  });
  const [restoreError, setRestoreError] = useState<string | null>(null);
  const [previousJobs, setPreviousJobs] = useState<ResearchResult[]>([]);

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

  // Load a specific research job by ID (restores from backend)
  const loadJob = useCallback(
    async (targetJobId: string): Promise<boolean> => {
      if (!targetJobId) return false;
      clearTimer();
      setIsRestoring(true);
      setRestoreError(null);
      setPollingError(null);

      try {
        const response = await getResearchResult(targetJobId);
        if (!isMountedRef.current) return false;

        setJobId(response.research_id);
        setStatus(response.status);
        setResult(response);

        try {
          localStorage.setItem(STORAGE_KEY_LAST_JOB_ID, response.research_id);
        } catch {}

        setIsRestoring(false);

        // Resume polling if job is still processing
        if (response.status === 'queued' || response.status === 'processing') {
          void executePoll(response.research_id);
        }
        return true;
      } catch (err) {
        if (!isMountedRef.current) return false;

        const message =
          err instanceof Error
            ? err.message
            : `Failed to restore research job '${targetJobId}'.`;
        setRestoreError(message);
        setIsRestoring(false);

        // Clear invalid saved ID so user can start fresh
        try {
          localStorage.removeItem(STORAGE_KEY_LAST_JOB_ID);
        } catch {}
        return false;
      }
    },
    [clearTimer, executePoll],
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

  // Fetch list of previous jobs from backend
  const refreshJobsList = useCallback(async () => {
    try {
      const jobs = await listResearchJobs();
      if (isMountedRef.current) {
        setPreviousJobs(jobs);
      }
    } catch {
      // Non-fatal if list endpoint is unavailable
    }
  }, []);

  // Restore active job on mount
  useEffect(() => {
    let targetId: string | null = null;
    try {
      const urlParams = new URLSearchParams(window.location.search);
      targetId = urlParams.get('job_id');
      if (!targetId) {
        targetId = localStorage.getItem(STORAGE_KEY_LAST_JOB_ID);
      }
    } catch {}

    if (targetId) {
      void loadJob(targetId);
    } else {
      setIsRestoring(false);
    }

    void refreshJobsList();
  }, [loadJob, refreshJobsList]);

  // Start research synthesis job
  const startJob = useCallback(
    async (payload: StartResearchPayload): Promise<boolean> => {
      clearTimer();
      setIsSubmitting(true);
      setSubmitError(null);
      setPollingError(null);
      setRestoreError(null);
      setConsecutivePollErrors(0);
      setResult(null);
      setIsVerifying(false);
      setVerificationError(null);

      // Save input configuration for restoration
      try {
        localStorage.setItem(STORAGE_KEY_LAST_PAPERS, JSON.stringify(payload.paper_ids));
        localStorage.setItem(STORAGE_KEY_LAST_TASKS, JSON.stringify(payload.tasks));
        if (payload.focus_query || payload.research_focus) {
          localStorage.setItem(
            STORAGE_KEY_LAST_FOCUS,
            payload.focus_query || payload.research_focus || '',
          );
        } else {
          localStorage.removeItem(STORAGE_KEY_LAST_FOCUS);
        }
      } catch {}

      try {
        const response = await startResearchSynthesis(payload);
        if (!isMountedRef.current) return false;

        setJobId(response.research_id);
        setStatus(response.status);

        try {
          localStorage.setItem(STORAGE_KEY_LAST_JOB_ID, response.research_id);
        } catch {}

        setIsSubmitting(false);
        // Refresh available jobs list
        void refreshJobsList();
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
    [clearTimer, refreshJobsList],
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

  // Reset active research state to start a new job
  const resetJob = useCallback(() => {
    clearTimer();
    isPollingActiveRef.current = false;
    setJobId(null);
    setStatus(null);
    setResult(null);
    setIsSubmitting(false);
    setSubmitError(null);
    setPollingError(null);
    setRestoreError(null);
    setConsecutivePollErrors(0);
    setIsVerifying(false);
    setVerificationError(null);

    try {
      localStorage.removeItem(STORAGE_KEY_LAST_JOB_ID);
    } catch {}
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
    isRestoring,
    restoreError,
    previousJobs,
    startJob,
    verifyClaims,
    retryPolling,
    resetJob,
    clearSubmitError,
    loadJob,
    refreshJobsList,
  };
}
