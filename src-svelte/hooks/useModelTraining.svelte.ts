import type {
  TrainingEvent_Deserialize,
  TrainingResult_Deserialize,
} from '@generated/bindings';
import { useTrainingConfig } from '../contexts/TrainingConfigContext';
import {
  createTrainingChannel,
  nativeClient,
  NativeCommandError,
  type NativeClient,
} from '../lib/native/client';
import { logger } from '../services/logging/logger';

export interface TrainingState {
  isTraining: boolean;
  isTrainingPipeline: boolean;
  progress: number;
  stage: 'idle' | 'validating' | 'processing' | 'training' | 'evaluating' | 'deploying';
  postureResult: TrainingResult_Deserialize | null;
  presenceResult: TrainingResult_Deserialize | null;
  error: string | null;
  warnings: string[];
  trainingQueued: boolean;
}

export interface UseModelTrainingReturn {
  readonly state: TrainingState;
  train(options?: { doCV?: boolean }): Promise<boolean>;
  trainAndDeploy(options?: { doCV?: boolean; onModelDeployed?: () => void }): Promise<boolean>;
  cancel(): Promise<void>;
  reconcile(): Promise<void>;
}

const initialState: TrainingState = {
  isTraining: false,
  isTrainingPipeline: false,
  progress: 0,
  stage: 'idle',
  postureResult: null,
  presenceResult: null,
  error: null,
  warnings: [],
  trainingQueued: false,
};

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Native training actor adapter. Concurrent starts are rejected, never queued. */
export function useModelTraining(client: NativeClient = nativeClient): UseModelTrainingReturn {
  let state = $state<TrainingState>({ ...initialState });
  const trainingConfig = useTrainingConfig();
  let active = false;
  let operationGeneration = 0;
  let terminalSeen = false;
  let cancelledSeen = false;
  let activeJobId: number | null = null;
  let lastSequence = -1;

  const isCurrentOperation = (operation: number): boolean =>
    operation === operationGeneration;

  const releaseOperation = (operation: number): void => {
    if (isCurrentOperation(operation)) active = false;
  };

  const acceptsSequence = (operation: number, event: TrainingEvent_Deserialize): boolean => {
    if (!isCurrentOperation(operation)) return false;
    if (activeJobId === null || event.jobId !== activeJobId) return false;
    if (event.sequence !== lastSequence + 1) return false;
    lastSequence = event.sequence;
    return true;
  };

  const setTerminal = (
    operation: number,
    event: Exclude<TrainingEvent_Deserialize, { type: 'started' } | { type: 'progress' }>,
  ): void => {
    if (terminalSeen || !acceptsSequence(operation, event)) return;
    terminalSeen = true;
    if (event.type === 'completed') {
      state = {
        ...state,
        isTraining: false,
        isTrainingPipeline: false,
        progress: 100,
        stage: 'idle',
        postureResult: event.result.postureResult,
        presenceResult: event.result.presenceResult,
        error: event.result.errors.length ? event.result.errors.join('; ') : null,
        // Warnings (e.g. "No AWAY frames collected") are non-fatal: a posture-only
        // run still succeeds, so surface them without turning them into an error.
        warnings: event.result.warnings,
      };
    } else if (event.type === 'failed') {
      state = { ...initialState, error: event.error };
    } else if (event.type === 'cancelled') {
      cancelledSeen = true;
      state = { ...initialState };
    }
  };

  const handleEvent = (operation: number, event: TrainingEvent_Deserialize): void => {
    if (!isCurrentOperation(operation)) return;
    if (event.type === 'started') {
      if (activeJobId !== null || event.sequence !== 0) return;
      activeJobId = event.jobId;
      lastSequence = event.sequence;
      state = { ...state, isTraining: true, isTrainingPipeline: true, stage: 'training', progress: 0 };
    } else if (event.type === 'progress') {
      if (!terminalSeen && acceptsSequence(operation, event)) {
        const stage = event.stage as TrainingState['stage'];
        state = { ...state, stage, progress: Math.max(state.progress, Math.min(99, event.progress)) };
      }
    } else {
      setTerminal(operation, event);
    }
  };

  const start = async (
    doCV: boolean,
    onModelDeployed?: () => void,
  ): Promise<boolean> => {
    if (active || state.isTraining || state.isTrainingPipeline) {
      throw new Error('Training is already running.');
    }

    // Reserve synchronously: no second caller can pass this guard while settings flush.
    active = true;
    const operation = ++operationGeneration;
    terminalSeen = false;
    cancelledSeen = false;
    activeJobId = null;
    lastSequence = -1;
    state = { ...initialState, isTraining: true, isTrainingPipeline: true, stage: 'validating' };

    try {
      await trainingConfig.flushToStorage();
      if (!isCurrentOperation(operation)) return false;
      const channel = createTrainingChannel((event) => handleEvent(operation, event));
      const result = await client.trainModels(doCV, channel);
      if (!terminalSeen) {
        throw new Error('Native training completed without a terminal event.');
      }
      if (cancelledSeen) return false;

      // A completed response is a logical result even when only one model was trained.
      // Keep nullable per-role results and warnings/errors in state instead of rejecting it.
      const deployed = result.success
        || result.postureResult?.success === true
        || result.presenceResult?.success === true;
      releaseOperation(operation);
      if (deployed) onModelDeployed?.();
      return true;
    } catch (cause) {
      if (cancelledSeen && cause instanceof NativeCommandError && cause.kind === 'cancelled') return false;
      if (!terminalSeen && isCurrentOperation(operation)) {
        state = { ...initialState, error: message(cause) };
      }
      logger.error('training', 'Native training failed:', cause);
      throw cause;
    } finally {
      releaseOperation(operation);
    }
  };

  const reconcile = async (expectedGeneration?: number): Promise<void> => {
    const status = await client.getTrainingStatus();
    if (expectedGeneration !== undefined && expectedGeneration !== operationGeneration) return;
    if (status.running) {
      active = true;
      state = { ...state, isTraining: true, isTrainingPipeline: true, stage: 'training' };
    } else if (!active) {
      // Only reset per-operation bookkeeping when no start() is in flight; otherwise an
      // in-flight start()/cancel() owns these flags and must reach its own settlement.
      activeJobId = null;
      terminalSeen = false;
      cancelledSeen = false;
      state = { ...state, isTraining: false, isTrainingPipeline: false, stage: 'idle' };
    }
  };

  $effect(() => {
    const expectedGeneration = operationGeneration;
    void reconcile(expectedGeneration).catch((cause: unknown) => {
      logger.error('training', 'Failed to reconcile native training status:', cause);
    });
  });

  return {
    get state() { return state; },
    train: (options) => start(options?.doCV ?? true),
    trainAndDeploy: (options) => start(options?.doCV ?? true, options?.onModelDeployed),
    async cancel() {
      if (!active && !state.isTraining) return;
      if (terminalSeen || state.stage === 'deploying') return;
      await client.cancelTraining();
      await reconcile();
    },
    reconcile,
  };
}
