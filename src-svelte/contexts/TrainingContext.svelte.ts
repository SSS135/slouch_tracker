import { getContext, setContext } from 'svelte';
import { useModelTraining } from '../hooks/useModelTraining';
import type { TrainingResult } from '../services/dataset/types';

/**
 * Training state exposed by the model-training hook.
 */
export interface TrainingContextState {
  isTraining: boolean;
  isTrainingPipeline: boolean;
  progress: number;
  stage: 'idle' | 'validating' | 'processing' | 'training' | 'evaluating' | 'deploying';
  postureResult: TrainingResult | null;
  presenceResult: TrainingResult | null;
  error: string | null;
  warnings: string[];
  trainingQueued: boolean;
}

/**
 * Training context value shared by the application.
 */
export interface TrainingContextValue {
  readonly state: TrainingContextState;
  train: (options?: { doCV?: boolean }) => Promise<boolean>;
  trainAndDeploy: (options?: { doCV?: boolean; onModelDeployed?: () => void }) => Promise<boolean>;
  cancel: () => Promise<void>;
  reconcile: () => Promise<void>;
}

const TRAINING_CONTEXT = Symbol('TrainingContext');

/**
 * Initializes the training context for the current Svelte component tree.
 *
 * Call this from the provider component's instance script. Svelte context is
 * used instead of a wrapper component because child markup is owned by the
 * consuming component.
 */
export function TrainingProvider(): TrainingContextValue {
  const training = useModelTraining();
  const value: TrainingContextValue = {
    get state() {
      return training.state;
    },
    train: training.train,
    trainAndDeploy: training.trainAndDeploy,
    cancel: training.cancel,
    reconcile: training.reconcile,
  };

  setContext(TRAINING_CONTEXT, value);
  return value;
}

/**
 * Returns the training context for the current component tree.
 *
 * @throws Error when called outside a component initialized with
 * TrainingProvider.
 */
export function useTraining(): TrainingContextValue {
  try {
    const context = getContext<TrainingContextValue | undefined>(TRAINING_CONTEXT);
    if (context !== undefined) {
      return context;
    }
  } catch {
    // getContext throws when called outside a Svelte component.
  }

  throw new Error('useTraining must be used within TrainingProvider');
}
