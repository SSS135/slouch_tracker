import type { ActiveModelMetadata, InferenceUiResult, ModelMetadata } from '@generated/bindings';
import { nativeClient } from '../lib/native/client';

type NativeClassificationResult = NonNullable<InferenceUiResult['classification']>;
export type ClassificationResult = Omit<NativeClassificationResult, 'presentProbability'> & {
  presentProbability: number;
};

export interface UsePostureClassifierReturn {
  readonly postureModel: ModelMetadata | null;
  readonly presenceModel: ModelMetadata | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  clearModel(): Promise<void>;
  reloadModel(): Promise<void>;
}

/** Read-only model metadata facade; model payloads remain owned by Rust. */
export function usePostureClassifier(): UsePostureClassifierReturn {
  let metadata = $state<ActiveModelMetadata>({ posture: null, presence: null });
  let isLoading = $state(false);
  let error = $state<string | null>(null);
  let requestGeneration = 0;
  const reloadModel = async (): Promise<void> => {
    const generation = ++requestGeneration;
    isLoading = true;
    error = null;
    try {
      const nextMetadata = await nativeClient.getActiveModelMetadata();
      if (generation === requestGeneration) metadata = nextMetadata;
    } catch (cause) {
      if (generation === requestGeneration) {
        error = cause instanceof Error ? cause.message : 'Unknown error';
      }
    } finally {
      if (generation === requestGeneration) isLoading = false;
    }
  };
  $effect(() => {
    void reloadModel();
    return () => { requestGeneration += 1; };
  });
  return {
    get postureModel() { return metadata.posture; },
    get presenceModel() { return metadata.presence; },
    get isLoading() { return isLoading; },
    get error() { return error; },
    async clearModel() {
      requestGeneration += 1;
      metadata = { posture: null, presence: null };
      error = null;
      isLoading = false;
    },
    reloadModel,
  };
}
