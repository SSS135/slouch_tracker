import type { InferenceUiResult } from '@generated/bindings';
import type { MultiTaskDetectionResult } from '../services/posture/types';

type ReactiveValue<T> = T | (() => T);
const read = <T>(value: ReactiveValue<T>): T => typeof value === 'function' ? (value as () => T)() : value;
export interface MultiTaskDetectionState { readonly detection: MultiTaskDetectionResult | null; }

/** Maps Rust-owned classification output to the legacy UI indicator shape. */
export function useMultiTaskDetection(
  inferenceResult: ReactiveValue<InferenceUiResult | null>,
): MultiTaskDetectionState {
  let detection = $state<MultiTaskDetectionResult | null>(null);
  $effect(() => {
    const result = read(inferenceResult);
    detection = {
      person_found: Boolean(result?.personFound),
      // These legacy cue flags are not aliases for classifier probability.
      // The oracle leaves them false until dedicated detectors populate them.
      slouching: false,
      forward_neck_tilt: false,
      hand_near_face: false,
      mouth_open: false,
    };
  });
  return { get detection() { return detection; } };
}
