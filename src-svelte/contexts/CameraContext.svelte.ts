/**
 * CameraContext
 *
 * Svelte context for camera and inference data shared across page tabs.
 * State management remains owned by the page-level camera lifecycle; this
 * module only provides and retrieves the shared value.
 */

import { getContext, setContext } from 'svelte';
import type { InferenceResult } from '../services/dataset/types';

/** Camera context value shared across tabs. */
export interface CameraContextValue {
  /** Latest inference result pushed from the native camera. */
  inferenceResult: InferenceResult | null;
  /** Current detection FPS (frames per second). */
  fps: number;
}

const CAMERA_CONTEXT = Symbol('CameraContext');

/**
 * Provides camera state to descendants of the current Svelte component.
 *
 * Call this once from the page-level component before rendering consumers.
 * The value may be a Svelte reactive object when its fields need to update.
 */
export function CameraProvider(value: CameraContextValue): void {
  setContext(CAMERA_CONTEXT, value);
}

/**
 * Access camera state from context.
 *
 * Must be called during component initialization beneath CameraProvider.
 */
export function useCameraContext(): CameraContextValue {
  const context = getContext<CameraContextValue | null>(CAMERA_CONTEXT);
  if (!context) {
    throw new Error('useCameraContext must be used within CameraProvider');
  }
  return context;
}
