import type { BoundingBox, InferenceUiResult, Keypoint } from '@generated/bindings';

export interface MockNativeInferenceOptions {
  requestId?: number;
  token?: number;
  personFound?: boolean;
  bbox?: InferenceUiResult['bbox'];
  keypoints?: Keypoint[] | null;
  classification?: InferenceUiResult['classification'];
}

export function createMockNativeBoundingBox(overrides: Partial<BoundingBox> = {}): BoundingBox {
  return {
    x1: 0.1,
    y1: 0.2,
    x2: 0.8,
    y2: 0.9,
    score: 0.95,
    width: 0.7,
    height: 0.7,
    ...overrides,
  };
}

export function createMockNativeKeypoints(score = 0.9): Keypoint[] {
  return Array.from({ length: 17 }, (_, index) => ({
    x: 0.2 + index * 0.01,
    y: 0.3 + index * 0.01,
    score,
  }));
}

/** Creates the UI-safe result returned by native inference; no feature payloads cross IPC. */
export function createMockNativeInferenceResult(
  options: MockNativeInferenceOptions = {},
): InferenceUiResult {
  const original = createMockNativeBoundingBox();
  const expanded = createMockNativeBoundingBox({
    x1: 0.05,
    y1: 0.1,
    x2: 0.9,
    y2: 0.95,
    width: 0.85,
    height: 0.85,
  });
  return {
    requestId: options.requestId ?? 7,
    token: options.token ?? 70,
    personFound: options.personFound ?? true,
    bbox: options.bbox === undefined ? { original, expanded } : options.bbox,
    keypoints: options.keypoints === undefined ? createMockNativeKeypoints() : options.keypoints,
    classification: options.classification === undefined
      ? { presentProbability: 0.9, goodProbability: 0.8 }
      : options.classification,
  };
}

export function createIncompleteNativeInferenceResult(
  missing: 'person' | 'token' | 'bbox',
): InferenceUiResult {
  if (missing === 'person') {
    return createMockNativeInferenceResult({
      personFound: false,
      bbox: null,
      keypoints: null,
      classification: null,
    });
  }
  if (missing === 'token') return createMockNativeInferenceResult({ token: 0 });
  return createMockNativeInferenceResult({ bbox: null });
}
