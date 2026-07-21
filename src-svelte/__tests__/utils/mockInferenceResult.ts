import type { InferenceUiResult } from '@generated/bindings';

export type MockInferenceResult = InferenceUiResult & {
  /** Legacy-only test input retained so parity tests can prove it is ignored. */
  features?: Record<string, Float32Array>;
};

const originalBbox = {
  x1: 0.16,
  y1: 0.31,
  x2: 0.84,
  y2: 0.9,
  score: 0.95,
  width: 0.68,
  height: 0.59,
};

const expandedBbox = {
  x1: 0.12,
  y1: 0.25,
  x2: 0.88,
  y2: 0.96,
  score: 0.95,
  width: 0.76,
  height: 0.71,
};

export function createMockInferenceResult(
  overrides: Partial<MockInferenceResult> = {},
): MockInferenceResult {
  return {
    requestId: 1,
    token: 2,
    personFound: true,
    bbox: {
      original: { ...originalBbox },
      expanded: { ...expandedBbox },
    },
    keypoints: Array.from({ length: 17 }, (_, index) => ({
      x: 0.2 + index * 0.03,
      y: 0.25 + index * 0.035,
      score: 0.9,
    })),
    classification: { presentProbability: 0.95, goodProbability: 0.8 },
    ...overrides,
  };
}

export function createMockInferenceResultWithoutFeatures(): MockInferenceResult {
  return createMockInferenceResult();
}
