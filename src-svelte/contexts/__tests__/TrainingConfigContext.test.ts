import { cleanup, render, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ParameterDefinition_Serialize, ParameterValue } from '@generated/bindings';
import { coerceParamValue, type TrainingConfigContextValue } from '../TrainingConfigContext.svelte';
import TrainingConfigHarness from './TrainingConfigHarness.svelte';

const native = vi.hoisted(() => ({
  getTrainingSettings: vi.fn(),
  getClassifierRegistry: vi.fn(),
  getFeatureRegistry: vi.fn(),
  saveTrainingSettings: vi.fn(),
}));

vi.mock('../../lib/native/client', () => ({ nativeClient: native }));

const feature = {
  id: 'engineered_features',
  name: 'Engineered features',
  description: 'Native posture geometry',
  dimensions: 64,
  storageCost: 0,
  computed: true,
  modelType: 'posture',
  userSelectable: true,
  requiresFitting: false,
} as const;

// A realistic registry in registry order that contains the default posture/presence feature ids, so
// canonicalization at the persistence boundary preserves them (the real registry always ships all 21
// features; a single-feature stub would drop the defaults as unknown ids).
const featureRegistry = [
  { ...feature, id: 'backbone_features_max', name: 'Backbone Max' },
  { ...feature, id: 'gau_features_max', name: 'GAU Max' },
  { ...feature, id: 'rtmdet_engineered', name: 'Detection', modelType: 'presence' },
  feature,
  { ...feature, id: 'keypoint_scores', name: 'Keypoint Scores', modelType: null },
  { ...feature, id: 'nlf_backbone_max', name: 'NLF Backbone Max', modelType: 'posture' },
  { ...feature, id: 'posture_geometry_3d', name: 'Posture Geometry 3D', modelType: 'posture' },
  { ...feature, id: 'torso_invariant_3d', name: 'Torso Invariant 3D', modelType: 'posture' },
];

const mlp = {
  id: 'mlp',
  name: 'Multi-layer Perceptron',
  description: 'Native MLP',
  params: {
    hiddenLayers: { type: 'integer', default: 0 },
    hiddenSize: { type: 'integer', default: 64 },
    weightDecay: { type: 'number', default: 0.03732501577957208 },
    maxIterations: { type: 'integer', default: 350 },
    learningRate: { type: 'number', default: 0.01 },
    useClassWeights: { type: 'boolean', default: false },
    labelSmoothing: { type: 'number', default: 0.05 },
  },
};

beforeEach(() => {
  native.getTrainingSettings.mockResolvedValue(null);
  native.getClassifierRegistry.mockResolvedValue([mlp]);
  native.getFeatureRegistry.mockResolvedValue(featureRegistry);
  native.saveTrainingSettings.mockResolvedValue(null);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('TrainingConfigContext native integration', () => {
  it('initializes Rust-owned defaults and classifier registry parameters', async () => {
    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });

    await waitFor(() => {
      expect(value?.classifiers).toHaveLength(1);
      expect(value?.config.classifierConfig.params).toEqual({
        hiddenLayers: 0,
        hiddenSize: 64,
        weightDecay: 0.03732501577957208,
        maxIterations: 350,
        learningRate: 0.01,
        useClassWeights: false,
        labelSmoothing: 0.05,
      });
    });
    expect(value?.config).toMatchObject({
      classifierConfig: { classifierId: 'mlp' },
      dimReductionConfig: { method: 'pca', components: 32 },
      postureFeatureTypes: ['posture_geometry_3d', 'torso_invariant_3d'],
      presenceFeatureTypes: ['rtmdet_engineered'],
      normalizationMode: 'z_score',
      cvFolds: 5,
    });
    expect(value?.features).toEqual(featureRegistry);
    expect(native.getTrainingSettings).toHaveBeenCalledTimes(1);

    value?.updateCvFolds(7);
    await waitFor(() => expect(native.saveTrainingSettings).toHaveBeenCalledTimes(1));
    const saved = native.saveTrainingSettings.mock.calls[0][0];
    expect(saved).toMatchObject({
      classifierConfig: {
        classifierId: 'mlp',
        params: {
          hiddenLayers: 0,
          hiddenSize: 64,
          weightDecay: 0.03732501577957208,
          maxIterations: 350,
          learningRate: 0.01,
          useClassWeights: false,
          labelSmoothing: 0.05,
        },
      },
      dimReductionConfig: { method: 'pca', components: 32 },
      postureFeatureTypes: ['posture_geometry_3d', 'torso_invariant_3d'],
      presenceFeatureTypes: ['rtmdet_engineered'],
      normalizationMode: 'z_score',
      cvFolds: 7,
      featureTypes: null,
    });
    expect(saved.lastUpdated).toEqual(expect.any(Number));
    expect(Number.isFinite(saved.lastUpdated)).toBe(true);
  });

  it('canonicalizes out-of-order and duplicate feature selections before persisting', async () => {
    // A multi-feature registry in registry order gives the canonicalizer a real ranking to sort by.
    const registry = [
      { ...feature, id: 'backbone_features_max', name: 'Backbone Max' },
      { ...feature, id: 'gau_features_max', name: 'GAU Max' },
      { ...feature, id: 'torso_invariant', name: 'Torso Invariant' },
    ];
    native.getFeatureRegistry.mockResolvedValue(registry);

    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });
    await waitFor(() => expect(value?.ready).toBe(true));

    // Click order that previously broke: the highest-index feature selected before lower ones.
    value?.updatePostureFeatureTypes(['torso_invariant', 'backbone_features_max', 'gau_features_max']);
    // Reverse order plus a duplicate for presence.
    value?.updatePresenceFeatureTypes(['gau_features_max', 'backbone_features_max', 'gau_features_max']);

    // The live config is normalized immediately, not only at the persistence boundary.
    expect(value?.config.postureFeatureTypes).toEqual(['backbone_features_max', 'gau_features_max', 'torso_invariant']);
    expect(value?.config.presenceFeatureTypes).toEqual(['backbone_features_max', 'gau_features_max']);

    await waitFor(() => expect(native.saveTrainingSettings).toHaveBeenCalled());
    const saved = native.saveTrainingSettings.mock.calls.at(-1)?.[0];
    expect(saved.postureFeatureTypes).toEqual(['backbone_features_max', 'gau_features_max', 'torso_invariant']);
    expect(saved.presenceFeatureTypes).toEqual(['backbone_features_max', 'gau_features_max']);
  });

  it('canonicalizes non-ascending feature lists loaded from stored settings', async () => {
    const registry = [
      { ...feature, id: 'backbone_features_max', name: 'Backbone Max' },
      { ...feature, id: 'gau_features_max', name: 'GAU Max' },
      { ...feature, id: 'torso_invariant', name: 'Torso Invariant' },
    ];
    native.getFeatureRegistry.mockResolvedValue(registry);
    // Simulate legacy/corrupted stored settings whose lists are out of registry order.
    native.getTrainingSettings.mockResolvedValue({
      classifierConfig: { classifierId: 'mlp', params: {} },
      dimReductionConfig: { method: 'pca', components: 32 },
      postureFeatureTypes: ['torso_invariant', 'backbone_features_max'],
      presenceFeatureTypes: ['gau_features_max', 'backbone_features_max'],
      normalizationMode: 'z_score',
      cvFolds: 5,
      featureTypes: null,
      lastUpdated: 1,
    });

    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });
    await waitFor(() => expect(value?.ready).toBe(true));

    expect(value?.config.postureFeatureTypes).toEqual(['backbone_features_max', 'torso_invariant']);
    expect(value?.config.presenceFeatureTypes).toEqual(['backbone_features_max', 'gau_features_max']);
  });

  it('flushes a pending debounced edit synchronously on window unload', async () => {
    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });

    await waitFor(() => expect(value?.ready).toBe(true));

    value?.updateCvFolds(11);
    // Let the debounce $effect schedule its timer, but stay inside the 300ms window.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(native.saveTrainingSettings).not.toHaveBeenCalled();
    window.dispatchEvent(new Event('beforeunload'));

    expect(native.saveTrainingSettings).toHaveBeenCalledTimes(1);
    expect(native.saveTrainingSettings.mock.calls[0][0]).toMatchObject({ cvFolds: 11 });

    // The debounce must have been cancelled: no second save fires after the window elapses.
    await new Promise((resolve) => setTimeout(resolve, 350));
    expect(native.saveTrainingSettings).toHaveBeenCalledTimes(1);
  });

  it('does not flush on unload when there are no pending edits', async () => {
    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });

    await waitFor(() => expect(value?.ready).toBe(true));
    window.dispatchEvent(new Event('beforeunload'));
    expect(native.saveTrainingSettings).not.toHaveBeenCalled();
  });

  it.each([
    ['classifier', native.getClassifierRegistry, 'classifier registry unavailable'],
    ['feature', native.getFeatureRegistry, 'feature registry unavailable'],
  ])('keeps configuration blocked when the %s registry fails before reconciliation', async (_name, registryRead, message) => {
    registryRead.mockRejectedValueOnce(new Error(message));
    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });

    await waitFor(() => expect(value?.error).toBe(message));
    value?.reconcile(null);
    value?.updateCvFolds(9);
    await new Promise((resolve) => setTimeout(resolve, 350));

    expect(value?.ready).toBe(false);
    expect(value?.error).toBe(message);
    expect(value?.config.cvFolds).toBe(5);
    expect(native.saveTrainingSettings).not.toHaveBeenCalled();
    await expect(value?.flushToStorage()).rejects.toThrow(`Training settings are unavailable: ${message}`);

    await value?.reload();
    expect(value?.ready).toBe(true);
    expect(value?.error).toBeNull();
    await new Promise((resolve) => setTimeout(resolve, 350));
    expect(native.saveTrainingSettings).not.toHaveBeenCalled();

    value?.updateCvFolds(7);
    await waitFor(() => expect(native.saveTrainingSettings).toHaveBeenCalledTimes(1));
  });

  it('does not overwrite native settings when the settings read fails', async () => {
    native.getTrainingSettings.mockRejectedValueOnce(new Error('settings unavailable'));
    let value: TrainingConfigContextValue | undefined;
    render(TrainingConfigHarness, { props: { onReady: (next) => { value = next; } } });

    await waitFor(() => expect(value?.error).toBe('settings unavailable'));
    await new Promise((resolve) => setTimeout(resolve, 350));
    expect(value?.ready).toBe(false);
    expect(native.saveTrainingSettings).not.toHaveBeenCalled();

    await value?.reload();
    expect(value?.ready).toBe(true);
  });

  it('reports the provider error from a component-valid consumer', () => {
    let error: Error | undefined;
    render(TrainingConfigHarness, {
      props: {
        provide: false,
        onError: (cause) => { error = cause; },
      },
    });
    expect(error?.message).toBe('useTrainingConfig must be used within TrainingConfigProvider');
  });
});

describe('integer parameter coercion', () => {
  const def = (partial: Record<string, unknown>): ParameterDefinition_Serialize =>
    partial as unknown as ParameterDefinition_Serialize;

  it('rounds integer-typed params but leaves float params untouched', () => {
    // Explicit integer type + integer-stepped ranges (maxIterations step 10, k/nClusters step 1).
    expect(coerceParamValue(def({ type: 'integer', default: 0 }), 2.0000001)).toBe(2);
    expect(coerceParamValue(def({ type: 'range', step: 10, default: 100 }), 99.6)).toBe(100);
    expect(coerceParamValue(def({ type: 'range', step: 1, default: 3 }), 3.999)).toBe(4);
    // Fractional-step / stepless numeric params are genuine floats and must be preserved.
    expect(coerceParamValue(def({ type: 'number', default: 0.01 }), 0.0123)).toBe(0.0123);
    expect(coerceParamValue(def({ type: 'range', step: 0.01, default: 0.05 }), 0.051)).toBe(0.051);
    expect(coerceParamValue(def({ type: 'range', default: 1 }), 1.5)).toBe(1.5);
    // Non-numeric values pass through regardless of declared type.
    expect(coerceParamValue(def({ type: 'select', default: 'cosine' }), 'rbf' as ParameterValue)).toBe('rbf');
    expect(coerceParamValue(def({ type: 'boolean', default: false }), true as ParameterValue)).toBe(true);
    expect(coerceParamValue(undefined, 42)).toBe(42);
  });

  it('serializes coerced integer params without a trailing decimal', () => {
    const value = coerceParamValue(def({ type: 'range', step: 10, default: 100 }), 100);
    expect(Number.isInteger(value)).toBe(true);
    expect(JSON.stringify({ maxIterations: value })).toBe('{"maxIterations":100}');
  });
});
