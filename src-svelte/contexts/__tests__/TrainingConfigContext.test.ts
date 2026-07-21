import { cleanup, render, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { TrainingConfigContextValue } from '../TrainingConfigContext.svelte';
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

const mlp = {
  id: 'mlp',
  name: 'Multi-layer Perceptron',
  description: 'Native MLP',
  params: {
    hiddenLayers: { type: 'integer', default: 0 },
    hiddenSize: { type: 'integer', default: 64 },
    weightDecay: { type: 'number', default: 1.0 },
    maxIterations: { type: 'integer', default: 100 },
    learningRate: { type: 'number', default: 0.01 },
    useClassWeights: { type: 'boolean', default: false },
    labelSmoothing: { type: 'number', default: 0.05 },
  },
};

beforeEach(() => {
  native.getTrainingSettings.mockResolvedValue(null);
  native.getClassifierRegistry.mockResolvedValue([mlp]);
  native.getFeatureRegistry.mockResolvedValue([feature]);
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
        weightDecay: 1.0,
        maxIterations: 100,
        learningRate: 0.01,
        useClassWeights: false,
        labelSmoothing: 0.05,
      });
    });
    expect(value?.config).toMatchObject({
      classifierConfig: { classifierId: 'mlp' },
      dimReductionConfig: { method: 'none', components: 64 },
      postureFeatureTypes: ['engineered_features'],
      presenceFeatureTypes: ['rtmdet_engineered', 'keypoint_scores'],
      normalizationMode: 'z_score',
      cvFolds: 5,
    });
    expect(value?.features).toEqual([feature]);
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
          weightDecay: 1.0,
          maxIterations: 100,
          learningRate: 0.01,
          useClassWeights: false,
          labelSmoothing: 0.05,
        },
      },
      dimReductionConfig: { method: 'none', components: 64 },
      postureFeatureTypes: ['engineered_features'],
      presenceFeatureTypes: ['rtmdet_engineered', 'keypoint_scores'],
      normalizationMode: 'z_score',
      cvFolds: 7,
      featureTypes: null,
    });
    expect(saved.lastUpdated).toEqual(expect.any(Number));
    expect(Number.isFinite(saved.lastUpdated)).toBe(true);
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
