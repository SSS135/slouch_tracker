import '@testing-library/jest-dom/vitest';
import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

// TrainingTab pulls its data from four context/hook modules and renders registry-driven
// children (ClassifierSelector, FeatureMultiSelector) that also read the training-config
// context. Mocking those modules lets the component render without a native backend or a
// svelte-query provider; the reservoir mock is the only per-test knob for the PCA label.
const mocks = vi.hoisted(() => {
  const reservoir: { data: { count: number; totalSeen: number; maxSamples: number }; error: null } = {
    data: { count: 0, totalSeen: 0, maxSamples: 1000 },
    error: null,
  };
  const query = (data: unknown) => ({ data, isLoading: false, error: null, refetch: vi.fn() });
  const mutation = () => ({ isPending: false, mutate: vi.fn() });

  const trainingConfig = {
    config: {
      classifierConfig: { classifierId: 'mlp', params: {} },
      // Keep the selected method off 'pca' so the reservoir auto-fallback effect stays inert
      // and the PCA slider stays hidden; the PCA radio option renders regardless.
      dimReductionConfig: { method: 'random_projection', components: 32 },
      postureFeatureTypes: ['nlf_depth'],
      presenceFeatureTypes: ['rtmdet_engineered'],
      normalizationMode: 'z_score',
      cvFolds: 5,
    },
    classifiers: [],
    features: [],
    ready: true,
    loading: false,
    error: null,
    updateClassifierConfig: vi.fn(),
    updateDimReductionConfig: vi.fn(),
    updatePostureFeatureTypes: vi.fn(),
    updatePresenceFeatureTypes: vi.fn(),
    updateNormalizationMode: vi.fn(),
    updateCvFolds: vi.fn(),
    flushToStorage: vi.fn(),
    reconcile: vi.fn(),
    reload: vi.fn(),
  };

  const training = {
    state: {
      isTraining: false,
      isTrainingPipeline: false,
      progress: 0,
      stage: 'idle',
      postureResult: null,
      presenceResult: null,
      error: null,
      warnings: [],
      trainingQueued: false,
    },
    train: vi.fn(),
    trainAndDeploy: vi.fn(),
    cancel: vi.fn(),
    reconcile: vi.fn(),
  };

  const datasetOps = {
    stats: query({
      total: 0,
      good: 0,
      bad: 0,
      away: 0,
      unused: 0,
      imbalanceRatio: 0,
      hasMinimumFrames: false,
      hasAwayFrames: false,
    }),
    reservoir,
    frames: query([]),
    needsRetraining: query(false),
    canUndo: query(false),
    exportDataset: mutation(),
    importDataset: mutation(),
    cleanupUnused: mutation(),
    resetDataset: mutation(),
    resetAllData: mutation(),
    deleteFrame: mutation(),
    updateLabel: mutation(),
    undo: mutation(),
    invalidateAll: vi.fn(),
    invalidateStats: vi.fn(),
  };

  return { reservoir, trainingConfig, training, datasetOps };
});

vi.mock('@/lib/native/client', () => ({ nativeClient: {} }));
vi.mock('@/contexts/TrainingConfigContext', () => ({
  useTrainingConfig: () => mocks.trainingConfig,
  coerceParamValue: (_definition: unknown, value: unknown) => value,
  defaultParams: () => ({}),
  DEFAULT_CONFIG: mocks.trainingConfig.config,
}));
vi.mock('@/contexts/TrainingContext', () => ({ useTraining: () => mocks.training }));
vi.mock('@/hooks/useDatasetOperations', () => ({ useDatasetOperations: () => mocks.datasetOps }));
vi.mock('@/hooks/useNotification', () => ({
  useNotification: () => ({
    showSuccess: vi.fn(),
    showError: vi.fn(),
    showWarning: vi.fn(),
    showInfo: vi.fn(),
    showConfirm: vi.fn(),
  }),
  NOTIFICATION_EVENT: 'slouch-tracker:notification',
}));

import TrainingTab from '../TrainingTab.svelte';

const PCA_INPUT_SELECTOR = 'input#dimensionality-reduction-method-pca';

function renderTab(reservoirCount: number, totalSeen = reservoirCount) {
  mocks.reservoir.data = { count: reservoirCount, totalSeen, maxSamples: 1000 };
  return render(TrainingTab, { props: { onTrainingComplete: () => Promise.resolve() } });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('TrainingTab PCA reduction option label', () => {
  it('reads as available (not a progress fraction) when the reservoir has enough samples', () => {
    const { container } = renderTab(1000, 1500);

    expect(screen.getByText('PCA (1000 samples available)')).toBeInTheDocument();
    expect(screen.queryByText(/PCA \(1000\/100 samples\)/)).not.toBeInTheDocument();

    const pcaInput = container.querySelector<HTMLInputElement>(PCA_INPUT_SELECTOR);
    expect(pcaInput).not.toBeNull();
    expect(pcaInput).toBeEnabled();
  });

  it('states the remaining requirement and stays disabled below the threshold', () => {
    const { container } = renderTab(37);

    expect(screen.getByText('PCA (needs 100 samples, have 37)')).toBeInTheDocument();

    const pcaInput = container.querySelector<HTMLInputElement>(PCA_INPUT_SELECTOR);
    expect(pcaInput).not.toBeNull();
    expect(pcaInput).toBeDisabled();
  });

  it('shows the exact-threshold count as available and enables the option', () => {
    const { container } = renderTab(100);

    expect(screen.getByText('PCA (100 samples available)')).toBeInTheDocument();

    const pcaInput = container.querySelector<HTMLInputElement>(PCA_INPUT_SELECTOR);
    expect(pcaInput).toBeEnabled();
  });

  it('keeps the reservoir capacity line as a genuine count/capacity fraction', () => {
    renderTab(1000, 1500);
    expect(
      screen.getByText('Feature reservoir: 1000/1000 samples (1500 observed).'),
    ).toBeInTheDocument();
  });
});
