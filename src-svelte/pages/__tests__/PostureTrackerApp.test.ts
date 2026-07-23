import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FrameLabel } from '@/services/dataset/types';
import { NativeCommandError } from '@/lib/native/client';
import { reactiveBox } from '@/__tests__/utils/reactiveBox.svelte';

const mocks = vi.hoisted(() => {
  const capturedFrame = {
    id: 'capture-1',
    requestId: 1,
    token: 101,
    timestamp: 100,
    label: 'good',
    thumbnail: new Blob(['thumbnail'], { type: 'image/webp' }),
    thumbnailMimeType: 'image/webp',
    keypoints: Array.from({ length: 17 }, (_, index) => ({
      x: 0.2 + index * 0.01,
      y: 0.3 + index * 0.01,
      score: 0.9,
    })),
    bbox: {
      original: { x1: 0.1, y1: 0.1, x2: 0.9, y2: 0.9, score: 0.95 },
      expanded: { x1: 0.05, y1: 0.05, x2: 0.95, y2: 0.95, score: 0.95 },
    },
  };
  return {
    capturedFrame,
    frameSampler: {
      recentFrames: [] as unknown[],
      isCapturing: false,
      canCapture: true,
      isLive: true,
      captureFrame: vi.fn(),
      requestCapture: vi.fn(),
      saveFrame: vi.fn(),
      updateFrameLabel: vi.fn(),
      clearFrames: vi.fn(),
      removeFrame: vi.fn(),
    },
    settings: {
      settings: {
        autoCaptureEnabled: false,
        autoCaptureIntervalSeconds: 5,
        captureIntervalSeconds: 1,
        alertVolume: 0.5,
        alertDelaySeconds: 5,
        cameraWidth: 800,
        cameraHeight: 600,
        privacyMode: false,
        claheStrength: 3.5,
        smoothingFrames: 1,
        tileMotionThreshold: 3,
        claheTemporalAlpha: 0.15,
        preprocessingDebugView: false,
        showDetectionOverlay: false,
      },
      ready: true,
      error: null,
      updateSettings: vi.fn(),
      resetSettings: vi.fn(),
      reload: vi.fn(),
      reconcile: vi.fn(),
      flush: vi.fn().mockResolvedValue(undefined),
    },
    autoCaptureHook: vi.fn(),
    postureChangeHook: vi.fn(),
    notification: {
      showSuccess: vi.fn(),
      showInfo: vi.fn(),
      showError: vi.fn(),
      showConfirm: vi.fn(),
    },
    dataset: {
      stats: { data: null as null | { hasMinimumFrames: boolean }, refetch: vi.fn() },
      frames: { data: [] },
      needsRetraining: { data: false as boolean, refetch: vi.fn() } as { data: boolean; refetch: ReturnType<typeof vi.fn> },
      canUndo: { data: { available: false, depth: 0, nextAction: null, revision: 0 } },
      invalidateAll: vi.fn(),
      invalidateStats: vi.fn(),
      undo: { mutateAsync: vi.fn() },
      resetDataset: { mutateAsync: vi.fn() },
      resetAllData: { mutateAsync: vi.fn() },
    },
    trainingConfig: { ready: true, persistedRevision: 0, reload: vi.fn().mockResolvedValue(undefined), flushToStorage: vi.fn().mockResolvedValue(undefined), reconcile: vi.fn() } as { ready: boolean; persistedRevision: number; reload: () => Promise<void>; flushToStorage: () => Promise<void>; reconcile: () => void },
    training: {
      train: vi.fn(),
      trainAndDeploy: vi.fn(),
      cancel: vi.fn(),
      reconcile: vi.fn(),
      state: {
        isTraining: false,
        isTrainingPipeline: false,
        progress: 0,
        stage: 'idle',
        postureResult: null,
        presenceResult: null,
        error: null,
        trainingQueued: false,
      },
    },
    background: { isVisible: true, flashTitle: vi.fn() },
    queryClient: { clear: vi.fn(), setQueryData: vi.fn() },
    native: {
      getActiveModelMetadata: vi.fn(),
      resetTrainingSettings: vi.fn(),
      onNativeStateChanged: vi.fn().mockResolvedValue(vi.fn()),
      onTrackingStateChanged: vi.fn().mockResolvedValue(vi.fn()),
      getAutostartEnabled: vi.fn().mockResolvedValue(false),
      setAutostartEnabled: vi.fn().mockResolvedValue(undefined),
    },
    nativeApp: {
      status: { inferenceReady: true } as { inferenceReady: boolean } | null,
      error: null as Error | null,
      initialize: vi.fn().mockResolvedValue(undefined),
      reconcile: vi.fn(),
    },
  };
});

vi.mock('@/components/PostureCamera.svelte', async () => ({
  default: (await import('./MockPostureCamera.svelte')).default,
}));
vi.mock('@tanstack/svelte-query', async () => {
  const actual = await vi.importActual<typeof import('@tanstack/svelte-query')>('@tanstack/svelte-query');
  return { ...actual, useQueryClient: () => mocks.queryClient };
});
vi.mock('@/contexts/TrainingConfigContext', () => ({
  useTrainingConfig: () => mocks.trainingConfig,
}));
vi.mock('@/contexts/TrainingContext', () => ({ useTraining: () => mocks.training }));
vi.mock('@/hooks/useCameraSettings', () => ({ useCameraSettings: () => mocks.settings }));
vi.mock('@/hooks/useNotification', () => ({ useNotification: () => mocks.notification }));
vi.mock('@/hooks/useFrameSampler', () => ({ useFrameSampler: () => mocks.frameSampler }));
vi.mock('@/hooks/useDatasetOperations', () => ({ useDatasetOperations: () => mocks.dataset }));
vi.mock('@/hooks/useAutoCapture', () => ({ useAutoCapture: mocks.autoCaptureHook }));
vi.mock('@/hooks/usePostureChangeDetector', () => ({ usePostureChangeDetector: mocks.postureChangeHook }));
vi.mock('@/hooks/usePostureSound', () => ({ usePostureSound: vi.fn() }));
vi.mock('@/hooks/useBackgroundProcessing', () => ({ useBackgroundProcessing: () => mocks.background }));
vi.mock('@/hooks/useGlobalShortcuts', () => ({ useGlobalShortcuts: vi.fn() }));
vi.mock('@/lib/state/nativeApp.svelte', () => ({
  useNativeAppState: () => mocks.nativeApp,
}));
vi.mock('@/lib/native/client', async () => {
  const actual = await vi.importActual<typeof import('@/lib/native/client')>('@/lib/native/client');
  return { nativeClient: mocks.native, NativeCommandError: actual.NativeCommandError };
});

import PostureTrackerApp from '../PostureTrackerApp.svelte';

async function renderReady(): Promise<ReturnType<typeof render>> {
  const view = render(PostureTrackerApp);
  await waitFor(() => expect(screen.getByTestId('mock-posture-camera')).toBeInTheDocument());
  return view;
}

async function loadInference(name = 'Load inference A'): Promise<void> {
  await fireEvent.click(screen.getByRole('button', { name }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Good' })).toBeEnabled());
}

beforeEach(() => {
  mocks.frameSampler.recentFrames = [];
  mocks.frameSampler.canCapture = true;
  mocks.frameSampler.isLive = true;
  mocks.frameSampler.captureFrame.mockResolvedValue(mocks.capturedFrame);
  mocks.frameSampler.requestCapture.mockResolvedValue({ status: 'captured', frame: mocks.capturedFrame });
  mocks.frameSampler.saveFrame.mockResolvedValue(undefined);
  mocks.settings.ready = true;
  mocks.settings.error = null;
  mocks.settings.settings.autoCaptureEnabled = false;
  mocks.settings.settings.showDetectionOverlay = false;
  mocks.nativeApp.status = { inferenceReady: true };
  mocks.nativeApp.error = null;
  mocks.settings.resetSettings.mockResolvedValue(undefined);
  mocks.dataset.stats.data = null;
  mocks.dataset.stats.refetch.mockResolvedValue({ data: null });
  mocks.dataset.needsRetraining = { data: false, refetch: vi.fn().mockResolvedValue({ data: false }) };
  mocks.trainingConfig = { ready: true, persistedRevision: 0, reload: vi.fn().mockResolvedValue(undefined), flushToStorage: vi.fn().mockResolvedValue(undefined), reconcile: vi.fn() };
  mocks.training.state.isTraining = false;
  mocks.dataset.invalidateAll.mockResolvedValue(undefined);
  mocks.dataset.undo.mutateAsync.mockResolvedValue(true);
  mocks.training.trainAndDeploy.mockResolvedValue(true);
  mocks.training.reconcile.mockResolvedValue(undefined);
  mocks.background.isVisible = true;
  mocks.dataset.resetDataset.mutateAsync.mockResolvedValue(undefined);
  mocks.dataset.resetAllData.mutateAsync.mockResolvedValue({
    app: { ready: true, inferenceReady: true, datasetVersion: 2, storage: { used: 0, available: 1, quota: 1 } },
    cameraSettings: { cameraWidth: 800, cameraHeight: 600, captureIntervalSeconds: 1, autoCaptureEnabled: false, autoCaptureIntervalSeconds: 5, privacyMode: false, claheStrength: 0, smoothingFrames: 1, tileMotionThreshold: 3, claheTemporalAlpha: 0.15, preprocessingDebugView: false, showDetectionOverlay: false },
    uiSettings: { alertVolume: 0.3, alertDelaySeconds: 5 },
    trainingSettings: null,
    activeModels: { posture: null, presence: null },
    undo: { available: false, depth: 0, nextAction: null, revision: 1 },
  });
  mocks.dataset.canUndo.data = { available: false, depth: 0, nextAction: null, revision: 0 };
  mocks.native.getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
  mocks.native.resetTrainingSettings.mockResolvedValue(null);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('PostureTrackerApp native view integration', () => {
  it('keeps the collapsed panel inert and exposes tab semantics after opening', async () => {
    const { container } = await renderReady();
    expect(container.querySelector('.viewport')).toBeInTheDocument();
    expect(screen.queryByRole('tablist', { name: 'Control panel tabs' })).not.toBeInTheDocument();
    await fireEvent.click(screen.getByRole('button', { name: 'Open control panel' }));
    expect(screen.getByRole('tablist', { name: 'Control panel tabs' })).toBeInTheDocument();
  });

  it('keeps capture controls disabled until native inference provides an opaque token', async () => {
    await renderReady();
    expect(screen.getByRole('button', { name: 'Good' })).toBeDisabled();
    await loadInference();
  });

  it('enables all capture labels for a complete native inference result', async () => {
    await renderReady();
    await loadInference();
    expect(screen.getByRole('button', { name: 'Bad' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Away' })).toBeEnabled();
  });

  it('keeps capture buttons enabled while the pipeline is live even when the current token is consumed', async () => {
    // Regression: the buttons used to gate on frameSampler.canCapture, so an
    // auto-capture consuming the current identity blinked every button disabled
    // until the next result. They now gate on isLive (pipeline liveness), so a
    // consumed token no longer disables them.
    mocks.frameSampler.canCapture = false;
    mocks.frameSampler.isLive = true;
    await renderReady();
    await loadInference();
    expect(screen.getByRole('button', { name: 'Good' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Bad' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Away' })).toBeEnabled();
  });

  it('enables capture while inference streams even when the inferenceReady snapshot is stale and overlay mode is on', async () => {
    // Regression: `nativeApp.status.inferenceReady` is a one-shot snapshot read
    // once during startup init and never refreshed, so a slow model load latched
    // it false while native inference was fully up and streaming results — which
    // permanently disabled every capture button. Capture readiness must follow the
    // live pipeline (frameSampler.isLive, itself proof inference is running), not
    // the stale flag, and must hold regardless of the detection-overlay preview mode.
    mocks.nativeApp.status = { inferenceReady: false };
    mocks.settings.settings.showDetectionOverlay = true;
    mocks.frameSampler.isLive = true;
    await renderReady();
    await loadInference();
    expect(screen.getByRole('button', { name: 'Good' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Bad' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Away' })).toBeEnabled();
  });

  it('disables capture buttons when the inference pipeline goes stale', async () => {
    mocks.frameSampler.isLive = false;
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference A' }));
    // A valid current result is present, but a stalled pipeline disables capture.
    expect(screen.getByRole('button', { name: 'Good' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Bad' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Away' })).toBeDisabled();
  });

  it('captures and saves a good frame through the token-backed sampler', async () => {
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.frameSampler.requestCapture).toHaveBeenCalledWith(FrameLabel.GOOD));
    expect(mocks.frameSampler.saveFrame).toHaveBeenCalledWith('capture-1', FrameLabel.GOOD);
  });

  it('routes bad captures without feature or IndexedDB payloads', async () => {
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Bad' }));
    await waitFor(() => expect(mocks.frameSampler.saveFrame).toHaveBeenCalledWith('capture-1', FrameLabel.BAD));
    expect(mocks.frameSampler.saveFrame.mock.calls[0]).toHaveLength(2);
  });

  it('routes away captures through the same native token path', async () => {
    await renderReady();
    await loadInference('Load inference B');
    await fireEvent.click(screen.getByRole('button', { name: 'Away' }));
    await waitFor(() => expect(mocks.frameSampler.saveFrame).toHaveBeenCalledWith('capture-1', FrameLabel.AWAY));
  });

  it('reports when the sampler rejects a capture without current detection data', async () => {
    mocks.frameSampler.requestCapture.mockResolvedValue({ status: 'unavailable' });
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.notification.showError).toHaveBeenCalledWith('No current person detection is available to capture.'));
    expect(mocks.frameSampler.saveFrame).not.toHaveBeenCalled();
  });

  it('stays silent when a labelled capture is superseded by a newer click', async () => {
    mocks.frameSampler.requestCapture.mockResolvedValue({ status: 'superseded' });
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.frameSampler.requestCapture).toHaveBeenCalledWith(FrameLabel.GOOD));
    expect(mocks.notification.showError).not.toHaveBeenCalled();
    expect(mocks.frameSampler.saveFrame).not.toHaveBeenCalled();
  });

  it('reports native save failures and keeps the page mounted', async () => {
    mocks.frameSampler.saveFrame.mockRejectedValue(new Error('token expired'));
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.notification.showError).toHaveBeenCalledWith('Failed to save frame: token expired'));
    expect(screen.getByTestId('mock-posture-camera')).toBeInTheDocument();
  });

  it('invalidates native dataset queries after a successful save', async () => {
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.dataset.invalidateAll).toHaveBeenCalledTimes(1));
  });

  it('starts no-CV retraining after a qualifying manual save', async () => {
    mocks.dataset.stats.refetch.mockResolvedValue({ data: { hasMinimumFrames: true } });
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.training.trainAndDeploy).toHaveBeenCalledWith(expect.objectContaining({ doCV: false })));
  });

  it('reloads active model metadata once a background auto-retrain deploys a model', async () => {
    // Regression: the silent auto-retrain (frame save -> trainAndDeploy) deployed
    // a posture model in the runtime but never reloaded metadata, so hasModel and
    // the status badge stayed at "No Model Trained" while the app already
    // classified. The deploy must refresh metadata via onModelDeployed.
    mocks.dataset.stats.refetch.mockResolvedValue({ data: { hasMinimumFrames: true } });
    mocks.native.getActiveModelMetadata
      .mockResolvedValueOnce({ posture: null, presence: null })
      .mockResolvedValue({
        posture: { classifierId: 'mlp', featureTypes: ['engineered_features'], trainedAt: 100 },
        presence: null,
      });
    mocks.training.trainAndDeploy.mockImplementation((options?: { onModelDeployed?: () => void }) => {
      options?.onModelDeployed?.();
      return Promise.resolve(true);
    });

    await renderReady();
    // Pre-deploy there is no posture model, so the runtime emits no goodProbability.
    // A presence-only result keeps the badge untrained via the snapshot and lets this
    // test prove the metadata-reload path rather than the live-signal self-heal.
    await fireEvent.click(screen.getByRole('button', { name: 'Load presence-only inference' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Good' })).toBeEnabled());
    expect(screen.getByText('No Model Trained')).toBeInTheDocument();
    const callsBefore = mocks.native.getActiveModelMetadata.mock.calls.length;

    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));

    await waitFor(() => expect(mocks.training.trainAndDeploy).toHaveBeenCalled());
    await waitFor(() =>
      expect(mocks.native.getActiveModelMetadata.mock.calls.length).toBeGreaterThan(callsBefore),
    );
    // Posture-only pair (presence null) must present as scoring, not "No Model Trained".
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
  });

  it('does not block capture handling on deferred native persistence', async () => {
    let resolveSave!: () => void;
    mocks.frameSampler.saveFrame.mockReturnValue(new Promise<void>((resolve) => { resolveSave = resolve; }));
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.frameSampler.saveFrame).toHaveBeenCalled());
    expect(screen.getByRole('button', { name: 'Bad' })).toBeEnabled();
    expect(mocks.dataset.invalidateAll).not.toHaveBeenCalled();
    resolveSave();
    await waitFor(() => expect(mocks.dataset.invalidateAll).toHaveBeenCalled());
  });

  it('does not maintain a parallel browser undo history after save', async () => {
    await renderReady();
    await loadInference();
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await waitFor(() => expect(mocks.dataset.invalidateAll).toHaveBeenCalled());
    expect(mocks.dataset.undo.mutateAsync).not.toHaveBeenCalled();
  });

  it('uses the G keyboard shortcut outside editable controls', async () => {
    await renderReady();
    await loadInference();
    await fireEvent.keyDown(window, { key: 'g' });
    await waitFor(() => expect(mocks.frameSampler.requestCapture).toHaveBeenCalledWith(FrameLabel.GOOD));
  });

  it('does not capture from keyboard shortcuts while editing settings', async () => {
    await renderReady();
    await loadInference();
    mocks.frameSampler.requestCapture.mockClear();
    const input = document.createElement('input');
    document.body.append(input);
    await fireEvent.keyDown(input, { key: 'g' });
    expect(mocks.frameSampler.requestCapture).not.toHaveBeenCalled();
    input.remove();
  });

  it('clears the transient capture buffer with the C shortcut', async () => {
    await renderReady();
    await fireEvent.keyDown(window, { key: 'c' });
    expect(mocks.frameSampler.clearFrames).toHaveBeenCalledTimes(1);
  });

  it('shows a truthful error when undo is unavailable', async () => {
    await renderReady();
    await fireEvent.keyDown(window, { key: 'u' });
    expect(mocks.notification.showError).toHaveBeenCalledWith('No actions to undo');
    expect(mocks.dataset.undo.mutateAsync).not.toHaveBeenCalled();
  });

  it('routes undo through native history and retrains after refreshed qualifying stats', async () => {
    mocks.dataset.canUndo.data = { available: true, depth: 1, nextAction: 'restoreFrame', revision: 1 };
    mocks.dataset.stats.refetch.mockResolvedValue({ data: { hasMinimumFrames: true } });
    await renderReady();
    await fireEvent.keyDown(window, { key: 'u' });
    await waitFor(() => expect(mocks.dataset.undo.mutateAsync).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mocks.training.trainAndDeploy).toHaveBeenCalledWith(expect.objectContaining({ doCV: false })));
    await waitFor(() => expect(mocks.notification.showSuccess).toHaveBeenCalledWith('Last dataset change undone.'));
  });

  it('buffers posture-change labels as interval captures without immediate persistence', async () => {
    mocks.settings.settings.autoCaptureEnabled = true;
    mocks.native.getActiveModelMetadata.mockResolvedValue({
      posture: { classifierId: 'mlp', featureTypes: ['engineered_features'], trainedAt: 100 },
      presence: null,
    });
    await renderReady();
    await waitFor(() => expect(mocks.postureChangeHook.mock.calls.at(-1)?.[1].enabled).toBe(true));
    const onCapture = mocks.postureChangeHook.mock.calls.at(-1)?.[1].onCapture;
    mocks.frameSampler.captureFrame.mockClear();
    mocks.frameSampler.saveFrame.mockClear();
    expect(onCapture).toBeTypeOf('function');
    for (const label of [FrameLabel.GOOD, FrameLabel.BAD, FrameLabel.AWAY]) onCapture?.(label);
    expect(mocks.frameSampler.captureFrame.mock.calls).toEqual([
      ['interval', FrameLabel.GOOD],
      ['interval', FrameLabel.BAD],
      ['interval', FrameLabel.AWAY],
    ]);
    expect(mocks.frameSampler.saveFrame).not.toHaveBeenCalled();
  });

  it('uses interval capture only without a model and posture-change capture with a model', async () => {
    mocks.settings.settings.autoCaptureEnabled = true;
    await renderReady();
    const intervalConfig = mocks.autoCaptureHook.mock.calls.at(-1)?.[0];
    const changeConfig = mocks.postureChangeHook.mock.calls.at(-1)?.[1];
    expect(intervalConfig.enabled).toBe(true);
    expect(changeConfig.enabled).toBe(false);

    cleanup();
    mocks.native.getActiveModelMetadata.mockResolvedValue({
      posture: { classifierId: 'mlp', featureTypes: ['engineered_features'], trainedAt: 100 },
      presence: null,
    });
    await renderReady();
    await waitFor(() => expect(mocks.native.getActiveModelMetadata).toHaveBeenCalled());
    await waitFor(() => expect(mocks.autoCaptureHook.mock.calls.at(-1)?.[0].enabled).toBe(false));
    expect(mocks.postureChangeHook.mock.calls.at(-1)?.[1].enabled).toBe(true);
  });

  it('opens the responsive control panel without removing tab semantics', async () => {
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Open control panel' }));
    expect(screen.getByRole('button', { name: 'Close control panel' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Runtime Settings' })).toHaveAttribute('aria-selected', 'true');
  });

  it('never flashes the document title for bad-posture classification (oracle parity)', async () => {
    mocks.background.isVisible = false;
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Load bad inference' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Good' })).toBeEnabled());
    expect(mocks.background.flashTitle).not.toHaveBeenCalled();
  });

  it('resets native dataset and settings only after confirmation', async () => {
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Open control panel' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset All Data' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    await waitFor(() => expect(mocks.dataset.resetAllData.mutateAsync).toHaveBeenCalledTimes(1));
    expect(mocks.settings.resetSettings).not.toHaveBeenCalled();
    expect(mocks.settings.reconcile).toHaveBeenCalledTimes(1);
    expect(mocks.queryClient.clear).not.toHaveBeenCalled();
  });
});

describe('PostureTrackerApp status badge model state', () => {
  it('shows the trained badge when a live classification proves a deployed model despite a stale null snapshot', async () => {
    // Regression (the reported bug): get_active_model_metadata reported no posture
    // model (a stale, unrefreshed snapshot) while the runtime already classified and
    // the alert sound fired. A live non-null goodProbability is authoritative proof of
    // a deployed posture classifier, so the badge must not claim "No Model Trained".
    mocks.native.getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
    await renderReady();
    expect(screen.getByText('No Model Trained')).toBeInTheDocument();
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference A' }));
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
    expect(screen.getByText('Good Posture')).toBeInTheDocument();
  });

  it('keeps the untrained badge when no model and no live posture score are present', async () => {
    // A present-but-unscored frame (goodProbability null: no posture model, or the
    // person is away) must not be mistaken for a deployed model.
    mocks.native.getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Load presence-only inference' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Good' })).toBeEnabled());
    expect(screen.getByText('No Model Trained')).toBeInTheDocument();
  });

  it('shows the trained badge from the metadata snapshot alone at startup with a pre-trained model', async () => {
    // Fresh start with a pre-trained model: get_active_model_metadata returns it on
    // mount, so the badge is correct before any live classification arrives.
    mocks.native.getActiveModelMetadata.mockResolvedValue({
      posture: { classifierId: 'mlp', featureTypes: ['engineered_features'], trainedAt: 100 },
      presence: null,
    });
    await renderReady();
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
  });

  it('returns the badge to untrained after Reset All Data clears the runtime model', async () => {
    // Reset unloads the runtime classifier (goodProbability -> null) and reconciles
    // metadata to null; the self-heal must not keep the badge trained from the
    // pre-reset live result.
    mocks.native.getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference A' }));
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Open control panel' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset All Data' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    await waitFor(() => expect(mocks.dataset.resetAllData.mutateAsync).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText('No Model Trained')).toBeInTheDocument());
  });

  it('keeps the trained badge in the away state when the person leaves after a live classification proved a model', async () => {
    // Regression fix: goodProbability is presence-gated, so it goes null when the
    // person leaves. With a still-stale null snapshot (getActiveModelMetadata keeps
    // returning null), the session latch must hold hasModel true so the badge shows
    // the away state, never "No Model Trained".
    mocks.native.getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference A' }));
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Load away inference' }));
    await waitFor(() => expect(screen.getByText('Person Away')).toBeInTheDocument());
    expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument();
  });

  it('returns the badge to untrained after Reset All Data even while the person is absent', async () => {
    // Clearing the latch on reconcile: a reset performed while the person is away
    // (no live classification) must still fall back to "No Model Trained".
    mocks.native.getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
    await renderReady();
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference A' }));
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Load away inference' }));
    await waitFor(() => expect(screen.getByText('Person Away')).toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Open control panel' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset All Data' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    await waitFor(() => expect(mocks.dataset.resetAllData.mutateAsync).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText('No Model Trained')).toBeInTheDocument());
  });

  it('heals a stale null snapshot with exactly one metadata refetch when a live classification contradicts it', async () => {
    // One-shot heal: a live goodProbability against a null snapshot triggers a single
    // refetch (no polling). Once the snapshot reads the model, further live results do
    // not refetch again.
    mocks.native.getActiveModelMetadata
      .mockResolvedValueOnce({ posture: null, presence: null })
      .mockResolvedValue({
        posture: { classifierId: 'mlp', featureTypes: ['engineered_features'], trainedAt: 100 },
        presence: null,
      });
    await renderReady();
    await waitFor(() => expect(mocks.native.getActiveModelMetadata).toHaveBeenCalledTimes(1));
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference A' }));
    await waitFor(() => expect(mocks.native.getActiveModelMetadata).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText('No Model Trained')).not.toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Load inference B' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Load bad inference' }));
    await waitFor(() => expect(screen.getByText('Bad Posture')).toBeInTheDocument());
    expect(mocks.native.getActiveModelMetadata).toHaveBeenCalledTimes(2);
  });
});

describe('PostureTrackerApp self-healing auto-train', () => {
  it('auto-trains exactly once when a sufficient dataset needs a model at startup', async () => {
    // The reported gap: frames were collected but no model existed and nothing retrained,
    // because the save-only trigger had already passed. A trainable-but-stale dataset must
    // train itself on load.
    mocks.dataset.stats.data = { hasMinimumFrames: true };
    mocks.dataset.needsRetraining = { data: true, refetch: vi.fn().mockResolvedValue({ data: false }) };
    await renderReady();
    await waitFor(() => expect(mocks.training.trainAndDeploy).toHaveBeenCalled());
    expect(mocks.training.trainAndDeploy).toHaveBeenCalledTimes(1);
    expect(mocks.training.trainAndDeploy).toHaveBeenCalledWith(expect.objectContaining({ doCV: false }));
  });

  it('auto-trains once when needsRetraining flips true after a training-settings save', async () => {
    // A settings edit refreshes needsRetraining (via TrainingConfigContext.persistedRevision);
    // when it reports the model is now stale, the reactive trigger fires exactly once.
    mocks.dataset.stats.data = { hasMinimumFrames: true };
    const needs = reactiveBox(false);
    mocks.dataset.needsRetraining = { get data() { return needs.value; }, refetch: vi.fn().mockResolvedValue({ data: false }) };
    await renderReady();
    await Promise.resolve();
    expect(mocks.training.trainAndDeploy).not.toHaveBeenCalled();
    needs.set(true);
    flushSync();
    await waitFor(() => expect(mocks.training.trainAndDeploy).toHaveBeenCalledTimes(1));
  });

  it('refreshes needsRetraining when training settings are persisted', async () => {
    // Requirement wiring: a persisted settings change refreshes the retraining flag so the
    // trigger sees it, with no polling. Held below the minimum here to isolate the refetch.
    mocks.dataset.stats.data = { hasMinimumFrames: false };
    const refetch = vi.fn().mockResolvedValue({ data: false });
    mocks.dataset.needsRetraining = { data: false, refetch };
    const revision = reactiveBox(0);
    mocks.trainingConfig = { ready: true, get persistedRevision() { return revision.value; }, reload: vi.fn().mockResolvedValue(undefined), flushToStorage: vi.fn().mockResolvedValue(undefined), reconcile: vi.fn() };
    await renderReady();
    expect(refetch).not.toHaveBeenCalled();
    revision.set(1);
    flushSync();
    await waitFor(() => expect(refetch).toHaveBeenCalled());
  });

  it('does not start a second training when one is already running', async () => {
    mocks.dataset.stats.data = { hasMinimumFrames: true };
    mocks.dataset.needsRetraining = { data: true, refetch: vi.fn().mockResolvedValue({ data: false }) };
    mocks.training.state.isTraining = true;
    await renderReady();
    await Promise.resolve();
    expect(mocks.training.trainAndDeploy).not.toHaveBeenCalled();
  });

  it('retries the auto-train exactly once when the deploy is superseded, then stops', async () => {
    // SnapshotChanged -> ApiError::DatasetChanged surfaces as NativeCommandError kind
    // 'datasetChanged'; a single retry recovers a deploy raced by a concurrent change.
    mocks.dataset.stats.data = { hasMinimumFrames: true };
    mocks.dataset.needsRetraining = { data: true, refetch: vi.fn().mockResolvedValue({ data: false }) };
    mocks.training.trainAndDeploy
      .mockRejectedValueOnce(new NativeCommandError({ kind: 'datasetChanged', message: 'training snapshot changed' }))
      .mockResolvedValue(true);
    await renderReady();
    await waitFor(() => expect(mocks.training.trainAndDeploy).toHaveBeenCalledTimes(2));
    await Promise.resolve();
    expect(mocks.training.trainAndDeploy).toHaveBeenCalledTimes(2);
  });

  it('never auto-trains when the dataset is below the minimum', async () => {
    mocks.dataset.stats.data = { hasMinimumFrames: false };
    mocks.dataset.needsRetraining = { data: true, refetch: vi.fn().mockResolvedValue({ data: true }) };
    await renderReady();
    await Promise.resolve();
    expect(mocks.training.trainAndDeploy).not.toHaveBeenCalled();
  });
});

describe('PostureTrackerApp camera-settings error recovery', () => {
  // A persisted settings row the native side can no longer deserialize surfaces as a
  // blocking "Camera settings unavailable" overlay. Retry alone re-reads the same
  // unreadable row, so the overlay must also offer a reset. ready/error are rune-backed
  // here so a successful reset can flip them and prove the overlay clears.
  let readyBox: { readonly value: boolean; set: (next: boolean) => void };
  let errorBox: { readonly value: string | null; set: (next: string | null) => void };

  beforeEach(() => {
    readyBox = reactiveBox<boolean>(false);
    errorBox = reactiveBox<string | null>('Native camera settings are unreadable.');
    Object.defineProperty(mocks.settings, 'ready', {
      configurable: true,
      get: () => readyBox.value,
      set: (value: boolean) => readyBox.set(value),
    });
    Object.defineProperty(mocks.settings, 'error', {
      configurable: true,
      get: () => errorBox.value,
      set: (value: string | null) => errorBox.set(value),
    });
  });

  afterEach(() => {
    // Restore plain data properties so the reactive accessors never leak into the
    // sibling suites that assign ready/error directly in the shared beforeEach.
    Object.defineProperty(mocks.settings, 'ready', { configurable: true, writable: true, value: true });
    Object.defineProperty(mocks.settings, 'error', { configurable: true, writable: true, value: null });
  });

  it('offers both retry and reset actions when camera settings fail to load', async () => {
    render(PostureTrackerApp);
    await waitFor(() => expect(screen.getByText('Camera settings unavailable')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Retry camera settings' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reset camera settings' })).toBeInTheDocument();
    expect(screen.getByText('Reset restores default preprocessing settings.')).toBeInTheDocument();
  });

  it('resets to defaults and clears the overlay when Reset camera settings succeeds', async () => {
    mocks.settings.resetSettings.mockImplementation(async () => {
      errorBox.set(null);
      readyBox.set(true);
    });
    render(PostureTrackerApp);
    await waitFor(() => expect(screen.getByText('Camera settings unavailable')).toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Reset camera settings' }));
    expect(mocks.settings.resetSettings).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByText('Camera settings unavailable')).not.toBeInTheDocument());
  });

  it('keeps the overlay recoverable and surfaces the new error when a reset fails', async () => {
    mocks.settings.resetSettings.mockImplementation(async () => {
      errorBox.set('Reset failed: native storage error.');
      throw new Error('Reset failed: native storage error.');
    });
    render(PostureTrackerApp);
    await waitFor(() => expect(screen.getByText('Camera settings unavailable')).toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Reset camera settings' }));
    await waitFor(() => expect(screen.getByText('Reset failed: native storage error.')).toBeInTheDocument());
    expect(screen.getByRole('button', { name: 'Reset camera settings' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry camera settings' })).toBeInTheDocument();
  });
});
