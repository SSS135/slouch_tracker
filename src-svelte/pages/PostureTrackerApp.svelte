<script lang="ts">
  import type { ActiveModelMetadata, InferenceUiResult, NativeStateSnapshot_Serialize } from '@generated/bindings';
  import { useQueryClient } from '@tanstack/svelte-query';
  import { CameraProvider } from '@/contexts/CameraContext';
  import { useTraining } from '@/contexts/TrainingContext';
  import { useTrainingConfig } from '@/contexts/TrainingConfigContext';
  import PostureCamera from '@/components/PostureCamera.svelte';
  import ErrorBoundary from '@/components/ErrorBoundary.svelte';
  import { useCameraSettings } from '@/hooks/useCameraSettings';
  import { useNotification } from '@/hooks/useNotification';
  import { useFrameSampler, type CapturedFrame } from '@/hooks/useFrameSampler';
  import type { PreviewFrameSource } from '@/services/dataset/thumbnailGenerator';
  import { useAutoCapture } from '@/hooks/useAutoCapture';
  import { useBackgroundProcessing } from '@/hooks/useBackgroundProcessing';
  import { useMultiTaskDetection } from '@/hooks/useMultiTaskDetection.svelte';
  import { usePostureChangeDetector } from '@/hooks/usePostureChangeDetector';
  import { datasetKeys, useDatasetOperations } from '@/hooks/useDatasetOperations';
  import { usePostureSound } from '@/hooks/usePostureSound';
  import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts';
  import { nativeClient } from '@/lib/native/client';
  import { useNativeAppState } from '@/lib/state/nativeApp.svelte';
  import { MAX_BUFFER_SIZE } from '@/services/dataset/constants';
  import { FrameLabel, type CaptureAction } from '@/services/dataset/types';
  import CameraViewport from '@/components/unified/CameraViewport.svelte';
  import ConfirmationModal from '@/components/dataset/ConfirmationModal.svelte';
  import SettingsTab from '@/components/unified/SettingsTab.svelte';
  import TrainingTab from '@/components/unified/TrainingTab.svelte';
  import ControlPanel, { type TabType } from '@/components/unified/ControlPanel.svelte';

  type ElementRef<T> = { current: T | null };
  let activeTab = $state<TabType>('runtime');
  let inferenceResult = $state<InferenceUiResult | null>(null);
  let fps = $state(0);
  let previewFrame = $state<{ blobUrl: string; label?: FrameLabel } | null>(null);
  let isPanelCollapsed = $state(true);
  let resetModalOpen = $state(false);
  let isCanvasReady = $state(false);
  // Ephemeral preview of the detector-input feed; deliberately not persisted.
  let processedView = $state(false);
  let modelMetadata = $state<ActiveModelMetadata>({ posture: null, presence: null });
  let modelMetadataLoading = $state(true);
  let modelMetadataError = $state<string | null>(null);
  let frozenFrames = $state<CapturedFrame[] | null>(null);
  let lastAction = $state<CaptureAction | null>(null);
  let lastActionUrl: string | null = null;
  let lastAutoRecordTime = -Infinity;
  let modelMetadataGeneration = 0;

  const canvasRef: ElementRef<HTMLCanvasElement> = { current: null };
  const previewFrameRef: ElementRef<PreviewFrameSource> = { current: null };
  const cameraSettings = useCameraSettings();
  const notification = useNotification();
  const datasetOps = useDatasetOperations();
  const queryClient = useQueryClient();
  const training = useTraining();
  const trainingConfig = useTrainingConfig();
  const nativeApp = useNativeAppState();
  const settings = $derived(cameraSettings.settings);
  const trainingState = $derived(training.state);

  CameraProvider({
    get inferenceResult() { return inferenceResult; },
    get fps() { return fps; },
  });

  const frameSampler = useFrameSampler({
    get inferenceResult() { return inferenceResult; },
    getPreviewFrame: () => previewFrameRef.current,
    get privacyMode() { return settings.privacyMode; },
    config: { maxBufferSize: MAX_BUFFER_SIZE },
  });

  const recentFrames = $derived(frameSampler.recentFrames);
  const visibleFrames = $derived(frozenFrames ?? recentFrames);
  const queuedFrameCount = $derived(frozenFrames ? Math.max(0, recentFrames.length - frozenFrames.length) : 0);
  const hasModel = $derived(Boolean(modelMetadata.posture));
  const systemReady = $derived(cameraSettings.ready && Boolean(nativeApp.status?.inferenceReady) && isCanvasReady);
  // Gate on pipeline liveness, not per-frame token consumption, so the buttons
  // stay enabled across auto-capture instead of blinking each interval. A click
  // during the consumed gap is deferred to the next result by requestCapture.
  const captureReady = $derived(systemReady && frameSampler.isLive);
  const canUndo = $derived(datasetOps.canUndo.data?.available ?? false);

  async function loadModelMetadata(): Promise<void> {
    const generation = ++modelMetadataGeneration;
    modelMetadataLoading = true;
    modelMetadataError = null;
    try {
      const nextMetadata = await nativeClient.getActiveModelMetadata();
      if (generation === modelMetadataGeneration) modelMetadata = nextMetadata;
    } catch (cause) {
      if (generation === modelMetadataGeneration) {
        modelMetadataError = cause instanceof Error ? cause.message : String(cause);
      }
      throw cause;
    } finally {
      if (generation === modelMetadataGeneration) modelMetadataLoading = false;
    }
  }

  $effect(() => {
    void loadModelMetadata().catch(() => undefined);
  });

  $effect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void nativeClient.onNativeStateChanged((event) => {
      void reconcileNativeState(event.state).catch((cause: unknown) => {
        notification.showError(`Failed to reconcile native state: ${cause instanceof Error ? cause.message : String(cause)}`);
      });
    }).then((cleanup) => disposed ? cleanup() : (unlisten = cleanup))
      .catch((cause: unknown) => {
        notification.showError(`Failed to subscribe to native state: ${cause instanceof Error ? cause.message : String(cause)}`);
      });
    return () => { disposed = true; unlisten?.(); };
  });

  useAutoCapture({
    get enabled() { return settings.autoCaptureEnabled && !hasModel && !trainingState.isTraining; },
    get intervalSeconds() { return settings.autoCaptureIntervalSeconds; },
    mode: 'interval',
    async onCapture() { await frameSampler.captureFrame('interval'); },
  });

  usePostureChangeDetector(
    () => inferenceResult?.classification ?? null,
    {
      get enabled() { return settings.autoCaptureEnabled && hasModel && !trainingState.isTraining; },
      onCapture: (label) => {
        void frameSampler.captureFrame('interval', label);
        lastAutoRecordTime = Date.now();
      },
    },
  );

  $effect(() => {
    if (!settings.autoCaptureEnabled || !hasModel || trainingState.isTraining) return;
    const interval = setInterval(() => {
      const now = Date.now();
      if (lastAutoRecordTime === -Infinity) {
        lastAutoRecordTime = now;
        return;
      }
      if (now - lastAutoRecordTime < 5000) return;
      const classification = inferenceResult?.classification;
      if (!classification) return;
      const label = classification.goodProbability === null
        ? FrameLabel.AWAY
        : classification.goodProbability >= 0.5 ? FrameLabel.GOOD : FrameLabel.BAD;
      void frameSampler.captureFrame('interval', label);
      lastAutoRecordTime = now;
    }, 500);
    return () => clearInterval(interval);
  });

  async function refreshAndRetrain(): Promise<void> {
    await datasetOps.invalidateAll();
    const refreshed = await datasetOps.stats.refetch();
    if (refreshed.data?.hasMinimumFrames) {
      void training.trainAndDeploy({ doCV: false }).catch((cause: unknown) => {
        notification.showError(`Automatic retraining failed: ${cause instanceof Error ? cause.message : String(cause)}`);
      });
    }
  }

  function replaceLastAction(frame: CapturedFrame | null, label?: FrameLabel): void {
    if (lastActionUrl) URL.revokeObjectURL(lastActionUrl);
    lastActionUrl = frame ? URL.createObjectURL(frame.thumbnail) : null;
    lastAction = frame && label
      ? {
          frameId: frame.id,
          timestamp: frame.timestamp,
          label,
          thumbnailUrl: lastActionUrl!,
        }
      : null;
  }

  $effect(() => () => replaceLastAction(null));

  function persistBufferedFrame(frameId: string, label: FrameLabel, showSaved: boolean): void {
    const frame = frameSampler.recentFrames.find((item) => item.id === frameId) ?? null;
    void frameSampler.saveFrame(frameId, label)
      .then(async () => {
        if (frozenFrames) frozenFrames = frozenFrames.filter((item) => item.id !== frameId);
        replaceLastAction(frame, label);
        await refreshAndRetrain();
        if (showSaved) notification.showSuccess('Frame saved.');
      })
      .catch((cause: unknown) => {
        notification.showError(`Failed to save frame: ${cause instanceof Error ? cause.message : String(cause)}`);
      });
  }

  async function captureWithLabel(label: FrameLabel): Promise<void> {
    const outcome = await frameSampler.requestCapture(label);
    // A newer labelled click replaced this one - stay silent, it is not an error.
    if (outcome.status === 'superseded') return;
    if (outcome.status !== 'captured') {
      notification.showError('No current person detection is available to capture.');
      return;
    }
    persistBufferedFrame(outcome.frame.id, label, true);
  }

  async function saveBufferedFrame(frameId: string, label: FrameLabel): Promise<void> {
    const frame = frameSampler.recentFrames.find((item) => item.id === frameId);
    if (!frame) {
      notification.showError('Frame not found in buffer');
      return;
    }
    persistBufferedFrame(frameId, label, false);
  }

  async function undo(): Promise<void> {
    if (!canUndo) {
      notification.showError('No actions to undo');
      return;
    }
    try {
      await datasetOps.undo.mutateAsync();
      replaceLastAction(null);
      await refreshAndRetrain();
      notification.showSuccess('Last dataset change undone.');
    } catch (cause) {
      notification.showError(`Failed to undo: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  }

  async function confirmReset(): Promise<void> {
    try {
      await prepareNativeReplace();
      const state = await datasetOps.resetAllData.mutateAsync();
      await reconcileNativeState(state);
      resetModalOpen = false;
      notification.showSuccess('Dataset and settings reset.');
    } catch (cause) {
      notification.showError(`Failed to reset data: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
  }

  async function prepareNativeReplace(): Promise<void> {
    await Promise.all([cameraSettings.flush(), trainingConfig.flushToStorage()]);
  }

  async function reconcileNativeState(state: NativeStateSnapshot_Serialize): Promise<void> {
    modelMetadataGeneration += 1;
    frameSampler.clearFrames();
    frozenFrames = null;
    inferenceResult = null;
    previewFrame = null;
    replaceLastAction(null);
    cameraSettings.reconcile(state.cameraSettings, state.uiSettings);
    trainingConfig.reconcile(state.trainingSettings);
    nativeApp.reconcile(state.app);
    modelMetadata = state.activeModels;
    modelMetadataError = null;
    modelMetadataLoading = false;
    queryClient.setQueryData(datasetKeys.undo(), state.undo);
    await Promise.all([training.reconcile(), datasetOps.invalidateAll()]);
  }

  async function handleTrainingComplete(): Promise<void> {
    await loadModelMetadata();
    await datasetOps.invalidateAll();
  }

  const backgroundProcessing = useBackgroundProcessing({});
  const multiTaskDetection = useMultiTaskDetection(() => inferenceResult);
  $effect(() => {
    if (!backgroundProcessing.isVisible && multiTaskDetection.detection?.slouching) {
      backgroundProcessing.flashTitle('Bad Posture Detected');
    }
  });

  const postureDataForSound = $derived.by(() => {
    const probability = inferenceResult?.classification?.goodProbability;
    if (!inferenceResult?.personFound || probability === null || probability === undefined) return null;
    const bad = probability < 0.5;
    return { person_found: true, slouching: bad, forward_neck_tilt: bad, hand_near_face: false, mouth_open: false };
  });
  usePostureSound(
    () => postureDataForSound,
    () => settings.alertVolume,
    () => trainingState.isTraining,
    () => settings.alertDelaySeconds,
  );

  function editableTarget(target: EventTarget | null): boolean {
    const element = target instanceof HTMLElement ? target : null;
    return Boolean(element?.closest('input, textarea, select, [contenteditable="true"]'));
  }
  $effect(() => {
    const handleKeydown = (event: KeyboardEvent): void => {
      if (event.repeat || editableTarget(event.target)) return;
      const key = event.key.toUpperCase();
      if (key === 'G' && captureReady) void captureWithLabel(FrameLabel.GOOD);
      else if (key === 'B' && captureReady) void captureWithLabel(FrameLabel.BAD);
      else if (key === 'A' && captureReady) void captureWithLabel(FrameLabel.AWAY);
      else if (key === 'C') frameSampler.clearFrames();
      else if (key === 'U') void undo();
    };
    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  });

  useGlobalShortcuts({
    onCaptureGood: () => { if (captureReady) void captureWithLabel(FrameLabel.GOOD); },
    onCaptureBad: () => { if (captureReady) void captureWithLabel(FrameLabel.BAD); },
    onCaptureAway: () => { if (captureReady) void captureWithLabel(FrameLabel.AWAY); },
  });

  const modelInfo = $derived(modelMetadata.posture ? {
    featureType: modelMetadata.posture.featureTypes.join(', '),
    accuracy: undefined,
    lastTrained: modelMetadata.posture.trainedAt ?? undefined,
    presenceFeatureType: modelMetadata.presence
      ? modelMetadata.presence.featureTypes.join(', ')
      : null,
  } : null);

</script>

<div class="app-shell">
  <div class="camera-layer">
    <ErrorBoundary>
      {#snippet fallback()}
        <div class="camera-error" role="alert">
          <strong>Camera Error</strong>
          <p>Reload the app to restart the camera.</p>
          <button type="button" onclick={() => window.location.reload()}>Reload</button>
        </div>
      {/snippet}
      <CameraViewport
        {hasModel}
        {previewFrame}
        frames={visibleFrames}
        {queuedFrameCount}
        onFrameListHoverStart={() => { frozenFrames = [...recentFrames]; }}
        onFrameListHoverEnd={() => { frozenFrames = null; }}
        onSaveFrameAsGood={(id) => saveBufferedFrame(id, FrameLabel.GOOD)}
        onSaveFrameAsBad={(id) => saveBufferedFrame(id, FrameLabel.BAD)}
        onSaveFrameAsAway={(id) => saveBufferedFrame(id, FrameLabel.AWAY)}
        onFramePreview={(blobUrl, label) => { previewFrame = { blobUrl, label }; }}
        onFramePreviewClear={() => { previewFrame = null; }}
        {isPanelCollapsed}
        onCaptureGood={() => captureWithLabel(FrameLabel.GOOD)}
        onCaptureBad={() => captureWithLabel(FrameLabel.BAD)}
        onCaptureAway={() => captureWithLabel(FrameLabel.AWAY)}
        {inferenceResult}
        isTraining={trainingState.isTraining}
        trainingProgress={trainingState.progress}
        isTrainingPipeline={trainingState.isTrainingPipeline}
        isSystemReady={captureReady}
        onUndo={() => { void undo(); }}
        {canUndo}
        {lastAction}
      >
        <PostureCamera
          onInferenceResult={(result) => { inferenceResult = result; }}
          onFps={(value) => { fps = value; }}
          onCanvasReady={(ready) => { isCanvasReady = ready; }}
          onBackgroundClick={() => { if (!isPanelCollapsed) isPanelCollapsed = true; }}
          {canvasRef}
          latestFrameRef={previewFrameRef}
          privacyMode={settings.privacyMode}
          {processedView}
          paused={!cameraSettings.ready}
        />
      </CameraViewport>
    </ErrorBoundary>
  </div>

  <button type="button" class:open={!isPanelCollapsed} class="panel-toggle" aria-label={isPanelCollapsed ? 'Open control panel' : 'Close control panel'} onclick={() => { isPanelCollapsed = !isPanelCollapsed; }}>
    <!-- SVG chevron instead of a text glyph: font metrics placed the character off-center. -->
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <path d={isPanelCollapsed ? 'M10.5 3 5.5 8l5 5' : 'M5.5 3l5 5-5 5'} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  </button>
  <div class:collapsed={isPanelCollapsed} class="panel-shell" aria-hidden={isPanelCollapsed} inert={isPanelCollapsed}>
    {#snippet runtimeContent()}
      <SettingsTab {settings} onUpdateSettings={cameraSettings.updateSettings} onResetSettings={() => { resetModalOpen = true; }} isModelLoaded={hasModel} {processedView} onProcessedViewChange={(value) => { processedView = value; }} {fps} {modelInfo} />
    {/snippet}
    {#snippet trainingContent()}
      <TrainingTab onTrainingComplete={handleTrainingComplete} onFramesChanged={() => { void datasetOps.invalidateAll(); }} onBeforeNativeReplace={prepareNativeReplace} onNativeStateChanged={reconcileNativeState} onFramePreview={(url, label) => { previewFrame = { blobUrl: url, label }; }} onFramePreviewClear={() => { previewFrame = null; }} />
    {/snippet}
    <ControlPanel {activeTab} onTabChange={(tab) => { activeTab = tab; }} collapsed={isPanelCollapsed} tabs={[{ id: 'runtime', label: 'Runtime Settings', content: runtimeContent }, { id: 'training', label: 'Training', content: trainingContent }]} />
  </div>
  {#if modelMetadataError}
    <div class="model-error" role="alert">
      <span>Failed to load active model metadata: {modelMetadataError}</span>
      <button type="button" onclick={() => { void loadModelMetadata().catch(() => undefined); }}>Retry model status</button>
    </div>
  {:else if modelMetadataLoading}
    <div class="model-loading" role="status">Loading model status…</div>
  {/if}
  {#if nativeApp.error}
    <div class="settings-loading" role="alert">
      <div class="blocking-error">
        <strong>Native initialization failed</strong>
        <span>{nativeApp.error.message}</span>
        <button type="button" onclick={() => { void nativeApp.initialize().catch(() => undefined); }}>Retry initialization</button>
      </div>
    </div>
  {:else if !cameraSettings.ready}
    <div class="settings-loading" role={cameraSettings.error ? 'alert' : 'status'}>
      {#if cameraSettings.error}
        <div class="blocking-error">
          <strong>Camera settings unavailable</strong>
          <span>{cameraSettings.error}</span>
          <button type="button" onclick={() => { void cameraSettings.reload().catch(() => undefined); }}>Retry camera settings</button>
        </div>
      {:else}
        Loading native camera settings…
      {/if}
    </div>
  {/if}
</div>

<ConfirmationModal visible={resetModalOpen} title="Reset Dataset and Settings" message="This removes collected frames and resets application settings." confirmText="Reset" cancelText="Cancel" confirmButtonColor="red" onConfirm={confirmReset} onCancel={() => { resetModalOpen = false; }} />

<style>
  .app-shell { position: relative; width: 100%; height: 100%; overflow: hidden; }
  .camera-layer { width: 100%; height: 100%; }
  .camera-error { display: flex; height: 100%; flex-direction: column; align-items: center; justify-content: center; color: white; background: #10181f; }
  .settings-loading { position: absolute; inset: 0; z-index: 200; display: grid; place-items: center; padding: 2rem; color: white; background: #10181f; text-align: center; }
  .blocking-error { display: flex; max-width: 32rem; flex-direction: column; gap: 0.75rem; align-items: center; }
  .model-error,
  .model-loading { position: absolute; top: 0.75rem; left: 50%; z-index: 150; display: flex; align-items: center; gap: 0.75rem; max-width: min(34rem, calc(100vw - 2rem)); padding: 0.625rem 0.75rem; border-radius: 0.375rem; color: white; background: rgb(201 42 42 / 92%); transform: translateX(-50%); }
  .model-loading { background: rgb(33 37 41 / 90%); }
  .model-error button { min-height: 2rem; border: 1px solid white; border-radius: 0.25rem; padding: 0.25rem 0.5rem; color: white; background: transparent; cursor: pointer; }
  .blocking-error button { min-height: 2.25rem; border: 1px solid #74c0fc; border-radius: 0.375rem; padding: 0.5rem 0.875rem; color: white; background: #1971c2; cursor: pointer; }
  .panel-toggle { position: absolute; top: 50%; right: 16px; z-index: 110; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; padding: 0; border: 0; border-radius: 4px; color: white; background: rgb(10 10 10 / 70%); cursor: pointer; transform: translateY(-50%); }
  .panel-toggle svg { display: block; }
  .panel-toggle.open { right: calc(var(--panel-width, 576px) + 16px); }
  .panel-shell { position: absolute; top: 0; right: 0; z-index: 100; width: var(--panel-width, 576px); max-width: 100vw; height: 100%; }
  .panel-shell.collapsed { pointer-events: none; transform: translateX(100%); }
  @media (max-width: 640px) { .app-shell { --panel-width: 100vw; } .panel-toggle.open { right: 16px; } }
</style>
