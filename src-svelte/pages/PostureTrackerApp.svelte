<script lang="ts">
  import type { ActiveModelMetadata, InferenceUiResult, NativeStateSnapshot_Serialize } from '@generated/bindings';
  import { useQueryClient } from '@tanstack/svelte-query';
  import { CameraProvider } from '@/contexts/CameraContext';
  import { useTraining } from '@/contexts/TrainingContext';
  import { useTrainingConfig } from '@/contexts/TrainingConfigContext';
  import PostureCamera from '@/components/PostureCamera.svelte';
  import ErrorBoundary from '@/components/ErrorBoundary.svelte';
  import { useCameraSettings } from '@/hooks/useCameraSettings';
  import { useTrackingToggle } from '@/hooks/useTrackingToggle.svelte';
  import { usePoseModelDownload } from '@/hooks/usePoseModelDownload.svelte';
  import { useNotification } from '@/hooks/useNotification';
  import { useFrameSampler, type CapturedFrame } from '@/hooks/useFrameSampler';
  import { useOnboarding } from '@/hooks/useOnboarding.svelte';
  import type { PreviewFrameSource } from '@/services/dataset/thumbnailGenerator';
  import { useAutoCapture } from '@/hooks/useAutoCapture';
  import { useBackgroundProcessing } from '@/hooks/useBackgroundProcessing';
  import { useMultiTaskDetection } from '@/hooks/useMultiTaskDetection.svelte';
  import { usePostureChangeDetector } from '@/hooks/usePostureChangeDetector';
  import { datasetKeys, useDatasetOperations } from '@/hooks/useDatasetOperations';
  import { usePostureSound } from '@/hooks/usePostureSound';
  import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts';
  import { nativeClient, NativeCommandError } from '@/lib/native/client';
  import { useNativeAppState } from '@/lib/state/nativeApp.svelte';
  import { logger } from '@/services/logging/logger';
  import { MAX_BUFFER_SIZE } from '@/services/dataset/constants';
  import { FrameLabel, type CaptureAction } from '@/services/dataset/types';
  import CameraViewport from '@/components/unified/CameraViewport.svelte';
  import OnboardingOverlay from '@/components/onboarding/OnboardingOverlay.svelte';
  import PoseModelDownloadScreen from '@/components/PoseModelDownloadScreen.svelte';
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
  // Latest native camera start error (null when clear); drives the resume-retry state.
  let cameraError = $state<string | null>(null);
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
  // Sticky proof that a posture model classified at least once this session. Live
  // `goodProbability` is presence-gated (null while the person is away), so without a
  // latch hasModel would drop back to "No Model Trained" whenever the person leaves
  // while the `modelMetadata` snapshot is still stale-null. Cleared only when native
  // state is reconciled (reset / import / native-state-changed), which carries truth.
  let liveModelProven = $state(false);
  // Guards the one-shot metadata refetch that heals a stale-null snapshot the first
  // time a live classification contradicts it — no polling.
  let modelMetadataHealed = false;
  // Arms the reactive self-heal auto-train to fire at most once per rising edge of the
  // "trainable but no current model" condition; re-armed only when that condition clears
  // again, so a completed train that still reports needsRetraining never loops.
  let autoTrainArmed = true;

  const canvasRef: ElementRef<HTMLCanvasElement> = { current: null };
  const previewFrameRef: ElementRef<PreviewFrameSource> = { current: null };
  const cameraRestartRef: ElementRef<() => Promise<void>> = { current: null };
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

  const onboarding = useOnboarding({
    settingsReady: () => cameraSettings.ready,
    settings: () => settings,
    updateSettings: cameraSettings.updateSettings,
    flushSettings: cameraSettings.flush,
    stats: () => datasetOps.stats.data,
    restartCamera: () => cameraRestartRef.current?.() ?? Promise.resolve(),
  });

  const recentFrames = $derived(frameSampler.recentFrames);
  const visibleFrames = $derived(frozenFrames ?? recentFrames);
  const queuedFrameCount = $derived(frozenFrames ? Math.max(0, recentFrames.length - frozenFrames.length) : 0);
  // `goodProbability` is populated only by a deployed posture classifier (the same
  // signal that fires the alert sound), so a live non-null value is authoritative
  // proof a model is running — even when the `modelMetadata` snapshot (refreshed only
  // on mount / reset / train-complete) missed a deploy and still reads null.
  const liveModelActive = $derived(
    typeof inferenceResult?.classification?.goodProbability === 'number',
  );
  // Latch that proof so leaving the frame (goodProbability -> null, presence-gated)
  // does not flip the badge back to "No Model Trained". Cleared on native reconcile.
  $effect(() => {
    if (liveModelActive) liveModelProven = true;
  });
  // Heal the underlying snapshot exactly once when a live classification contradicts a
  // null snapshot, so hasModel and every other reader of modelMetadata (auto-capture
  // gating, model info, settings) reflect the truth instead of leaning on the latch.
  $effect(() => {
    if (liveModelActive && !modelMetadata.posture && !modelMetadataLoading && !modelMetadataHealed) {
      modelMetadataHealed = true;
      void loadModelMetadata().catch(() => undefined);
    }
  });
  const hasModel = $derived(Boolean(modelMetadata.posture) || liveModelActive || liveModelProven);
  // Capture readiness follows the LIVE detection pipeline, not the one-shot
  // `inferenceReady` status snapshot. That snapshot is read once during startup
  // init and never refreshed, so a slow model load can latch it false while
  // inference is fully up and streaming results — which permanently disabled
  // capture. `frameSampler.isLive` (fresh valid detections arriving) is itself
  // proof the native inference pipeline is ready, so it is the authoritative gate.
  const systemReady = $derived(cameraSettings.ready && isCanvasReady);
  // Gate on pipeline liveness, not per-frame token consumption, so the buttons
  // stay enabled across auto-capture instead of blinking each interval. A click
  // during the consumed gap is deferred to the next result by requestCapture.
  const captureReady = $derived(systemReady && frameSampler.isLive);
  const canUndo = $derived(datasetOps.canUndo.data?.available ?? false);

  // Session-only pause/resume of tracking. `paused` OR's into PostureCamera's
  // `paused` prop below, so pausing is a real native stop_camera (not a frontend
  // freeze) and resume a real start_camera. `isCanvasReady` (preview frames
  // flowing) is the "running" signal; `cameraError` keeps the button retryable
  // after a failed resume. Never persisted.
  const trackingToggle = useTrackingToggle({
    get cameraRunning() { return isCanvasReady; },
    get cameraError() { return cameraError; },
    get settingsReady() { return cameraSettings.ready; },
  });

  // First-run gate: when the native pose-model file is absent, block the app on a
  // one-time download. On completion, re-run inference init so the app proceeds
  // without a restart (Rust resolves the freshly downloaded file lazily). This
  // screen takes precedence over the GPU/DX12 init-error overlay below, so a
  // missing model is never misreported as a hardware failure.
  const poseModelGate = usePoseModelDownload({
    onReady: () => nativeApp.initialize(),
  });

  // Clear the last detection the instant tracking pauses so capture buttons
  // disable, the status classification clears, and no posture alert fires.
  $effect(() => {
    if (trackingToggle.paused) inferenceResult = null;
  });

  // Native is the single source of truth for pause state: tray menu / global
  // hotkey toggles arrive as `tracking-state-changed`. Adopting the payload here
  // keeps the UI button, overlay and camera gate in lockstep with the backend.
  // Frontend-initiated toggles round-trip through start/stop_camera -> the shared
  // native helper -> this same event; `applyNativePaused` absorbs those echoes.
  $effect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    void nativeClient.onTrackingStateChanged((payload) => {
      trackingToggle.applyNativePaused(payload.paused);
    }).then((cleanup) => disposed ? cleanup() : (unlisten = cleanup))
      .catch((cause: unknown) => {
        notification.showError(`Failed to subscribe to tracking state: ${cause instanceof Error ? cause.message : String(cause)}`);
      });
    return () => { disposed = true; unlisten?.(); };
  });

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

  // Self-healing auto-train: whenever the dataset is trainable but the deployed model is
  // missing or stale (needsRetraining), train once. This covers the cases the save-only
  // trigger misses — app start with data already collected, a training-settings change,
  // or an earlier deploy that was superseded — so "our models auto train" holds without a
  // fresh capture. Purely reactive on already-fetched query state; no polling.
  // Suppressed mid-wizard: auto-train would deploy a model and beep at the user while they
  // deliberately slouch for the "bad" step; this effect fires as soon as the wizard closes.
  const retrainNeeded = $derived(
    cameraSettings.ready
    && datasetOps.stats.data?.hasMinimumFrames === true
    && datasetOps.needsRetraining.data === true
    && !onboarding.active,
  );
  $effect(() => {
    if (!retrainNeeded) { autoTrainArmed = true; return; }
    if (trainingState.isTraining) return;
    if (!autoTrainArmed) return;
    autoTrainArmed = false;
    void autoTrain();
  });
  // needsRetraining already refetches on dataset-changed and every save invalidation, but
  // not on a training-settings edit; TrainingConfigContext bumps persistedRevision after it
  // persists, so refresh the flag here to let the trigger above see a settings-driven change.
  $effect(() => {
    const revision = trainingConfig.persistedRevision;
    if (!trainingConfig.ready || revision === 0) return;
    void datasetOps.needsRetraining.refetch();
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
    if (refreshed.data?.hasMinimumFrames && !onboarding.active) void autoTrain();
  }

  // Single training entry point shared by the save-triggered retrain and the reactive
  // self-heal above. onModelDeployed reloads the model metadata so hasModel/the badge stop
  // reading the pre-train value. Retries once when the deploy is superseded by a concurrent
  // dataset/settings change (SnapshotChanged -> datasetChanged); stays silent when another
  // job already owns training; warns without looping if the dataset still needs retraining.
  async function autoTrain(attempt = 0): Promise<void> {
    try {
      await training.trainAndDeploy({
        doCV: false,
        onModelDeployed: () => { void handleTrainingComplete(); },
      });
      const stillNeeded = await datasetOps.needsRetraining.refetch();
      if (stillNeeded.data === true) {
        logger.warn('training', 'Auto-train completed but the dataset still reports needsRetraining; stopping to avoid a loop.');
      }
    } catch (cause) {
      if (cause instanceof Error && /already running/i.test(cause.message)) return;
      if (cause instanceof NativeCommandError && cause.kind === 'datasetChanged' && attempt === 0) {
        logger.warn('training', 'Auto-train deploy was superseded by a dataset or settings change; retrying once.');
        await autoTrain(attempt + 1);
        return;
      }
      notification.showError(`Automatic training failed: ${cause instanceof Error ? cause.message : String(cause)}`);
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
        onboarding.notifyFramePersisted(label);
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
    // Native state was replaced (reset / import / native-state-changed): drop the
    // session latch and re-arm the one-shot heal so the reconciled snapshot is
    // authoritative and a removed model correctly returns the badge to untrained.
    liveModelProven = false;
    modelMetadataHealed = false;
    autoTrainArmed = true;
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
  // The alert beeps on bad-posture detections only (no wall-clock timer): a beep fires when
  // a bad detection arrives at least `alertDelaySeconds` after the streak start / last beep,
  // so it re-alerts every `alertDelaySeconds` of continued bad posture.
  usePostureSound(
    () => postureDataForSound,
    () => settings.alertVolume,
    () => trainingState.isTraining || onboarding.active,
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
        trackingPaused={trackingToggle.paused}
        onToggleTracking={trackingToggle.toggle}
        toggleTrackingDisabled={trackingToggle.disabled}
        chromeHidden={onboarding.active}
      >
        <PostureCamera
          onInferenceResult={(result) => { inferenceResult = result; }}
          onFps={(value) => { fps = value; }}
          onCanvasReady={(ready) => { isCanvasReady = ready; }}
          onBackgroundClick={() => { if (!isPanelCollapsed) isPanelCollapsed = true; }}
          {canvasRef}
          {cameraRestartRef}
          latestFrameRef={previewFrameRef}
          privacyMode={settings.privacyMode}
          {processedView}
          preprocessingDebugView={settings.preprocessingDebugView}
          showDetectionOverlay={settings.showDetectionOverlay}
          paused={!cameraSettings.ready || trackingToggle.paused}
          onCameraError={(error) => { cameraError = error; }}
        />
      </CameraViewport>
    </ErrorBoundary>
  </div>

  {#if !onboarding.active}
    <button type="button" class:open={!isPanelCollapsed} class="panel-toggle" aria-label={isPanelCollapsed ? 'Open control panel' : 'Close control panel'} onclick={() => { isPanelCollapsed = !isPanelCollapsed; }}>
      <!-- SVG chevron instead of a text glyph: font metrics placed the character off-center. -->
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
        <path d={isPanelCollapsed ? 'M10.5 3 5.5 8l5 5' : 'M5.5 3l5 5-5 5'} fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>
    <div class:collapsed={isPanelCollapsed} class="panel-shell" aria-hidden={isPanelCollapsed} inert={isPanelCollapsed}>
      {#snippet runtimeContent()}
        <SettingsTab {settings} onUpdateSettings={cameraSettings.updateSettings} onResetSettings={() => { resetModalOpen = true; }} onRunSetupAgain={() => { isPanelCollapsed = true; onboarding.begin(); }} isModelLoaded={hasModel} {processedView} onProcessedViewChange={(value) => { processedView = value; }} {fps} {modelInfo} />
      {/snippet}
      {#snippet trainingContent()}
        <TrainingTab onTrainingComplete={handleTrainingComplete} onFramesChanged={() => { void datasetOps.invalidateAll(); }} onBeforeNativeReplace={prepareNativeReplace} onNativeStateChanged={reconcileNativeState} onFramePreview={(url, label) => { previewFrame = { blobUrl: url, label }; }} onFramePreviewClear={() => { previewFrame = null; }} />
      {/snippet}
      <ControlPanel {activeTab} onTabChange={(tab) => { activeTab = tab; }} collapsed={isPanelCollapsed} tabs={[{ id: 'runtime', label: 'Runtime Settings', content: runtimeContent }, { id: 'training', label: 'Training', content: trainingContent }]} />
    </div>
  {/if}
  {#if modelMetadataError}
    <div class="model-error" role="alert">
      <span>Failed to load active model metadata: {modelMetadataError}</span>
      <button type="button" onclick={() => { void loadModelMetadata().catch(() => undefined); }}>Retry model status</button>
    </div>
  {:else if modelMetadataLoading}
    <div class="model-loading" role="status">Loading model status…</div>
  {/if}
  {#if poseModelGate.blocking}
    <PoseModelDownloadScreen state={poseModelGate.phase} onCancel={poseModelGate.cancel} onRetry={poseModelGate.retry} />
  {:else if nativeApp.error}
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
          <div class="blocking-error-actions">
            <button type="button" onclick={() => { void cameraSettings.reload().catch(() => undefined); }}>Retry camera settings</button>
            <button type="button" class="secondary" onclick={() => { void cameraSettings.resetSettings().catch(() => undefined); }}>Reset camera settings</button>
          </div>
          <span class="blocking-error-hint">Reset restores default preprocessing settings.</span>
        </div>
      {:else}
        Loading native camera settings…
      {/if}
    </div>
  {:else if onboarding.active}
    <OnboardingOverlay
      {onboarding}
      cameraOk={isCanvasReady && frameSampler.isLive}
      personFound={inferenceResult?.personFound ?? false}
      {captureReady}
      {cameraError}
      selectedCameraIndex={settings.cameraIndex}
      onCapture={(label) => void captureWithLabel(label)}
    />
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
  .blocking-error-actions { display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; }
  .blocking-error button.secondary { border-color: rgb(255 255 255 / 45%); background: transparent; }
  .blocking-error-hint { font-size: 0.8125rem; color: rgb(255 255 255 / 70%); }
  .panel-toggle { position: absolute; top: 50%; right: 16px; z-index: 110; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; padding: 0; border: 0; border-radius: 4px; color: white; background: rgb(10 10 10 / 70%); cursor: pointer; transform: translateY(-50%); }
  .panel-toggle svg { display: block; }
  .panel-toggle.open { right: calc(var(--panel-width, 576px) + 16px); }
  .panel-shell { position: absolute; top: 0; right: 0; z-index: 100; width: var(--panel-width, 576px); max-width: 100vw; height: 100%; }
  .panel-shell.collapsed { pointer-events: none; transform: translateX(100%); }
  @media (max-width: 640px) { .app-shell { --panel-width: 100vw; } .panel-toggle.open { right: 16px; } }
</style>
