<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { CapturedFrame } from '@/hooks/useFrameSampler';
  import { useCameraContext } from '@/contexts/CameraContext';
  import { FrameLabel, type InferenceResult, type CaptureAction } from '@/services/dataset/types';
  import PostureStatusBadge from '../PostureStatusBadge.svelte';
  import CaptureButtonsOverlay from './CaptureButtonsOverlay.svelte';
  import FrameListOverlay from './FrameListOverlay.svelte';
  import TrackingToggleButton from './TrackingToggleButton.svelte';
  import UndoButton from './UndoButton.svelte';

  export interface CameraViewportProps {
    children: Snippet;
    hasModel: boolean;
    previewFrame?: { blobUrl: string; label?: FrameLabel } | null;
    frames?: CapturedFrame[];
    onSaveFrameAsGood?: (id: string) => Promise<void>;
    onSaveFrameAsBad?: (id: string) => Promise<void>;
    onSaveFrameAsAway?: (id: string) => Promise<void>;
    onFramePreview?: (blobUrl: string, label: FrameLabel) => void;
    onFramePreviewClear?: () => void;
    isPanelCollapsed?: boolean;
    queuedFrameCount?: number;
    onFrameListHoverStart?: () => void;
    onFrameListHoverEnd?: () => void;
    onCaptureGood?: () => Promise<void>;
    onCaptureBad?: () => Promise<void>;
    onCaptureAway?: () => Promise<void>;
    inferenceResult?: InferenceResult | null;
    isTraining?: boolean;
    trainingProgress?: number;
    isTrainingPipeline?: boolean;
    isSystemReady?: boolean;
    onUndo?: () => void;
    canUndo?: boolean;
    lastAction?: CaptureAction | null;
    trackingPaused?: boolean;
    onToggleTracking?: () => void;
    toggleTrackingDisabled?: boolean;
  }

  let {
    children,
    hasModel,
    previewFrame,
    frames = [],
    onSaveFrameAsGood,
    onSaveFrameAsBad,
    onSaveFrameAsAway,
    onFramePreview,
    onFramePreviewClear,
    isPanelCollapsed = true,
    queuedFrameCount = 0,
    onFrameListHoverStart,
    onFrameListHoverEnd,
    onCaptureGood,
    onCaptureBad,
    onCaptureAway,
    inferenceResult,
    isTraining = false,
    trainingProgress = 0,
    isTrainingPipeline = false,
    isSystemReady = true,
    onUndo,
    canUndo = false,
    lastAction = null,
    trackingPaused = false,
    onToggleTracking,
    toggleTrackingDisabled = false,
  }: CameraViewportProps = $props();

  const cameraContext = useCameraContext();
  const activeInferenceResult = $derived(inferenceResult ?? cameraContext.inferenceResult);
  const classification = $derived(activeInferenceResult?.classification ?? undefined);

  let isPreviewVisible = $state(false);
  let displayedPreviewFrame = $state<{
    blobUrl: string;
    label?: FrameLabel;
  } | null>(null);

  $effect(() => {
    if (previewFrame) {
      displayedPreviewFrame = previewFrame;
      const timer = setTimeout(() => (isPreviewVisible = true), 10);
      return () => clearTimeout(timer);
    }

    isPreviewVisible = false;
    const timer = setTimeout(() => (displayedPreviewFrame = null), 200);
    return () => clearTimeout(timer);
  });

  const previewMaxWidth = 800;
  const previewMaxHeight = '50%';

  function labelBadgeColor(label: FrameLabel): string {
    switch (label) {
      case FrameLabel.GOOD:
        return '#40c057';
      case FrameLabel.BAD:
        return '#fa5252';
      case FrameLabel.AWAY:
        return '#868e96';
      default:
        return '#25262b';
    }
  }
</script>

<div class="viewport">
  {@render children()}

  {#if trackingPaused}
    <div class="paused-overlay" aria-hidden="true">
      <div class="paused-card">
        <svg width="40" height="40" viewBox="0 0 24 24" aria-hidden="true">
          <rect x="7" y="5" width="4" height="14" rx="1" fill="currentColor" />
          <rect x="13" y="5" width="4" height="14" rx="1" fill="currentColor" />
        </svg>
        <span class="paused-title">Tracking paused</span>
        <span class="paused-hint">Press Resume to continue</span>
      </div>
    </div>
  {/if}

  {#if onToggleTracking}
    <div class="tracking-toggle-position">
      <TrackingToggleButton
        paused={trackingPaused}
        disabled={toggleTrackingDisabled}
        onToggle={onToggleTracking}
      />
    </div>
  {/if}

  {#if onSaveFrameAsGood && onSaveFrameAsBad && onSaveFrameAsAway}
    <FrameListOverlay
      {frames}
      onSaveAsGood={onSaveFrameAsGood}
      onSaveAsBad={onSaveFrameAsBad}
      onSaveAsAway={onSaveFrameAsAway}
      {onFramePreview}
      {onFramePreviewClear}
      {queuedFrameCount}
      onHoverStart={onFrameListHoverStart}
      onHoverEnd={onFrameListHoverEnd}
    />
  {/if}

  {#if onUndo && canUndo}
    <div class="undo-position">
      <UndoButton
        onUndo={onUndo}
        {canUndo}
        {lastAction}
      />
    </div>
  {/if}

  {#if onCaptureGood && onCaptureBad && onCaptureAway}
    <CaptureButtonsOverlay
      onCaptureGood={onCaptureGood}
      onCaptureBad={onCaptureBad}
      onCaptureAway={onCaptureAway}
      disabled={!isSystemReady}
      inferenceResult={activeInferenceResult}
    />
  {/if}

  <div
    class="status-position"
    class:panel-open={!isPanelCollapsed}
  >
    <PostureStatusBadge
      data={classification}
      {hasModel}
    />

    {#if isTraining || isTrainingPipeline}
      <div class="training-badge" role="status" aria-label="Training in progress">
        <span class="training-spinner" aria-hidden="true"></span>
        Training...
      </div>
    {/if}
  </div>

  {#if displayedPreviewFrame}
    <div
      class:visible={isPreviewVisible}
      class="preview-overlay"
      class:panel-open={!isPanelCollapsed}
      style:pointer-events={isPreviewVisible ? 'auto' : 'none'}
    >
      <div
        class="preview-content"
        style={`max-width: ${previewMaxWidth}px; height: ${previewMaxHeight};`}
      >
        <img
          class="preview-image"
          src={displayedPreviewFrame.blobUrl}
          alt="Frame preview"
        />
        {#if displayedPreviewFrame.label && displayedPreviewFrame.label !== FrameLabel.UNUSED}
          <span
            class="preview-label"
            style={`--badge-color: ${labelBadgeColor(displayedPreviewFrame.label)};`}
          >
            {displayedPreviewFrame.label === FrameLabel.GOOD ? 'Good Posture' : 'Bad Posture'}
          </span>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .viewport {
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #000;
  }

  .undo-position {
    position: absolute;
    top: 16px;
    left: 184px;
    z-index: 40;
  }

  .tracking-toggle-position {
    position: absolute;
    top: 16px;
    left: 50%;
    z-index: 70;
    transform: translateX(-50%);
  }

  /* Calm, informational paused state — deliberately not styled as an error. */
  .paused-overlay {
    position: absolute;
    inset: 0;
    z-index: 45;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }

  .paused-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 20px 28px;
    border-radius: 16px;
    color: rgb(255 255 255 / 82%);
    background: rgb(0 0 0 / 55%);
    box-shadow: 0 4px 24px rgb(0 0 0 / 45%);
    backdrop-filter: blur(4px);
    text-align: center;
  }

  .paused-title {
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.2;
  }

  .paused-hint {
    color: rgb(255 255 255 / 60%);
    font-size: 0.8rem;
    line-height: 1.2;
  }

  .status-position {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 160px;
    transition: right 0.3s ease-in-out;
  }

  .status-position.panel-open {
    right: 592px;
  }

  .training-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    padding: 4px 8px;
    border-radius: 4px;
    color: #fff;
    background: #fd7e14;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.25;
    opacity: 0.9;
  }

  .training-spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid rgb(255 255 255 / 35%);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .preview-overlay {
    position: absolute;
    inset: 0;
    padding-left: 176px;
    padding-right: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgb(0 0 0 / 75%);
    opacity: 0;
    transition: opacity 0.2s ease-in-out;
  }

  .preview-overlay.visible {
    opacity: 1;
  }

  .preview-overlay.panel-open {
    padding-right: 576px;
  }

  .preview-content {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
  }

  .preview-image {
    display: block;
    max-width: 100%;
    /* Bound the height to the preview box (set via inline height on
       .preview-content) so object-fit: contain has a box to work against.
       Without an explicit height the img keeps its intrinsic aspect, object-fit
       becomes a no-op, and the frame overflows and anchors to a corner instead
       of centering in the panel. */
    max-height: 100%;
    border-radius: var(--mantine-radius-md, 8px);
    object-fit: contain;
    box-shadow: 0 4px 25px rgb(0 0 0 / 55%);
  }

  .preview-label {
    position: absolute;
    top: 16px;
    right: 16px;
    padding: 6px 12px;
    border-radius: 4px;
    color: #fff;
    background: var(--badge-color);
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.25;
  }


  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
