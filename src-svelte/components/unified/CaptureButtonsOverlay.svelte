<script lang="ts">
  import type { InferenceResult } from '@/services/dataset/types';
  import AnimatedCaptureButton from './AnimatedCaptureButton.svelte';

  export interface CaptureButtonsOverlayProps {
    onCaptureGood: () => Promise<void>;
    onCaptureBad: () => Promise<void>;
    onCaptureAway: () => Promise<void>;
    disabled?: boolean;
    inferenceResult?: InferenceResult | null;
  }

  let {
    onCaptureGood,
    onCaptureBad,
    onCaptureAway,
    disabled = false,
    inferenceResult,
  }: CaptureButtonsOverlayProps = $props();

  const hasInferenceData = $derived(Boolean(
    inferenceResult?.personFound &&
    Number.isSafeInteger(inferenceResult.token) &&
    inferenceResult.token > 0 &&
    inferenceResult.bbox &&
    inferenceResult.keypoints?.length === 17 &&
    // Keypoint scores are SimCC activation means, not probabilities, so values > 1
    // are legitimate on real frames. Only finiteness is required here, matching
    // useFrameSampler.validKeypoints and the native validate_keypoints check; a
    // score-<=1 gate wrongly disabled every capture button for well-detected people.
    inferenceResult.keypoints.every((keypoint) => {
      const score = keypoint.score;
      return Number.isFinite(keypoint.x) &&
        Number.isFinite(keypoint.y) &&
        typeof score === 'number' &&
        Number.isFinite(score);
    }),
  ));
  const shouldDisable = $derived(disabled || !hasInferenceData);
</script>

<div class="overlay">
  <div class="button-group">
    <AnimatedCaptureButton
      label="Good"
      color="var(--mantine-color-green-6, #40c057)"
      onPress={onCaptureGood}
      disabled={shouldDisable}
    />
    <AnimatedCaptureButton
      label="Bad"
      color="var(--mantine-color-red-6, #fa5252)"
      onPress={onCaptureBad}
      disabled={shouldDisable}
    />
    <AnimatedCaptureButton
      label="Away"
      color="var(--mantine-color-blue-6, #228be6)"
      onPress={onCaptureAway}
      disabled={shouldDisable}
    />
  </div>
</div>

<style>
  .overlay {
    position: absolute;
    bottom: 16px;
    left: 50%;
    z-index: 40;
    padding: 8px 12px;
    background: rgb(0 0 0 / 75%);
    border-radius: 12px;
    box-shadow: 0 4px 16px rgb(0 0 0 / 50%);
    backdrop-filter: blur(6px);
    transform: translateX(-50%);
  }

  .button-group {
    display: flex;
    align-items: center;
    gap: var(--mantine-spacing-sm, 12px);
  }

</style>
