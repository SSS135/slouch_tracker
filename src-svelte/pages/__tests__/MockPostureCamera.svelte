<script lang="ts">
  import { onMount } from 'svelte';
  import type { InferenceUiResult } from '@generated/bindings';

  interface Props {
    onInferenceResult: (result: InferenceUiResult | null) => void;
    onFps: (fps: number) => void;
    onCanvasReady?: (ready: boolean) => void;
    // Accepted but ignored: the mock has no native camera to restart.
    cameraRestartRef?: { current: (() => Promise<void>) | null };
  }

  let { onInferenceResult, onFps, onCanvasReady }: Props = $props();

  const result = (requestId: number, token: number, goodProbability = 0.8): InferenceUiResult => ({
    requestId,
    token,
    personFound: true,
    bbox: {
      original: {
        x1: 0.1,
        y1: 0.1,
        x2: 0.9,
        y2: 0.9,
        width: 0.8,
        height: 0.8,
        score: 0.95,
      },
      expanded: {
        x1: 0.05,
        y1: 0.05,
        x2: 0.95,
        y2: 0.95,
        width: 0.9,
        height: 0.9,
        score: 0.95,
      },
    },
    keypoints: Array.from({ length: 17 }, (_, index) => ({
      x: 0.2 + index * 0.01,
      y: 0.3 + index * 0.01,
      score: 0.9,
    })),
    classification: { presentProbability: 0.95, goodProbability },
  });

  // Person present but no posture model deployed: presence comes from the RTMDet
  // fallback, goodProbability is null (the runtime produces it only from a loaded
  // posture classifier).
  const presenceOnly = (requestId: number, token: number): InferenceUiResult => ({
    ...result(requestId, token),
    classification: { presentProbability: 0.95, goodProbability: null },
  });

  // Person left the frame: no detection, so the runtime emits no classification.
  const away = (requestId: number, token: number): InferenceUiResult => ({
    ...result(requestId, token),
    personFound: false,
    bbox: null,
    keypoints: null,
    classification: null,
  });

  onMount(() => {
    onCanvasReady?.(true);
    onFps(30);
  });
</script>

<div data-testid="mock-posture-camera">
  <button type="button" onclick={() => onInferenceResult(result(1, 101))}>Load inference A</button>
  <button type="button" onclick={() => onInferenceResult(result(2, 102))}>Load inference B</button>
  <button type="button" onclick={() => onInferenceResult(result(3, 103, 0.2))}>Load bad inference</button>
  <button type="button" onclick={() => onInferenceResult(presenceOnly(4, 104))}>Load presence-only inference</button>
  <button type="button" onclick={() => onInferenceResult(away(5, 105))}>Load away inference</button>
</div>
