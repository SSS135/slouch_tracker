<script lang="ts">
  import type { BoundingBox, InferenceUiResult } from '@generated/bindings';
  import { useNativeCamera } from '@/hooks/useNativeCamera.svelte';
  import {
    useCanvasRenderer,
    type CanvasRefObject,
    type ImageRefObject,
  } from '@/hooks/useCanvasRenderer.svelte';
  import { useWindowAspect } from '@/hooks/useWindowAspect.svelte';
  import type { PreviewFrameSource } from '@/services/dataset/thumbnailGenerator';
  import {
    drawHumanLikeSkeleton,
    drawKeypointOverlay,
    drawDetectionBox,
    type SmoothedKeypoint,
    type Keypoint,
  } from '@/utils/canvasDrawing';
  import { logger } from '@/services/logging';
  const KEYPOINT_DRAW_THRESHOLD = 0.3;

  // Custom Tauri protocol serving preview frames. Windows uses the
  // `http://<scheme>.localhost` form; macOS/Linux use `<scheme>://localhost`.
  // `/frame` is the raw ~30fps feed; `/processed` is the detector-input frame
  // (post CLAHE/blur/smoothing), refreshed at detection rate.
  const FRAME_BASE =
    typeof navigator !== 'undefined' && navigator.userAgent.includes('Windows')
      ? 'http://slouchcam.localhost'
      : 'slouchcam://localhost';
  const FRAME_URL = `${FRAME_BASE}/frame`;
  const PROCESSED_FRAME_URL = `${FRAME_BASE}/processed`;
  // Tile-accumulation heatmap of the processed feed (green = averaging, red = live);
  // shares /processed's demand-stamp contract, so it drives the same processed refresh.
  const DEBUG_TILES_FRAME_URL = `${FRAME_BASE}/debug-tiles`;
  // Detector-input frame served without stamping processed-view demand: stays the
  // inferred frame at detection cadence, for the diagnostic detection overlay.
  const INFERRED_FRAME_URL = `${FRAME_BASE}/inferred`;

  interface InferenceResultForDrawing {
    keypoints: Keypoint[];
  }

  export interface PostureCameraProps {
    onInferenceResult: (result: InferenceUiResult | null) => void;
    onFps: (fps: number) => void;
    onCanvasReady?: (ready: boolean) => void;
    canvasRef?: CanvasRefObject;
    latestFrameRef?: { current: PreviewFrameSource | null };
    /** Ref object a parent can call to restart the native camera (e.g. after a camera-index change). */
    cameraRestartRef?: { current: (() => Promise<void>) | null };
    paused?: boolean;
    privacyMode?: boolean;
    /** Show the preprocessed detector-input feed instead of the raw feed. */
    processedView?: boolean;
    /** In the processed view, show the tile-accumulation heatmap instead of the plain processed feed. */
    preprocessingDebugView?: boolean;
    /** Draw the diagnostic skeleton + detection box (with confidence) over the live video. */
    showDetectionOverlay?: boolean;
    // Fired when the bare video area (outside every overlay control) is clicked.
    onBackgroundClick?: () => void;
    /** Reports the native camera start error (null when clear) so a parent can gate a resume control. */
    onCameraError?: (error: string | null) => void;
  }

  let {
    onInferenceResult,
    onFps,
    onCanvasReady,
    canvasRef: externalCanvasRef,
    latestFrameRef,
    cameraRestartRef,
    paused = false,
    privacyMode = false,
    processedView = false,
    preprocessingDebugView = false,
    showDetectionOverlay = false,
    onBackgroundClick,
    onCameraError,
  }: PostureCameraProps = $props();

  let canvasElement = $state<HTMLCanvasElement | null>(null);
  let imgElement = $state<HTMLImageElement | null>(null);
  let frameWidth = $state(0);
  let frameHeight = $state(0);

  const internalCanvasRef: CanvasRefObject = {
    get current() {
      return canvasElement;
    },
    set current(value: HTMLCanvasElement | null) {
      canvasElement = value;
      if (externalCanvasRef) externalCanvasRef.current = value;
    },
  };

  const internalImgRef: ImageRefObject = {
    get current() {
      return imgElement;
    },
    set current(value: HTMLImageElement | null) {
      imgElement = value;
    },
  };

  const lastResult = $state<{ current: InferenceResultForDrawing | null }>({ current: null });
  const detectionBbox = $state<{ current: BoundingBox | null }>({ current: null });
  // Monotonic per-result counter. The renderer refreshes the inferred-frame
  // display + overlay only when this changes, stepping the video at detection
  // cadence in overlay mode. Plain counter (polled by the renderer, not rendered).
  let detectionSeq = 0;
  const smoothedKeypoints = $state<{ current: SmoothedKeypoint[] }>({ current: [] });
  const targetKeypoints = $state<{ current: Keypoint[] }>({ current: [] });
  const lastRenderTime = $state({ current: Date.now() });
  const lastDetectionTime = $state({ current: Date.now() });
  const detectionInterval = $state({ current: 500 });

  function calculateSmoothingAlpha(deltaTimeMs: number, smoothTimeMs: number): number {
    if (smoothTimeMs <= 0) return 1;
    return Math.min(1, deltaTimeMs / smoothTimeMs);
  }

  function interpolateKeypoints(
    previous: SmoothedKeypoint[],
    next: Keypoint[],
    alpha: number,
    confidenceThreshold = KEYPOINT_DRAW_THRESHOLD,
  ): SmoothedKeypoint[] {
    if (previous.length === 0) {
      return next.map((keypoint) => ({
        ...keypoint,
        opacity: keypoint.score > confidenceThreshold ? 1 : 0,
      }));
    }

    if (next.length === 0) {
      return previous.map((keypoint) => ({
        ...keypoint,
        opacity: keypoint.opacity * (1 - alpha),
      }));
    }

    return next.map((nextKeypoint, index) => {
      const previousKeypoint = previous[index] ?? { ...nextKeypoint, opacity: 0 };
      const targetOpacity = nextKeypoint.score > confidenceThreshold ? 1 : 0;
      return {
        x: previousKeypoint.x * (1 - alpha) + nextKeypoint.x * alpha,
        y: previousKeypoint.y * (1 - alpha) + nextKeypoint.y * alpha,
        score: nextKeypoint.score,
        opacity: previousKeypoint.opacity * (1 - alpha) + targetOpacity * alpha,
      };
    });
  }

  function handleResult(result: InferenceUiResult): void {
    const now = Date.now();
    detectionInterval.current = now - lastDetectionTime.current;
    lastDetectionTime.current = now;

    const keypoints = (result.keypoints ?? []).map((point) => ({
      x: point.x ?? 0,
      y: point.y ?? 0,
      score: point.score ?? 0,
    }));
    targetKeypoints.current = result.personFound ? keypoints : [];
    if (result.personFound) {
      lastResult.current = { keypoints };
    }
    detectionBbox.current = result.personFound ? (result.bbox?.original ?? null) : null;
    detectionSeq += 1;
    onInferenceResult(result);
  }

  const camera = useNativeCamera({
    onResult: handleResult,
    get enabled() {
      return !paused;
    },
  });

  $effect(() => {
    onFps(camera.detectionFps);
  });

  $effect(() => {
    if (camera.error) {
      lastResult.current = null;
      targetKeypoints.current = [];
      onInferenceResult(null);
    }
  });

  $effect(() => {
    onCameraError?.(camera.error);
  });

  $effect(() => {
    if (!cameraRestartRef) return;
    cameraRestartRef.current = () => camera.retry();
    return () => {
      cameraRestartRef.current = null;
    };
  });

  $effect(() => {
    return () => {
      lastResult.current = null;
    };
  });

  // Keep the native window content area matched to the live frame aspect ratio
  // (no-op outside the Tauri runtime).
  useWindowAspect({
    get cameraWidth() {
      return frameWidth;
    },
    get cameraHeight() {
      return frameHeight;
    },
  });

  function handleDraw(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
    if (!lastResult.current || (!privacyMode && !showDetectionOverlay)) return;

    if (privacyMode) {
      // End-user human-like avatar with temporal smoothing (unchanged).
      const now = Date.now();
      const deltaTime = now - lastRenderTime.current;
      lastRenderTime.current = now;

      if (
        smoothedKeypoints.current.length === 0 &&
        targetKeypoints.current.length === 0 &&
        lastResult.current.keypoints.length > 0
      ) {
        targetKeypoints.current = lastResult.current.keypoints.map((keypoint) => ({ ...keypoint }));
        smoothedKeypoints.current = lastResult.current.keypoints.map((keypoint) => ({
          ...keypoint,
          opacity: keypoint.score > KEYPOINT_DRAW_THRESHOLD ? 1 : 0,
        }));
      }

      smoothedKeypoints.current = interpolateKeypoints(
        smoothedKeypoints.current,
        targetKeypoints.current,
        calculateSmoothingAlpha(deltaTime, detectionInterval.current),
      );

      drawHumanLikeSkeleton(
        ctx,
        smoothedKeypoints.current,
        canvas.width,
        canvas.height,
        frameWidth || canvas.width,
        frameHeight || canvas.height,
        {
          color: '#4dabf7',
          fillOpacity: 0.8,
          noseColor: '#ffa94d',
          earColor: '#ffa94d',
        },
      );
    } else {
      // Diagnostic detection overlay: raw keypoint dots + COCO skeleton lines,
      // exactly as the latest detection produced them (no smoothing). targetKeypoints
      // holds the latest raw keypoints, or [] when no person is currently detected.
      drawKeypointOverlay(ctx, targetKeypoints.current, canvas.width, canvas.height, {
        color: '#00e676',
      });
    }

    if (showDetectionOverlay && detectionBbox.current) {
      drawDetectionBox(ctx, detectionBbox.current, canvas.width, canvas.height, { color: '#4dabf7' });
    }
  }

  const renderer = useCanvasRenderer({
    frameUrl: FRAME_URL,
    processedFrameUrl: PROCESSED_FRAME_URL,
    debugTilesFrameUrl: DEBUG_TILES_FRAME_URL,
    get enabled() {
      return !paused;
    },
    get onDraw() {
      return handleDraw;
    },
    get privacyMode() {
      return privacyMode;
    },
    get processedView() {
      return processedView;
    },
    get preprocessingDebugView() {
      return preprocessingDebugView;
    },
    get showDetectionOverlay() {
      return showDetectionOverlay;
    },
    inferredFrameUrl: INFERRED_FRAME_URL,
    detectionSequence: () => detectionSeq,
    canvasRef: internalCanvasRef,
    imgRef: internalImgRef,
    get latestFrameRef() {
      return latestFrameRef;
    },
    onFrameSize: (width, height) => {
      frameWidth = width;
      frameHeight = height;
    },
  });

  $effect(() => {
    onCanvasReady?.(renderer.isCanvasReady);
  });
</script>

<div class="camera-container">
  {#if camera.error}
    <div class="alert alert-error inference-error" role="alert">
      <strong>Camera unavailable</strong>
      <span>{camera.error}</span>
      <button
        type="button"
        onclick={() => {
          void camera
            .retry()
            .catch((cause: unknown) => logger.error('detection', 'Camera retry failed:', cause));
        }}
      >
        Retry camera
      </button>
    </div>
  {/if}

  <!-- Clicking the bare video collapses an open control panel. The capture
       buttons, undo, status badge, preview and toggle live in CameraViewport as
       siblings on top of this container, so their clicks never reach it; only a
       click on the empty video area does. Pointer-only affordance — the toggle
       button stays the keyboard-accessible control. -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="canvas-container" onclick={() => onBackgroundClick?.()}>
    <!-- Video layer: the real feed decoded + GPU-composited by the browser
         (privacy off). Hidden in privacy mode, where the overlay shows the grid. -->
    <img
      bind:this={imgElement}
      class="video-layer"
      class:hidden={privacyMode}
      alt=""
      draggable="false"
      aria-hidden="true"
    />
    <!-- Overlay layer: transparent skeleton/bbox over the video; opaque privacy grid. -->
    <canvas bind:this={canvasElement} class="overlay-canvas"></canvas>
  </div>
</div>

<style>
  .camera-container {
    position: relative;
    width: 100%;
    height: 100%;
    background-color: #10181f;
  }

  .alert {
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border: 1px solid transparent;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  .alert strong {
    font-weight: 700;
  }

  .alert span {
    flex: 1;
  }

  .alert-error {
    color: #c92a2a;
    background: #fff5f5;
    border-color: #ffa8a8;
  }

  .alert button {
    padding: 4px 8px;
    border: 0;
    border-radius: 4px;
    color: #fff;
    background: #c92a2a;
    cursor: pointer;
    font-size: 0.75rem;
  }

  .canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
    /* Keep the preview layers on their own compositing group. */
    isolation: isolate;
  }

  .video-layer,
  .overlay-canvas {
    position: absolute;
    inset: 0;
    display: block;
    /* Cover-fit fills the window edge-to-edge; any transient aspect mismatch
       (during a window snap or min-size clamp) shows as an invisible crop
       sliver instead of letterbox/pillarbox bars. Both layers share the frame
       aspect and object-fit, so the skeleton overlay crops and scales in
       lockstep with the video frame and stays aligned. */
    width: 100%;
    height: 100%;
    object-fit: cover;
    /* Promote each layer to a dedicated GPU compositing layer so per-frame
       updates are composited, not rastered on the main thread, and never force
       the slide-in panel's backdrop-filter to re-raster (parity with the old
       native <video>). */
    transform: translateZ(0);
    will-change: transform;
  }

  .video-layer.hidden {
    display: none;
  }
</style>
