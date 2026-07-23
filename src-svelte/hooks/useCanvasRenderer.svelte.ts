/**
 * Native-decode preview driver for the Rust `slouchcam` camera protocol.
 *
 * The window shows two stacked, GPU-composited layers (see PostureCamera):
 *   - an <img> "video layer" for the real feed (privacy OFF)
 *   - a transparent <canvas> "overlay layer" for the privacy grid + skeleton
 *
 * privacy OFF — the real feed is driven straight into the <img> by swapping its
 * `src` to a cache-busted `slouchcam` URL and chaining the next request on
 * `img.decode()`. The browser fetches, decodes and composites each JPEG natively
 * (no JS `fetch`, no `createImageBitmap`, no full-window 2D-canvas raster), so a
 * per-frame video update never forces the slide-in control panel's
 * `backdrop-filter` to re-raster on the main thread — this is what reclaims the
 * foreground-CPU regression and restores the old native-<video> smoothness. A
 * low-fps sample loop (below) keeps `latestFrameRef` fed for capture thumbnails.
 *
 * privacy ON — the real frame is never shown. A low-fps sample loop fetches +
 * decodes frames and samples a blurred colour grid from them; a separate rAF
 * render loop paints that grid (temporally smoothed) plus the skeleton overlay.
 * Decoupling the expensive fetch/decode/grid-sample (low fps) from the cheap
 * grid-upscale + skeleton render (rAF) keeps the default privacy view as smooth
 * as before while cutting the per-frame JPEG-decode cost.
 *
 * The `slouchcam` protocol is served without CORS headers, so the cross-origin
 * <img> must never be drawn to a canvas (it would taint it and break `toBlob`).
 * `latestFrameRef` therefore always holds an ImageBitmap decoded from a same-origin
 * fetched blob — untainted — so `save_capture` thumbnails keep working in both modes.
 *
 * Both loops run whenever the window is VISIBLE, at a focus-dependent rate: focused
 * → smooth ~30fps; visible but unfocused → only the ~1fps detection frames (nearly
 * free — Rust already captures them for inference); minimized/hidden → stopped. They
 * are also hard-stopped on dispose / `pagehide` / `beforeunload` so a torn-down or
 * backgrounded webview can never keep pulling frames. Detection runs natively either way.
 */

import { sampleImageGrid, type RGB } from '@/utils/colorUtils';
import { renderSmoothedBicubicGrid } from '@/utils/bicubicGridRenderer';
import { MonotonicFrameGate } from '@/utils/frameSequence';
import type { PreviewFrameSource } from '@/services/dataset/thumbnailGenerator';

export interface DrawCallback {
  (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void;
}

export interface CanvasRefObject {
  current: HTMLCanvasElement | null;
}

export interface ImageRefObject {
  current: HTMLImageElement | null;
}

export interface UseCanvasRendererOptions {
  /** The native `slouchcam` frame URL to pull preview frames from. */
  frameUrl: string;
  /** The `slouchcam` processed-frame URL (detector input, updated at detection rate). */
  processedFrameUrl?: string;
  enabled?: boolean;
  onDraw?: DrawCallback;
  privacyMode?: boolean;
  /**
   * Drive the video layer from the processed (detector-input) feed instead of
   * the raw one. Ignored in privacy mode — the grid must keep obscuring the feed.
   */
  processedView?: boolean;
  /**
   * Detection-overlay diagnostic mode (privacy OFF). When on, the video layer
   * shows ONLY the inferred (detector-input) frame, swapped once per detection
   * result (see `detectionSequence`) instead of the ~30fps raw feed, and `onDraw`
   * paints the raw keypoints/bbox over it — so the frame and overlay come from the
   * same detection. When off, the privacy-off path is the unchanged 30fps preview.
   */
  showDetectionOverlay?: boolean;
  /** `slouchcam` inferred-frame URL (detector-input JPEG, served without a demand stamp). */
  inferredFrameUrl?: string;
  /**
   * Monotonic counter incremented once per inference result. In detection-overlay
   * mode the renderer refreshes the inferred frame + overlay only when it changes,
   * so the displayed video steps at detection cadence (~1 fps), not 30 fps.
   */
  detectionSequence?: () => number;
  /** Transparent overlay canvas (skeleton, and the privacy grid). */
  canvasRef?: CanvasRefObject;
  /** Native <img> video layer used for the real feed when privacy is off. */
  imgRef?: ImageRefObject;
  /** Reports the decoded frame dimensions (used to snap the window aspect). */
  onFrameSize?: (width: number, height: number) => void;
  /** Exposes the latest decoded frame so captures can build a thumbnail from it. */
  latestFrameRef?: { current: PreviewFrameSource | null };
}

export interface UseCanvasRendererReturn {
  canvasRef: CanvasRefObject;
  readonly isRendering: boolean;
  readonly isCanvasReady: boolean;
  readonly isForeground: boolean;
}

// Preview rates depend on window state. FOCUSED: smooth ~30fps. UNFOCUSED but
// visible: only the ~1-2fps detection frames (nearly free — Rust already captures
// them for inference), so a glanced-at window still shows a live, if choppy, feed.
// MINIMIZED/hidden: loops stop entirely (see computeVisible + the visibility gate).
// Native <img> refresh cap for the real feed (privacy off). Source is ~30fps.
const VIDEO_FOCUSED_INTERVAL_MS = 1000 / 30;
const VIDEO_UNFOCUSED_INTERVAL_MS = 1000 / 2;
// Cadence of the fetch+decode loop that feeds thumbnails and the privacy grid; the
// expensive JPEG decode + grid sample runs here, the cheap paint runs at rAF.
const SAMPLE_FOCUSED_INTERVAL_MS = 1000 / 3;
const SAMPLE_UNFOCUSED_INTERVAL_MS = 1000;
// Privacy grid + skeleton repaint cap (cheap: tiny-grid upscale + vector skeleton).
const RENDER_FOCUSED_INTERVAL_MS = 1000 / 30;
const RENDER_UNFOCUSED_INTERVAL_MS = 1000 / 2;
// Retry gap after a 204 (no fresh frame) or a decode error.
const NO_FRAME_BACKOFF_MS = 150;
const GRID_SMOOTHING_ALPHA = 0.1;
const GRID_SIZE = 4;

// Whether the preview should render at all: visible (not minimized/hidden). Losing
// focus no longer stops the preview — it only slows it (see computeFocused).
function computeVisible(): boolean {
  if (typeof document === 'undefined') return true;
  return !document.hidden;
}

// Whether the window is focused: drives the preview RATE (fast vs interval-only),
// not whether it renders.
function computeFocused(): boolean {
  if (typeof document === 'undefined') return true;
  return document.hasFocus();
}

export function useCanvasRenderer(
  options: UseCanvasRendererOptions
): UseCanvasRendererReturn {
  const internalCanvasRef = $state<CanvasRefObject>({ current: null });
  const canvasRef = options.canvasRef ?? internalCanvasRef;

  let isRendering = $state(false);
  let isCanvasReady = $state(false);
  let isForeground = $state(computeVisible());

  $effect(() => {
    const enabled = options.enabled ?? true;
    const frameUrl = options.frameUrl;
    const overlay = canvasRef.current;
    const img = options.imgRef?.current ?? null;
    const privacyMode = options.privacyMode ?? false;
    // Reading the toggle here makes a mid-run flip rerun the whole effect, so the
    // loops restart cleanly on the new URL/rate (same pattern as privacyMode).
    // Privacy wins: the processed view must never leak the real image.
    const processedView =
      (options.processedView ?? false) && !privacyMode && Boolean(options.processedFrameUrl);
    const videoUrl = processedView ? options.processedFrameUrl! : frameUrl;
    // Reading the flag here reruns the effect on a flip, so the overlay loop
    // starts/stops cleanly (same pattern as privacyMode/processedView).
    const showDetectionOverlay = options.showDetectionOverlay ?? false;
    // Detector-input frame served without a demand stamp, so it stays the
    // dispatcher-written inferred frame (detection cadence).
    const inferredUrl = options.inferredFrameUrl ?? options.processedFrameUrl ?? frameUrl;
    if (!enabled || !overlay || !frameUrl) {
      return;
    }

    const ctx = overlay.getContext('2d');
    if (!ctx) {
      return;
    }

    isRendering = true;

    let disposed = false;
    // Focus drives the preview RATE (fast when focused, interval-only otherwise),
    // not whether it renders (that is `isForeground`, i.e. window visibility).
    let focused = computeFocused();
    let seq = 0;
    // Bumped whenever the loops are (re)started or stopped; every scheduled
    // continuation checks its captured epoch so a stale chain left over from a
    // focus flap can never double up with the fresh one.
    let epoch = 0;
    let latestFrame: PreviewFrameSource | null = null;
    let targetGrid: RGB[][] | null = null;
    let colorGrid: RGB[][] | null = null;
    let sampleTimer: ReturnType<typeof setTimeout> | null = null;
    let videoTimer: ReturnType<typeof setTimeout> | null = null;
    let rafId: number | null = null;
    let lastRenderAt = 0;

    const setLatestFrame = (frame: PreviewFrameSource): void => {
      const previous = latestFrame;
      latestFrame = frame;
      if (options.latestFrameRef) options.latestFrameRef.current = frame;
      if (previous && previous.image instanceof ImageBitmap && previous.image !== frame.image) {
        previous.image.close();
      }
    };

    // Fetch + decode one frame into an untainted ImageBitmap for thumbnails, and
    // (privacy) sample the blurred colour grid from it. Same-origin blob → the
    // bitmap and any grid readback stay untainted.
    const runSample = async (): Promise<void> => {
      if (disposed || !isForeground) return;
      try {
        const response = await fetch(`${frameUrl}?seq=${seq++}`, { cache: 'no-store' });
        if (disposed) return;
        // 204 = no fresh frame yet (background / not started).
        if (response.status === 204 || !response.ok) return;
        const blob = await response.blob();
        if (disposed || blob.size === 0) return;
        const bitmap = await createImageBitmap(blob);
        if (disposed) {
          bitmap.close();
          return;
        }
        setLatestFrame({ image: bitmap, width: bitmap.width, height: bitmap.height });
        options.onFrameSize?.(bitmap.width, bitmap.height);
        if (privacyMode) {
          targetGrid = sampleImageGrid(bitmap, bitmap.width, bitmap.height, GRID_SIZE);
        }
        // Frames are flowing; the capture pipeline may enable.
        isCanvasReady = true;
      } catch {
        // Network/decoding hiccup: skip this frame, the next tick retries.
      }
    };

    // Size the overlay to the frame aspect, cover-fitted to the window, matching
    // the <img>'s object-fit: cover so the skeleton stays aligned to the video.
    const sizeOverlayToAspect = (frameWidth: number, frameHeight: number): void => {
      if (!frameWidth || !frameHeight) return;
      const aspect = frameHeight / frameWidth;
      let width: number;
      let height: number;
      if (window.innerWidth > window.innerHeight) {
        height = window.innerHeight;
        width = height / aspect;
      } else {
        width = window.innerWidth;
        height = width * aspect;
      }
      if (overlay.width !== width || overlay.height !== height) {
        overlay.width = width;
        overlay.height = height;
      }
      isCanvasReady = true;
    };

    const renderPrivacy = (): void => {
      if (!targetGrid || !latestFrame) return;
      sizeOverlayToAspect(latestFrame.width, latestFrame.height);
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      colorGrid = renderSmoothedBicubicGrid(ctx, overlay, colorGrid, targetGrid, GRID_SMOOTHING_ALPHA);
      options.onDraw?.(ctx, overlay);
    };

    const startAll = (): void => {
      const myEpoch = ++epoch;
      // Both the raw and processed feeds pump at the focus-dependent video rate. A
      // focused processed view pumps at ~30fps: this displays smoothly AND is the
      // demand signal that drives the Rust capture-rate processed-frame refresh
      // (unfocused stays at the slower rate, matching the Rust fallback).
      const videoInterval = focused ? VIDEO_FOCUSED_INTERVAL_MS : VIDEO_UNFOCUSED_INTERVAL_MS;
      const sampleInterval = focused ? SAMPLE_FOCUSED_INTERVAL_MS : SAMPLE_UNFOCUSED_INTERVAL_MS;
      const renderInterval = focused ? RENDER_FOCUSED_INTERVAL_MS : RENDER_UNFOCUSED_INTERVAL_MS;
      const alive = (): boolean => !disposed && isForeground && epoch === myEpoch;

      const sampleTick = (): void => {
        if (!alive()) return;
        void runSample().finally(() => {
          if (alive()) sampleTimer = setTimeout(sampleTick, sampleInterval);
        });
      };

      // Privacy OFF: drive the native <img> from the protocol, chaining on decode.
      // `videoUrl` is the raw feed, or the processed detector-input feed when the
      // processed view is on.
      const pumpVideo = async (): Promise<void> => {
        if (!alive() || !img) return;
        const started = performance.now();
        img.src = `${videoUrl}?seq=${seq++}`;
        try {
          await img.decode();
        } catch {
          // 204 / decode error: back off, then retry.
          if (alive()) videoTimer = setTimeout(() => void pumpVideo(), NO_FRAME_BACKOFF_MS);
          return;
        }
        if (!alive()) return;
        options.onFrameSize?.(img.naturalWidth, img.naturalHeight);
        isCanvasReady = true;
        const wait = Math.max(0, videoInterval - (performance.now() - started));
        videoTimer = setTimeout(() => void pumpVideo(), wait);
      };

      const renderTick = (timestamp: number): void => {
        if (!alive()) {
          rafId = null;
          return;
        }
        rafId = requestAnimationFrame(renderTick);
        if (timestamp - lastRenderAt < renderInterval) return;
        lastRenderAt = timestamp;
        renderPrivacy();
      };

      // Detection-overlay (privacy OFF): show ONLY the inferred (detector-input)
      // frame, swapped once per detection result, with the raw keypoints/bbox
      // painted over it — the shown frame and overlay come from the same
      // detection. No 30fps pump: the video visibly steps at detection cadence.
      const refreshInferredFrame = (): void => {
        if (!alive() || !img) return;
        img.src = `${inferredUrl}?seq=${seq++}`;
        img
          .decode()
          .then(() => {
            if (!alive()) return;
            options.onFrameSize?.(img.naturalWidth, img.naturalHeight);
            isCanvasReady = true;
            sizeOverlayToAspect(img.naturalWidth, img.naturalHeight);
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            options.onDraw?.(ctx, overlay);
          })
          .catch(() => {
            // 204 (no inferred frame yet) or decode error: the next detection retries.
          });
      };

      // Strictly-newer gate: refresh the inferred frame + overlay only when the
      // detection sequence advances, so a stale/reset sequence can never re-drive
      // an older frame over a newer one (mirrors the native ordering guard).
      const overlayGate = new MonotonicFrameGate();
      const overlayTick = (): void => {
        if (!alive()) {
          rafId = null;
          return;
        }
        rafId = requestAnimationFrame(overlayTick);
        const current = options.detectionSequence?.() ?? 0;
        if (overlayGate.admit(current)) {
          refreshInferredFrame();
        }
      };

      if (privacyMode) {
        sampleTick();
        lastRenderAt = 0;
        rafId = requestAnimationFrame(renderTick);
      } else if (showDetectionOverlay) {
        // Feed latestFrameRef for capture thumbnails; the display + overlay are
        // driven by detection results (overlayTick), not a 30fps timer.
        sampleTick();
        rafId = requestAnimationFrame(overlayTick);
      } else {
        void pumpVideo();
        sampleTick();
      }
    };

    const stopAll = (): void => {
      epoch++;
      if (sampleTimer) {
        clearTimeout(sampleTimer);
        sampleTimer = null;
      }
      if (videoTimer) {
        clearTimeout(videoTimer);
        videoTimer = null;
      }
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    };

    const updateForeground = (): void => {
      isForeground = computeVisible();
      focused = computeFocused();
      // Always tear the loops down first so a repeated focus/visibility event can
      // never leave two chains racing (stopAll bumps the epoch either way); a focus
      // change (visible both sides) restarts the loops at the new focus-dependent rate.
      stopAll();
      if (isForeground) startAll();
    };

    // Hard-stop for an abnormal teardown so a leaked webview cannot keep looping.
    const hardStop = (): void => {
      disposed = true;
      stopAll();
      if (img) img.removeAttribute('src');
    };

    window.addEventListener('focus', updateForeground);
    window.addEventListener('blur', updateForeground);
    document.addEventListener('visibilitychange', updateForeground);
    window.addEventListener('pagehide', hardStop);
    window.addEventListener('beforeunload', hardStop);

    startAll();

    return () => {
      disposed = true;
      isRendering = false;
      isCanvasReady = false;
      stopAll();
      window.removeEventListener('focus', updateForeground);
      window.removeEventListener('blur', updateForeground);
      document.removeEventListener('visibilitychange', updateForeground);
      window.removeEventListener('pagehide', hardStop);
      window.removeEventListener('beforeunload', hardStop);
      if (latestFrame && latestFrame.image instanceof ImageBitmap) latestFrame.image.close();
      latestFrame = null;
      if (options.latestFrameRef) options.latestFrameRef.current = null;
      if (img) img.removeAttribute('src');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      overlay.width = 0;
      overlay.height = 0;
    };
  });

  return {
    canvasRef,
    get isRendering() {
      return isRendering;
    },
    get isCanvasReady() {
      return isCanvasReady;
    },
    get isForeground() {
      return isForeground;
    },
  };
}
