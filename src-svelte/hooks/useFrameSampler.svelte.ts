import type { FrameLabel as NativeFrameLabel, InferenceUiResult } from '@generated/bindings';
import { untrack } from 'svelte';
import { SvelteSet } from 'svelte/reactivity';
import { nativeClient, type NativeClient } from '../lib/native/client';
import {
  renderCaptureThumbnail,
  type PreviewFrameSource,
} from '../services/dataset/thumbnailGenerator';
import { FrameLabel } from '../services/dataset/types';
import { logger } from '../services/logging/logger';

export interface CapturedFrame {
  id: string;
  timestamp: number;
  requestId: number;
  token: number;
  thumbnail: Blob;
  keypoints: Array<{ x: number; y: number; score: number }>;
  bbox: InferenceUiResult['bbox'];
  label: FrameLabel;
  saveError?: string;
}

/**
 * Outcome of a stable-button capture request.
 *
 * `captured` carries the buffered frame ready to persist. `unavailable` means no
 * live detection could satisfy the request (now or within the stall window).
 * `superseded` means a newer labelled request replaced this one (single-slot,
 * last-click-wins) so the caller must stay silent - it is not an error.
 */
export type CaptureRequestOutcome =
  | { status: 'captured'; frame: CapturedFrame }
  | { status: 'unavailable' }
  | { status: 'superseded' };

export interface FrameSamplerConfig { maxBufferSize: number; }
export interface FrameSamplerOptions {
  inferenceResult: InferenceUiResult | null;
  /** Supplies the latest decoded preview frame for the real-image thumbnail. */
  getPreviewFrame?: () => PreviewFrameSource | null;
  config?: Partial<FrameSamplerConfig>;
  privacyMode?: boolean;
  client?: NativeClient;
}
export interface FrameSamplerState {
  readonly recentFrames: CapturedFrame[];
  readonly isCapturing: boolean;
  readonly canCapture: boolean;
  /**
   * True while the inference pipeline is streaming fresh results (a valid
   * detection arrived within the stall window). Independent of per-frame token
   * consumption, so the capture buttons stay enabled across auto-capture instead
   * of blinking each interval. Goes false only when results genuinely stall.
   */
  readonly isLive: boolean;
  captureFrame(source?: 'interval' | 'manual', initialLabel?: FrameLabel): Promise<CapturedFrame | null>;
  /**
   * Capture the current detection, or - when its token was already consumed but
   * the pipeline is live - defer a single-slot intent fulfilled by the next
   * inference result ("capture now"). Never double-captures a token.
   */
  requestCapture(label: FrameLabel): Promise<CaptureRequestOutcome>;
  saveFrame(id: string, label?: FrameLabel): Promise<void>;
  clearFrames(): void;
  removeFrame(id: string): void;
  updateFrameLabel(id: string, label: FrameLabel): void;
}

const DEFAULT_CONFIG: FrameSamplerConfig = { maxBufferSize: 20 };
/** No fresh inference result for this long marks the pipeline stale (disables capture). */
export const STALL_MS = 3_000;
const FRAME_LABELS = new Set<string>(Object.values(FrameLabel));

function validIdentity(result: InferenceUiResult): boolean {
  return Number.isSafeInteger(result.requestId)
    && result.requestId >= 0
    && Number.isSafeInteger(result.token)
    && result.token > 0;
}

function validLabel(label: unknown): label is FrameLabel {
  return typeof label === 'string' && FRAME_LABELS.has(label);
}

function validBbox(box: InferenceUiResult['bbox']): box is NonNullable<InferenceUiResult['bbox']> {
  if (!box) return false;
  return [box.original, box.expanded].every((bounds) => {
    const { x1, y1, x2, y2, score, width, height } = bounds;
    return [x1, y1, x2, y2, score, width, height].every(
      (value) => typeof value === 'number' && Number.isFinite(value),
    )
      && x1! <= x2!
      && y1! <= y2!
      && width! >= 0
      && height! >= 0
      && score! >= 0
      && score! <= 1;
  });
}

function validKeypoints(
  keypoints: InferenceUiResult['keypoints'],
): keypoints is NonNullable<InferenceUiResult['keypoints']> {
  // Keypoint scores are SimCC activation means, not probabilities, so values > 1
  // are legitimate on real frames. Only finiteness is required.
  return Boolean(keypoints && keypoints.length === 17 && keypoints.every((point) =>
    typeof point.x === 'number'
      && Number.isFinite(point.x)
      && typeof point.y === 'number'
      && Number.isFinite(point.y)
      && typeof point.score === 'number'
      && Number.isFinite(point.score)));
}

/** A well-formed, person-found detection with a usable opaque token - capturable given a fresh token. */
function hasCapturableShape(result: InferenceUiResult | null): boolean {
  return Boolean(result
    && result.personFound
    && validIdentity(result)
    && validBbox(result.bbox)
    && validKeypoints(result.keypoints));
}

function frameId(timestamp: number, requestId: number, token: number): string {
  return `${timestamp}_${requestId}_${token}`;
}

/** Buffers only opaque native tokens and UI/thumbnail data; features stay in Rust. */
export function useFrameSampler(options: FrameSamplerOptions): FrameSamplerState {
  const client = options.client ?? nativeClient;
  let recentFrames = $state<CapturedFrame[]>([]);
  let capturePending = $state(false);
  let savePending = $state(false);
  let reservedIdentity = $state<string | null>(null);
  const consumedIdentities = new SvelteSet<string>();
  // Starts stale until the first result confirms a streaming pipeline.
  let stalled = $state(true);
  // Single-slot deferred capture intent (last-click-wins).
  let pending: { label: FrameLabel; resolve: (outcome: CaptureRequestOutcome) => void } | null = null;

  const currentIdentity = (): string | null => {
    const result = options.inferenceResult;
    return result && validIdentity(result) ? `${result.requestId}:${result.token}` : null;
  };
  const canCapture = (): boolean => {
    const identity = currentIdentity();
    return Boolean(
      identity &&
      identity !== reservedIdentity &&
      !consumedIdentities.has(identity) &&
      !capturePending &&
      !savePending,
    );
  };
  // Liveness is decoupled from token consumption: a consumed current token still
  // leaves the pipeline live, so the buttons stay enabled instead of blinking.
  const isLive = (): boolean => !stalled && currentIdentity() !== null;

  const settlePending = (outcome: CaptureRequestOutcome): void => {
    const intent = pending;
    pending = null;
    intent?.resolve(outcome);
  };

  const captureFrame = async (
    _source: 'interval' | 'manual' = 'manual',
    initialLabel = FrameLabel.UNUSED,
  ): Promise<CapturedFrame | null> => {
    const result = options.inferenceResult;
    const label = initialLabel ?? FrameLabel.UNUSED;
    if (!result
      || !result.personFound
      || !validIdentity(result)
      || !validBbox(result.bbox)
      || !validKeypoints(result.keypoints)
      || !validLabel(label)) {
      return null;
    }
    const identity = `${result.requestId}:${result.token}`;
    if (!canCapture()) return null;

    reservedIdentity = identity;
    capturePending = true;
    try {
      const keypoints = result.keypoints.map((point) => ({
        x: point.x!,
        y: point.y!,
        score: point.score!,
      }));
      const timestamp = Date.now();
      const thumbnail = await renderCaptureThumbnail({
        privacyMode: options.privacyMode ?? false,
        keypoints,
        previewFrame: options.getPreviewFrame?.() ?? null,
      });
      const frame: CapturedFrame = {
        id: frameId(timestamp, result.requestId, result.token),
        timestamp,
        requestId: result.requestId,
        token: result.token,
        thumbnail,
        keypoints,
        bbox: result.bbox,
        label,
      };
      consumedIdentities.add(identity);
      const max = options.config?.maxBufferSize ?? DEFAULT_CONFIG.maxBufferSize;
      const next = [...recentFrames, frame];
      const dropCount = Math.max(0, next.length - max);
      for (const dropped of next.slice(0, dropCount)) {
        consumedIdentities.delete(`${dropped.requestId}:${dropped.token}`);
      }
      recentFrames = next.slice(dropCount);
      return frame;
    } catch (cause) {
      if (reservedIdentity === identity) reservedIdentity = null;
      logger.error('detection', 'Failed to prepare capture:', cause);
      return null;
    } finally {
      capturePending = false;
    }
  };

  const saveFrame = async (id: string, label?: FrameLabel): Promise<void> => {
    const frame = recentFrames.find((item) => item.id === id);
    if (!frame) throw new Error('Frame not found in capture buffer.');
    if (savePending) throw new Error('A capture is already being saved.');
    const effectiveLabel = label ?? frame.label;
    if (!validLabel(effectiveLabel)) throw new Error('Invalid capture label.');
    savePending = true;
    try {
      await client.saveCapture(new Uint8Array(await frame.thumbnail.arrayBuffer()), {
        requestId: frame.requestId,
        token: frame.token,
        frameId: frame.id,
        timestamp: frame.timestamp,
        label: effectiveLabel as NativeFrameLabel,
        mimeType: frame.thumbnail.type === 'image/png' || frame.thumbnail.type === 'image/jpeg'
          ? frame.thumbnail.type
          : 'image/webp',
      });
      consumedIdentities.delete(`${frame.requestId}:${frame.token}`);
      recentFrames = recentFrames.filter((item) => item.id !== id);
    } catch (cause) {
      const identity = `${frame.requestId}:${frame.token}`;
      if (reservedIdentity === identity) reservedIdentity = null;
      consumedIdentities.delete(identity);
      const message = cause instanceof Error ? cause.message : String(cause);
      recentFrames = recentFrames.map((item) => item.id === id ? { ...item, label: effectiveLabel, saveError: message } : item);
      throw cause;
    } finally {
      savePending = false;
    }
  };

  // Fulfil a deferred intent once a fresh, capturable token is available. Runs on
  // each new inference result; a no-op when there is no pending intent.
  const fulfillPending = async (): Promise<void> => {
    const intent = pending;
    if (!intent || !canCapture()) return;
    const frame = await captureFrame('manual', intent.label);
    if (!frame) return; // token raced away mid-await; a later result retries.
    if (pending !== intent) return; // superseded during await; leave for the newer intent.
    pending = null;
    intent.resolve({ status: 'captured', frame });
  };

  const requestCapture = async (label: FrameLabel): Promise<CaptureRequestOutcome> => {
    const frame = await captureFrame('manual', label);
    if (frame) return { status: 'captured', frame };
    // Immediate capture failed. Defer only when the pipeline is live and the
    // current detection is well-formed - i.e. its token was merely already
    // consumed (the auto-capture gap). Otherwise there is nothing to capture.
    if (!isLive() || !hasCapturableShape(options.inferenceResult)) {
      return { status: 'unavailable' };
    }
    settlePending({ status: 'superseded' }); // single-slot, last-click-wins.
    return new Promise<CaptureRequestOutcome>((resolve) => { pending = { label, resolve }; });
  };

  // Track pipeline liveness and fulfil deferred intents. Re-runs on every fresh
  // inference identity, re-arming the stall timer; if results dry up past the
  // threshold the pipeline is marked stale (buttons disable) and any pending
  // intent is reported unavailable so its awaiting caller is released.
  $effect(() => {
    const identity = currentIdentity();
    if (!identity) return;
    stalled = false;
    untrack(() => { void fulfillPending(); });
    const timer = setTimeout(() => {
      stalled = true;
      settlePending({ status: 'unavailable' });
    }, STALL_MS);
    return () => clearTimeout(timer);
  });
  // Release any awaiting caller if the sampler is torn down mid-wait.
  $effect(() => () => settlePending({ status: 'superseded' }));

  return {
    get recentFrames() { return recentFrames; },
    get isCapturing() { return capturePending || savePending; },
    get canCapture() { return canCapture(); },
    get isLive() { return isLive(); },
    captureFrame,
    requestCapture,
    saveFrame,
    clearFrames: () => { recentFrames = []; consumedIdentities.clear(); reservedIdentity = null; settlePending({ status: 'superseded' }); },
    removeFrame: (id) => {
      const frame = recentFrames.find((item) => item.id === id);
      if (frame) {
        const identity = `${frame.requestId}:${frame.token}`;
        consumedIdentities.delete(identity);
        if (reservedIdentity === identity) reservedIdentity = null;
      }
      recentFrames = recentFrames.filter((item) => item.id !== id);
    },
    updateFrameLabel: (id, label) => {
      if (!validLabel(label)) throw new Error('Invalid capture label.');
      recentFrames = recentFrames.map((frame) => frame.id === id ? { ...frame, label } : frame);
    },
  };
}
