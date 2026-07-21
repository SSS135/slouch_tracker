import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { InferenceUiResult } from '@generated/bindings';
import { createMockNativeInferenceResult } from '../../__tests__/utils/mockNativeInferenceResult';
import type { NativeClient } from '../../lib/native/client';
import { FrameLabel } from '../../services/dataset/types';
import { renderCaptureThumbnail } from '../../services/dataset/thumbnailGenerator';
import { STALL_MS, useFrameSampler, type FrameSamplerOptions } from '../useFrameSampler';

const mockRenderCaptureThumbnail = vi.mocked(renderCaptureThumbnail);

vi.mock('../../services/dataset/thumbnailGenerator', () => ({
  renderCaptureThumbnail: vi.fn().mockImplementation(async () => {
    const blob = new Blob(['thumbnail'], { type: 'image/webp' });
    Object.defineProperty(blob, 'arrayBuffer', {
      value: async () => new TextEncoder().encode('thumbnail').buffer,
    });
    return blob;
  }),
}));

function inference(requestId = 7, token = 70): InferenceUiResult {
  return createMockNativeInferenceResult({ requestId, token });
}

function client() {
  return { saveCapture: vi.fn().mockResolvedValue(undefined) };
}

const disposers: Array<() => void> = [];
function mount(options: FrameSamplerOptions) {
  let result!: ReturnType<typeof useFrameSampler>;
  const dispose = $effect.root(() => { result = useFrameSampler(options); });
  disposers.push(dispose);
  flushSync();
  return result;
}

/** Drain the microtask queue so a deferred requestCapture registers its pending intent. */
async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 5; index += 1) await Promise.resolve();
}

const previewFrame = { image: {} as CanvasImageSource, width: 640, height: 480 };

function options(mockClient: ReturnType<typeof client>, result: InferenceUiResult | null = inference()): FrameSamplerOptions {
  return {
    inferenceResult: result,
    getPreviewFrame: () => previewFrame,
    client: mockClient as unknown as NativeClient,
  };
}

beforeEach(() => vi.useFakeTimers({ now: 1_000 }));
afterEach(() => {
  while (disposers.length) disposers.pop()?.();
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe('useFrameSampler native token buffer', () => {
  it('initializes with an empty buffer and is not capturing', () => {
    const mock = client();
    const sampler = mount(options(mock));

    expect(sampler.recentFrames).toEqual([]);
    expect(sampler.isCapturing).toBe(false);
  });

  it('returns null when the inference result is not available', async () => {
    const mock = client();
    const sampler = mount(options(mock, null));

    await expect(sampler.captureFrame()).resolves.toBeNull();
    expect(mockRenderCaptureThumbnail).not.toHaveBeenCalled();
    expect(sampler.recentFrames).toEqual([]);
  });

  it('defaults privacyMode to false when no privacy option is provided', async () => {
    const mock = client();
    const sampler = mount(options(mock));

    await sampler.captureFrame();

    expect(mockRenderCaptureThumbnail).toHaveBeenCalledWith(
      expect.objectContaining({ privacyMode: false }),
    );
  });

  it('buffers only UI data and the opaque request/token pair', async () => {
    const mock = client();
    const sampler = mount(options(mock));
    const frame = await sampler.captureFrame('manual', FrameLabel.GOOD);

    expect(frame).toMatchObject({ requestId: 7, token: 70, label: 'good', timestamp: 1_000 });
    expect(frame).not.toHaveProperty('features');
    expect(frame?.keypoints).toHaveLength(17);
    expect(frame?.bbox).toEqual(inference().bbox);
    expect(frame?.bbox?.original.x1).toBe(0.1);
    expect(frame?.bbox?.original.score).toBeCloseTo(0.95);
    expect(sampler.recentFrames[0].bbox).toEqual(inference().bbox);
  });

  it('sends raw thumbnail bytes and removes a capture only after success', async () => {
    const mock = client();
    const sampler = mount(options(mock));
    const frame = await sampler.captureFrame();
    await sampler.saveFrame(frame!.id, FrameLabel.BAD);

    expect(mock.saveCapture).toHaveBeenCalledWith(expect.any(Uint8Array), expect.objectContaining({
      requestId: 7,
      token: 70,
      label: 'bad',
      mimeType: 'image/webp',
    }));
    expect(sampler.recentFrames).toEqual([]);
  });

  it('retains the same opaque token for a retry after a save failure', async () => {
    const mock = client();
    mock.saveCapture.mockRejectedValueOnce(new Error('temporary storage failure'));
    const sampler = mount(options(mock));
    const frame = await sampler.captureFrame();

    await expect(sampler.saveFrame(frame!.id, FrameLabel.AWAY)).rejects.toThrow('temporary storage failure');
    expect(sampler.recentFrames[0]).toMatchObject({ token: 70, requestId: 7, saveError: 'temporary storage failure' });

    await sampler.saveFrame(frame!.id, FrameLabel.AWAY);
    expect(mock.saveCapture).toHaveBeenCalledTimes(2);
    expect(sampler.recentFrames).toEqual([]);
  });

  it('reserves an opaque token across thumbnail creation and native save', async () => {
    let finishSave!: () => void;
    const mock = client();
    mock.saveCapture.mockReturnValue(new Promise<void>((resolve) => { finishSave = resolve; }));
    const state = $state({ result: inference(7, 70) as InferenceUiResult | null });
    const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });
    const frame = await sampler.captureFrame('manual', FrameLabel.GOOD);
    const save = sampler.saveFrame(frame!.id, FrameLabel.GOOD);

    // A fresh, never-reserved/never-consumed token arrives mid-save: the savePending
    // gate alone must block a new capture, independent of the identity dedup gates.
    state.result = inference(8, 80);
    flushSync();
    expect(sampler.canCapture).toBe(false);
    await expect(sampler.captureFrame('manual', FrameLabel.BAD)).resolves.toBeNull();
    expect(mock.saveCapture).toHaveBeenCalledTimes(1);
    // isCapturing must stay true while the save is still in flight.
    expect(sampler.isCapturing).toBe(true);

    finishSave();
    await save;
    expect(sampler.isCapturing).toBe(false);
  });

  it('allows capture again only after a newer inference token arrives', async () => {
    const mock = client();
    const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
    const sampler = mount({
      ...options(mock),
      get inferenceResult() { return state.result; },
    });
    const first = await sampler.captureFrame();
    await sampler.saveFrame(first!.id);
    expect(sampler.canCapture).toBe(false);
    state.result = inference(2, 22);
    flushSync();
    expect(sampler.canCapture).toBe(true);
  });

  it('enforces the bounded capture buffer', async () => {
    const mock = client();
    const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
    const sampler = mount({
      ...options(mock),
      get inferenceResult() { return state.result; },
      config: { maxBufferSize: 2 },
    });

    await sampler.captureFrame();
    vi.setSystemTime(2_000);
    state.result = inference(2, 22);
    await sampler.captureFrame();
    vi.setSystemTime(3_000);
    state.result = inference(3, 33);
    await sampler.captureFrame();

    expect(sampler.recentFrames.map((frame) => frame.token)).toEqual([22, 33]);
  });

  it('releases an evicted identity so it no longer blocks capture', async () => {
    const mock = client();
    const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
    const sampler = mount({
      ...options(mock),
      get inferenceResult() { return state.result; },
      config: { maxBufferSize: 1 },
    });

    await sampler.captureFrame();
    state.result = inference(2, 22);
    flushSync();
    await sampler.captureFrame();

    // Frame (1,11) was evicted from the buffer; its identity must not linger in
    // the consumed-identity set, so re-presenting it can be captured again.
    state.result = inference(1, 11);
    flushSync();
    expect(sampler.canCapture).toBe(true);
  });

  it('re-enables capture of the current identity after removing its frame', async () => {
    const mock = client();
    const sampler = mount(options(mock));
    const frame = await sampler.captureFrame();

    // Same inference result is still current (detection runs at 1-2 fps), so both
    // the consumed-identity and reserved-identity gates would otherwise block re-capture.
    expect(sampler.canCapture).toBe(false);
    sampler.removeFrame(frame!.id);
    expect(sampler.canCapture).toBe(true);
  });

  it('re-enables capture of the current identity after clearing the buffer', async () => {
    const mock = client();
    const sampler = mount(options(mock));
    await sampler.captureFrame();

    expect(sampler.canCapture).toBe(false);
    sampler.clearFrames();
    expect(sampler.canCapture).toBe(true);
  });

  it('forwards the preview frame, keypoints, and privacy mode to the thumbnail renderer', async () => {
    const mock = client();
    const sampler = mount({
      ...options(mock),
      privacyMode: true,
    });

    await sampler.captureFrame();

    expect(mockRenderCaptureThumbnail).toHaveBeenCalledWith(
      expect.objectContaining({
        privacyMode: true,
        previewFrame,
        keypoints: expect.any(Array),
      }),
    );
    expect(mockRenderCaptureThumbnail.mock.calls[0][0]?.keypoints).toHaveLength(17);
  });

  it('preserves default and explicit labels and supports buffer mutations', async () => {
    const mock = client();
    const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
    const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

    const first = await sampler.captureFrame();
    expect(first?.label).toBe(FrameLabel.UNUSED);
    sampler.updateFrameLabel(first!.id, FrameLabel.GOOD);
    expect(sampler.recentFrames[0].label).toBe(FrameLabel.GOOD);
    sampler.removeFrame(first!.id);
    expect(sampler.recentFrames).toEqual([]);

    state.result = inference(2, 22);
    const second = await sampler.captureFrame('interval', FrameLabel.BAD);
    expect(second?.label).toBe(FrameLabel.BAD);
    sampler.clearFrames();
    expect(sampler.recentFrames).toEqual([]);
  });

  it.each([
    { name: 'sixteen keypoints', mutate: (value: InferenceUiResult) => ({ ...value, keypoints: value.keypoints?.slice(0, 16) }) },
    { name: 'eighteen keypoints', mutate: (value: InferenceUiResult) => ({ ...value, keypoints: [...(value.keypoints ?? []), { x: 0, y: 0, score: 1 }] }) },
    { name: 'null keypoint lane', mutate: (value: InferenceUiResult) => ({ ...value, keypoints: value.keypoints?.map((point, index) => index === 0 ? { ...point, x: null } : point) }) },
    { name: 'NaN keypoint lane', mutate: (value: InferenceUiResult) => ({ ...value, keypoints: value.keypoints?.map((point, index) => index === 0 ? { ...point, x: Number.NaN } : point) }) },
    {
      name: 'infinite bbox lane',
      mutate: (value: InferenceUiResult) => ({
        ...value,
        bbox: value.bbox
          ? {
              ...value.bbox,
              original: { ...value.bbox.original, x1: Number.POSITIVE_INFINITY },
            }
          : null,
      }),
    },
  ])('rejects $name before thumbnail generation', async ({ mutate }) => {
    const mock = client();
    const malformed = mutate(inference()) as InferenceUiResult;
    const sampler = mount(options(mock, malformed));

    await expect(sampler.captureFrame()).resolves.toBeNull();
    expect(mockRenderCaptureThumbnail).not.toHaveBeenCalled();
    expect(sampler.recentFrames).toEqual([]);
  });

  it('never reuses an A to B to A token identity', async () => {
    const mock = client();
    const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
    const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });
    expect(await sampler.captureFrame()).not.toBeNull();
    state.result = inference(2, 22);
    expect(await sampler.captureFrame()).not.toBeNull();
    state.result = inference(1, 11);
    await expect(sampler.captureFrame()).resolves.toBeNull();
  });

  it('creates distinct IDs for distinct tokens captured at the same time', async () => {
    const mock = client();
    const state = $state({ result: inference(7, 70) as InferenceUiResult | null });
    const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });
    const first = await sampler.captureFrame();
    state.result = inference(7, 71);
    const second = await sampler.captureFrame();
    expect(second?.id).not.toBe(first?.id);
  });

  it('does not capture without a complete person result', async () => {
    const mock = client();
    const missingPerson = { ...inference(), personFound: false };
    const sampler = mount(options(mock, missingPerson));
    await expect(sampler.captureFrame()).resolves.toBeNull();
    expect(sampler.recentFrames).toEqual([]);
  });

  describe('stable-button liveness and deferred capture', () => {
    it('stays live after an interval capture consumes the current identity', async () => {
      const mock = client();
      const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
      const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

      expect(sampler.isLive).toBe(true);
      // Auto-capture consumes the current token, closing the per-frame gate...
      await sampler.captureFrame('interval');
      expect(sampler.canCapture).toBe(false);
      // ...but the pipeline stays live, so the capture buttons must not blink disabled.
      expect(sampler.isLive).toBe(true);
    });

    it('defers a request during the consumed gap and fulfils it with the next result exactly once', async () => {
      const mock = client();
      const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
      const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

      await sampler.captureFrame('interval'); // consume (1,11)
      expect(sampler.canCapture).toBe(false);

      let settled = false;
      const request = sampler.requestCapture(FrameLabel.BAD).then((outcome) => { settled = true; return outcome; });
      await flushMicrotasks();
      // Deferred: nothing new captured yet, still waiting for a fresh result.
      expect(settled).toBe(false);
      expect(sampler.recentFrames.map((frame) => frame.token)).toEqual([11]);

      state.result = inference(2, 22);
      flushSync();
      const outcome = await request;
      expect(settled).toBe(true);
      expect(outcome).toEqual({
        status: 'captured',
        frame: expect.objectContaining({ token: 22, label: FrameLabel.BAD }),
      });
      // Exactly one new capture for the fresh token (plus the earlier interval frame).
      expect(sampler.recentFrames.map((frame) => frame.token)).toEqual([11, 22]);
    });

    it('saves a deferred-fulfilled token exactly once and never reuses it', async () => {
      const mock = client();
      const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
      const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

      await sampler.captureFrame('interval'); // consume (1,11)
      const request = sampler.requestCapture(FrameLabel.GOOD);
      await flushMicrotasks();
      state.result = inference(2, 22);
      flushSync();
      const outcome = await request;
      expect(outcome.status).toBe('captured');
      const frame = outcome.status === 'captured' ? outcome.frame : null;

      await sampler.saveFrame(frame!.id, FrameLabel.GOOD);
      expect(mock.saveCapture).toHaveBeenCalledTimes(1);
      expect(mock.saveCapture).toHaveBeenCalledWith(
        expect.any(Uint8Array),
        expect.objectContaining({ token: 22, label: 'good' }),
      );

      // The token stays reserved after saving, so re-presenting it never captures
      // or saves the same one-use token twice.
      state.result = inference(2, 22);
      flushSync();
      await expect(sampler.captureFrame('manual', FrameLabel.GOOD)).resolves.toBeNull();
      expect(mock.saveCapture).toHaveBeenCalledTimes(1);
    });

    it('reports a deferred request as unavailable when the pipeline stalls past the threshold', async () => {
      const mock = client();
      const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
      const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

      await sampler.captureFrame('interval'); // consume (1,11)
      const request = sampler.requestCapture(FrameLabel.BAD);
      await flushMicrotasks();
      expect(sampler.isLive).toBe(true);

      // No fresh result arrives: the pipeline goes stale and releases the intent.
      await vi.advanceTimersByTimeAsync(STALL_MS + 100);
      expect(sampler.isLive).toBe(false);
      await expect(request).resolves.toEqual({ status: 'unavailable' });
    });

    it('marks the pipeline stale after the stall threshold and revives it on a fresh result', async () => {
      const mock = client();
      const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
      const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

      expect(sampler.isLive).toBe(true);
      await vi.advanceTimersByTimeAsync(STALL_MS + 100);
      expect(sampler.isLive).toBe(false);

      state.result = inference(2, 22);
      flushSync();
      expect(sampler.isLive).toBe(true);
    });

    it('applies last-click-wins when a second request supersedes an unfulfilled one', async () => {
      const mock = client();
      const state = $state({ result: inference(1, 11) as InferenceUiResult | null });
      const sampler = mount({ ...options(mock), get inferenceResult() { return state.result; } });

      await sampler.captureFrame('interval'); // consume (1,11)
      const first = sampler.requestCapture(FrameLabel.GOOD);
      await flushMicrotasks();
      const second = sampler.requestCapture(FrameLabel.BAD);
      await flushMicrotasks();

      // The superseded request resolves without error; only the latest label captures.
      await expect(first).resolves.toEqual({ status: 'superseded' });

      state.result = inference(2, 22);
      flushSync();
      const outcome = await second;
      expect(outcome).toEqual({
        status: 'captured',
        frame: expect.objectContaining({ token: 22, label: FrameLabel.BAD }),
      });
      expect(sampler.recentFrames.map((frame) => frame.token)).toEqual([11, 22]);
    });
  });
});
