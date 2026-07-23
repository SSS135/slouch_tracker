import { cleanup, render, waitFor } from '@testing-library/svelte';
import { QueryClient } from '@tanstack/svelte-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DatasetPage, DatasetStats } from '@generated/bindings';
import {
  bindMockNativePersistence,
  createMockNativePersistence,
} from '../../__tests__/utils/mockNativePersistence';
import { FrameLabel } from '../../services/dataset/types';
import type { UseDatasetOperationsResult } from '../useDatasetOperations.svelte';
import { datasetKeys } from '../useDatasetOperations.svelte';
import DatasetOperationsHarness from './DatasetOperationsHarness.svelte';

const native = vi.hoisted(() => ({
  getDatasetStats: vi.fn(),
  getNeedsRetraining: vi.fn(),
  getReservoirMetadata: vi.fn(),
  getDatasetPage: vi.fn(),
  getThumbnail: vi.fn(),
  updateFrameLabel: vi.fn(),
  deleteFrame: vi.fn(),
  undoLastDatasetChange: vi.fn(),
  getUndoStatus: vi.fn(),
  cleanupUnusedFrames: vi.fn(),
  resetDataset: vi.fn(),
  resetAllData: vi.fn(),
  exportDataset: vi.fn(),
  importDataset: vi.fn(),
  onDatasetChanged: vi.fn(),
  onUndoStatusChanged: vi.fn(),
}));

vi.mock('../../lib/native/client', () => ({ nativeClient: native }));

const stats: DatasetStats = {
  total: 2,
  good: 1,
  bad: 1,
  away: 0,
  unused: 0,
  imbalanceRatio: 0,
  hasMinimumFrames: true,
  hasAwayFrames: false,
};

function metadata(id: string, label: 'good' | 'bad' = 'good') {
  return {
    id,
    timestamp: 100,
    label,
    thumbnailMimeType: 'image/webp',
    keypoints: Array.from({ length: 17 }, (_, index) => ({
      x: 0.2 + index * 0.01,
      y: 0.4 + index * 0.01,
      score: 0.9,
    })),
    bbox: {
      x1: 0.1,
      y1: 0.2,
      x2: 0.8,
      y2: 0.9,
      score: 0.95,
      width: 0.7,
      height: 0.7,
    },
  };
}

function page(frames = [metadata('frame-1'), metadata('frame-2', 'bad')], offset = 0, total = frames.length): DatasetPage {
  return {
    frames,
    offset,
    limit: 100,
    total,
    version: 3,
    lastModified: 200,
  } as DatasetPage;
}

let client: QueryClient;
let result: UseDatasetOperationsResult;
let datasetChanged: (() => void) | undefined;
let undoChanged: ((status: { available: boolean; depth: number; nextAction: 'restoreFrame' | null; revision: number }) => void) | undefined;
let unlisten: ReturnType<typeof vi.fn>;
let undoUnlisten: ReturnType<typeof vi.fn>;

async function mountHook(): Promise<UseDatasetOperationsResult> {
  render(DatasetOperationsHarness, {
    props: {
      client,
      onReady: (value) => { result = value; },
    },
  });
  await waitFor(() => expect(result).toBeDefined());
  return result;
}

async function waitForFrames(): Promise<void> {
  await waitFor(() => expect(result.frames.isSuccess).toBe(true));
}

beforeEach(() => {
  client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Number.POSITIVE_INFINITY },
      mutations: { retry: false },
    },
  });
  result = undefined as unknown as UseDatasetOperationsResult;
  datasetChanged = undefined;
  undoChanged = undefined;
  unlisten = vi.fn();
  undoUnlisten = vi.fn();
  const persistence = createMockNativePersistence({
    cameraSettings: {
      cameraWidth: 800,
      cameraHeight: 600,
      captureIntervalSeconds: 0.5,
      autoCaptureEnabled: true,
      autoCaptureIntervalSeconds: 2,
      privacyMode: true,
      claheStrength: 3.5,
      smoothingFrames: 3,
      showDetectionOverlay: false,
    },
    uiSettings: { alertVolume: 0.3, alertDelaySeconds: 5 },
    datasetPage: page(),
    datasetStats: stats,
  });
  bindMockNativePersistence(native, persistence.client);
  native.getNeedsRetraining.mockResolvedValue(false);
  native.getReservoirMetadata.mockResolvedValue({ count: 0, totalSeen: 0, maxSamples: 1000 });
  native.cleanupUnusedFrames.mockResolvedValue(0);
  native.onDatasetChanged.mockImplementation(async (handler: () => void) => {
    datasetChanged = handler;
    return unlisten;
  });
  native.onUndoStatusChanged.mockImplementation(async (handler: typeof undoChanged) => {
    undoChanged = handler;
    return undoUnlisten;
  });
});

afterEach(() => {
  cleanup();
  client.clear();
  vi.clearAllMocks();
});

describe('useDatasetOperations native integration', () => {
  it('fetches native dataset stats', async () => {
    await mountHook();
    await waitFor(() => expect(result.stats.isSuccess).toBe(true));
    expect(result.stats.data).toEqual(stats);
    expect(native.getDatasetStats).toHaveBeenCalledTimes(1);
  });

  it('surfaces native stats errors', async () => {
    const error = new Error('stats unavailable');
    native.getDatasetStats.mockRejectedValue(error);
    await mountHook();
    await waitFor(() => expect(result.stats.isError).toBe(true));
    expect(result.stats.error).toBe(error);
  });

  it('preserves a stats error across observer remounts until an explicit retry', async () => {
    native.getDatasetStats.mockRejectedValueOnce(new Error('stats unavailable'));
    await mountHook();
    await waitFor(() => expect(result.stats.isError).toBe(true));
    expect(native.getDatasetStats).toHaveBeenCalledTimes(1);

    cleanup();
    result = undefined as unknown as UseDatasetOperationsResult;
    await mountHook();
    await waitFor(() => expect(result.stats.isError).toBe(true));
    expect(native.getDatasetStats).toHaveBeenCalledTimes(1);

    const refreshed = await result.stats.refetch();
    expect(refreshed.data).toEqual(stats);
    expect(client.getQueryData(datasetKeys.stats())).toEqual(stats);
    expect(native.getDatasetStats).toHaveBeenCalledTimes(2);
  });

  it('loads dataset frame metadata without eagerly downloading thumbnails', async () => {
    await mountHook();
    await waitForFrames();
    expect(result.frames.data).toHaveLength(2);
    expect(result.frames.data?.[0]).toMatchObject({ id: 'frame-1', label: FrameLabel.GOOD, thumbnailMimeType: 'image/webp' });
    expect(result.frames.data?.[0].thumbnail).toBeUndefined();
    expect(native.getThumbnail).not.toHaveBeenCalled();
  });

  it('loads the entire dataset in a single query without paging controls', async () => {
    await mountHook();
    await waitForFrames();
    expect(result.frames.data?.map((frame) => frame.id)).toEqual(['frame-1', 'frame-2']);
    expect(native.getDatasetPage).toHaveBeenCalledWith(0, 100);
    expect(native.getDatasetPage).toHaveBeenCalledTimes(1);
    expect(native.getThumbnail).not.toHaveBeenCalled();
  });

  it('surfaces native page errors', async () => {
    const error = new Error('page unavailable');
    native.getDatasetPage.mockRejectedValue(error);
    await mountHook();
    await waitFor(() => expect(result.frames.isError).toBe(true));
    expect(result.frames.error).toBe(error);
  });

  it('optimistically relabels a cached frame while native persistence is pending', async () => {
    let resolve!: () => void;
    native.updateFrameLabel.mockReturnValue(new Promise<void>((done) => { resolve = done; }));
    await mountHook();
    await waitForFrames();
    const mutation = result.updateLabel.mutateAsync({ id: 'frame-1', label: FrameLabel.BAD });
    await waitFor(() => expect(client.getQueryData<any[]>(datasetKeys.frames())?.[0].label).toBe(FrameLabel.BAD));
    expect(native.updateFrameLabel).toHaveBeenCalledWith('frame-1', FrameLabel.BAD);
    resolve();
    await mutation;
  });

  it('rolls a relabel back when native persistence fails', async () => {
    native.updateFrameLabel.mockRejectedValue(new Error('relabel failed'));
    await mountHook();
    await waitForFrames();
    await expect(result.updateLabel.mutateAsync({ id: 'frame-1', label: FrameLabel.BAD })).rejects.toThrow('relabel failed');
    await waitFor(() => expect(client.getQueryData<any[]>(datasetKeys.frames())?.[0].label).toBe(FrameLabel.GOOD));
  });

  it('invalidates native dataset queries after relabel settles', async () => {
    await mountHook();
    await waitForFrames();
    const before = native.getDatasetPage.mock.calls.length;
    await result.updateLabel.mutateAsync({ id: 'frame-1', label: FrameLabel.BAD });
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });

  it('optimistically removes a frame while native deletion is pending', async () => {
    let resolve!: () => void;
    native.deleteFrame.mockReturnValue(new Promise<void>((done) => { resolve = done; }));
    await mountHook();
    await waitForFrames();
    const mutation = result.deleteFrame.mutateAsync('frame-1');
    await waitFor(() => expect(client.getQueryData<any[]>(datasetKeys.frames())).toHaveLength(1));
    expect(native.deleteFrame).toHaveBeenCalledWith('frame-1');
    resolve();
    await mutation;
  });

  it('rolls a deletion back when the native command fails', async () => {
    native.deleteFrame.mockRejectedValue(new Error('delete failed'));
    await mountHook();
    await waitForFrames();
    await expect(result.deleteFrame.mutateAsync('frame-1')).rejects.toThrow('delete failed');
    await waitFor(() => expect(client.getQueryData<any[]>(datasetKeys.frames())).toHaveLength(2));
  });

  it('invalidates native dataset queries after deletion settles', async () => {
    await mountHook();
    await waitForFrames();
    const before = native.getDatasetPage.mock.calls.length;
    await result.deleteFrame.mutateAsync('frame-1');
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });

  it('loads native undo availability as the authoritative status', async () => {
    native.getUndoStatus.mockResolvedValue({ available: true, depth: 2, nextAction: 'restoreFrame', revision: 4 });
    await mountHook();
    await waitFor(() => expect(result.canUndo.data).toMatchObject({ available: true, depth: 2, revision: 4 }));
  });

  it('routes undo through the native undo command', async () => {
    await mountHook();
    await waitForFrames();
    await result.updateLabel.mutateAsync({ id: 'frame-1', label: FrameLabel.BAD });
    await expect(result.undo.mutateAsync()).resolves.toBeUndefined();
    expect(native.undoLastDatasetChange).toHaveBeenCalledTimes(1);
  });

  it('invalidates dataset queries after native undo', async () => {
    await mountHook();
    await waitForFrames();
    await result.updateLabel.mutateAsync({ id: 'frame-1', label: FrameLabel.BAD });
    const before = native.getDatasetPage.mock.calls.length;
    await result.undo.mutateAsync();
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });

  it('surfaces native undo errors', async () => {
    native.undoLastDatasetChange.mockRejectedValue(new Error('nothing to undo'));
    await mountHook();
    await expect(result.undo.mutateAsync()).rejects.toThrow('nothing to undo');
  });

  it('routes reset through the native dataset command', async () => {
    await mountHook();
    await result.resetDataset.mutateAsync();
    expect(native.resetDataset).toHaveBeenCalledTimes(1);
  });

  it('routes a complete reset through the distinct native command', async () => {
    await mountHook();
    await result.resetAllData.mutateAsync();
    expect(native.resetAllData).toHaveBeenCalledTimes(1);
  });

  it('invalidates dataset queries after native reset', async () => {
    await mountHook();
    await waitForFrames();
    const before = native.getDatasetPage.mock.calls.length;
    await result.resetDataset.mutateAsync();
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });

  it('surfaces native reset errors', async () => {
    native.resetDataset.mockRejectedValue(new Error('reset failed'));
    await mountHook();
    await expect(result.resetDataset.mutateAsync()).rejects.toThrow('reset failed');
  });

  it('returns the native export dialog summary', async () => {
    await mountHook();
    await expect(result.exportDataset.mutateAsync()).resolves.toMatchObject({ frameCount: 2 });
    expect(native.exportDataset).toHaveBeenCalledTimes(1);
  });

  it('surfaces native export errors', async () => {
    native.exportDataset.mockRejectedValue(new Error('export failed'));
    await mountHook();
    await expect(result.exportDataset.mutateAsync()).rejects.toThrow('export failed');
  });

  it('returns the native import summary and invalidates dataset queries', async () => {
    await mountHook();
    await waitForFrames();
    const before = native.getDatasetPage.mock.calls.length;
    await expect(result.importDataset.mutateAsync()).resolves.toMatchObject({ frameCount: 2 });
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });

  it('routes cleanup through the native command and invalidates dataset queries', async () => {
    native.cleanupUnusedFrames.mockResolvedValue(3);
    await mountHook();
    await waitForFrames();
    const before = native.getDatasetPage.mock.calls.length;
    await expect(result.cleanupUnused.mutateAsync()).resolves.toBe(3);
    expect(native.cleanupUnusedFrames).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });

  it('surfaces native cleanup errors', async () => {
    native.cleanupUnusedFrames.mockRejectedValue(new Error('cleanup failed'));
    await mountHook();
    await expect(result.cleanupUnused.mutateAsync()).rejects.toThrow('cleanup failed');
  });

  it('surfaces the native retraining flag', async () => {
    native.getNeedsRetraining.mockResolvedValue(true);
    await mountHook();
    await waitFor(() => expect(result.needsRetraining.isSuccess).toBe(true));
    expect(result.needsRetraining.data).toBe(true);
  });

  it('defaults to needing retraining when the native check fails', async () => {
    native.getNeedsRetraining.mockRejectedValue(new Error('retraining check failed'));
    await mountHook();
    await waitFor(() => expect(result.needsRetraining.isSuccess).toBe(true));
    expect(result.needsRetraining.data).toBe(true);
  });

  it('surfaces native import errors', async () => {
    native.importDataset.mockRejectedValue(new Error('import failed'));
    await mountHook();
    await expect(result.importDataset.mutateAsync()).rejects.toThrow('import failed');
  });

  it('rejects frame metadata whose bbox score is out of range', async () => {
    const invalid = metadata('frame-oob');
    invalid.bbox.score = 5;
    native.getDatasetPage.mockResolvedValueOnce(page([invalid], 0, 1));
    await mountHook();
    await waitFor(() => expect(result.frames.isError).toBe(true));
    expect(result.frames.error).toBeInstanceOf(RangeError);
    expect((result.frames.error as Error).message).toContain('bbox score must be between 0 and 1');
  });

  it('accepts frame metadata whose keypoint score exceeds one', async () => {
    // Keypoint scores are SimCC activation means, not probabilities: a score > 1 is
    // legitimate on real frames and must load without error.
    const activation = metadata('frame-activation');
    activation.keypoints[0].score = 3.2;
    native.getDatasetPage.mockResolvedValueOnce(page([activation], 0, 1));
    await mountHook();
    await waitForFrames();
    expect(result.frames.isError).toBe(false);
    expect(result.frames.data).toHaveLength(1);
  });

  it('pages through the whole dataset in 100-frame chunks until every frame is loaded', async () => {
    const all = Array.from({ length: 150 }, (_, index) => metadata(`frame-${index + 1}`));
    native.getDatasetPage.mockImplementation(async (offset: number, limit: number) =>
      page(all.slice(offset, offset + limit), offset, all.length));
    await mountHook();
    await waitForFrames();
    expect(result.frames.data).toHaveLength(150);
    expect(result.frames.data?.at(-1)?.id).toBe('frame-150');
    expect(native.getDatasetPage).toHaveBeenNthCalledWith(1, 0, 100);
    expect(native.getDatasetPage).toHaveBeenNthCalledWith(2, 100, 100);
    expect(native.getDatasetPage).toHaveBeenCalledTimes(2);
  });

  it('subscribes to native dataset and undo events and cleans up on unmount', async () => {
    await mountHook();
    await waitFor(() => expect(datasetChanged).toBeTypeOf('function'));
    await waitFor(() => expect(undoChanged).toBeTypeOf('function'));
    await waitFor(() => expect(result.canUndo.isSuccess).toBe(true));
    const changedStatus = { available: true, depth: 1, nextAction: 'restoreFrame' as const, revision: 2 };
    native.getUndoStatus.mockResolvedValue(changedStatus);
    datasetChanged?.();
    undoChanged?.(changedStatus);
    await waitFor(() => expect(native.getDatasetStats.mock.calls.length).toBeGreaterThan(1));
    await waitFor(() => expect(client.getQueryData(datasetKeys.undo())).toMatchObject({ available: true, revision: 2 }));
    cleanup();
    expect(unlisten).toHaveBeenCalledTimes(1);
    expect(undoUnlisten).toHaveBeenCalledTimes(1);
  });

  // BUG 1: a capture (or any dataset mutation) emits `dataset-changed`; the listener
  // must invalidate the frame-page query so the grid reflects the change without a
  // reload. staleTime is Infinity here, so only invalidation can force the refetch.
  it('refetches the visible frame page when a native dataset-changed event fires', async () => {
    await mountHook();
    await waitForFrames();
    await waitFor(() => expect(datasetChanged).toBeTypeOf('function'));
    const before = native.getDatasetPage.mock.calls.length;
    datasetChanged?.();
    await waitFor(() => expect(native.getDatasetPage.mock.calls.length).toBeGreaterThan(before));
  });
});
