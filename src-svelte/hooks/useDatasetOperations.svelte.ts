import { onMount } from 'svelte';
import { createMutation, createQuery, useQueryClient } from '@tanstack/svelte-query';
import type { DatasetPage, FrameLabel as NativeFrameLabel } from '@generated/bindings';
import { nativeClient } from '../lib/native/client';
import type { FrameLabel, PostureFrame } from '../services/dataset/types';
import { logger } from '../services/logging/logger';

export const datasetKeys = {
  all: ['native-dataset'] as const,
  stats: () => ['native-dataset', 'stats'] as const,
  frames: () => ['native-dataset', 'frames'] as const,
  reservoir: () => ['native-dataset', 'reservoir'] as const,
  retraining: () => ['native-dataset', 'retraining'] as const,
  undo: () => ['native-dataset', 'undo'] as const,
};

function requiredFinite(value: number | null, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${path} must be a finite number.`);
  }
  return value;
}

function frameFromMetadata(metadata: DatasetPage['frames'][number]): PostureFrame {
  const timestamp = requiredFinite(metadata.timestamp, `frame ${metadata.id} timestamp`);
  if (!Number.isSafeInteger(timestamp) || timestamp <= 0) {
    throw new RangeError(`frame ${metadata.id} timestamp must be a positive safe integer.`);
  }
  if (metadata.keypoints.length !== 17) {
    throw new TypeError(`frame ${metadata.id} must contain exactly 17 keypoints.`);
  }
  const keypoints = metadata.keypoints.map((point, index) => {
    // Keypoint scores are SimCC activation means, not probabilities, so values > 1
    // are legitimate on real frames. Only finiteness is required.
    const score = requiredFinite(point.score, `frame ${metadata.id} keypoint ${index} score`);
    return {
      x: requiredFinite(point.x, `frame ${metadata.id} keypoint ${index} x`),
      y: requiredFinite(point.y, `frame ${metadata.id} keypoint ${index} y`),
      score,
    };
  });
  const { x1, y1, x2, y2, width, height } = metadata.bbox;
  const bounds = [x1, y1, x2, y2, metadata.bbox.score, width, height].map((value, index) =>
    requiredFinite(value, `frame ${metadata.id} bbox lane ${index}`));
  if (bounds[4] < 0 || bounds[4] > 1) {
    throw new RangeError(`frame ${metadata.id} bbox score must be between 0 and 1.`);
  }
  if (
    bounds[0] > bounds[2]
    || bounds[1] > bounds[3]
    || bounds[5] < 0
    || bounds[6] < 0
    || Math.abs(bounds[5] - (bounds[2] - bounds[0])) > 1e-9
    || Math.abs(bounds[6] - (bounds[3] - bounds[1])) > 1e-9
  ) {
    throw new RangeError(`frame ${metadata.id} bbox geometry is invalid.`);
  }
  return {
    id: metadata.id,
    timestamp,
    thumbnailMimeType: metadata.thumbnailMimeType,
    keypoints,
    bbox: metadata.bbox,
    label: metadata.label as FrameLabel,
  };
}

export function useDatasetOperations() {
  const queryClient = useQueryClient();
  // get_dataset_page caps `limit` at 100 (Rust MAX_PAGE_SIZE), so page through in
  // 100-frame chunks until the whole dataset is loaded and shown on one page.
  const PAGE_LIMIT = 100;
  onMount(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    let undoUnlisten: (() => void) | undefined;
    void nativeClient.onDatasetChanged(() => {
      void queryClient.invalidateQueries({ queryKey: datasetKeys.all });
    }).then((cleanup) => disposed ? cleanup() : (unlisten = cleanup)).catch((cause: unknown) => {
      logger.error('storage', 'Failed to register dataset change listener:', cause);
    });
    void nativeClient.onUndoStatusChanged((status) => {
      queryClient.setQueryData(datasetKeys.undo(), status);
    }).then((cleanup) => disposed ? cleanup() : (undoUnlisten = cleanup)).catch((cause: unknown) => {
      logger.error('storage', 'Failed to register undo status listener:', cause);
    });
    return () => {
      disposed = true;
      for (const cleanup of [unlisten, undoUnlisten]) {
        try {
          cleanup?.();
        } catch (cause) {
          logger.error('storage', 'Failed to remove dataset listener:', cause);
        }
      }
    };
  });

  const invalidateAll = async (): Promise<void> => {
    await queryClient.invalidateQueries({ queryKey: datasetKeys.all });
  };
  const stats = createQuery(() => ({
    queryKey: datasetKeys.stats(),
    queryFn: () => nativeClient.getDatasetStats(),
    retryOnMount: false,
  }));
  const frames = createQuery(() => ({
    queryKey: datasetKeys.frames(),
    queryFn: async () => {
      const collected: DatasetPage['frames'] = [];
      for (;;) {
        const page = await nativeClient.getDatasetPage(collected.length, PAGE_LIMIT);
        collected.push(...page.frames);
        if (page.frames.length === 0 || collected.length >= page.total) break;
      }
      return collected.map(frameFromMetadata);
    },
  }));
  const reservoir = createQuery(() => ({
    queryKey: datasetKeys.reservoir(),
    queryFn: () => nativeClient.getReservoirMetadata(),
  }));
  const needsRetraining = createQuery(() => ({
    queryKey: datasetKeys.retraining(),
    queryFn: async () => {
      try {
        return await nativeClient.getNeedsRetraining();
      } catch (cause) {
        logger.error('storage', 'Failed to check retraining status:', cause);
        return true;
      }
    },
    staleTime: 0,
  }));
  const canUndo = createQuery(() => ({ queryKey: datasetKeys.undo(), queryFn: () => nativeClient.getUndoStatus() }));
  const updateLabel = createMutation(() => ({
    mutationFn: ({ id, label }: { id: string; label: FrameLabel }) => nativeClient.updateFrameLabel(id, label as NativeFrameLabel),
    onMutate: async ({ id, label }) => {
      const queryKey = datasetKeys.frames();
      await queryClient.cancelQueries({ queryKey });
      const previous = queryClient.getQueryData<PostureFrame[]>(queryKey);
      queryClient.setQueryData<PostureFrame[]>(queryKey, previous?.map((frame) => frame.id === id ? { ...frame, label } : frame));
      return { previous, queryKey };
    },
    onError: (_error, _variables, context) => queryClient.setQueryData(context?.queryKey ?? datasetKeys.frames(), context?.previous),
    onSettled: invalidateAll,
  }));
  const deleteFrame = createMutation(() => ({
    mutationFn: (id: string) => nativeClient.deleteFrame(id),
    onMutate: async (id) => {
      const queryKey = datasetKeys.frames();
      await queryClient.cancelQueries({ queryKey });
      const previous = queryClient.getQueryData<PostureFrame[]>(queryKey);
      queryClient.setQueryData<PostureFrame[]>(queryKey, previous?.filter((frame) => frame.id !== id));
      return { previous, queryKey };
    },
    onError: (_error, _variables, context) => queryClient.setQueryData(context?.queryKey ?? datasetKeys.frames(), context?.previous),
    onSettled: invalidateAll,
  }));
  const undo = createMutation(() => ({ mutationFn: () => nativeClient.undoLastDatasetChange(), onSuccess: invalidateAll }));
  const cleanupUnused = createMutation(() => ({
    mutationFn: () => nativeClient.cleanupUnusedFrames(),
    onSuccess: invalidateAll,
  }));
  const resetDataset = createMutation(() => ({
    mutationFn: () => nativeClient.resetDataset(),
    onSuccess: invalidateAll,
  }));
  const resetAllData = createMutation(() => ({
    mutationFn: () => nativeClient.resetAllData(),
    onSuccess: invalidateAll,
  }));
  const exportDataset = createMutation(() => ({ mutationFn: () => nativeClient.exportDataset() }));
  const importDataset = createMutation(() => ({
    mutationFn: () => nativeClient.importDataset(),
    onSuccess: invalidateAll,
  }));

  return {
    stats,
    frames,
    reservoir,
    needsRetraining,
    canUndo,
    updateLabel,
    deleteFrame,
    undo,
    cleanupUnused,
    resetDataset,
    resetAllData,
    exportDataset,
    importDataset,
    invalidateAll,
    invalidateStats: async (): Promise<void> => {
      await queryClient.invalidateQueries({ queryKey: datasetKeys.stats() });
    },
  };
}

export type UseDatasetOperationsResult = ReturnType<typeof useDatasetOperations>;
