import {
  createMutation,
  createQuery,
  useQueryClient,
  type QueryClient,
} from '@tanstack/svelte-query';
import type { DatasetPage, FrameLabel } from '@generated/bindings';
import { nativeClient, type NativeClient } from '../native/client';

export const datasetQueryKeys = {
  all: ['native-dataset'] as const,
  pages: () => [...datasetQueryKeys.all, 'page'] as const,
  page: (offset: number, limit: number) => [...datasetQueryKeys.pages(), offset, limit] as const,
  stats: () => [...datasetQueryKeys.all, 'stats'] as const,
};

type PageSnapshot = Array<[readonly unknown[], DatasetPage | undefined]>;

function snapshotPages(queryClient: QueryClient): PageSnapshot {
  return queryClient.getQueriesData<DatasetPage>({
    queryKey: datasetQueryKeys.pages(),
  });
}

function restorePages(queryClient: QueryClient, snapshot: PageSnapshot | undefined): void {
  for (const [queryKey, page] of snapshot ?? []) {
    queryClient.setQueryData(queryKey, page);
  }
}

function updateFrame(
  queryClient: QueryClient,
  id: string,
  update: (page: DatasetPage) => DatasetPage,
): void {
  queryClient.setQueriesData<DatasetPage>(
    { queryKey: datasetQueryKeys.pages() },
    (page) => page && page.frames.some((frame) => frame.id === id) ? update(page) : page,
  );
}

export function createDatasetQueries(
  offset: () => number,
  limit: () => number,
  client: NativeClient = nativeClient,
) {
  const queryClient = useQueryClient();

  const page = createQuery(() => ({
    queryKey: datasetQueryKeys.page(offset(), limit()),
    queryFn: () => client.getDatasetPage(offset(), limit()),
  }));

  const stats = createQuery(() => ({
    queryKey: datasetQueryKeys.stats(),
    queryFn: () => client.getDatasetStats(),
  }));

  const relabel = createMutation<void, Error, { id: string; label: FrameLabel }, { snapshot: PageSnapshot }>(() => ({
    mutationFn: ({ id, label }: { id: string; label: FrameLabel }) =>
      client.updateFrameLabel(id, label),
    onMutate: async ({ id, label }: { id: string; label: FrameLabel }) => {
      await queryClient.cancelQueries({ queryKey: datasetQueryKeys.pages() });
      const snapshot = snapshotPages(queryClient);
      updateFrame(queryClient, id, (current) => ({
        ...current,
        frames: current.frames.map((frame) => frame.id === id ? { ...frame, label } : frame),
      }));
      return { snapshot };
    },
    onError: (_error: unknown, _variables: unknown, context: { snapshot: PageSnapshot } | undefined) => {
      restorePages(queryClient, context?.snapshot);
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetQueryKeys.all });
    },
  }));

  const remove = createMutation<void, Error, string, { snapshot: PageSnapshot }>(() => ({
    mutationFn: (id: string) => client.deleteFrame(id),
    onMutate: async (id: string) => {
      await queryClient.cancelQueries({ queryKey: datasetQueryKeys.pages() });
      const snapshot = snapshotPages(queryClient);
      updateFrame(queryClient, id, (current) => ({
        ...current,
        total: Math.max(0, current.total - 1),
        frames: current.frames.filter((frame) => frame.id !== id),
      }));
      return { snapshot };
    },
    onError: (_error: unknown, _variables: unknown, context: { snapshot: PageSnapshot } | undefined) => {
      restorePages(queryClient, context?.snapshot);
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetQueryKeys.all });
    },
  }));

  const undo = createMutation<void, Error, void>(() => ({
    mutationFn: () => client.undoLastDatasetChange(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetQueryKeys.all });
    },
  }));

  const reset = createMutation(() => ({
    mutationFn: () => client.resetDataset(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetQueryKeys.all });
    },
  }));

  const exportDataset = createMutation(() => ({
    mutationFn: () => client.exportDataset(),
  }));

  const importDataset = createMutation(() => ({
    mutationFn: () => client.importDataset(),
    onSuccess: async (summary) => {
      if (summary) {
        await queryClient.invalidateQueries({ queryKey: datasetQueryKeys.all });
      }
    },
  }));

  return {
    page,
    stats,
    relabel,
    remove,
    undo,
    reset,
    exportDataset,
    importDataset,
  };
}
