import { vi, type Mock } from 'vitest';
import type {
  ArchiveImportResult_Serialize,
  ArchiveSummaryDto,
  CameraSettings,
  DatasetPage,
  DatasetStats,
  FrameLabel,
  FrameMetadataDto,
  NativeStateSnapshot_Serialize,
  UiSettings,
  UndoStatus,
} from '@generated/bindings';

export interface MockNativePersistenceSeed {
  cameraSettings: CameraSettings;
  uiSettings: UiSettings;
  datasetPage: DatasetPage;
  datasetStats: DatasetStats;
}

export interface MockNativePersistenceClient {
  getCameraSettings: Mock<() => Promise<CameraSettings>>;
  saveCameraSettings: Mock<(settings: CameraSettings) => Promise<void>>;
  resetCameraSettings: Mock<() => Promise<CameraSettings>>;
  getUiSettings: Mock<() => Promise<UiSettings>>;
  saveUiSettings: Mock<(settings: UiSettings) => Promise<void>>;
  resetUiSettings: Mock<() => Promise<UiSettings>>;
  getDatasetPage: Mock<(offset?: number, limit?: number) => Promise<DatasetPage>>;
  getDatasetStats: Mock<() => Promise<DatasetStats>>;
  getThumbnail: Mock<(id: string) => Promise<Uint8Array>>;
  updateFrameLabel: Mock<(id: string, label: FrameLabel) => Promise<void>>;
  deleteFrame: Mock<(id: string) => Promise<void>>;
  undoLastDatasetChange: Mock<() => Promise<void>>;
  getUndoStatus: Mock<() => Promise<UndoStatus>>;
  resetDataset: Mock<() => Promise<NativeStateSnapshot_Serialize>>;
  resetAllData: Mock<() => Promise<NativeStateSnapshot_Serialize>>;
  exportDataset: Mock<() => Promise<ArchiveSummaryDto | null>>;
  importDataset: Mock<() => Promise<ArchiveImportResult_Serialize | null>>;
  onDatasetChanged: Mock<(handler: () => void) => Promise<() => void>>;
  onUndoStatusChanged: Mock<(handler: (status: UndoStatus) => void) => Promise<() => void>>;
}

export interface MockNativePersistence {
  client: MockNativePersistenceClient;
  readCameraSettings(): CameraSettings;
  readUiSettings(): UiSettings;
}

const clone = <T>(value: T): T => structuredClone(value);
const MAX_UNDO_DEPTH = 100;

type UndoEntry = {
  frame: FrameMetadataDto;
  index: number;
  deleted: boolean;
};

function assertPageArguments(offset: number, limit: number): void {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new Error('dataset page offset must be a non-negative integer');
  }
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100) {
    throw new Error('dataset page limit must be between 1 and 100');
  }
}

/**
 * Creates a stateful mocked-Tauri persistence boundary. It models the typed
 * native commands used by settings and dataset tests without browser storage.
 */
export function createMockNativePersistence(seed: MockNativePersistenceSeed): MockNativePersistence {
  const cameraDefaults = clone(seed.cameraSettings);
  const uiDefaults = clone(seed.uiSettings);
  let cameraSettings = clone(cameraDefaults);
  let uiSettings = clone(uiDefaults);
  let frames = clone(seed.datasetPage.frames);
  let datasetTotal = seed.datasetPage.total;
  let datasetVersion = seed.datasetPage.version;
  let lastModified = seed.datasetPage.lastModified;
  let datasetStats = clone(seed.datasetStats);
  const undoEntries: UndoEntry[] = [];
  const datasetHandlers = new Set<() => void>();
  const undoHandlers = new Set<(status: UndoStatus) => void>();

  const currentUndoStatus = (): UndoStatus => ({
    available: undoEntries.length > 0,
    depth: undoEntries.length,
    nextAction: undoEntries.length > 0 ? 'restoreFrame' : null,
    revision: datasetVersion,
  });

  const snapshot = (): NativeStateSnapshot_Serialize => ({
    app: {
      ready: true,
      inferenceReady: true,
      datasetVersion,
      storage: { used: 0, available: 1, quota: 1 },
    },
    cameraSettings: clone(cameraSettings),
    uiSettings: clone(uiSettings),
    trainingSettings: null,
    activeModels: { posture: null, presence: null },
    undo: currentUndoStatus(),
  });

  const emitUndo = (): void => {
    const status = currentUndoStatus();
    for (const handler of undoHandlers) handler(clone(status));
  };

  const commitDatasetMutation = (): void => {
    datasetVersion += 1;
    lastModified = (lastModified ?? 0) + 1;
    emitUndo();
    for (const handler of datasetHandlers) handler();
  };

  const adjustLabelStats = (from: FrameLabel, to: FrameLabel): void => {
    if (from === to) return;
    datasetStats = {
      ...datasetStats,
      [from]: Math.max(0, datasetStats[from] - 1),
      [to]: datasetStats[to] + 1,
    };
  };

  const pushUndo = (entry: UndoEntry): void => {
    undoEntries.push(clone(entry));
    if (undoEntries.length > MAX_UNDO_DEPTH) undoEntries.shift();
  };

  const client: MockNativePersistenceClient = {
    getCameraSettings: vi.fn(async () => clone(cameraSettings)),
    saveCameraSettings: vi.fn(async (settings) => {
      cameraSettings = clone(settings);
    }),
    resetCameraSettings: vi.fn(async () => {
      cameraSettings = clone(cameraDefaults);
      return clone(cameraSettings);
    }),
    getUiSettings: vi.fn(async () => clone(uiSettings)),
    saveUiSettings: vi.fn(async (settings) => {
      uiSettings = clone(settings);
    }),
    resetUiSettings: vi.fn(async () => {
      uiSettings = clone(uiDefaults);
      return clone(uiSettings);
    }),
    getDatasetPage: vi.fn(async (
      offset = seed.datasetPage.offset,
      limit = seed.datasetPage.limit,
    ) => {
      assertPageArguments(offset, limit);
      return {
        frames: clone(frames.slice(offset, offset + limit)),
        offset,
        limit,
        total: datasetTotal,
        version: datasetVersion,
        lastModified,
      };
    }),
    getDatasetStats: vi.fn(async () => clone(datasetStats)),
    getThumbnail: vi.fn(async (id) => {
      if (!frames.some((frame) => frame.id === id)) throw new Error(`frame with id ${id} not found`);
      return new Uint8Array([1, 2, 3]);
    }),
    updateFrameLabel: vi.fn(async (id, label) => {
      const index = frames.findIndex((frame) => frame.id === id);
      if (index < 0) throw new Error(`frame with id ${id} not found`);
      const previous = clone(frames[index]);
      adjustLabelStats(previous.label, label);
      frames[index] = { ...frames[index], label };
      pushUndo({ frame: previous, index, deleted: false });
      commitDatasetMutation();
    }),
    deleteFrame: vi.fn(async (id) => {
      const index = frames.findIndex((frame) => frame.id === id);
      if (index < 0) throw new Error(`frame with id ${id} not found`);
      const [removed] = frames.splice(index, 1);
      datasetTotal -= 1;
      datasetStats = {
        ...datasetStats,
        total: Math.max(0, datasetStats.total - 1),
        [removed.label]: Math.max(0, datasetStats[removed.label] - 1),
      };
      pushUndo({ frame: removed, index, deleted: true });
      commitDatasetMutation();
    }),
    undoLastDatasetChange: vi.fn(async () => {
      const entry = undoEntries.pop();
      if (!entry) throw new Error('there is no dataset change to undo');
      if (entry.deleted) {
        frames.splice(Math.min(entry.index, frames.length), 0, clone(entry.frame));
        datasetTotal += 1;
        datasetStats = {
          ...datasetStats,
          total: datasetStats.total + 1,
          [entry.frame.label]: datasetStats[entry.frame.label] + 1,
        };
      } else {
        const index = frames.findIndex((frame) => frame.id === entry.frame.id);
        if (index < 0) {
          undoEntries.push(entry);
          throw new Error(`frame with id ${entry.frame.id} not found`);
        }
        adjustLabelStats(frames[index].label, entry.frame.label);
        frames[index] = clone(entry.frame);
      }
      commitDatasetMutation();
    }),
    getUndoStatus: vi.fn(async () => clone(currentUndoStatus())),
    resetDataset: vi.fn(async () => {
      frames = [];
      datasetTotal = 0;
      datasetStats = {
        ...datasetStats,
        total: 0,
        good: 0,
        bad: 0,
        away: 0,
        unused: 0,
      };
      undoEntries.length = 0;
      commitDatasetMutation();
      return snapshot();
    }),
    resetAllData: vi.fn(async () => {
      cameraSettings = clone(cameraDefaults);
      uiSettings = clone(uiDefaults);
      frames = [];
      datasetTotal = 0;
      datasetStats = {
        ...datasetStats,
        total: 0,
        good: 0,
        bad: 0,
        away: 0,
        unused: 0,
      };
      undoEntries.length = 0;
      commitDatasetMutation();
      return snapshot();
    }),
    exportDataset: vi.fn(async () => ({
      frameCount: datasetTotal,
      datasetVersion,
    })),
    importDataset: vi.fn(async () => ({
      frameCount: datasetTotal,
      datasetVersion,
      state: snapshot(),
    })),
    onDatasetChanged: vi.fn(async (handler) => {
      datasetHandlers.add(handler);
      return () => {
        datasetHandlers.delete(handler);
      };
    }),
    onUndoStatusChanged: vi.fn(async (handler) => {
      undoHandlers.add(handler);
      return () => {
        undoHandlers.delete(handler);
      };
    }),
  };

  return {
    client,
    readCameraSettings: () => clone(cameraSettings),
    readUiSettings: () => clone(uiSettings),
  };
}

/** Binds stateful implementations to the hoisted spies used by module-mocked tests. */
export function bindMockNativePersistence(
  target: Partial<Record<keyof MockNativePersistenceClient, Mock>>,
  source: MockNativePersistenceClient,
): void {
  for (const key of Object.keys(source) as Array<keyof MockNativePersistenceClient>) {
    target[key]?.mockImplementation(source[key] as never);
  }
}
