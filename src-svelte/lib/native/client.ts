import { Channel } from '@tauri-apps/api/core';
import type { UnlistenFn } from '@tauri-apps/api/event';
import {
  commands,
  events,
  getThumbnail as getThumbnailBytes,
  onDatasetChanged,
  onShortcutCapture,
  saveCapture as invokeSaveCapture,
  type ActiveModelMetadata,
  type ApiError,
  type AppStatus,
  type ArchiveImportResult_Serialize,
  type ArchiveSummaryDto,
  type CameraDeviceInfo,
  type CameraSettings,
  type ClassifierMetadata_Serialize,
  type DatasetChangedEvent,
  type DatasetPage,
  type DatasetStats,
  type FeatureMetadata_Serialize,
  type FrameLabel,
  type InferenceUiResult,
  type NativeStateChangedEvent_Deserialize,
  type NativeStateSnapshot_Serialize,
  type ReservoirMetadata,
  type ShortcutStatus,
  type TrainingEvent_Deserialize,
  type TrainingResultResponse_Serialize,
  type TrainingSettings_Deserialize,
  type TrainingSettings_Serialize,
  type TrainingStatus,
  type UiSettings,
  type UndoStatus,
} from '@generated/bindings';

export class NativeCommandError extends Error {
  readonly kind: ApiError['kind'];

  constructor(error: ApiError) {
    super(error.message);
    this.name = 'NativeCommandError';
    this.kind = error.kind;
  }
}

type NativeResult<T> =
  | { status: 'ok'; data: T }
  | { status: 'error'; error: ApiError };

export function unwrapNativeResult<T>(result: NativeResult<T>): T {
  if (result.status === 'error') {
    throw new NativeCommandError(result.error);
  }
  return result.data;
}

function isApiError(value: unknown): value is ApiError {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as Record<string, unknown>).kind === 'string' &&
    typeof (value as Record<string, unknown>).message === 'string'
  );
}

// Raw-byte commands reject with the serialized error payload instead of the
// typed Result envelope; normalize so consumers always receive an Error with a
// readable message (never "[object Object]").
function toNativeError(cause: unknown): Error {
  if (cause instanceof Error) return cause;
  if (isApiError(cause)) return new NativeCommandError(cause);
  if (typeof cause === 'string') return new Error(cause);
  try {
    return new Error(JSON.stringify(cause));
  } catch {
    return new Error(String(cause));
  }
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive safe integer.`);
  }
}

function assertRequestId(requestId: number): void {
  if (!Number.isSafeInteger(requestId) || requestId < 0) {
    throw new TypeError('requestId must be a non-negative safe integer.');
  }
}

export interface SaveCaptureHeaders {
  requestId: number;
  token: number;
  frameId: string;
  timestamp: number;
  label: FrameLabel;
  mimeType: 'image/jpeg' | 'image/png' | 'image/webp';
}

export async function getThumbnail(id: string): Promise<Uint8Array> {
  if (id.trim().length === 0) {
    throw new TypeError('Thumbnail id must not be empty.');
  }
  try {
    return await getThumbnailBytes(id);
  } catch (cause) {
    throw toNativeError(cause);
  }
}

export async function saveCapture(
  thumbnail: Uint8Array,
  headers: SaveCaptureHeaders,
): Promise<void> {
  assertRequestId(headers.requestId);
  assertPositiveInteger(headers.token, 'token');
  assertPositiveInteger(headers.timestamp, 'timestamp');

  if (headers.frameId.trim().length === 0) {
    throw new TypeError('frameId must not be empty.');
  }
  if (thumbnail.byteLength === 0 || thumbnail.byteLength > 2 * 1024 * 1024) {
    throw new RangeError('Thumbnail must contain between 1 byte and 2 MiB.');
  }

  try {
    await invokeSaveCapture(thumbnail, headers);
  } catch (cause) {
    throw toNativeError(cause);
  }
}

export function createTrainingChannel(
  handler: (event: TrainingEvent_Deserialize) => void,
): Channel<TrainingEvent_Deserialize> {
  const channel = new Channel<TrainingEvent_Deserialize>();
  channel.onmessage = handler;
  return channel;
}

export const nativeClient = {
  async appStatus(): Promise<AppStatus> {
    return unwrapNativeResult(await commands.appStatus());
  },
  async initializeInference(): Promise<void> {
    unwrapNativeResult(await commands.initializeInference());
  },
  async trainModels(
    doCv: boolean | null,
    onEvent: Channel<TrainingEvent_Deserialize>,
  ): Promise<TrainingResultResponse_Serialize> {
    return unwrapNativeResult(await commands.trainModels(doCv, onEvent));
  },
  async getTrainingStatus(): Promise<TrainingStatus> {
    return unwrapNativeResult(await commands.getTrainingStatus());
  },
  async cancelTraining(): Promise<void> {
    unwrapNativeResult(await commands.cancelTraining());
  },
  async getDatasetPage(offset = 0, limit = 100): Promise<DatasetPage> {
    return unwrapNativeResult(await commands.getDatasetPage(offset, limit));
  },
  async getDatasetStats(): Promise<DatasetStats> {
    return unwrapNativeResult(await commands.getDatasetStats());
  },
  async getNeedsRetraining(): Promise<boolean> {
    return unwrapNativeResult(await commands.getNeedsRetraining());
  },
  async getReservoirMetadata(): Promise<ReservoirMetadata> {
    return unwrapNativeResult(await commands.getReservoirMetadata());
  },
  async getCameraSettings(): Promise<CameraSettings> {
    return unwrapNativeResult(await commands.getCameraSettings());
  },
  async saveCameraSettings(settings: CameraSettings): Promise<void> {
    unwrapNativeResult(await commands.saveCameraSettings(settings));
  },
  async resetCameraSettings(): Promise<CameraSettings> {
    return unwrapNativeResult(await commands.resetCameraSettings());
  },
  async getUiSettings(): Promise<UiSettings> {
    return unwrapNativeResult(await commands.getUiSettings());
  },
  async saveUiSettings(settings: UiSettings): Promise<void> {
    unwrapNativeResult(await commands.saveUiSettings(settings));
  },
  async resetUiSettings(): Promise<UiSettings> {
    return unwrapNativeResult(await commands.resetUiSettings());
  },
  async getTrainingSettings(): Promise<TrainingSettings_Serialize | null> {
    return unwrapNativeResult(await commands.getTrainingSettings());
  },
  async resetTrainingSettings(): Promise<void> {
    unwrapNativeResult(await commands.resetTrainingSettings());
  },
  async saveTrainingSettings(settings: TrainingSettings_Deserialize): Promise<void> {
    unwrapNativeResult(await commands.saveTrainingSettings(settings));
  },
  async updateFrameLabel(id: string, label: FrameLabel): Promise<void> {
    unwrapNativeResult(await commands.updateFrameLabel(id, label));
  },
  async deleteFrame(id: string): Promise<void> {
    unwrapNativeResult(await commands.deleteFrame(id));
  },
  async undoLastDatasetChange(): Promise<void> {
    unwrapNativeResult(await commands.undoLastDatasetChange());
  },
  async getUndoStatus(): Promise<UndoStatus> {
    return unwrapNativeResult(await commands.getUndoStatus());
  },
  async cleanupUnusedFrames(): Promise<number> {
    return unwrapNativeResult(await commands.cleanupUnusedFrames());
  },
  async resetDataset(): Promise<NativeStateSnapshot_Serialize> {
    return unwrapNativeResult(await commands.resetDataset());
  },
  async resetAllData(): Promise<NativeStateSnapshot_Serialize> {
    return unwrapNativeResult(await commands.resetAllData());
  },
  async getClassifierRegistry(): Promise<ClassifierMetadata_Serialize[]> {
    return unwrapNativeResult(await commands.getClassifierRegistry());
  },
  async getFeatureRegistry(): Promise<FeatureMetadata_Serialize[]> {
    return unwrapNativeResult(await commands.getFeatureRegistry());
  },
  async getActiveModelMetadata(): Promise<ActiveModelMetadata> {
    return unwrapNativeResult(await commands.getActiveModelMetadata());
  },
  async exportDataset(): Promise<ArchiveSummaryDto | null> {
    return unwrapNativeResult(await commands.exportDataset());
  },
  async importDataset(): Promise<ArchiveImportResult_Serialize | null> {
    return unwrapNativeResult(await commands.importDataset());
  },
  async getShortcutStatus(): Promise<ShortcutStatus> {
    return unwrapNativeResult(await commands.getShortcutStatus());
  },
  async startCamera(onResult: Channel<InferenceUiResult>): Promise<void> {
    unwrapNativeResult(await commands.startCamera(onResult));
  },
  async stopCamera(): Promise<void> {
    unwrapNativeResult(await commands.stopCamera());
  },
  async listCameras(): Promise<CameraDeviceInfo[]> {
    return unwrapNativeResult(await commands.listCameras());
  },
  getThumbnail,
  saveCapture,
  onDatasetChanged(handler: (event: DatasetChangedEvent) => void): Promise<UnlistenFn> {
    return onDatasetChanged(handler);
  },
  onShortcutCapture(
    handler: (label: 'good' | 'bad' | 'away') => void,
  ): Promise<UnlistenFn> {
    return onShortcutCapture(handler);
  },
  onUndoStatusChanged(handler: (status: UndoStatus) => void): Promise<UnlistenFn> {
    return events.undoStatusChanged.listen((event) => handler(event.payload.status));
  },
  onNativeStateChanged(
    handler: (event: NativeStateChangedEvent_Deserialize) => void,
  ): Promise<UnlistenFn> {
    return events.nativeStateChanged.listen((event) => handler(event.payload));
  },
};

export type NativeClient = typeof nativeClient;
