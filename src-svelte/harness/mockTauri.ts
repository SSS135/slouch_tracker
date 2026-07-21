import type { Channel, InvokeArgs } from '@tauri-apps/api/core';
import { clearMocks, mockIPC, mockWindows } from '@tauri-apps/api/mocks';
import type {
  ActiveModelMetadata,
  AppStatus,
  DatasetPage,
  FrameMetadataDto,
  NativeStateSnapshot_Serialize,
  TrainingEvent_Deserialize,
  TrainingResultResponse_Deserialize,
  TrainingResultResponse_Serialize,
  TrainingSettings_Serialize,
  UndoStatus,
} from '@generated/bindings';
import type { InferenceUiResult } from '@generated/bindings';

const emptyTrainingResult: TrainingResultResponse_Deserialize = {
  postureResult: null,
  presenceResult: null,
  success: true,
  errors: [],
  warnings: [],
};

const initialFrame: FrameMetadataDto = {
  id: 'frame-1',
  timestamp: 1,
  keypoints: Array.from({ length: 17 }, (_, index) => ({
    x: 0.2 + index * 0.01,
    y: 0.3 + index * 0.01,
    score: 0.9,
  })),
  bbox: { x1: 0.1, y1: 0.1, x2: 0.9, y2: 0.9, score: 0.9, width: 0.8, height: 0.8 },
  label: 'good',
  thumbnailMimeType: 'image/webp',
};

interface HarnessMetrics {
  captureBytes: number;
  captureCalls: number;
  failStatsRequests: number;
}

const metrics: HarnessMetrics = {
  captureBytes: 0,
  captureCalls: 0,
  failStatsRequests: 0,
};

// 1x1 PNG used as the mocked native `slouchcam` preview frame so the renderer's
// createImageBitmap/canvas-ready path works under the browser harness.
const FAKE_FRAME_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

function fakeFrameBlob(): Blob {
  const binary = atob(FAKE_FRAME_BASE64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: 'image/png' });
}

function isSlouchcamUrl(input: RequestInfo | URL): boolean {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
  return url.includes('slouchcam');
}

export function getHarnessMetrics(): Readonly<HarnessMetrics> {
  return metrics;
}

function getArg<T>(args: InvokeArgs | undefined, key: string): T {
  return (args as Record<string, unknown> | undefined)?.[key] as T;
}

export function installMockTauri(): () => void {
  clearMocks();
  mockWindows('main');
  metrics.captureBytes = 0;
  metrics.captureCalls = 0;
  metrics.failStatsRequests = typeof window === 'undefined'
    ? 0
    : Number.parseInt(new URLSearchParams(window.location.search).get('failStats') ?? '0', 10);

  const failModelMetadata = typeof window !== 'undefined'
    && new URLSearchParams(window.location.search).get('failModelMeta') === '1';

  let inferenceReady = false;
  let cameraSettings = {
    cameraWidth: 800,
    cameraHeight: 600,
    captureIntervalSeconds: 0.2,
    autoCaptureEnabled: false,
    autoCaptureIntervalSeconds: 2,
    privacyMode: true,
    claheStrength: 3.5,
    gaussianBlurKernel: 5,
    smoothingFrames: 3,
  };
  let uiSettings = { alertVolume: 0.3, alertDelaySeconds: 5 };
  let datasetVersion = 1;
  let frames: FrameMetadataDto[] = [{ ...initialFrame }];
  let undoFrames: FrameMetadataDto[] | null = null;
  let undoRevision = 0;
  let activeModels: ActiveModelMetadata = {
    posture: { classifierId: 'mlp', featureTypes: ['engineered_features'], trainedAt: 1_700_000_000_000 },
    presence: null,
  };
  let trainingSettings: TrainingSettings_Serialize | null = null;
  let trainingChannel: Channel<TrainingEvent_Deserialize> | null = null;
  let finishTraining: ((value: TrainingResultResponse_Serialize) => void) | null = null;
  let failTraining: ((reason: unknown) => void) | null = null;
  let cameraInterval: ReturnType<typeof setInterval> | null = null;
  let cameraResultSeq = 0;

  const makeCameraResult = (seq: number): InferenceUiResult => ({
    requestId: seq,
    token: 100 + seq,
    personFound: true,
    bbox: {
      original: { x1: 0.2, y1: 0.1, x2: 0.8, y2: 0.9, width: 0.6, height: 0.8, score: 0.95 },
      expanded: { x1: 0.15, y1: 0.05, x2: 0.85, y2: 0.95, width: 0.7, height: 0.9, score: 0.95 },
    },
    keypoints: Array.from({ length: 17 }, (_, index) => ({ x: 0.25 + index * 0.01, y: 0.25 + index * 0.01, score: 0.9 })),
    classification: { presentProbability: 0.95, goodProbability: 0.8 },
  });

  const appStatus = (): AppStatus => ({
    ready: inferenceReady,
    inferenceReady,
    datasetVersion,
    storage: { used: 1024, available: 4096, quota: 5120 },
  });
  const undoStatus = (): UndoStatus => ({
    available: undoFrames !== null,
    depth: undoFrames === null ? 0 : 1,
    nextAction: undoFrames === null ? null : 'restoreFrame',
    revision: undoRevision,
  });
  const snapshot = (): NativeStateSnapshot_Serialize => ({
    app: appStatus(),
    cameraSettings,
    uiSettings,
    trainingSettings,
    activeModels,
    undo: undoStatus(),
  });

  mockIPC((command, args) => {
    switch (command) {
      case 'app_status':
        return appStatus();
      case 'initialize_inference':
        inferenceReady = true;
        return null;
      case 'get_classifier_registry':
        return [];
      case 'get_feature_registry':
        return [];
      case 'get_shortcut_status':
        return { registered: true };
      case 'start_camera': {
        const channel = getArg<Channel<InferenceUiResult>>(args, 'onResult');
        const push = (): void => {
          cameraResultSeq += 1;
          channel.onmessage(makeCameraResult(cameraResultSeq));
        };
        // Push an immediate result so consumers have a token synchronously, then
        // keep streaming to mimic the native detection cadence.
        push();
        if (cameraInterval) clearInterval(cameraInterval);
        cameraInterval = setInterval(push, 200);
        return null;
      }
      case 'stop_camera':
        if (cameraInterval) {
          clearInterval(cameraInterval);
          cameraInterval = null;
        }
        return null;
      case 'list_cameras':
        return [{ index: '0', name: 'Mock Camera', description: 'harness capture device' }];
      case 'save_capture':
        metrics.captureBytes = (args as unknown as Uint8Array).byteLength;
        metrics.captureCalls += 1;
        undoFrames = frames.map((frame) => ({ ...frame }));
        undoRevision += 1;
        if (!frames.some((frame) => frame.id === 'captured-frame')) {
          frames = [...frames, { ...initialFrame, id: 'captured-frame', timestamp: 2 }];
          datasetVersion += 1;
        }
        return new Promise<null>((resolve) => setTimeout(() => resolve(null), 100));
      case 'get_thumbnail':
        return new Uint8Array([1, 2, 3]);
      case 'get_dataset_page': {
        const offset = getArg<number | null>(args, 'offset') ?? 0;
        const limit = getArg<number | null>(args, 'limit') ?? 100;
        const page: DatasetPage = {
          frames: frames.slice(offset, offset + limit),
          offset,
          limit,
          total: frames.length,
          version: datasetVersion,
          lastModified: 1,
        };
        return page;
      }
      case 'get_dataset_stats':
        if (metrics.failStatsRequests > 0) {
          metrics.failStatsRequests -= 1;
          throw { kind: 'storage', message: 'deterministic statistics failure' };
        }
        return {
          total: frames.length,
          good: frames.filter((frame) => frame.label === 'good').length,
          bad: frames.filter((frame) => frame.label === 'bad').length,
          away: frames.filter((frame) => frame.label === 'away').length,
          unused: frames.filter((frame) => frame.label === 'unused').length,
          imbalanceRatio: null,
          hasMinimumFrames: false,
          hasAwayFrames: false,
        };
      case 'update_frame_label': {
        undoFrames = frames.map((frame) => ({ ...frame }));
        undoRevision += 1;
        const id = getArg<string>(args, 'id');
        const label = getArg<FrameMetadataDto['label']>(args, 'label');
        frames = frames.map((frame) => frame.id === id ? { ...frame, label } : frame);
        datasetVersion += 1;
        return null;
      }
      case 'delete_frame': {
        undoFrames = frames.map((frame) => ({ ...frame }));
        undoRevision += 1;
        const id = getArg<string>(args, 'id');
        frames = frames.filter((frame) => frame.id !== id);
        datasetVersion += 1;
        return null;
      }
      case 'get_undo_status':
        return undoStatus();
      case 'undo_last_dataset_change':
        if (undoFrames) {
          frames = undoFrames;
          undoFrames = null;
          undoRevision += 1;
          datasetVersion += 1;
        }
        return null;
      case 'reset_dataset':
        undoFrames = null;
        undoRevision += 1;
        frames = [];
        datasetVersion = 0;
        activeModels = { posture: null, presence: null };
        return snapshot();
      case 'reset_all_data':
        undoFrames = null;
        undoRevision += 1;
        frames = [];
        datasetVersion = 0;
        activeModels = { posture: null, presence: null };
        trainingSettings = null;
        cameraSettings = { ...cameraSettings, cameraWidth: 800, cameraHeight: 600, captureIntervalSeconds: 0.2, autoCaptureEnabled: false, autoCaptureIntervalSeconds: 2, privacyMode: true, claheStrength: 3.5, gaussianBlurKernel: 5, smoothingFrames: 3 };
        uiSettings = { alertVolume: 0.3, alertDelaySeconds: 5 };
        return snapshot();
      case 'get_training_status':
        return { running: finishTraining !== null };
      case 'train_models': {
        const doCv = getArg<boolean | null>(args, 'doCv');
        const channel = getArg<Channel<TrainingEvent_Deserialize>>(args, 'onEvent');
        const jobId = 1;
        channel.onmessage({ type: 'started', jobId, sequence: 0 });
        channel.onmessage({ type: 'progress', jobId, sequence: 1, stage: 'processing', progress: 5 });
        if (doCv === false) {
          const error = 'deterministic training failure';
          channel.onmessage({ type: 'failed', jobId, sequence: 2, error });
          throw { kind: 'training', message: error };
        }
        if (doCv === null) {
          channel.onmessage({ type: 'progress', jobId, sequence: 2, stage: 'evaluating', progress: 85 });
          channel.onmessage({ type: 'progress', jobId, sequence: 3, stage: 'deploying', progress: 95 });
          channel.onmessage({ type: 'completed', jobId, sequence: 4, result: emptyTrainingResult });
          return emptyTrainingResult;
        }
        trainingChannel = channel;
        return new Promise<TrainingResultResponse_Serialize>((resolve, reject) => {
          finishTraining = resolve;
          failTraining = reject;
        });
      }
      case 'cancel_training':
        trainingChannel?.onmessage({ type: 'cancelled', jobId: 1, sequence: 2 });
        failTraining?.({ kind: 'cancelled', message: 'Training cancelled.' });
        trainingChannel = null;
        finishTraining = null;
        failTraining = null;
        return null;
      case 'get_camera_settings':
        return cameraSettings;
      case 'save_camera_settings':
        cameraSettings = getArg<typeof cameraSettings>(args, 'settings');
        return null;
      case 'reset_camera_settings':
        cameraSettings = { ...cameraSettings, cameraWidth: 800, cameraHeight: 600, captureIntervalSeconds: 0.2, autoCaptureEnabled: false, autoCaptureIntervalSeconds: 2, privacyMode: true, claheStrength: 3.5, gaussianBlurKernel: 5, smoothingFrames: 3 };
        return cameraSettings;
      case 'get_ui_settings':
        return uiSettings;
      case 'save_ui_settings':
        uiSettings = getArg<typeof uiSettings>(args, 'settings');
        return null;
      case 'reset_ui_settings':
        uiSettings = { alertVolume: 0.3, alertDelaySeconds: 5 };
        return uiSettings;
      case 'get_training_settings':
        return trainingSettings;
      case 'reset_training_settings':
        trainingSettings = null;
        return null;
      case 'save_training_settings':
        trainingSettings = getArg<TrainingSettings_Serialize>(args, 'settings');
        return null;
      case 'get_active_model_metadata':
        if (failModelMetadata) throw { kind: 'storage', message: 'deterministic model metadata failure' };
        return activeModels;
      case 'export_dataset':
        return { frameCount: frames.length, datasetVersion };
      case 'import_dataset':
        undoFrames = null;
        undoRevision += 1;
        return { frameCount: frames.length, datasetVersion, state: snapshot() };
      default:
        throw { kind: 'invalidRequest', message: `Unexpected mocked command: ${command}` };
    }
  }, { shouldMockEvents: true });

  // Serve the native `slouchcam` preview protocol; delegate all other requests.
  const originalFetch = typeof globalThis.fetch === 'function' ? globalThis.fetch.bind(globalThis) : null;
  if (typeof globalThis !== 'undefined') {
    globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      if (isSlouchcamUrl(input)) {
        return Promise.resolve(new Response(fakeFrameBlob(), { status: 200, headers: { 'content-type': 'image/png' } }));
      }
      if (originalFetch) return originalFetch(input, init);
      return Promise.reject(new Error(`Unmocked fetch: ${String(input)}`));
    }) as typeof globalThis.fetch;
  }

  return () => {
    if (cameraInterval) {
      clearInterval(cameraInterval);
      cameraInterval = null;
    }
    if (originalFetch) globalThis.fetch = originalFetch;
    clearMocks();
  };
}
