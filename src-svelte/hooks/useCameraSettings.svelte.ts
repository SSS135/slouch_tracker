import type {
  CameraSettings as NativeCameraSettings,
  UiSettings as NativeUiSettings,
} from '@generated/bindings';
import { nativeClient, type NativeClient } from '../lib/native/client';
import { logger } from '../services/logging/logger';

export interface CameraSettings {
  cameraWidth: number;
  cameraHeight: number;
  captureIntervalSeconds: number;
  alertVolume: number;
  autoCaptureEnabled: boolean;
  autoCaptureIntervalSeconds: number;
  alertDelaySeconds: number;
  privacyMode: boolean;
  claheStrength: number;
  smoothingFrames: number;
  tileMotionThreshold: number;
  claheTemporalAlpha: number;
  preprocessingDebugView: boolean;
  showDetectionOverlay: boolean;
  minimizeToTrayOnClose: boolean;
  startHiddenOnLogin: boolean;
}

export interface CameraSettingsState {
  readonly settings: CameraSettings;
  readonly ready: boolean;
  readonly error: string | null;
  updateSettings(updates: Partial<CameraSettings>): void;
  resetSettings(): Promise<void>;
  reconcile(camera: NativeCameraSettings, ui: NativeUiSettings): void;
  reload(): Promise<void>;
  flush(): Promise<void>;
}

const LOADING_SETTINGS: CameraSettings = {
  cameraWidth: 0,
  cameraHeight: 0,
  captureIntervalSeconds: 0,
  alertVolume: 0,
  autoCaptureEnabled: false,
  autoCaptureIntervalSeconds: 0,
  alertDelaySeconds: 0,
  privacyMode: false,
  claheStrength: 0,
  smoothingFrames: 0,
  tileMotionThreshold: 0,
  claheTemporalAlpha: 0,
  preprocessingDebugView: false,
  showDetectionOverlay: false,
  minimizeToTrayOnClose: false,
  startHiddenOnLogin: false,
};

function requiredNumber(value: number | null, name: string): number {
  if (value === null || !Number.isFinite(value)) {
    throw new Error(`Native ${name} setting is invalid.`);
  }
  return value;
}

function validateSettings(settings: CameraSettings): CameraSettings {
  if (!Number.isSafeInteger(settings.cameraWidth) || settings.cameraWidth <= 0
    || !Number.isSafeInteger(settings.cameraHeight) || settings.cameraHeight <= 0) {
    throw new Error('Camera dimensions must be positive safe integers.');
  }
  if (!Number.isFinite(settings.captureIntervalSeconds) || settings.captureIntervalSeconds <= 0
    || !Number.isFinite(settings.autoCaptureIntervalSeconds) || settings.autoCaptureIntervalSeconds <= 0
    || !Number.isFinite(settings.alertDelaySeconds) || settings.alertDelaySeconds < 0) {
    throw new Error('Camera timing settings are invalid.');
  }
  if (!Number.isFinite(settings.alertVolume)
    || settings.alertVolume < 0
    || settings.alertVolume > 1) {
    throw new Error('Alert volume must be between 0 and 1.');
  }
  if (typeof settings.autoCaptureEnabled !== 'boolean' || typeof settings.privacyMode !== 'boolean'
    || typeof settings.showDetectionOverlay !== 'boolean'
    || typeof settings.preprocessingDebugView !== 'boolean'
    || typeof settings.minimizeToTrayOnClose !== 'boolean'
    || typeof settings.startHiddenOnLogin !== 'boolean') {
    throw new Error('Camera toggle settings are invalid.');
  }
  if (!Number.isFinite(settings.claheStrength)
    || settings.claheStrength < 0
    || settings.claheStrength > 10) {
    throw new Error('CLAHE strength must be between 0 and 10.');
  }
  if (!Number.isInteger(settings.smoothingFrames)
    || settings.smoothingFrames < 1
    || settings.smoothingFrames > 10) {
    throw new Error('Smoothing frames must be an integer between 1 and 10.');
  }
  if (!Number.isFinite(settings.tileMotionThreshold)
    || settings.tileMotionThreshold < 0.5
    || settings.tileMotionThreshold > 20) {
    throw new Error('Motion threshold must be between 0.5 and 20.');
  }
  if (!Number.isFinite(settings.claheTemporalAlpha)
    || settings.claheTemporalAlpha < 0.05
    || settings.claheTemporalAlpha > 1) {
    throw new Error('CLAHE smoothing must be between 0.05 and 1.');
  }
  return settings;
}

function normalizeSettings(settings: CameraSettings): CameraSettings {
  const normalized = { ...settings };
  normalized.claheStrength = Math.max(0, Math.min(10, Number(normalized.claheStrength) || 0));
  normalized.claheStrength = Math.round(normalized.claheStrength * 10) / 10;
  const frames = Number(normalized.smoothingFrames) || 1;
  normalized.smoothingFrames = Number.isInteger(frames) && frames >= 1 && frames <= 10
    ? frames
    : 1;
  const motion = Number(normalized.tileMotionThreshold);
  normalized.tileMotionThreshold = Number.isFinite(motion) ? Math.max(0.5, Math.min(20, motion)) : 1.5;
  const alpha = Number(normalized.claheTemporalAlpha);
  normalized.claheTemporalAlpha = Number.isFinite(alpha) ? Math.max(0.05, Math.min(1, alpha)) : 0.20;
  normalized.preprocessingDebugView = Boolean(normalized.preprocessingDebugView);
  return validateSettings(normalized);
}

function combine(camera: NativeCameraSettings, ui: NativeUiSettings): CameraSettings {
  return validateSettings({
    cameraWidth: camera.cameraWidth,
    cameraHeight: camera.cameraHeight,
    captureIntervalSeconds: requiredNumber(camera.captureIntervalSeconds, 'capture interval'),
    alertVolume: requiredNumber(ui.alertVolume, 'alert volume'),
    autoCaptureEnabled: camera.autoCaptureEnabled,
    autoCaptureIntervalSeconds: requiredNumber(camera.autoCaptureIntervalSeconds, 'auto-capture interval'),
    alertDelaySeconds: requiredNumber(ui.alertDelaySeconds, 'alert delay'),
    privacyMode: camera.privacyMode,
    claheStrength: requiredNumber(camera.claheStrength, 'CLAHE strength'),
    smoothingFrames: camera.smoothingFrames,
    // Optional in the generated bindings (serde default on the Rust field); a
    // settings row from a prior app version omits them, so coalesce to the native
    // defaults (per-tile motion gate 1.5, CLAHE LUT-EMA 0.20, debug view off).
    tileMotionThreshold: camera.tileMotionThreshold ?? 1.5,
    claheTemporalAlpha: camera.claheTemporalAlpha ?? 0.20,
    preprocessingDebugView: camera.preprocessingDebugView ?? false,
    // Optional in the generated bindings (serde default on the Rust field); a
    // settings row from a prior app version omits it, so coalesce to off.
    showDetectionOverlay: camera.showDetectionOverlay ?? false,
    // UiSettings tray toggles default true natively (serde default_true); a
    // settings row that predates them omits the fields, so coalesce to on.
    minimizeToTrayOnClose: ui.minimizeToTrayOnClose ?? true,
    startHiddenOnLogin: ui.startHiddenOnLogin ?? true,
  });
}

function split(settings: CameraSettings): { camera: NativeCameraSettings; ui: NativeUiSettings } {
  return {
    camera: {
      cameraWidth: settings.cameraWidth,
      cameraHeight: settings.cameraHeight,
      captureIntervalSeconds: settings.captureIntervalSeconds,
      autoCaptureEnabled: settings.autoCaptureEnabled,
      autoCaptureIntervalSeconds: settings.autoCaptureIntervalSeconds,
      privacyMode: settings.privacyMode,
      claheStrength: settings.claheStrength,
      smoothingFrames: settings.smoothingFrames,
      tileMotionThreshold: settings.tileMotionThreshold,
      claheTemporalAlpha: settings.claheTemporalAlpha,
      preprocessingDebugView: settings.preprocessingDebugView,
      showDetectionOverlay: settings.showDetectionOverlay,
    },
    ui: {
      alertVolume: settings.alertVolume,
      alertDelaySeconds: settings.alertDelaySeconds,
      minimizeToTrayOnClose: settings.minimizeToTrayOnClose,
      startHiddenOnLogin: settings.startHiddenOnLogin,
    },
  };
}

function message(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

/** Rust-owned camera/UI settings with one linearizable native mutation queue. */
export function useCameraSettings(client: NativeClient = nativeClient): CameraSettingsState {
  let current = $state<CameraSettings>({ ...LOADING_SETTINGS });
  let ready = $state(false);
  let error = $state<string | null>(null);
  let generation = 0;
  let operationChain: Promise<void> = Promise.resolve();

  const enqueue = (operation: () => Promise<void>): Promise<void> => {
    const task = operationChain.then(operation, operation);
    operationChain = task.catch(() => undefined);
    return task;
  };

  const fetchSettings = async (): Promise<CameraSettings> => {
    const [camera, ui] = await Promise.all([
      client.getCameraSettings(),
      client.getUiSettings(),
    ]);
    return combine(camera, ui);
  };

  const commit = (token: number, settings: CameraSettings, nextError: string | null): void => {
    if (token !== generation) return;
    current = settings;
    error = nextError;
    ready = true;
  };

  const recover = async (token: number, mutationError: string): Promise<void> => {
    try {
      commit(token, await fetchSettings(), mutationError);
    } catch (reloadCause) {
      if (token !== generation) return;
      error = mutationError;
      ready = false;
      logger.error('storage', 'Failed to reconcile native camera/UI settings:', reloadCause);
    }
  };

  $effect(() => {
    const token = ++generation;
    void fetchSettings().then((settings) => commit(token, settings, null)).catch((cause: unknown) => {
      if (token !== generation) return;
      error = message(cause);
      ready = false;
      logger.error('storage', 'Failed to load native camera/UI settings:', cause);
    });
    return () => {
      if (token === generation) generation += 1;
    };
  });

  const updateSettings = (updates: Partial<CameraSettings>): void => {
    if (!ready) return;
    let next: CameraSettings;
    try {
      next = normalizeSettings({ ...current, ...updates });
    } catch (cause) {
      error = message(cause);
      return;
    }

    const token = ++generation;
    current = next;
    const snapshot = split(next);
    void enqueue(async () => {
      const results = await Promise.allSettled([
        client.saveCameraSettings(snapshot.camera),
        client.saveUiSettings(snapshot.ui),
      ]);
      const failure = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (failure) {
        const mutationError = message(failure.reason);
        logger.error('storage', 'Failed to persist native camera/UI settings:', failure.reason);
        await recover(token, mutationError);
      } else if (token === generation) {
        error = null;
      }
    });
  };

  const resetSettings = (): Promise<void> => {
    const token = ++generation;
    ready = false;
    return enqueue(async () => {
      const results = await Promise.allSettled([
        client.resetCameraSettings(),
        client.resetUiSettings(),
      ]);
      const failure = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
      if (failure) {
        const mutationError = message(failure.reason);
        await recover(token, mutationError);
        throw failure.reason;
      }
      commit(
        token,
        combine(
          (results[0] as PromiseFulfilledResult<NativeCameraSettings>).value,
          (results[1] as PromiseFulfilledResult<NativeUiSettings>).value,
        ),
        null,
      );
    });
  };

  return {
    get settings() { return current; },
    get ready() { return ready; },
    get error() { return error; },
    updateSettings,
    resetSettings,
    reconcile(camera, ui) {
      let authoritative: CameraSettings;
      try {
        authoritative = combine(camera, ui);
      } catch (cause) {
        error = message(cause);
        return;
      }
      const token = ++generation;
      ready = false;
      const snapshot = split(authoritative);
      void enqueue(async () => {
        const results = await Promise.allSettled([
          client.saveCameraSettings(snapshot.camera),
          client.saveUiSettings(snapshot.ui),
        ]);
        const failure = results.find((result): result is PromiseRejectedResult => result.status === 'rejected');
        if (failure) {
          await recover(token, message(failure.reason));
          return;
        }
        commit(token, authoritative, null);
      });
    },
    reload() {
      const token = ++generation;
      ready = false;
      return enqueue(async () => {
        try {
          commit(token, await fetchSettings(), null);
        } catch (cause) {
          if (token === generation) {
            error = message(cause);
            ready = false;
          }
          throw cause;
        }
      });
    },
    async flush() { await operationChain; },
  };
}
