import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createMockNativePersistence } from '../../__tests__/utils/mockNativePersistence';
import type { NativeClient } from '../../lib/native/client';
import { useCameraSettings } from '../useCameraSettings';

const cameraDefaults = {
  cameraWidth: 800,
  cameraHeight: 600,
  captureIntervalSeconds: 0.5,
  autoCaptureEnabled: true,
  autoCaptureIntervalSeconds: 2,
  privacyMode: true,
  claheStrength: 3.5,
  smoothingFrames: 3,
  tileMotionThreshold: 3,
  claheTemporalAlpha: 0.15,
  preprocessingDebugView: false,
  showDetectionOverlay: false,
};
const uiDefaults = {
  alertVolume: 0.3,
  alertDelaySeconds: 5,
  minimizeToTrayOnClose: true,
  startHiddenOnLogin: true,
};

function client() {
  return createMockNativePersistence({
    cameraSettings: cameraDefaults,
    uiSettings: uiDefaults,
    datasetPage: {
      frames: [],
      offset: 0,
      limit: 24,
      total: 0,
      version: 0,
      lastModified: 0,
    },
    datasetStats: {
      total: 0,
      good: 0,
      bad: 0,
      away: 0,
      unused: 0,
      imbalanceRatio: 0,
      hasMinimumFrames: false,
      hasAwayFrames: false,
    },
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

const disposers: Array<() => void> = [];
function mount(mock: ReturnType<typeof client>) {
  let result!: ReturnType<typeof useCameraSettings>;
  const dispose = $effect.root(() => {
    result = useCameraSettings(mock.client as unknown as NativeClient);
  });
  disposers.push(dispose);
  flushSync();
  return result;
}

async function loaded(result: ReturnType<typeof useCameraSettings>): Promise<void> {
  await vi.waitFor(() => expect(result.ready).toBe(true));
  flushSync();
}

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
  vi.restoreAllMocks();
});

describe('useCameraSettings native persistence', () => {
  it('loads Rust-owned camera and UI defaults through typed native commands', async () => {
    const mock = client();
    const result = mount(mock);

    expect(result.ready).toBe(false);
    await loaded(result);
    expect(result.settings).toEqual({
      ...cameraDefaults,
      ...uiDefaults,
    });
    expect(mock.client.getCameraSettings).toHaveBeenCalledOnce();
    expect(mock.client.getUiSettings).toHaveBeenCalledOnce();
  });

  it('serializes updates through typed camera and UI save commands', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ cameraWidth: 1280, alertVolume: 0.75 }));
    await result.flush();

    expect(mock.client.saveCameraSettings).toHaveBeenCalledWith({
      ...cameraDefaults,
      cameraWidth: 1280,
    });
    expect(mock.client.saveUiSettings).toHaveBeenCalledWith({
      ...uiDefaults,
      alertVolume: 0.75,
    });
    expect(mock.readCameraSettings().cameraWidth).toBe(1280);
    expect(mock.readUiSettings().alertVolume).toBe(0.75);
  });

  it('uses native reset results rather than local defaults', async () => {
    const mock = client();
    mock.client.resetCameraSettings.mockResolvedValueOnce({ ...cameraDefaults, cameraWidth: 640 });
    mock.client.resetUiSettings.mockResolvedValueOnce({ ...uiDefaults, alertDelaySeconds: 8 });
    const result = mount(mock);
    await loaded(result);

    await result.resetSettings();
    expect(result.settings.cameraWidth).toBe(640);
    expect(result.settings.alertDelaySeconds).toBe(8);
  });

  it('fails closed when native settings cannot be loaded', async () => {
    const mock = client();
    mock.client.getCameraSettings.mockRejectedValueOnce(new Error('settings unavailable'));
    const result = mount(mock);

    await vi.waitFor(() => expect(result.error).toBe('settings unavailable'));
    expect(result.ready).toBe(false);
    expect(result.settings.cameraWidth).toBe(0);
    flushSync(() => result.updateSettings({ cameraWidth: 1920 }));
    expect(mock.client.saveCameraSettings).not.toHaveBeenCalled();
  });

  it('normalizes preprocessing settings before publishing and persists complete payloads', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({
      claheStrength: 15.04,
      smoothingFrames: 20,
      alertVolume: 0,
    }));
    await result.flush();

    expect(result.settings).toEqual({
      ...cameraDefaults,
      ...uiDefaults,
      claheStrength: 10,
      smoothingFrames: 1,
      alertVolume: 0,
    });
    expect(mock.client.saveCameraSettings).toHaveBeenLastCalledWith({
      ...cameraDefaults,
      claheStrength: 10,
      smoothingFrames: 1,
    });
    expect(mock.client.saveUiSettings).toHaveBeenLastCalledWith({
      ...uiDefaults,
      alertVolume: 0,
    });
  });

  it('clamps claheStrength to the 0-10 range and passes valid values through unchanged', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ claheStrength: -5 }));
    await result.flush();
    expect(result.settings.claheStrength).toBe(0);
    expect(result.error).toBeNull();

    flushSync(() => result.updateSettings({ claheStrength: 15 }));
    await result.flush();
    expect(result.settings.claheStrength).toBe(10);

    flushSync(() => result.updateSettings({ claheStrength: 5.5 }));
    await result.flush();
    expect(result.settings.claheStrength).toBe(5.5);
    expect(result.error).toBeNull();
  });

  it('passes valid smoothingFrames boundaries through unchanged', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ smoothingFrames: 5 }));
    await result.flush();
    expect(result.settings.smoothingFrames).toBe(5);

    flushSync(() => result.updateSettings({ smoothingFrames: 1 }));
    await result.flush();
    expect(result.settings.smoothingFrames).toBe(1);

    flushSync(() => result.updateSettings({ smoothingFrames: 10 }));
    await result.flush();
    expect(result.settings.smoothingFrames).toBe(10);
    expect(result.error).toBeNull();
  });

  it('defaults the new preprocessing fields when native omits them', async () => {
    const mock = client();
    mock.client.getCameraSettings.mockResolvedValueOnce({
      cameraWidth: 800,
      cameraHeight: 600,
      captureIntervalSeconds: 0.5,
      autoCaptureEnabled: true,
      autoCaptureIntervalSeconds: 2,
      privacyMode: true,
      claheStrength: 3.5,
      smoothingFrames: 3,
      showDetectionOverlay: false,
    });
    const result = mount(mock);
    await loaded(result);

    expect(result.settings.tileMotionThreshold).toBe(1.5);
    expect(result.settings.claheTemporalAlpha).toBe(0.20);
    expect(result.settings.preprocessingDebugView).toBe(false);
  });

  it('clamps tileMotionThreshold and claheTemporalAlpha into range and passes valid values through', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ tileMotionThreshold: 50, claheTemporalAlpha: 5 }));
    await result.flush();
    expect(result.settings.tileMotionThreshold).toBe(20);
    expect(result.settings.claheTemporalAlpha).toBe(1);
    expect(result.error).toBeNull();

    flushSync(() => result.updateSettings({ tileMotionThreshold: 0.1, claheTemporalAlpha: 0.01 }));
    await result.flush();
    expect(result.settings.tileMotionThreshold).toBe(0.5);
    expect(result.settings.claheTemporalAlpha).toBe(0.05);

    flushSync(() => result.updateSettings({ tileMotionThreshold: 7.5, claheTemporalAlpha: 0.3 }));
    await result.flush();
    expect(result.settings.tileMotionThreshold).toBe(7.5);
    expect(result.settings.claheTemporalAlpha).toBe(0.3);
    expect(result.error).toBeNull();
  });

  it('fails closed when native returns an out-of-range motion threshold', async () => {
    const mock = client();
    mock.client.getCameraSettings.mockResolvedValueOnce({ ...cameraDefaults, tileMotionThreshold: 40 });
    const result = mount(mock);

    await vi.waitFor(() => expect(result.error).toBe('Motion threshold must be between 0.5 and 20.'));
    expect(result.ready).toBe(false);
  });

  it('fails closed when native returns an out-of-range CLAHE smoothing alpha', async () => {
    const mock = client();
    mock.client.getCameraSettings.mockResolvedValueOnce({ ...cameraDefaults, claheTemporalAlpha: 2 });
    const result = mount(mock);

    await vi.waitFor(() => expect(result.error).toBe('CLAHE smoothing must be between 0.05 and 1.'));
    expect(result.ready).toBe(false);
  });

  it('persists a flipped preprocessingDebugView through saveCameraSettings', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ preprocessingDebugView: true }));
    await result.flush();

    expect(mock.client.saveCameraSettings).toHaveBeenLastCalledWith({
      ...cameraDefaults,
      preprocessingDebugView: true,
    });
    expect(mock.readCameraSettings().preprocessingDebugView).toBe(true);
    expect(result.settings.preprocessingDebugView).toBe(true);
  });

  it('rejects malformed non-preprocessing updates before optimistic publication', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);
    flushSync(() => result.updateSettings({ cameraWidth: Number.NaN }));
    expect(result.settings.cameraWidth).toBe(800);
    expect(result.error).toBe('Camera dimensions must be positive safe integers.');
    expect(mock.client.saveCameraSettings).not.toHaveBeenCalled();
  });

  it('does not allow a stale initial load to overwrite reconciliation', async () => {
    const mock = client();
    const cameraLoad = deferred<typeof cameraDefaults>();
    const uiLoad = deferred<typeof uiDefaults>();
    mock.client.getCameraSettings.mockReturnValueOnce(cameraLoad.promise);
    mock.client.getUiSettings.mockReturnValueOnce(uiLoad.promise);
    const result = mount(mock);

    result.reconcile(
      { ...cameraDefaults, cameraWidth: 1440 },
      { ...uiDefaults, alertDelaySeconds: 12 },
    );
    cameraLoad.resolve({ ...cameraDefaults, cameraWidth: 320 });
    uiLoad.resolve({ ...uiDefaults, alertDelaySeconds: 1 });
    await result.flush();
    flushSync();

    expect(result.settings.cameraWidth).toBe(1440);
    expect(result.settings.alertDelaySeconds).toBe(12);
    expect(mock.client.saveCameraSettings).toHaveBeenLastCalledWith({
      ...cameraDefaults,
      cameraWidth: 1440,
    });
  });

  it('preserves a write error after authoritative readback reconciliation', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);
    mock.client.saveCameraSettings.mockRejectedValueOnce(new Error('camera write failed'));

    flushSync(() => result.updateSettings({ cameraWidth: 1280, alertVolume: 0.8 }));
    await result.flush();
    flushSync();

    expect(result.error).toBe('camera write failed');
    expect(result.ready).toBe(true);
    expect(result.settings.cameraWidth).toBe(800);
    expect(result.settings.alertVolume).toBe(0.8);
  });

  it('serializes reset after an in-flight update so stale saves cannot finish last', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);
    const cameraSave = deferred<void>();
    const uiSave = deferred<void>();
    mock.client.saveCameraSettings.mockReturnValueOnce(cameraSave.promise);
    mock.client.saveUiSettings.mockReturnValueOnce(uiSave.promise);

    flushSync(() => result.updateSettings({ cameraWidth: 1280 }));
    const reset = result.resetSettings();
    expect(mock.client.resetCameraSettings).not.toHaveBeenCalled();
    cameraSave.resolve();
    uiSave.resolve();
    await reset;
    flushSync();

    expect(mock.client.resetCameraSettings).toHaveBeenCalledOnce();
    expect(result.settings).toEqual({ ...cameraDefaults, ...uiDefaults });
  });

  it('surfaces the tray toggles as true when native omits the optional UI fields', async () => {
    const mock = client();
    mock.client.getUiSettings.mockResolvedValueOnce({ alertVolume: 0.3, alertDelaySeconds: 5 });
    const result = mount(mock);
    await loaded(result);

    expect(result.settings.minimizeToTrayOnClose).toBe(true);
    expect(result.settings.startHiddenOnLogin).toBe(true);
  });

  it('persists a flipped minimizeToTrayOnClose through saveUiSettings and re-reads it', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ minimizeToTrayOnClose: false }));
    await result.flush();

    expect(mock.client.saveUiSettings).toHaveBeenLastCalledWith({
      ...uiDefaults,
      minimizeToTrayOnClose: false,
    });
    expect(mock.readUiSettings().minimizeToTrayOnClose).toBe(false);
    expect(result.settings.minimizeToTrayOnClose).toBe(false);

    await result.reload();
    expect(result.settings.minimizeToTrayOnClose).toBe(false);
    expect(result.settings.startHiddenOnLogin).toBe(true);
  });

  it('persists a flipped startHiddenOnLogin through saveUiSettings', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ startHiddenOnLogin: false }));
    await result.flush();

    expect(mock.client.saveUiSettings).toHaveBeenLastCalledWith({
      ...uiDefaults,
      startHiddenOnLogin: false,
    });
    expect(mock.readUiSettings().startHiddenOnLogin).toBe(false);
    expect(result.settings.startHiddenOnLogin).toBe(false);
  });

  it('restores the tray toggles to true on reset', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ minimizeToTrayOnClose: false, startHiddenOnLogin: false }));
    await result.flush();
    expect(result.settings.minimizeToTrayOnClose).toBe(false);
    expect(result.settings.startHiddenOnLogin).toBe(false);

    await result.resetSettings();
    expect(result.settings.minimizeToTrayOnClose).toBe(true);
    expect(result.settings.startHiddenOnLogin).toBe(true);
  });
});
