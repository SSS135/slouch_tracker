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
  gaussianBlurKernel: 5,
  smoothingFrames: 3,
};
const uiDefaults = { alertVolume: 0.3, alertDelaySeconds: 5 };

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
      alertVolume: 0.75,
      alertDelaySeconds: 5,
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
      gaussianBlurKernel: 8,
      smoothingFrames: 20,
      alertVolume: 0,
    }));
    await result.flush();

    expect(result.settings).toEqual({
      ...cameraDefaults,
      ...uiDefaults,
      claheStrength: 10,
      gaussianBlurKernel: 7,
      smoothingFrames: 1,
      alertVolume: 0,
    });
    expect(mock.client.saveCameraSettings).toHaveBeenLastCalledWith({
      ...cameraDefaults,
      claheStrength: 10,
      gaussianBlurKernel: 7,
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

  it('normalizes gaussianBlurKernel to an odd value in 0-15, keeping the zero disable case', async () => {
    const mock = client();
    const result = mount(mock);
    await loaded(result);

    flushSync(() => result.updateSettings({ gaussianBlurKernel: 4 }));
    await result.flush();
    expect(result.settings.gaussianBlurKernel).toBe(3);

    flushSync(() => result.updateSettings({ gaussianBlurKernel: 5 }));
    await result.flush();
    expect(result.settings.gaussianBlurKernel).toBe(5);

    flushSync(() => result.updateSettings({ gaussianBlurKernel: 0 }));
    await result.flush();
    expect(result.settings.gaussianBlurKernel).toBe(0);
    expect(result.error).toBeNull();

    flushSync(() => result.updateSettings({ gaussianBlurKernel: 20 }));
    await result.flush();
    expect(result.settings.gaussianBlurKernel).toBe(15);
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
});
