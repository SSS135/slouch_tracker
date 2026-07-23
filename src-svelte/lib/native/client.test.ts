import { beforeEach, describe, expect, it, vi } from 'vitest';

const bridge = vi.hoisted(() => ({
  appStatus: vi.fn(),
  getCameraSettings: vi.fn(),
  saveCameraSettings: vi.fn(),
  resetCameraSettings: vi.fn(),
  getUiSettings: vi.fn(),
  saveUiSettings: vi.fn(),
  resetUiSettings: vi.fn(),
  startCamera: vi.fn(),
  stopCamera: vi.fn(),
  listCameras: vi.fn(),
  getAutostartEnabled: vi.fn(),
  setAutostartEnabled: vi.fn(),
  getThumbnail: vi.fn(),
  saveCapture: vi.fn(),
}));

vi.mock('@generated/bindings', () => ({
  commands: {
    appStatus: bridge.appStatus,
    getCameraSettings: bridge.getCameraSettings,
    saveCameraSettings: bridge.saveCameraSettings,
    resetCameraSettings: bridge.resetCameraSettings,
    getUiSettings: bridge.getUiSettings,
    saveUiSettings: bridge.saveUiSettings,
    resetUiSettings: bridge.resetUiSettings,
    startCamera: bridge.startCamera,
    stopCamera: bridge.stopCamera,
    listCameras: bridge.listCameras,
    getAutostartEnabled: bridge.getAutostartEnabled,
    setAutostartEnabled: bridge.setAutostartEnabled,
  },
  getThumbnail: bridge.getThumbnail,
  saveCapture: bridge.saveCapture,
  onDatasetChanged: vi.fn(),
  onShortcutCapture: vi.fn(),
}));

import {
  NativeCommandError,
  nativeClient,
  saveCapture,
  unwrapNativeResult,
} from './client';

describe('Svelte native client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('unwraps generated command envelopes and preserves typed errors', async () => {
    bridge.appStatus.mockResolvedValueOnce({
      status: 'ok',
      data: {
        ready: true,
        inferenceReady: true,
        datasetVersion: 2,
        storage: { used: 1, available: 2, quota: 3 },
      },
    });
    await expect(nativeClient.appStatus()).resolves.toMatchObject({ ready: true });

    expect(() => unwrapNativeResult({
      status: 'error',
      error: { kind: 'busy', message: 'mailbox full' },
    })).toThrow(NativeCommandError);
  });

  it('routes camera and UI settings only through generated native commands', async () => {
    const camera = {
      cameraWidth: 800,
      cameraHeight: 600,
      captureIntervalSeconds: 0.5,
      autoCaptureEnabled: true,
      autoCaptureIntervalSeconds: 2,
      privacyMode: true,
      claheStrength: 3.5,
      gaussianBlurKernel: 5,
      smoothingFrames: 3,
      showDetectionOverlay: false,
    };
    const ui = { alertVolume: 0.3, alertDelaySeconds: 5 };
    bridge.getCameraSettings.mockResolvedValueOnce({ status: 'ok', data: camera });
    bridge.getUiSettings.mockResolvedValueOnce({ status: 'ok', data: ui });
    bridge.saveCameraSettings.mockResolvedValueOnce({ status: 'ok', data: null });
    bridge.saveUiSettings.mockResolvedValueOnce({ status: 'ok', data: null });

    await expect(nativeClient.getCameraSettings()).resolves.toEqual(camera);
    await expect(nativeClient.getUiSettings()).resolves.toEqual(ui);
    await nativeClient.saveCameraSettings(camera);
    await nativeClient.saveUiSettings(ui);
    expect(bridge.saveCameraSettings).toHaveBeenCalledWith(camera);
    expect(bridge.saveUiSettings).toHaveBeenCalledWith(ui);
  });

  it('drives the native camera lifecycle through generated commands', async () => {
    const channel = { onmessage: () => undefined } as never;
    bridge.startCamera.mockResolvedValueOnce({ status: 'ok', data: null });
    bridge.stopCamera.mockResolvedValueOnce({ status: 'ok', data: null });
    bridge.listCameras.mockResolvedValueOnce({
      status: 'ok',
      data: [{ index: '0', name: 'Mock Camera', description: 'device' }],
    });

    await nativeClient.startCamera(channel);
    await nativeClient.stopCamera();
    await expect(nativeClient.listCameras()).resolves.toEqual([
      { index: '0', name: 'Mock Camera', description: 'device' },
    ]);
    expect(bridge.startCamera).toHaveBeenCalledWith(channel);
    expect(bridge.stopCamera).toHaveBeenCalledTimes(1);
  });

  it('surfaces typed errors when starting the camera fails', async () => {
    bridge.startCamera.mockResolvedValueOnce({ status: 'error', error: { kind: 'inference', message: 'no device' } });
    await expect(nativeClient.startCamera({ onmessage: () => undefined } as never)).rejects.toMatchObject({
      name: 'NativeCommandError',
      kind: 'inference',
      message: 'no device',
    });
  });

  it('reads and writes autostart through generated commands', async () => {
    bridge.getAutostartEnabled.mockResolvedValueOnce({ status: 'ok', data: true });
    bridge.setAutostartEnabled.mockResolvedValueOnce({ status: 'ok', data: null });

    await expect(nativeClient.getAutostartEnabled()).resolves.toBe(true);
    await nativeClient.setAutostartEnabled(false);
    expect(bridge.setAutostartEnabled).toHaveBeenCalledWith(false);
  });

  it('surfaces typed errors when toggling autostart fails', async () => {
    bridge.setAutostartEnabled.mockResolvedValueOnce({
      status: 'error',
      error: { kind: 'internal', message: 'autostart registry operation failed: denied' },
    });
    await expect(nativeClient.setAutostartEnabled(true)).rejects.toMatchObject({
      name: 'NativeCommandError',
      kind: 'internal',
    });
  });

  it('enforces the raw thumbnail size contract', async () => {
    await expect(saveCapture(new Uint8Array(), {
      requestId: 1,
      token: 2,
      frameId: 'frame-1',
      timestamp: 1,
      label: 'good',
      mimeType: 'image/webp',
    })).rejects.toThrow('Thumbnail must contain between 1 byte and 2 MiB');
    expect(bridge.saveCapture).not.toHaveBeenCalled();
  });
});
