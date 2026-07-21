import { beforeEach, describe, expect, it, vi } from 'vitest';

const { invoke, listen } = vi.hoisted(() => ({
  invoke: vi.fn(),
  listen: vi.fn(),
}));

vi.mock('@tauri-apps/api/core', () => ({
  invoke,
  Channel: class Channel<T> {
    onmessage: (message: T) => void = () => undefined;
  },
}));
vi.mock('@tauri-apps/api/event', () => ({ listen }));

import {
  commands,
  getThumbnail,
  onDatasetChanged,
  saveCapture,
} from './bindings';

describe('native public bridge contract', () => {
  beforeEach(() => {
    invoke.mockReset();
    listen.mockReset();
  });

  it('wraps every generated JSON command in the same success and error envelope', async () => {
    const generatedCalls: Array<() => Promise<unknown>> = [
      () => commands.appStatus(),
      () => commands.initializeInference(),
      () => commands.trainModels(null, {} as never),
      () => commands.getTrainingStatus(),
      () => commands.cancelTraining(),
      () => commands.getDatasetPage(null, null),
      () => commands.getDatasetStats(),
      () => commands.getCameraSettings(),
      () => commands.saveCameraSettings({} as never),
      () => commands.resetCameraSettings(),
      () => commands.getUiSettings(),
      () => commands.saveUiSettings({} as never),
      () => commands.resetUiSettings(),
      () => commands.getTrainingSettings(),
      () => commands.resetTrainingSettings(),
      () => commands.saveTrainingSettings({} as never),
      () => commands.updateFrameLabel('frame-1', 'good'),
      () => commands.deleteFrame('frame-1'),
      () => commands.undoLastDatasetChange(),
      () => commands.resetDataset(),
      () => commands.getClassifierRegistry(),
      () => commands.getFeatureRegistry(),
      () => commands.getActiveModelMetadata(),
      () => commands.exportDataset(),
      () => commands.importDataset(),
      () => commands.getShortcutStatus(),
      () => commands.startCamera({ onmessage: () => undefined } as never),
      () => commands.stopCamera(),
      () => commands.listCameras(),
    ];

    for (const call of generatedCalls) {
      invoke.mockResolvedValueOnce(undefined);
      await expect(call()).resolves.toEqual({ status: 'ok', data: undefined });
    }
    const error = { kind: 'busy', message: 'busy' };
    for (const call of generatedCalls) {
      invoke.mockRejectedValueOnce(error);
      await expect(call()).resolves.toEqual({ status: 'error', error });
    }
    expect(invoke).toHaveBeenCalledTimes(generatedCalls.length * 2);
  });

  it('preserves generated command argument casing', async () => {
    invoke.mockResolvedValueOnce({
      frames: [],
      offset: 0,
      limit: 100,
      total: 0,
      version: 0,
      lastModified: 1,
    });
    await commands.getDatasetPage(0, 100);
    expect(invoke).toHaveBeenCalledWith('get_dataset_page', {
      offset: 0,
      limit: 100,
    });
  });

  it('retains raw bytes only for thumbnails and capture', async () => {
    const thumbnail = new Uint8Array([1, 2, 3]);
    invoke.mockResolvedValueOnce(thumbnail);
    await expect(getThumbnail('frame-1')).resolves.toBe(thumbnail);

    invoke.mockResolvedValueOnce(undefined);
    await saveCapture(thumbnail, {
      requestId: 10,
      token: 11,
      frameId: 'frame-1',
      timestamp: 1,
      label: 'good',
      mimeType: 'image/webp',
    });
    expect(invoke.mock.calls.map(([command]) => command)).toEqual([
      'get_thumbnail',
      'save_capture',
    ]);
  });

  it('keeps typed event adaptation outside generated command wrappers', async () => {
    listen.mockResolvedValue(() => undefined);
    const handler = vi.fn();
    await onDatasetChanged(handler);
    expect(listen).toHaveBeenCalledWith('dataset-changed', expect.any(Function));
  });
});
