import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export { commands } from './bindings.generated';
export { events } from './bindings.generated';
export type * from './bindings.generated';

export interface DatasetChangedEvent {
  version: number;
  reason: string;
}

export function getThumbnail(id: string): Promise<Uint8Array> {
  return invoke<Uint8Array>('get_thumbnail', { id });
}

export async function saveCapture(
  thumbnail: Uint8Array,
  headers: {
    requestId: number;
    token: number;
    frameId: string;
    timestamp: number;
    label: 'good' | 'bad' | 'away' | 'unused';
    mimeType: 'image/jpeg' | 'image/png' | 'image/webp';
  },
): Promise<void> {
  await invoke('save_capture', thumbnail, {
    headers: {
      'x-slouch-ipc-version': '1',
      'x-slouch-request-id': String(headers.requestId),
      'x-slouch-token': String(headers.token),
      'x-slouch-frame-id': headers.frameId,
      'x-slouch-timestamp': String(headers.timestamp),
      'x-slouch-label': headers.label,
      'x-slouch-mime-type': headers.mimeType,
    },
  });
}

export function onShortcutCapture(
  handler: (label: 'good' | 'bad' | 'away') => void,
): Promise<UnlistenFn> {
  return listen<{ label: 'good' | 'bad' | 'away' }>(
    'shortcut-capture',
    (event) => handler(event.payload.label),
  );
}

export function onDatasetChanged(
  handler: (event: DatasetChangedEvent) => void,
): Promise<UnlistenFn> {
  return listen<DatasetChangedEvent>(
    'dataset-changed',
    (event) => handler(event.payload),
  );
}
