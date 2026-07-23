import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Channel } from '@tauri-apps/api/core';
import type { InferenceUiResult } from '@generated/bindings';
import { createMockNativeInferenceResult } from '../../__tests__/utils/mockNativeInferenceResult';
import type { NativeClient } from '../../lib/native/client';
import { useNativeCamera } from '../useNativeCamera.svelte';

vi.mock('@tauri-apps/api/core', () => ({
  Channel: class {
    onmessage: (message: unknown) => void = () => undefined;
  },
}));

const disposers: Array<() => void> = [];

function mount(client: Partial<NativeClient>, onResult = vi.fn()) {
  let hook!: ReturnType<typeof useNativeCamera>;
  const dispose = $effect.root(() => {
    hook = useNativeCamera({ client: client as NativeClient, onResult });
  });
  disposers.push(dispose);
  flushSync();
  return { hook, onResult, dispose };
}

function readyClient() {
  return {
    appStatus: vi.fn().mockResolvedValue({ inferenceReady: true }),
    startCamera: vi.fn().mockResolvedValue(undefined),
    stopCamera: vi.fn().mockResolvedValue(undefined),
  };
}

function mountToggle(client: Partial<NativeClient>, initialEnabled = true) {
  const state = $state({ enabled: initialEnabled });
  let hook!: ReturnType<typeof useNativeCamera>;
  const dispose = $effect.root(() => {
    hook = useNativeCamera({
      client: client as NativeClient,
      onResult: vi.fn(),
      get enabled() {
        return state.enabled;
      },
    });
  });
  disposers.push(dispose);
  flushSync();
  return {
    hook,
    setEnabled: (value: boolean) => {
      state.enabled = value;
      flushSync();
    },
  };
}

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
  vi.clearAllMocks();
});

describe('useNativeCamera lifecycle', () => {
  it('starts the native camera once inference is ready', async () => {
    const client = readyClient();
    const { hook } = mount(client);
    await vi.waitFor(() => expect(client.startCamera).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(hook.ready).toBe(true));
    expect(hook.error).toBeNull();
  });

  it('forwards pushed channel results to the consumer', async () => {
    const client = readyClient();
    const { onResult } = mount(client);
    await vi.waitFor(() => expect(client.startCamera).toHaveBeenCalled());
    const channel = client.startCamera.mock.calls[0][0] as Channel<InferenceUiResult>;
    const result = createMockNativeInferenceResult({ requestId: 3, token: 33 });
    channel.onmessage(result);
    expect(onResult).toHaveBeenCalledWith(result);
  });

  it('stops the native camera on teardown', async () => {
    const client = readyClient();
    const { hook, dispose } = mount(client);
    await vi.waitFor(() => expect(hook.ready).toBe(true));
    dispose();
    expect(client.stopCamera).toHaveBeenCalledTimes(1);
  });

  it('surfaces a start failure through the error field', async () => {
    const client = {
      appStatus: vi.fn().mockResolvedValue({ inferenceReady: true }),
      startCamera: vi.fn().mockRejectedValue(new Error('no capture device')),
      stopCamera: vi.fn().mockResolvedValue(undefined),
    };
    const { hook } = mount(client);
    await vi.waitFor(() => expect(hook.error).toBe('no capture device'));
    expect(hook.ready).toBe(false);
  });
});

describe('useNativeCamera enabled toggle (pause/resume wiring)', () => {
  it('stops the camera when disabled and restarts it when re-enabled', async () => {
    const client = readyClient();
    const { hook, setEnabled } = mountToggle(client, true);
    await vi.waitFor(() => expect(hook.ready).toBe(true));
    expect(client.startCamera).toHaveBeenCalledTimes(1);

    setEnabled(false);
    expect(client.stopCamera).toHaveBeenCalledTimes(1);
    expect(hook.ready).toBe(false);

    setEnabled(true);
    await vi.waitFor(() => expect(client.startCamera).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(hook.ready).toBe(true));
    // Never issues a second stop for a single pause.
    expect(client.stopCamera).toHaveBeenCalledTimes(1);
  });

  it('never starts the camera while disabled from the outset', async () => {
    const client = readyClient();
    const { hook } = mountToggle(client, false);
    await Promise.resolve();
    flushSync();
    expect(client.startCamera).not.toHaveBeenCalled();
    expect(hook.ready).toBe(false);
  });
});
