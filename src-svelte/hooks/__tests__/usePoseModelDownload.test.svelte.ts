import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { clearMocks, mockIPC, mockWindows } from '@tauri-apps/api/mocks';
import type { Channel } from '@tauri-apps/api/core';
import type { PoseModelDownloadEvent, PoseModelStatus } from '@generated/bindings';
import { usePoseModelDownload } from '../usePoseModelDownload.svelte';

const TOTAL = 245 * 1024 * 1024;

interface Harness {
  status: PoseModelStatus;
  channel: Channel<PoseModelDownloadEvent> | null;
  ensureCalls: number;
}

const disposers: Array<() => void> = [];

function installIpc(harness: Harness): void {
  clearMocks();
  mockWindows('main');
  mockIPC((command, args) => {
    switch (command) {
      case 'get_pose_model_status':
        return harness.status;
      case 'ensure_pose_model':
        harness.ensureCalls += 1;
        harness.channel = (args as { onEvent: Channel<PoseModelDownloadEvent> }).onEvent;
        return null;
      default:
        throw { kind: 'invalidRequest', message: `unexpected ${command}` };
    }
  });
}

function mount(harness: Harness) {
  const onReady = vi.fn();
  let hook!: ReturnType<typeof usePoseModelDownload>;
  const dispose = $effect.root(() => {
    hook = usePoseModelDownload({ onReady });
  });
  disposers.push(dispose);
  flushSync();
  return { hook, onReady };
}

function emit(harness: Harness, event: PoseModelDownloadEvent): void {
  harness.channel?.onmessage?.(event);
  flushSync();
}

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
  clearMocks();
});

describe('usePoseModelDownload', () => {
  it('proceeds without a screen when the model is already present', async () => {
    const harness: Harness = { status: { type: 'ready', path: 'X:/models/nlf.onnx' }, channel: null, ensureCalls: 0 };
    installIpc(harness);
    const { hook } = mount(harness);

    await vi.waitFor(() => expect(hook.phase.kind).toBe('ready'));
    expect(hook.blocking).toBe(false);
    expect(harness.ensureCalls).toBe(0);
  });

  it('auto-starts the download and blocks when the model must be fetched', async () => {
    const harness: Harness = { status: { type: 'downloadRequired', totalBytes: TOTAL }, channel: null, ensureCalls: 0 };
    installIpc(harness);
    const { hook } = mount(harness);

    await vi.waitFor(() => expect(hook.phase.kind).toBe('downloading'));
    expect(harness.ensureCalls).toBe(1);
    expect(hook.blocking).toBe(true);
  });

  it('adopts progress events then reaches ready and re-initializes inference', async () => {
    const harness: Harness = { status: { type: 'downloadRequired', totalBytes: TOTAL }, channel: null, ensureCalls: 0 };
    installIpc(harness);
    const { hook, onReady } = mount(harness);
    await vi.waitFor(() => expect(hook.phase.kind).toBe('downloading'));

    emit(harness, { type: 'started', totalBytes: TOTAL });
    expect(hook.phase).toEqual({ kind: 'downloading', received: 0, total: TOTAL });

    emit(harness, { type: 'progress', received: TOTAL / 2, total: TOTAL });
    expect(hook.phase).toEqual({ kind: 'downloading', received: TOTAL / 2, total: TOTAL });

    emit(harness, { type: 'verifying' });
    expect(hook.phase.kind).toBe('verifying');

    emit(harness, { type: 'ready' });
    expect(hook.phase.kind).toBe('ready');
    expect(hook.blocking).toBe(false);
    await vi.waitFor(() => expect(onReady).toHaveBeenCalledTimes(1));
  });

  it('surfaces a failure with a retry that restarts the download', async () => {
    const harness: Harness = { status: { type: 'downloadRequired', totalBytes: TOTAL }, channel: null, ensureCalls: 0 };
    installIpc(harness);
    const { hook } = mount(harness);
    await vi.waitFor(() => expect(hook.phase.kind).toBe('downloading'));

    emit(harness, { type: 'failed', reason: 'network connection failed' });
    expect(hook.phase).toMatchObject({ kind: 'failed', offline: true });
    expect(hook.blocking).toBe(true);

    hook.retry();
    flushSync();
    expect(hook.phase.kind).toBe('downloading');
    expect(harness.ensureCalls).toBe(2);
  });

  it('flags a non-network failure as not offline', async () => {
    const harness: Harness = { status: { type: 'downloadRequired', totalBytes: TOTAL }, channel: null, ensureCalls: 0 };
    installIpc(harness);
    const { hook } = mount(harness);
    await vi.waitFor(() => expect(hook.phase.kind).toBe('downloading'));

    emit(harness, { type: 'failed', reason: 'checksum mismatch after download' });
    expect(hook.phase).toMatchObject({ kind: 'failed', offline: false });
  });

  it('cancels the in-flight download and ignores its late events', async () => {
    const harness: Harness = { status: { type: 'downloadRequired', totalBytes: TOTAL }, channel: null, ensureCalls: 0 };
    installIpc(harness);
    const { hook } = mount(harness);
    await vi.waitFor(() => expect(hook.phase.kind).toBe('downloading'));
    const stale = harness.channel;

    hook.cancel();
    flushSync();
    expect(hook.phase.kind).toBe('cancelled');

    // A late event from the abandoned channel must not resurrect progress.
    stale?.onmessage?.({ type: 'progress', received: TOTAL, total: TOTAL });
    flushSync();
    expect(hook.phase.kind).toBe('cancelled');

    hook.retry();
    flushSync();
    expect(hook.phase.kind).toBe('downloading');
    expect(harness.ensureCalls).toBe(2);
  });
});
