import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Channel } from '@tauri-apps/api/core';
import type { TrainingEvent_Deserialize, TrainingResultResponse_Serialize } from '@generated/bindings';
import { NativeCommandError, type NativeClient } from '../../lib/native/client';
import { useModelTraining } from '../useModelTraining';

const flushToStorage = vi.fn().mockResolvedValue(undefined);
vi.mock('../../lib/native/client', async () => {
  const actual = await vi.importActual<typeof import('../../lib/native/client')>('../../lib/native/client');
  return {
    ...actual,
    createTrainingChannel: (handler: (event: TrainingEvent_Deserialize) => void) => ({ onmessage: handler }),
  };
});
vi.mock('../../contexts/TrainingConfigContext', () => ({
  useTrainingConfig: () => ({ flushToStorage }),
}));

const result: TrainingResultResponse_Serialize = {
  postureResult: null,
  presenceResult: null,
  success: true,
  errors: [],
  warnings: [],
};

const disposers: Array<() => void> = [];
function mount(client: Partial<NativeClient>) {
  let training!: ReturnType<typeof useModelTraining>;
  const dispose = $effect.root(() => { training = useModelTraining(client as NativeClient); });
  disposers.push(dispose);
  flushSync();
  return training;
}

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
  vi.clearAllMocks();
});

describe('useModelTraining keyed native event stream', () => {
  it('accepts contiguous active-job progress and ignores foreign or regressing events', async () => {
    const client = {
      getTrainingStatus: vi.fn().mockResolvedValue({ running: false }),
      trainModels: vi.fn().mockImplementation(async (_doCv: boolean | null, channel: Channel<TrainingEvent_Deserialize>) => {
        channel.onmessage({ type: 'started', jobId: 7, sequence: 0 });
        channel.onmessage({ type: 'progress', jobId: 8, sequence: 1, stage: 'processing', progress: 50 });
        channel.onmessage({ type: 'progress', jobId: 7, sequence: 1, stage: 'processing', progress: 5 });
        channel.onmessage({ type: 'progress', jobId: 7, sequence: 2, stage: 'evaluating', progress: 85 });
        channel.onmessage({ type: 'progress', jobId: 7, sequence: 2, stage: 'deploying', progress: 95 });
        channel.onmessage({ type: 'progress', jobId: 7, sequence: 3, stage: 'deploying', progress: 95 });
        channel.onmessage({ type: 'completed', jobId: 7, sequence: 4, result });
        return result;
      }),
    };
    const training = mount(client);
    await expect(training.train()).resolves.toBe(true);
    expect(training.state.progress).toBe(100);
    expect(training.state.error).toBeNull();
    expect(flushToStorage).toHaveBeenCalledTimes(1);
  });

  it('treats an acknowledged cancellation and cancelled command error as idle, not failure', async () => {
    let channel!: Channel<TrainingEvent_Deserialize>;
    let rejectTraining!: (cause: unknown) => void;
    const client = {
      getTrainingStatus: vi.fn().mockResolvedValue({ running: false }),
      trainModels: vi.fn().mockImplementation((_doCv: boolean | null, nextChannel: Channel<TrainingEvent_Deserialize>) => {
        channel = nextChannel;
        channel.onmessage({ type: 'started', jobId: 9, sequence: 0 });
        channel.onmessage({ type: 'progress', jobId: 9, sequence: 1, stage: 'processing', progress: 5 });
        return new Promise<TrainingResultResponse_Serialize>((_resolve, reject) => { rejectTraining = reject; });
      }),
      cancelTraining: vi.fn().mockImplementation(async () => {
        channel.onmessage({ type: 'cancelled', jobId: 9, sequence: 2 });
        rejectTraining(new NativeCommandError({ kind: 'cancelled', message: 'cancelled' }));
      }),
    };
    const training = mount(client);
    const pending = training.train();
    await vi.waitFor(() => expect(training.state.isTraining).toBe(true));
    await training.cancel();
    await expect(pending).resolves.toBe(false);
    expect(training.state).toMatchObject({ isTraining: false, stage: 'idle', error: null });
  });

  it('keeps a cancellation idle when reconcile runs before training settles', async () => {
    let channel!: Channel<TrainingEvent_Deserialize>;
    let rejectTraining!: (cause: unknown) => void;
    const client = {
      getTrainingStatus: vi.fn().mockResolvedValue({ running: false }),
      trainModels: vi.fn().mockImplementation((_doCv: boolean | null, nextChannel: Channel<TrainingEvent_Deserialize>) => {
        channel = nextChannel;
        channel.onmessage({ type: 'started', jobId: 11, sequence: 0 });
        channel.onmessage({ type: 'progress', jobId: 11, sequence: 1, stage: 'processing', progress: 5 });
        // Long-running: settles only after cancel()+reconcile have fully completed.
        return new Promise<TrainingResultResponse_Serialize>((_resolve, reject) => { rejectTraining = reject; });
      }),
      cancelTraining: vi.fn().mockImplementation(async () => {
        // Acknowledge cancellation via terminal event, but defer trainModels rejection.
        channel.onmessage({ type: 'cancelled', jobId: 11, sequence: 2 });
      }),
    };
    const training = mount(client);
    const pending = training.train();
    await vi.waitFor(() => expect(training.state.isTraining).toBe(true));
    // cancelTraining resolves and reconcile completes here, while trainModels is still pending.
    await training.cancel();
    expect(client.getTrainingStatus).toHaveBeenCalled();
    // Now settle the training promise on a later turn.
    rejectTraining(new NativeCommandError({ kind: 'cancelled', message: 'cancelled' }));
    await expect(pending).resolves.toBe(false);
    expect(training.state).toMatchObject({ isTraining: false, stage: 'idle', error: null });
  });
});
