import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const getActiveModelMetadata = vi.hoisted(() => vi.fn());
vi.mock('../../lib/native/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../lib/native/client')>();
  return {
    ...actual,
    nativeClient: { getActiveModelMetadata },
  };
});

import { NativeCommandError } from '../../lib/native/client';
import { usePostureClassifier } from '../usePostureClassifier';

const metadata = {
  posture: { classifierId: 'knn', trainedAt: 1_000, featureTypes: ['gau_features'] },
  presence: { classifierId: 'mlp', trainedAt: 1_000, featureTypes: ['rtmdet_engineered'] },
};

const disposers: Array<() => void> = [];
function mount() {
  let result!: ReturnType<typeof usePostureClassifier>;
  const dispose = $effect.root(() => { result = usePostureClassifier(); });
  disposers.push(dispose);
  flushSync();
  return result;
}

beforeEach(() => {
  getActiveModelMetadata.mockReset();
  getActiveModelMetadata.mockResolvedValue({ posture: null, presence: null });
});
afterEach(() => {
  while (disposers.length) disposers.pop()?.();
});

describe('usePostureClassifier native metadata facade', () => {
  it('loads only native model metadata and never model payloads', async () => {
    getActiveModelMetadata.mockResolvedValueOnce(metadata);
    const result = mount();
    expect(result.isLoading).toBe(true);
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    expect(result.postureModel).toEqual(metadata.posture);
    expect(result.presenceModel).toEqual(metadata.presence);
  });

  it('reconciles metadata on demand', async () => {
    const result = mount();
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    getActiveModelMetadata.mockResolvedValueOnce(metadata);
    await result.reloadModel();
    expect(result.postureModel?.classifierId).toBe('knn');
  });

  it('clears a prior native failure when a reload succeeds', async () => {
    getActiveModelMetadata.mockRejectedValueOnce(new NativeCommandError({
      kind: 'notReady',
      message: 'native unavailable',
    }));
    const result = mount();
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    expect(result.error).toBe('native unavailable');

    getActiveModelMetadata.mockResolvedValueOnce(metadata);
    await result.reloadModel();
    expect(result.error).toBeNull();
    expect(result.postureModel?.classifierId).toBe('knn');
  });

  it('surfaces failures that occur during a reload', async () => {
    const result = mount();
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    expect(result.error).toBeNull();

    getActiveModelMetadata.mockRejectedValueOnce(new NativeCommandError({
      kind: 'notReady',
      message: 'reload boom',
    }));
    await result.reloadModel();
    expect(result.error).toBe('reload boom');
    expect(result.isLoading).toBe(false);
  });

  it('exposes typed native failures without a browser model fallback', async () => {
    getActiveModelMetadata.mockRejectedValueOnce(new NativeCommandError({
      kind: 'notReady',
      message: 'native unavailable',
    }));
    const result = mount();
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    expect(result.error).toBe('native unavailable');
    expect(result.postureModel).toBeNull();
  });

  it('falls back to a generic message for non-Error native rejections', async () => {
    getActiveModelMetadata.mockRejectedValueOnce('String error');
    const result = mount();
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    expect(result.error).toBe('Unknown error');
    expect(result.postureModel).toBeNull();
  });

  it('clears transient metadata and error without reloading native models', async () => {
    getActiveModelMetadata.mockRejectedValueOnce(new NativeCommandError({
      kind: 'notReady',
      message: 'native unavailable',
    }));
    const result = mount();
    await vi.waitFor(() => expect(result.isLoading).toBe(false));
    expect(result.error).toBe('native unavailable');

    getActiveModelMetadata.mockResolvedValueOnce(metadata);
    await result.reloadModel();
    await vi.waitFor(() => expect(result.postureModel).not.toBeNull());

    const callsBeforeClear = getActiveModelMetadata.mock.calls.length;
    await result.clearModel();
    expect(result.postureModel).toBeNull();
    expect(result.presenceModel).toBeNull();
    expect(result.error).toBeNull();
    expect(getActiveModelMetadata.mock.calls.length).toBe(callsBeforeClear);
  });
});
