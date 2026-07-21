import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MultiTaskDetectionResult } from '../../services/posture/types';
import { logger } from '../../services/logging/logger';
import { usePostureSound } from '../usePostureSound';

vi.mock('../../services/logging/logger', () => ({
  logger: { warn: vi.fn(), error: vi.fn(), info: vi.fn() },
}));
vi.mock('../../utils/runtimeEnv', () => ({
  resolveAssetUrl: vi.fn((path: string) => `/assets/${path}`),
}));

const bad: MultiTaskDetectionResult = {
  person_found: true,
  slouching: true,
  forward_neck_tilt: false,
  hand_near_face: false,
  mouth_open: false,
};
const good: MultiTaskDetectionResult = { ...bad, slouching: false };

let play: ReturnType<typeof vi.fn>;
let pause: ReturnType<typeof vi.fn>;
let audio: {
  play: typeof play;
  pause: typeof pause;
  currentTime: number;
  volume: number;
  loop: boolean;
  preload: string;
};
const disposers: Array<() => void> = [];

function installAudio(): void {
  class AudioMock {
    constructor() { return audio; }
  }
  globalThis.Audio = AudioMock as unknown as typeof Audio;
  window.Audio = AudioMock as unknown as typeof Audio;
}

function mount(
  initial: MultiTaskDetectionResult | null = bad,
  volume = 0.3,
  paused = false,
  delay = 5,
) {
  const values = $state({ posture: initial, volume, paused, delay });
  let result!: ReturnType<typeof usePostureSound>;
  const dispose = $effect.root(() => {
    result = usePostureSound(
      () => values.posture,
      () => values.volume,
      () => values.paused,
      () => values.delay,
    );
  });
  disposers.push(dispose);
  flushSync();
  return { result, values, dispose };
}

function reevaluate(values: ReturnType<typeof mount>['values']): void {
  values.posture = values.posture ? { ...values.posture } : null;
  flushSync();
}

beforeEach(() => {
  vi.useFakeTimers({ now: 0 });
  play = vi.fn().mockResolvedValue(undefined);
  pause = vi.fn();
  audio = { play, pause, currentTime: 0, volume: 0, loop: false, preload: '' };
  installAudio();
});

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('usePostureSound', () => {
  it('does not autonomously alert without a fresh posture input', () => {
    const harness = mount();
    vi.advanceTimersByTime(10_000);
    expect(play).not.toHaveBeenCalled();
    reevaluate(harness.values);
    expect(play).toHaveBeenCalledOnce();
    expect(harness.result.isPlaying).toBe(true);
  });

  it('waits for the configured delay and rate-limits fresh inputs to one second', () => {
    const harness = mount();
    vi.advanceTimersByTime(4_999);
    reevaluate(harness.values);
    expect(play).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    reevaluate(harness.values);
    expect(play).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(999);
    reevaluate(harness.values);
    expect(play).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(1);
    reevaluate(harness.values);
    expect(play).toHaveBeenCalledTimes(2);
  });

  it('preserves the original bad-posture deadline while paused', () => {
    const harness = mount();
    vi.advanceTimersByTime(4_000);
    harness.values.paused = true;
    flushSync();
    vi.advanceTimersByTime(1_000);
    harness.values.paused = false;
    flushSync();
    expect(play).toHaveBeenCalledOnce();
  });

  it('preserves the original bad-posture deadline while muted', () => {
    const harness = mount();
    vi.advanceTimersByTime(4_000);
    harness.values.volume = 0;
    flushSync();
    vi.advanceTimersByTime(1_000);
    harness.values.volume = 0.3;
    flushSync();
    expect(play).toHaveBeenCalledOnce();
  });

  it('resets timing only when posture becomes good', () => {
    const harness = mount();
    vi.advanceTimersByTime(4_000);
    harness.values.posture = good;
    flushSync();
    expect(pause).toHaveBeenCalled();
    harness.values.posture = bad;
    flushSync();
    vi.advanceTimersByTime(1_000);
    reevaluate(harness.values);
    expect(play).not.toHaveBeenCalled();
  });

  it('blocks an otherwise-eligible alert while paused, then plays when unpaused', () => {
    const harness = mount(bad, 0.3, true, 0);
    expect(harness.result.isPlaying).toBe(false);
    expect(play).not.toHaveBeenCalled();
    harness.values.paused = false;
    flushSync();
    expect(harness.result.isPlaying).toBe(true);
    expect(play).toHaveBeenCalledOnce();
  });

  it('monitors forward neck tilt but ignores hand and mouth flags', () => {
    const forward = mount({ ...good, forward_neck_tilt: true }, 0.3, false, 0);
    expect(forward.result.isPlaying).toBe(true);
    expect(play).toHaveBeenCalledOnce();

    mount({ ...good, hand_near_face: true }, 0.3, false, 0);
    mount({ ...good, mouth_open: true }, 0.3, false, 0);
    expect(play).toHaveBeenCalledOnce();
  });

  it('does not alert for null data or when no person is found', () => {
    expect(mount(null, 0.3, false, 0).result.isPlaying).toBe(false);
    expect(mount({ ...bad, person_found: false }, 0.3, false, 0).result.isPlaying).toBe(false);
    expect(play).not.toHaveBeenCalled();
  });

  it('clamps only the assigned audio volume while retaining raw eligibility', () => {
    mount(bad, Infinity, false, 0);
    expect(audio.volume).toBe(1);
    expect(play).toHaveBeenCalledOnce();

    const negative = mount(bad, -0.5, false, 0);
    expect(audio.volume).toBe(0);
    expect(negative.result.isPlaying).toBe(false);
    expect(play).toHaveBeenCalledOnce();

    const nan = mount(bad, Number.NaN, false, 0);
    expect(audio.volume).toBeNaN();
    expect(nan.result.isPlaying).toBe(false);
  });

  it('handles non-finite and negative delay values without scheduling timers', () => {
    expect(mount(bad, 0.3, false, Infinity).result.isPlaying).toBe(false);
    expect(mount(bad, 0.3, false, -Infinity).result.isPlaying).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('catches synchronous play failures without recording a false rate-limit time', () => {
    play.mockImplementationOnce(() => { throw new Error('sync blocked'); });
    const harness = mount(bad, 0.3, false, 0);
    expect(logger.error).toHaveBeenCalledWith(
      'detection',
      'Failed to play posture alert:',
      expect.any(Error),
    );
    vi.advanceTimersByTime(100);
    reevaluate(harness.values);
    expect(play).toHaveBeenCalledTimes(2);
  });

  it('preserves oracle playing state after asynchronous play rejection', async () => {
    play.mockRejectedValueOnce(new Error('blocked'));
    const harness = mount(bad, 0.3, false, 0);
    await Promise.resolve();
    expect(harness.result.isPlaying).toBe(true);
    expect(logger.warn).not.toHaveBeenCalledWith(
      'detection',
      'Posture alert playback was blocked:',
      expect.anything(),
    );
  });

  it('does not rewind active playback during ordinary reactive reevaluation', () => {
    const harness = mount(bad, 0.3, false, 0);
    audio.currentTime = 5;
    harness.values.volume = 0.4;
    flushSync();
    expect(audio.currentTime).toBe(5);
    expect(audio.volume).toBe(0.4);
    expect(harness.result.isPlaying).toBe(true);
  });

  it('pauses and rewinds only when the hook is disposed', () => {
    const harness = mount(bad, 0.3, false, 0);
    audio.currentTime = 4;
    harness.dispose();
    disposers.pop();
    expect(pause).toHaveBeenCalled();
    expect(audio.currentTime).toBe(0);
  });

  it('warns and returns a stable disabled result when Audio is unavailable', () => {
    Reflect.deleteProperty(window, 'Audio');
    Reflect.deleteProperty(globalThis, 'Audio');
    const harness = mount(bad, 0.3, false, 0);
    expect(harness.result.isPlaying).toBe(false);
    expect(logger.warn).toHaveBeenCalledWith(
      'detection',
      'HTMLAudioElement not available. Posture sound disabled.',
    );
    expect(Object.keys(harness.result)).toEqual(['isPlaying']);
  });
});
