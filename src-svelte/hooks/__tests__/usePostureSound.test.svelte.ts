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
const away: MultiTaskDetectionResult = { ...bad, person_found: false };

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
  alertDelaySeconds = 5,
) {
  const values = $state({ posture: initial, volume, paused, alertDelaySeconds });
  let result!: ReturnType<typeof usePostureSound>;
  const dispose = $effect.root(() => {
    result = usePostureSound(
      () => values.posture,
      () => values.volume,
      () => values.paused,
      () => values.alertDelaySeconds,
    );
  });
  disposers.push(dispose);
  flushSync();
  return { result, values, dispose };
}

/** Feeds a fresh detection (new object reference == a new detection) at wall-clock `atMs`. */
function detectAt(
  values: ReturnType<typeof mount>['values'],
  next: MultiTaskDetectionResult | null,
  atMs: number,
): void {
  vi.setSystemTime(atMs);
  values.posture = next ? { ...next } : null;
  flushSync();
}

beforeEach(() => {
  // Fake timers give a controllable Date.now via setSystemTime. The hook schedules no
  // timers of its own, so getTimerCount() stays 0 throughout.
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
  it('stays silent while the bad streak is shorter than the delay, across many detections', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 1_000);
    detectAt(harness.values, bad, 2_000);
    detectAt(harness.values, bad, 4_000);
    expect(play).not.toHaveBeenCalled();
    expect(harness.result.isPlaying).toBe(true);
  });

  it('beeps once on the detection that crosses the delay', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 4_000);
    expect(play).not.toHaveBeenCalled();
    detectAt(harness.values, bad, 5_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('requires another full delay of continued bad posture between beeps', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 5_000);
    expect(play).toHaveBeenCalledOnce();
    detectAt(harness.values, bad, 9_000);
    expect(play).toHaveBeenCalledOnce();
    detectAt(harness.values, bad, 10_000);
    expect(play).toHaveBeenCalledTimes(2);
  });

  it('never emits from elapsed time alone - only a detection arrival can beep, no timers', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    vi.setSystemTime(60_000);
    flushSync();
    expect(play).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
    detectAt(harness.values, bad, 60_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('resets the streak deadline when posture becomes good', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 4_000);
    detectAt(harness.values, good, 4_500);
    expect(pause).toHaveBeenCalled();
    expect(harness.result.isPlaying).toBe(false);
    // A fresh bad streak starts at 5s, so the old anchor at 0 no longer counts.
    detectAt(harness.values, bad, 5_000);
    detectAt(harness.values, bad, 9_000);
    expect(play).not.toHaveBeenCalled();
    detectAt(harness.values, bad, 10_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('resets the streak when the person leaves', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 4_000);
    detectAt(harness.values, away, 4_500);
    expect(harness.result.isPlaying).toBe(false);
    detectAt(harness.values, bad, 5_000);
    detectAt(harness.values, bad, 9_000);
    expect(play).not.toHaveBeenCalled();
    detectAt(harness.values, bad, 10_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('resets the streak when detections stop (null result)', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 4_000);
    detectAt(harness.values, null, 4_500);
    expect(harness.result.isPlaying).toBe(false);
    detectAt(harness.values, bad, 5_000);
    detectAt(harness.values, bad, 9_000);
    expect(play).not.toHaveBeenCalled();
    detectAt(harness.values, bad, 10_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('does not evaluate the alert on settings-only re-evaluations, even past the delay', () => {
    const harness = mount(null, 0.3, false, 5);
    detectAt(harness.values, bad, 0);
    // Time is now well past the delay, but only settings churn - no beep.
    vi.setSystemTime(20_000);
    harness.values.volume = 0.5;
    flushSync();
    harness.values.alertDelaySeconds = 5;
    flushSync();
    harness.values.paused = true;
    flushSync();
    harness.values.paused = false;
    flushSync();
    expect(play).not.toHaveBeenCalled();
    detectAt(harness.values, bad, 20_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('monitors forward neck tilt but ignores hand and mouth flags', () => {
    const forward = mount(null, 0.3, false, 0);
    detectAt(forward.values, { ...good, forward_neck_tilt: true }, 0);
    expect(play).toHaveBeenCalledOnce();

    detectAt(forward.values, { ...good, hand_near_face: true }, 1_000);
    detectAt(forward.values, { ...good, mouth_open: true }, 2_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('does not alert for null data or when no person is found', () => {
    const harness = mount(null, 0.3, false, 0);
    detectAt(harness.values, null, 0);
    detectAt(harness.values, { ...bad, person_found: false }, 1_000);
    expect(play).not.toHaveBeenCalled();
    expect(harness.result.isPlaying).toBe(false);
  });

  it('suppresses the beep while paused, then beeps on the next bad detection past the delay', () => {
    const harness = mount(null, 0.3, true, 0);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 1_000);
    expect(play).not.toHaveBeenCalled();
    expect(harness.result.isPlaying).toBe(false);
    harness.values.paused = false;
    flushSync();
    // Unpausing alone is not a detection, so nothing beeps yet.
    expect(play).not.toHaveBeenCalled();
    expect(harness.result.isPlaying).toBe(true);
    detectAt(harness.values, bad, 2_000);
    expect(play).toHaveBeenCalledOnce();
  });

  it('honors volume: mute blocks the beep, and the assigned volume is clamped', () => {
    const harness = mount(null, 0, false, 0);
    detectAt(harness.values, bad, 0);
    detectAt(harness.values, bad, 1_000);
    expect(play).not.toHaveBeenCalled();
    expect(harness.result.isPlaying).toBe(false);

    harness.values.volume = Infinity;
    flushSync();
    detectAt(harness.values, bad, 2_000);
    expect(audio.volume).toBe(1);
    expect(play).toHaveBeenCalledOnce();
  });

  it('catches synchronous play failures and still beeps on the next crossing', () => {
    play.mockImplementationOnce(() => { throw new Error('sync blocked'); });
    const harness = mount(null, 0.3, false, 0);
    detectAt(harness.values, bad, 0);
    expect(logger.error).toHaveBeenCalledWith(
      'detection',
      'Failed to play posture alert:',
      expect.any(Error),
    );
    detectAt(harness.values, bad, 1_000);
    expect(play).toHaveBeenCalledTimes(2);
  });

  it('preserves playing state after an asynchronous play rejection', async () => {
    play.mockRejectedValueOnce(new Error('blocked'));
    const harness = mount(null, 0.3, false, 0);
    detectAt(harness.values, bad, 0);
    await Promise.resolve();
    expect(harness.result.isPlaying).toBe(true);
  });

  it('does not rewind active playback during ordinary settings re-evaluation', () => {
    const harness = mount(null, 0.3, false, 0);
    detectAt(harness.values, bad, 0);
    audio.currentTime = 5;
    harness.values.volume = 0.4;
    flushSync();
    expect(audio.currentTime).toBe(5);
    expect(audio.volume).toBe(0.4);
    expect(harness.result.isPlaying).toBe(true);
  });

  it('pauses and rewinds only when the hook is disposed', () => {
    const harness = mount(null, 0.3, false, 0);
    detectAt(harness.values, bad, 0);
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
