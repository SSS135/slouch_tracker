import { logger } from '../services/logging/logger';
import type { MultiTaskDetectionResult } from '../services/posture/types';
import { resolveAssetUrl } from '../utils/runtimeEnv';

type AudioHandle = Pick<HTMLAudioElement, 'play' | 'pause' | 'currentTime'> & {
  volume: number;
  loop: boolean;
};
type ReactiveValue<T> = T | (() => T);
const read = <T>(value: ReactiveValue<T>): T =>
  typeof value === 'function' ? (value as () => T)() : value;

// Sentinel distinct from every possible posture value (including null) so the very
// first evaluation registers as a fresh detection.
const UNSEEN = Symbol('unseen-posture');

function createAudio(): AudioHandle | null {
  if (typeof window === 'undefined' || typeof window.Audio === 'undefined') {
    logger.warn('detection', 'HTMLAudioElement not available. Posture sound disabled.');
    return null;
  }
  const source = resolveAssetUrl('posture-alert.mp3');
  let audio: HTMLAudioElement;
  try {
    audio = new window.Audio(source);
  } catch {
    audio = (window.Audio as unknown as (url: string) => HTMLAudioElement)(source);
  }
  audio.preload = 'auto';
  audio.loop = false;
  return audio;
}

/**
 * Emits the bad-posture alert purely as a consequence of detection results — there is no
 * wall-clock timer, so a beep can only fire when a fresh detection arrives (~1-2 fps).
 * The threshold stays time-based: a beep fires when a bad detection lands at least
 * `alertDelaySeconds` after the anchor, where the anchor is the start of the current bad
 * streak and is reset to "now" on every beep. Repeats therefore also require
 * `alertDelaySeconds` of continued bad posture, evaluated only at detection arrivals. Any
 * good/away/no-person result clears the anchor, so the sound stops the instant posture
 * recovers or detections stop being bad.
 */
export function usePostureSound(
  postureData: ReactiveValue<MultiTaskDetectionResult | null>,
  volume: ReactiveValue<number> = 0.3,
  paused: ReactiveValue<boolean> = false,
  alertDelaySeconds: ReactiveValue<number> = 5,
): { readonly isPlaying: boolean } {
  const audio = createAudio();
  let isPlaying = $state(false);
  // Start of the current bad streak / timestamp of the last beep; null when not bad.
  let anchorMs: number | null = null;
  let lastSeen: MultiTaskDetectionResult | null | typeof UNSEEN = UNSEEN;

  const playAlert = (): void => {
    if (!audio) return;
    try {
      audio.pause();
      audio.currentTime = 0;
      void audio.play();
    } catch (error) {
      logger.error('detection', 'Failed to play posture alert:', error);
    }
  };

  $effect(() => {
    if (!audio) return;
    const data = read(postureData);
    const rawVolume = read(volume);
    const isPaused = read(paused);
    const delaySeconds = read(alertDelaySeconds);
    audio.volume = Math.max(0, Math.min(1, rawVolume));

    // A genuinely new detection carries a fresh object reference; settings-driven re-runs
    // (volume/paused/delay) keep the same reference. Only detections move the streak anchor
    // and may emit — settings changes must never evaluate the alert.
    const isNewDetection = data !== lastSeen;
    lastSeen = data;

    const bad = Boolean(data?.person_found && (data.slouching || data.forward_neck_tilt));
    if (isNewDetection) {
      anchorMs = bad ? (anchorMs ?? Date.now()) : null;
    }

    const eligible = bad && rawVolume > 0 && !isPaused;
    isPlaying = eligible;
    if (!eligible) {
      audio.pause();
      return;
    }

    if (isNewDetection && anchorMs !== null && (Date.now() - anchorMs) / 1000 >= delaySeconds) {
      anchorMs = Date.now();
      playAlert();
    }
  });

  $effect(() => () => {
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
  });

  return { get isPlaying() { return isPlaying; } };
}
