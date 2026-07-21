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

/** Evaluates posture audio only when reactive inputs change, as the oracle does. */
export function usePostureSound(
  postureData: ReactiveValue<MultiTaskDetectionResult | null>,
  volume: ReactiveValue<number> = 0.3,
  paused: ReactiveValue<boolean> = false,
  alertDelaySeconds: ReactiveValue<number> = 5,
): { readonly isPlaying: boolean } {
  const audio = createAudio();
  let isPlaying = $state(false);
  let badSince: number | null = null;
  let lastPlayedAt: number | null = null;

  $effect(() => {
    if (!audio) return;
    const data = read(postureData);
    const rawVolume = read(volume);
    const isPaused = read(paused);
    const delaySeconds = read(alertDelaySeconds);
    audio.volume = Math.max(0, Math.min(1, rawVolume));

    const bad = Boolean(data?.person_found && (data.slouching || data.forward_neck_tilt));
    const now = Date.now();
    if (bad) {
      badSince ??= now;
    } else {
      badSince = null;
      lastPlayedAt = null;
    }

    const elapsedSeconds = badSince === null ? 0 : (now - badSince) / 1000;
    const shouldPlay = bad
      && elapsedSeconds >= delaySeconds
      && rawVolume > 0
      && !isPaused;
    isPlaying = shouldPlay;

    if (!shouldPlay) {
      audio.pause();
      return;
    }

    const timeSinceLastPlay = lastPlayedAt === null ? Infinity : now - lastPlayedAt;
    if (timeSinceLastPlay < 1000) return;

    try {
      audio.pause();
      audio.currentTime = 0;
      void audio.play();
      lastPlayedAt = now;
    } catch (error) {
      logger.error('detection', 'Failed to play posture alert:', error);
    }
  });

  $effect(() => () => {
    if (!audio) return;
    audio.pause();
    audio.currentTime = 0;
  });

  return { get isPlaying() { return isPlaying; } };
}
