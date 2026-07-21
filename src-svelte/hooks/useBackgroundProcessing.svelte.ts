import { SvelteSet } from 'svelte/reactivity';
import { logger } from '../services/logging/logger';

interface BackgroundProcessingOptions {
  onVisibilityChange?: (isVisible: boolean) => void;
}

type WakeLockHandle = {
  release: () => Promise<void>;
  addEventListener: (type: 'release', listener: () => void) => void;
};

function errorMessage(error: unknown): unknown {
  if (typeof error === 'object' && error !== null && 'message' in error) {
    const message = (error as { message?: unknown }).message;
    if (message) return message;
  }
  return error;
}

/** Manages visibility, wake-lock ownership, and background title flashes. */
export function useBackgroundProcessing(options: BackgroundProcessingOptions) {
  let isVisible = $state(!document.hidden);
  let isBackgroundCapable = $state(false);
  let visibilityCallback = options.onVisibilityChange;
  let wakeLock: WakeLockHandle | null = null;
  let wakeLockGeneration = 0;
  const titleTimers = new SvelteSet<ReturnType<typeof setTimeout>>();
  const originalTitle = document.title;

  const clearTitleTimers = (): void => {
    for (const timer of titleTimers) clearTimeout(timer);
    titleTimers.clear();
  };

  $effect(() => {
    visibilityCallback = options.onVisibilityChange;
  });

  $effect(() => {
    const handleVisibilityChange = (): void => {
      const visible = !document.hidden;
      isVisible = visible;
      logger.debug('detection', `Tab ${visible ? 'visible' : 'hidden'} - Background processing: enabled`);
      visibilityCallback?.(visible);
      if (visible) {
        clearTitleTimers();
        document.title = originalTitle;
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      clearTitleTimers();
      document.title = originalTitle;
    };
  });

  $effect(() => {
    const visible = isVisible;
    const generation = ++wakeLockGeneration;
    if (!visible) {
      return;
    }

    let ownedLock: WakeLockHandle | null = null;
    const requestWakeLock = async (): Promise<void> => {
      try {
        const wakeLockApi = (navigator as Navigator & {
          wakeLock?: { request: (type: 'screen') => Promise<WakeLockHandle> };
        }).wakeLock;
        if (!wakeLockApi) {
          logger.debug('detection', 'Wake Lock API not supported');
          if (generation === wakeLockGeneration) isBackgroundCapable = false;
          return;
        }

        const acquired = await wakeLockApi.request('screen');
        if (generation !== wakeLockGeneration || !isVisible) {
          await acquired.release().catch(() => undefined);
          return;
        }

        ownedLock = acquired;
        wakeLock = acquired;
        isBackgroundCapable = true;
        logger.debug('detection', 'Wake Lock acquired - background processing improved');
        acquired.addEventListener('release', () => {
          logger.debug('detection', 'Wake Lock released');
          if (generation === wakeLockGeneration && wakeLock === acquired) {
            wakeLock = null;
            ownedLock = null;
          }
        });
      } catch (error) {
        if (generation !== wakeLockGeneration) return;
        logger.debug('detection', `Wake Lock error: ${String(errorMessage(error))}`);
        isBackgroundCapable = false;
      }
    };

    void requestWakeLock();
    return () => {
      if (generation === wakeLockGeneration) wakeLockGeneration += 1;
      const lock = ownedLock;
      ownedLock = null;
      if (wakeLock === lock) wakeLock = null;
      if (lock) void lock.release().catch(() => undefined);
    };
  });

  const flashTitle = (message: string, duration = 3000): void => {
    if (isVisible) return;
    document.title = `⚠️ ${message}`;
    const timer = setTimeout(() => {
      titleTimers.delete(timer);
      if (!isVisible) document.title = originalTitle;
    }, duration);
    titleTimers.add(timer);
  };

  return {
    get isVisible() { return isVisible; },
    get isBackgroundCapable() { return isBackgroundCapable; },
    flashTitle,
  };
}
