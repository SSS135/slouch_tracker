/**
 * Auto-Capture Hook
 *
 * Manages automatic frame capture timing with dynamic interval adjustment.
 * Subtracts processing time from next interval to maintain accurate capture rate.
 *
 * Example: With 1-second interval, if capture takes 200ms, next timeout is 800ms.
 * This maintains exactly 1 capture/second regardless of processing time.
 */

import { logger } from '../services/logging/logger';

export interface AutoCaptureConfig {
  /** Enable/disable auto-capture */
  enabled: boolean;
  /** Capture interval in seconds */
  intervalSeconds: number;
  /** Callback to execute on each interval (async operation to measure) */
  onCapture: () => Promise<unknown>;
  /** Capture mode - only 'interval' mode uses this hook */
  mode: 'interval' | 'posture-change';
}

/**
 * Auto-Capture Hook
 *
 * Manages interval-based auto-capture with dynamic timing adjustment.
 *
 * @param config - Auto-capture configuration
 */
export function useAutoCapture(config: AutoCaptureConfig): void {
  let timeoutRef: ReturnType<typeof setTimeout> | null = null;
  let callbackRef: (() => Promise<unknown>) = config.onCapture;

  $effect(() => {
    callbackRef = config.onCapture;
  });

  $effect(() => {
    if (timeoutRef) {
      clearTimeout(timeoutRef);
      timeoutRef = null;
    }

    if (!config.enabled || config.mode !== 'interval') {
      return;
    }

    const intervalMs = config.intervalSeconds * 1000;
    let disposed = false;

    const scheduleNextCapture = () => {
      if (disposed) return;
      const captureStartTime = Date.now();

      void callbackRef()
        .then(() => {
          if (disposed) return;
          const captureEndTime = Date.now();
          const processingDuration = captureEndTime - captureStartTime;

          // Calculate delay for next capture (subtract processing time from interval)
          // Use Math.max(0, ...) to handle case where processing exceeds interval
          const targetDelay = intervalMs - processingDuration;
          const actualDelay = Math.max(0, targetDelay);

          logger.debug(
            'detection',
            `[AutoCapture] Processing: ${processingDuration}ms, Next delay: ${actualDelay}ms (interval: ${intervalMs}ms)`,
          );

          timeoutRef = setTimeout(scheduleNextCapture, actualDelay);
        })
        .catch(error => {
          if (disposed) return;
          logger.warn(
            'detection',
            '[AutoCapture] Capture failed, scheduling next:',
            error,
          );
          timeoutRef = setTimeout(scheduleNextCapture, intervalMs);
        });
    };

    logger.debug('detection', `[AutoCapture] Starting with ${intervalMs}ms interval`);
    timeoutRef = setTimeout(scheduleNextCapture, intervalMs);

    return () => {
      disposed = true;
      if (timeoutRef) {
        logger.debug('detection', '[AutoCapture] Stopping');
        clearTimeout(timeoutRef);
        timeoutRef = null;
      }
    };
  });
}
