/**
 * Posture Change Detector
 *
 * Detects good, bad, and away posture transitions and invokes the capture
 * callback once for a newly entered posture when that posture's cooldown has
 * expired.
 */

import { FrameLabel } from '../services/dataset/types';
import { logger } from '../services/logging/logger';
import type { ClassificationResult } from '../services/types';

export interface PostureChangeDetectorConfig {
  /** Cooldown period in milliseconds per posture type (default: 2000ms). */
  cooldownMs?: number;
  /** Enable or disable transition detection (default: true). */
  enabled?: boolean;
  /** Callback to execute when a posture capture is triggered. */
  onCapture: (
    label: FrameLabel.GOOD | FrameLabel.BAD | FrameLabel.AWAY,
  ) => void | Promise<void>;
}

const DEFAULT_COOLDOWN_MS = 2000;
const DEFAULT_ENABLED = true;

type ReactiveGetter<T> = () => T;

/**
 * Monitors ML classification for posture state transitions.
 *
 * The classification is supplied as a getter so replacements of Svelte state
 * are observed by the effect.
 */
export function usePostureChangeDetector(
  classification: ReactiveGetter<ClassificationResult | null>,
  config: PostureChangeDetectorConfig,
): void {
  let previousPrediction: FrameLabel.GOOD | FrameLabel.BAD | FrameLabel.AWAY | null =
    null;
  const lastCaptureTimeByPrediction: Partial<Record<FrameLabel, number>> = {
    [FrameLabel.GOOD]: -Infinity,
    [FrameLabel.BAD]: -Infinity,
    [FrameLabel.AWAY]: -Infinity,
  };
  let lastTriggeredPrediction:
    | FrameLabel.GOOD
    | FrameLabel.BAD
    | FrameLabel.AWAY
    | null = null;

  $effect(() => {
    const currentClassification = classification();
    const cooldownMs = config.cooldownMs ?? DEFAULT_COOLDOWN_MS;
    const enabled = config.enabled ?? DEFAULT_ENABLED;
    const prediction = currentClassification
      ? currentClassification.goodProbability === null
        ? FrameLabel.AWAY
        : currentClassification.goodProbability >= 0.5
          ? FrameLabel.GOOD
          : FrameLabel.BAD
      : null;

    if (enabled && prediction && previousPrediction) {
      const hasTransition = prediction !== previousPrediction;

      if (hasTransition) {
        logger.debug('detection', '[PostureChangeDetector] Transition detected', {
          from: previousPrediction,
          to: prediction,
        });

        if (previousPrediction !== lastTriggeredPrediction) {
          lastTriggeredPrediction = null;
        }

        const alreadyTriggered = prediction === lastTriggeredPrediction;

        if (alreadyTriggered) {
          logger.debug(
            'detection',
            '[PostureChangeDetector] Already triggered for this prediction (one-shot block)',
            { predictionValue: prediction },
          );
        }

        if (!alreadyTriggered) {
          const now = Date.now();
          const lastCaptureTime =
            lastCaptureTimeByPrediction[prediction] ?? -Infinity;
          const timeSinceLastCapture = now - lastCaptureTime;

          logger.debug('detection', '[PostureChangeDetector] Cooldown check', {
            predictionValue: prediction,
            lastCaptureTime,
            timeSinceLastCapture,
            cooldownMs,
            allowed: timeSinceLastCapture >= cooldownMs,
          });

          if (timeSinceLastCapture >= cooldownMs) {
            logger.debug('detection', '[PostureChangeDetector] ✓ CAPTURE TRIGGERED!', {
              captureLabel: prediction,
              transition: `${previousPrediction} → ${prediction}`,
            });

            lastCaptureTimeByPrediction[prediction] = now;
            lastTriggeredPrediction = prediction;
            // Commit the transition before user code runs so synchronous
            // callback re-entry cannot observe the previous posture.
            previousPrediction = prediction;

            logger.debug(
              'detection',
              '[PostureChangeDetector] Calling onCapture callback',
              { label: prediction },
            );
            void config.onCapture(prediction);
          } else {
            logger.debug(
              'detection',
              '[PostureChangeDetector] ✗ Capture blocked by cooldown',
              {
                predictionValue: prediction,
                timeRemaining: cooldownMs - timeSinceLastCapture,
              },
            );
          }
        }
      }
    }

    previousPrediction = prediction;
  });
}
