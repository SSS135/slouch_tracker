import type { InferenceUiResult } from '@generated/bindings';
import { Channel } from '@tauri-apps/api/core';
import { nativeClient, type NativeClient } from '../lib/native/client';
import { initializeNativeInference } from '../lib/state/nativeApp.svelte';
import { logger } from '../services/logging/logger';

export interface UseNativeCameraOptions {
  /** Called for every inference result the Rust camera pushes over the channel. */
  onResult: (result: InferenceUiResult) => void;
  /** When false the camera is stopped (e.g. settings not yet loaded). */
  enabled?: boolean;
  client?: NativeClient;
}

export interface UseNativeCameraReturn {
  /** True once the native camera has started streaming results. */
  readonly ready: boolean;
  /** Non-null when initialization or camera start failed. */
  readonly error: string | null;
  /** Detection cadence derived from the interval between pushed results. */
  readonly detectionFps: number;
  retry(): Promise<void>;
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

/**
 * Owns the native camera lifecycle: it initializes inference, opens a
 * `Channel<InferenceUiResult>`, and asks Rust to drive the detection loop
 * (`start_camera`). Rust now owns capture, preprocessing, and inference; the
 * frontend only consumes the pushed results. The channel is torn down and the
 * camera stopped on unmount or when disabled.
 */
export function useNativeCamera(options: UseNativeCameraOptions): UseNativeCameraReturn {
  const client = options.client ?? nativeClient;

  let ready = $state(false);
  let error = $state<string | null>(null);
  let detectionFps = $state(0);
  let retryTrigger = $state(0);

  let onResult = options.onResult;
  $effect(() => {
    onResult = options.onResult;
  });

  $effect(() => {
    const enabled = options.enabled ?? true;
    // Reading retryTrigger re-runs the effect (restarting the camera) on retry.
    if (retryTrigger < 0) return;
    if (!enabled) {
      ready = false;
      detectionFps = 0;
      return;
    }

    let disposed = false;
    let started = false;
    let lastMessageAt = 0;

    const channel = new Channel<InferenceUiResult>();
    channel.onmessage = (result) => {
      if (disposed) return;
      const now = performance.now();
      if (lastMessageAt > 0) {
        const delta = now - lastMessageAt;
        if (delta > 0) detectionFps = 1000 / delta;
      }
      lastMessageAt = now;
      try {
        onResult(result);
      } catch (cause) {
        logger.error('detection', 'Inference result callback failed:', cause);
      }
    };

    void (async () => {
      ready = false;
      error = null;
      detectionFps = 0;
      try {
        await initializeNativeInference(client);
        if (disposed) return;
        await client.startCamera(channel);
        if (disposed) {
          void client.stopCamera().catch(() => undefined);
          return;
        }
        started = true;
        ready = true;
      } catch (cause) {
        if (disposed) return;
        error = asError(cause).message;
        ready = false;
        logger.error('detection', 'Native camera start failed:', cause);
      }
    })();

    return () => {
      disposed = true;
      ready = false;
      detectionFps = 0;
      if (started) {
        void client.stopCamera().catch((cause: unknown) => {
          logger.warn('detection', 'Native camera stop failed:', cause);
        });
      }
    };
  });

  return {
    get ready() {
      return ready;
    },
    get error() {
      return error;
    },
    get detectionFps() {
      return detectionFps;
    },
    async retry() {
      retryTrigger += 1;
    },
  };
}
