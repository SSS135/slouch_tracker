import { Channel } from '@tauri-apps/api/core';
import type { PoseModelDownloadEvent } from '@generated/bindings';
import { nativeClient, type NativeClient } from '../lib/native/client';
import { logger } from '../services/logging/logger';

/**
 * First-run pose-model gate state. `checking`/`ready` never render a screen
 * (the app proceeds normally); the other phases drive the blocking download UI.
 */
export type PoseModelPhase =
  | { kind: 'checking' }
  | { kind: 'ready' }
  | { kind: 'downloading'; received: number; total: number }
  | { kind: 'verifying' }
  | { kind: 'failed'; reason: string; offline: boolean }
  | { kind: 'cancelled' };

export interface UsePoseModelDownloadOptions {
  /**
   * Invoked once the model is on disk (a `ready` event). Callers re-run native
   * inference init here so the app proceeds without a restart; the real Rust
   * resolves the freshly downloaded file lazily.
   */
  onReady: () => void | Promise<void>;
  client?: NativeClient;
}

export interface UsePoseModelDownloadReturn {
  readonly phase: PoseModelPhase;
  /** True while a blocking download/verify/failure screen must cover the app. */
  readonly blocking: boolean;
  /** Abandon the in-flight download (stops reacting to its events) and await a retry. */
  cancel(): void;
  /** (Re)start the download; the Rust side resumes a partial file via Range. */
  retry(): void;
}

// Heuristic: does a failure reason read like "no network" rather than a real
// server/verification error? Drives the offline install hint, nothing load-bearing.
const OFFLINE_HINTS = [
  'offline',
  'network',
  'dns',
  'connect',
  'connection',
  'timed out',
  'timeout',
  'unreachable',
  'resolve',
  'failed to fetch',
  'no route',
  'name or service',
];

function looksOffline(reason: string): boolean {
  const lower = reason.toLowerCase();
  return OFFLINE_HINTS.some((hint) => lower.includes(hint));
}

function toReason(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

export function usePoseModelDownload(
  options: UsePoseModelDownloadOptions,
): UsePoseModelDownloadReturn {
  const client = options.client ?? nativeClient;

  let phase = $state<PoseModelPhase>({ kind: 'checking' });
  // Bumped on cancel/retry/unmount so a superseded channel's late events are ignored.
  let generation = 0;

  function fail(reason: string): void {
    phase = { kind: 'failed', reason, offline: looksOffline(reason) };
  }

  function startDownload(): void {
    const token = (generation += 1);
    phase = { kind: 'downloading', received: 0, total: 0 };

    const channel = new Channel<PoseModelDownloadEvent>();
    channel.onmessage = (event) => {
      if (token !== generation) return;
      switch (event.type) {
        case 'started':
          phase = { kind: 'downloading', received: 0, total: event.totalBytes };
          break;
        case 'progress':
          phase = { kind: 'downloading', received: event.received, total: event.total };
          break;
        case 'verifying':
          phase = { kind: 'verifying' };
          break;
        case 'ready':
          phase = { kind: 'ready' };
          void Promise.resolve(options.onReady()).catch((cause: unknown) => {
            logger.error('detection', 'Pose-model post-download init failed:', cause);
          });
          break;
        case 'failed':
          fail(event.reason);
          break;
      }
    };

    void client.ensurePoseModel(channel).catch((cause: unknown) => {
      // A late rejection after the model already resolved (or a superseded run) is noise.
      if (token !== generation || phase.kind === 'ready' || phase.kind === 'verifying') return;
      fail(toReason(cause));
    });
  }

  async function check(): Promise<void> {
    try {
      const status = await client.getPoseModelStatus();
      if (status.type === 'ready') {
        phase = { kind: 'ready' };
        return;
      }
      // `downloadRequired` or an in-progress `downloading`: auto-start (Rust attaches
      // to / resumes the existing download) so first launch just begins fetching.
      startDownload();
    } catch (cause) {
      fail(toReason(cause));
    }
  }

  $effect(() => {
    void check();
    return () => {
      generation += 1;
    };
  });

  return {
    get phase() {
      return phase;
    },
    get blocking() {
      return phase.kind !== 'ready' && phase.kind !== 'checking';
    },
    cancel() {
      generation += 1;
      phase = { kind: 'cancelled' };
    },
    retry() {
      startDownload();
    },
  };
}
