/**
 * Snaps the Tauri window content area to the live camera aspect ratio so the
 * video fills the window without letterbox/pillarbox bars.
 *
 * Frontend-only: it reads the window's inner size + scale factor and adjusts
 * the off-axis dimension with `setSize`. Everything is guarded on the Tauri
 * runtime and wrapped in best-effort try/catch, so harness/vitest/browser
 * contexts (which lack the window APIs) are complete no-ops.
 */

import { computeAspectSnap } from '@/utils/windowAspect';
import { logger } from '@/services/logging';

// Mirrors the minWidth/minHeight declared for the main window in tauri.conf.json.
const MIN_WINDOW_WIDTH = 800;
const MIN_WINDOW_HEIGHT = 600;
const RESIZE_DEBOUNCE_MS = 200;

export interface UseWindowAspectOptions {
  readonly cameraWidth: number;
  readonly cameraHeight: number;
}

/** True only inside the packaged Tauri webview, where the window IPC exists. */
function tauriRuntimePresent(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof (window as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== 'undefined'
  );
}

export function useWindowAspect(options: UseWindowAspectOptions): void {
  // Non-Tauri contexts register no effects and never load the window module.
  if (!tauriRuntimePresent()) return;

  // Serializes overlapping snaps and lets the resize listener ignore the
  // resize events emitted by our own setSize call while it is in flight.
  let snapInFlight = false;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  async function snapToAspect(): Promise<void> {
    const cameraWidth = options.cameraWidth;
    const cameraHeight = options.cameraHeight;
    if (snapInFlight || cameraWidth <= 0 || cameraHeight <= 0) return;

    snapInFlight = true;
    try {
      const { getCurrentWindow, LogicalSize } = await import('@tauri-apps/api/window');
      const appWindow = getCurrentWindow();
      const physical = await appWindow.innerSize();
      const scaleFactor = await appWindow.scaleFactor();
      const current = {
        width: physical.width / scaleFactor,
        height: physical.height / scaleFactor,
      };

      const target = computeAspectSnap(
        current,
        cameraWidth,
        cameraHeight,
        MIN_WINDOW_WIDTH,
        MIN_WINDOW_HEIGHT,
      );
      // A null target means the window already matches within tolerance; that
      // is what terminates the setSize -> onResized -> setSize loop.
      if (target) {
        await appWindow.setSize(new LogicalSize(target.width, target.height));
      }
    } catch (cause) {
      logger.debug('detection', 'Window aspect snap skipped', cause);
    } finally {
      snapInFlight = false;
    }
  }

  // Snap whenever the camera reports (new) stream dimensions.
  $effect(() => {
    if (options.cameraWidth > 0 && options.cameraHeight > 0) {
      void snapToAspect();
    }
  });

  // Re-snap the off-axis dimension after a debounced user resize.
  $effect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;

    void (async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window');
        const cleanup = await getCurrentWindow().onResized(() => {
          if (snapInFlight) return;
          if (debounceTimer) clearTimeout(debounceTimer);
          debounceTimer = setTimeout(() => void snapToAspect(), RESIZE_DEBOUNCE_MS);
        });
        if (disposed) cleanup();
        else unlisten = cleanup;
      } catch (cause) {
        logger.debug('detection', 'Window resize listener skipped', cause);
      }
    })();

    return () => {
      disposed = true;
      if (debounceTimer) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      unlisten?.();
    };
  });
}
