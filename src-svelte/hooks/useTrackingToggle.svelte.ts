/**
 * Session-only pause/resume intent for posture tracking.
 *
 * The camera lifecycle is driven declaratively by `PostureCamera`'s `paused`
 * prop (via `useNativeCamera`'s `enabled` gate, which calls `start_camera` /
 * `stop_camera`). This hook only owns the user's pause *intent* plus an
 * in-flight ("settling") guard, decoupled from the camera stack through injected
 * getters so it stays trivially testable. It never persists anything.
 */

export interface UseTrackingToggleOptions {
  /** Actual "tracking is live" signal (preview frames flowing == resume succeeded). */
  readonly cameraRunning: boolean;
  /** Non-null when the last camera start failed; breaks the resume "settling" lock so the button stays retryable. */
  readonly cameraError: string | null;
  /** False while native camera settings are still loading. */
  readonly settingsReady: boolean;
}

export interface UseTrackingToggleReturn {
  /** True when the user has paused tracking (session only). */
  readonly paused: boolean;
  /** True while settings load or a start/stop is settling — the button is disabled. */
  readonly disabled: boolean;
  /** Flip the pause intent. No-op while disabled, so overlapping start/stop is impossible. */
  toggle(): void;
  /**
   * Adopt an authoritative paused state pushed by native (tray menu / global
   * hotkey) via the `tracking-state-changed` event. Native is the single source
   * of truth, so this is the reconciliation entry:
   * - Idempotent: a payload matching the current intent writes nothing, so the
   *   command echoes a UI-initiated toggle produces (frontend stop/start ->
   *   shared helper -> event) are absorbed as no-ops and never re-toggle.
   * - Bypasses the settling guard (unlike `toggle`): a native flip mid-start/stop
   *   must still reconcile, so an event during settling can't deadlock.
   */
  applyNativePaused(paused: boolean): void;
}

export function useTrackingToggle(options: UseTrackingToggleOptions): UseTrackingToggleReturn {
  let paused = $state(false);

  // In-flight while the actual camera state has not caught up to the intent.
  // Pausing: still running -> settling until it stops.
  // Resuming: not running and no error -> settling until it streams (an error
  // clears the lock so a failed resume leaves a retryable, enabled button).
  const settling = (): boolean =>
    paused ? options.cameraRunning : !options.cameraRunning && !options.cameraError;

  const isDisabled = (): boolean => !options.settingsReady || settling();

  return {
    get paused() {
      return paused;
    },
    get disabled() {
      return isDisabled();
    },
    toggle() {
      if (isDisabled()) return;
      paused = !paused;
    },
    applyNativePaused(next: boolean) {
      // Equal-value guard: the only write path. Keeps the sync convergent and
      // idempotent — echoes and repeated events cause zero state churn.
      if (paused === next) return;
      paused = next;
    },
  };
}
