import { flushSync } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { useTrackingToggle } from '../useTrackingToggle.svelte';

interface Signals {
  cameraRunning: boolean;
  cameraError: string | null;
  settingsReady: boolean;
}

const disposers: Array<() => void> = [];

function mount(initial: Partial<Signals> = {}) {
  const signals = $state<Signals>({
    cameraRunning: true,
    cameraError: null,
    settingsReady: true,
    ...initial,
  });
  let hook!: ReturnType<typeof useTrackingToggle>;
  // Every distinct `paused` transition appends one entry, so its length proves
  // exactly how many reactive flips happened (a no-op adds nothing).
  const pausedLog: boolean[] = [];
  const dispose = $effect.root(() => {
    hook = useTrackingToggle({
      get cameraRunning() {
        return signals.cameraRunning;
      },
      get cameraError() {
        return signals.cameraError;
      },
      get settingsReady() {
        return signals.settingsReady;
      },
    });
    $effect(() => {
      pausedLog.push(hook.paused);
    });
  });
  disposers.push(dispose);
  flushSync();
  const set = (changes: Partial<Signals>): void => {
    Object.assign(signals, changes);
    flushSync();
  };
  const toggle = (): void => {
    hook.toggle();
    flushSync();
  };
  const applyNative = (paused: boolean): void => {
    hook.applyNativePaused(paused);
    flushSync();
  };
  return { hook, set, toggle, applyNative, pausedLog };
}

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
});

describe('useTrackingToggle', () => {
  it('starts active and enabled once settings ready and frames flow', () => {
    const { hook } = mount({ cameraRunning: true });
    expect(hook.paused).toBe(false);
    expect(hook.disabled).toBe(false);
  });

  it('is disabled while camera settings are still loading', () => {
    const { hook, set } = mount({ settingsReady: false, cameraRunning: false });
    expect(hook.disabled).toBe(true);
    set({ settingsReady: true, cameraRunning: true });
    expect(hook.disabled).toBe(false);
  });

  it('is disabled while resuming until preview frames flow again', () => {
    // Not paused but no frames yet == booting/resuming -> in flight.
    const { hook, set } = mount({ cameraRunning: false });
    expect(hook.disabled).toBe(true);
    set({ cameraRunning: true });
    expect(hook.disabled).toBe(false);
  });

  it('pauses, disables while the stop settles, then re-enables as Resume', () => {
    const { hook, set, toggle } = mount({ cameraRunning: true });
    toggle();
    expect(hook.paused).toBe(true);
    // Camera still reports running for an instant -> in flight.
    expect(hook.disabled).toBe(true);
    set({ cameraRunning: false });
    expect(hook.disabled).toBe(false);
  });

  it('resumes, disables while the start settles, then re-enables as Pause', () => {
    const { hook, set, toggle } = mount({ cameraRunning: false });
    // Reach a settled paused state first.
    set({ cameraRunning: true });
    toggle(); // pause
    set({ cameraRunning: false });
    expect(hook.paused).toBe(true);
    expect(hook.disabled).toBe(false);

    toggle(); // resume
    expect(hook.paused).toBe(false);
    expect(hook.disabled).toBe(true); // starting, no frames yet
    set({ cameraRunning: true });
    expect(hook.disabled).toBe(false);
  });

  it('ignores a toggle while disabled', () => {
    const { hook, toggle } = mount({ settingsReady: false, cameraRunning: false });
    expect(hook.disabled).toBe(true);
    toggle();
    expect(hook.paused).toBe(false);
  });

  it('leaves the button enabled and retryable after a failed resume', () => {
    // Intent is running, but the start failed: error breaks the settling lock.
    const { hook, toggle } = mount({ cameraRunning: false, cameraError: 'no capture device' });
    expect(hook.paused).toBe(false);
    expect(hook.disabled).toBe(false);
    // The user can still act on it.
    toggle();
    expect(hook.paused).toBe(true);
  });
});

describe('useTrackingToggle native event sync', () => {
  it('treats a native event matching current state as a pure no-op', () => {
    const { hook, applyNative, pausedLog } = mount({ cameraRunning: true });
    expect(hook.paused).toBe(false);
    const before = pausedLog.length;
    applyNative(false); // already false
    expect(hook.paused).toBe(false);
    expect(pausedLog.length).toBe(before); // no reactive churn, no re-issue
  });

  it('flips the UI exactly once for a tray-initiated pause and absorbs the echo', () => {
    const { hook, applyNative, pausedLog } = mount({ cameraRunning: true });
    const before = pausedLog.length;
    applyNative(true); // tray paused
    expect(hook.paused).toBe(true);
    expect(pausedLog.length).toBe(before + 1); // exactly one transition
    // The redundant stop_camera the flip triggers echoes tracking-state-changed{true}.
    applyNative(true);
    expect(hook.paused).toBe(true);
    expect(pausedLog.length).toBe(before + 1); // echo absorbed, still one flip
  });

  it('converges to the last value across rapid alternating native events', () => {
    const { hook, applyNative } = mount({ cameraRunning: true });
    applyNative(true);
    applyNative(false);
    applyNative(true);
    applyNative(false);
    applyNative(true);
    expect(hook.paused).toBe(true);
  });

  it('does not double-toggle when a UI-initiated toggle echoes back as an event', () => {
    const { hook, toggle, applyNative, pausedLog } = mount({ cameraRunning: true });
    const before = pausedLog.length;
    toggle(); // user pauses
    expect(hook.paused).toBe(true);
    expect(pausedLog.length).toBe(before + 1);
    // stop_camera -> shared native helper -> tracking-state-changed{true} echo.
    applyNative(true);
    expect(hook.paused).toBe(true);
    expect(pausedLog.length).toBe(before + 1); // no second flip
  });

  it('adopts a native event even while a start/stop is settling (no deadlock)', () => {
    const { hook, applyNative } = mount({ cameraRunning: true });
    // Tray pause while frames still flow -> paused but settling -> disabled.
    applyNative(true);
    expect(hook.paused).toBe(true);
    expect(hook.disabled).toBe(true);
    // A resume event arrives mid-settle; it must still reconcile, not lock out.
    applyNative(false);
    expect(hook.paused).toBe(false);
    // paused=false with frames flowing -> settled -> enabled again.
    expect(hook.disabled).toBe(false);
  });
});
