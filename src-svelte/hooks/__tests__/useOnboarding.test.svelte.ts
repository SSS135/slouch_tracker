import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { FrameLabel } from '../../services/dataset/types';
import { ONBOARDING_TARGETS, useOnboarding } from '../useOnboarding.svelte';

interface Env {
  ready: boolean;
  onboardingCompleted: boolean;
  cameraIndex: number;
  stats: { good?: number; bad?: number; away?: number } | null | undefined;
}

interface MountOptions {
  env?: Partial<Env>;
  /** When false, updateSettings records calls without touching env (simulates a lagging write). */
  applyUpdates?: boolean;
}

const disposers: Array<() => void> = [];

function mount(options: MountOptions = {}) {
  const env = $state<Env>({
    ready: true,
    onboardingCompleted: false,
    cameraIndex: 0,
    stats: { good: 0, bad: 0, away: 0 },
    ...options.env,
  });
  const apply = options.applyUpdates ?? true;
  // Ordered call log proving the selectCamera sequence: update -> flush -> restart.
  const calls: string[] = [];
  const updateSettings = vi.fn(
    (updates: Partial<{ onboardingCompleted: boolean; cameraIndex: number }>) => {
      calls.push('update');
      if (!apply) return;
      if (updates.onboardingCompleted !== undefined) env.onboardingCompleted = updates.onboardingCompleted;
      if (updates.cameraIndex !== undefined) env.cameraIndex = updates.cameraIndex;
    },
  );
  const flushSettings = vi.fn(async () => {
    calls.push('flush');
  });
  const restartCamera = vi.fn(async () => {
    calls.push('restart');
  });
  let hook!: ReturnType<typeof useOnboarding>;
  const dispose = $effect.root(() => {
    hook = useOnboarding({
      settingsReady: () => env.ready,
      settings: () => ({ onboardingCompleted: env.onboardingCompleted, cameraIndex: env.cameraIndex }),
      updateSettings,
      flushSettings,
      stats: () => env.stats,
      restartCamera,
    });
  });
  disposers.push(dispose);
  flushSync();
  const set = (changes: Partial<Env>): void => {
    Object.assign(env, changes);
    flushSync();
  };
  const persist = (label: FrameLabel, times = 1): void => {
    for (let index = 0; index < times; index += 1) hook.notifyFramePersisted(label);
    flushSync();
  };
  return { hook, env, set, persist, calls, updateSettings, flushSettings, restartCamera };
}

afterEach(() => {
  while (disposers.length) disposers.pop()?.();
});

describe('useOnboarding first-run gate', () => {
  it('opens on a true first run: flag unset and zero labeled frames', () => {
    const { hook } = mount();
    expect(hook.active).toBe(true);
    expect(hook.step).toBe('camera');
    expect(hook.capturedGood).toBe(0);
    expect(hook.capturedBad).toBe(0);
    expect(hook.capturedAway).toBe(0);
  });

  it('stays closed when onboarding is already completed', () => {
    const { hook, updateSettings } = mount({ env: { onboardingCompleted: true } });
    expect(hook.active).toBe(false);
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it('stays undecided while settings are not ready', () => {
    const { hook, set } = mount({ env: { ready: false } });
    expect(hook.active).toBe(false);
    set({ ready: true });
    expect(hook.active).toBe(true);
  });

  it('stays undecided while stats are not fetched, then decides', () => {
    const { hook, set, updateSettings } = mount({ env: { stats: null } });
    expect(hook.active).toBe(false);
    expect(updateSettings).not.toHaveBeenCalled();
    set({ stats: { good: 0, bad: 0, away: 0 } });
    expect(hook.active).toBe(true);
  });

  it('silently completes an existing install with labeled frames', () => {
    const { hook, updateSettings } = mount({ env: { stats: { good: 0, bad: 0, away: 2 } } });
    expect(hook.active).toBe(false);
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ onboardingCompleted: true });
  });

  it('treats missing label counts as zero', () => {
    const { hook } = mount({ env: { stats: {} } });
    expect(hook.active).toBe(true);
  });

  it('writes the silent completion at most once even while the settings write lags', () => {
    const { hook, set, updateSettings } = mount({
      env: { stats: { good: 1 } },
      applyUpdates: false,
    });
    expect(hook.active).toBe(false);
    expect(updateSettings).toHaveBeenCalledTimes(1);
    // A stats refresh re-runs the gate; the plain autoCompleted flag must hold.
    set({ stats: { good: 2 } });
    expect(updateSettings).toHaveBeenCalledTimes(1);
    expect(hook.active).toBe(false);
  });

  it('never re-decides mid-run when stats gain labeled frames', () => {
    const { hook, set, updateSettings } = mount();
    expect(hook.active).toBe(true);
    set({ stats: { good: 3, bad: 2, away: 0 } });
    expect(hook.active).toBe(true);
    expect(updateSettings).not.toHaveBeenCalled();
  });
});

describe('useOnboarding step machine', () => {
  it('advances camera -> good on next() and ignores next() elsewhere', () => {
    const { hook } = mount();
    hook.next();
    expect(hook.step).toBe('good');
    hook.next();
    expect(hook.step).toBe('good');
  });

  it('counts only the label matching the current step', () => {
    const { hook, persist } = mount();
    // Camera step: every label is ignored.
    persist(FrameLabel.GOOD);
    persist(FrameLabel.BAD);
    expect(hook.capturedGood).toBe(0);
    expect(hook.capturedBad).toBe(0);
    hook.next();
    persist(FrameLabel.BAD);
    persist(FrameLabel.AWAY);
    persist(FrameLabel.UNUSED);
    expect(hook.capturedGood).toBe(0);
    expect(hook.capturedBad).toBe(0);
    expect(hook.capturedAway).toBe(0);
    persist(FrameLabel.GOOD);
    expect(hook.capturedGood).toBe(1);
    expect(hook.step).toBe('good');
  });

  it('ignores persisted frames while inactive', () => {
    const { hook } = mount({ env: { onboardingCompleted: true } });
    hook.notifyFramePersisted(FrameLabel.GOOD);
    expect(hook.capturedGood).toBe(0);
  });

  it('auto-advances good -> bad -> away and finishes after the away target', () => {
    const { hook, persist, updateSettings } = mount();
    hook.next();
    persist(FrameLabel.GOOD, ONBOARDING_TARGETS.good);
    expect(hook.step).toBe('bad');
    expect(hook.capturedGood).toBe(ONBOARDING_TARGETS.good);
    persist(FrameLabel.BAD, ONBOARDING_TARGETS.bad);
    expect(hook.step).toBe('away');
    persist(FrameLabel.AWAY, ONBOARDING_TARGETS.away);
    expect(hook.active).toBe(false);
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ onboardingCompleted: true });
  });

  it('completes and closes from any step on skip()', () => {
    const { hook, persist, updateSettings } = mount();
    hook.next();
    persist(FrameLabel.GOOD, 2);
    hook.skip();
    flushSync();
    expect(hook.active).toBe(false);
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ onboardingCompleted: true });
  });

  it('completes on skipAwayStep() without away captures', () => {
    const { hook, persist, updateSettings } = mount();
    hook.next();
    persist(FrameLabel.GOOD, ONBOARDING_TARGETS.good);
    persist(FrameLabel.BAD, ONBOARDING_TARGETS.bad);
    expect(hook.step).toBe('away');
    hook.skipAwayStep();
    flushSync();
    expect(hook.active).toBe(false);
    expect(hook.capturedAway).toBe(0);
    expect(updateSettings).toHaveBeenCalledExactlyOnceWith({ onboardingCompleted: true });
  });

  it('does not reopen through the gate after a completed run', () => {
    const { hook, set } = mount();
    hook.skip();
    flushSync();
    expect(hook.active).toBe(false);
    // Later stats refreshes must not resurrect the wizard.
    set({ stats: { good: 0, bad: 0, away: 0 } });
    expect(hook.active).toBe(false);
  });
});

describe('useOnboarding begin (Run Setup Again)', () => {
  it('reopens with reset session counters over an existing dataset', () => {
    const { hook, persist, set, updateSettings } = mount();
    hook.next();
    persist(FrameLabel.GOOD, 3);
    hook.skip();
    flushSync();
    expect(hook.active).toBe(false);
    set({ stats: { good: 3, bad: 0, away: 0 } });

    updateSettings.mockClear();
    hook.begin();
    flushSync();
    expect(updateSettings).toHaveBeenCalledWith({ onboardingCompleted: false });
    expect(hook.active).toBe(true);
    expect(hook.step).toBe('camera');
    // Session-local progress: never derived from the dataset totals.
    expect(hook.capturedGood).toBe(0);
    expect(hook.capturedBad).toBe(0);
    expect(hook.capturedAway).toBe(0);
  });

  it('runs a full second pass after begin()', () => {
    const { hook, persist } = mount();
    hook.skip();
    flushSync();
    hook.begin();
    flushSync();
    hook.next();
    persist(FrameLabel.GOOD, ONBOARDING_TARGETS.good);
    expect(hook.step).toBe('bad');
  });
});

describe('useOnboarding selectCamera', () => {
  it('is a no-op when the index is unchanged', async () => {
    const { hook, updateSettings, flushSettings, restartCamera } = mount({ env: { cameraIndex: 2 } });
    await hook.selectCamera(2);
    expect(updateSettings).not.toHaveBeenCalled();
    expect(flushSettings).not.toHaveBeenCalled();
    expect(restartCamera).not.toHaveBeenCalled();
  });

  it('updates, flushes, then restarts the camera in order', async () => {
    const { hook, env, calls, updateSettings } = mount();
    await hook.selectCamera(1);
    expect(updateSettings).toHaveBeenCalledWith({ cameraIndex: 1 });
    expect(env.cameraIndex).toBe(1);
    expect(calls).toEqual(['update', 'flush', 'restart']);
  });
});
