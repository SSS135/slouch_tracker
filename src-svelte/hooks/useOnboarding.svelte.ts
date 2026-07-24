import { FrameLabel } from '../services/dataset/types';

export const ONBOARDING_TARGETS = { good: 5, bad: 5, away: 3 } as const;

export type OnboardingStep = 'camera' | 'good' | 'bad' | 'away';

export interface UseOnboardingOptions {
  settingsReady: () => boolean;
  settings: () => { onboardingCompleted: boolean; cameraIndex: number };
  updateSettings: (updates: Partial<{ onboardingCompleted: boolean; cameraIndex: number }>) => void;
  flushSettings: () => Promise<void>;
  stats: () => { good?: number; bad?: number; away?: number } | null | undefined;
  restartCamera: () => Promise<void>;
}

export interface OnboardingState {
  readonly active: boolean;
  readonly step: OnboardingStep;
  readonly capturedGood: number;
  readonly capturedBad: number;
  readonly capturedAway: number;
  /** Run Setup Again: reopens the wizard regardless of the completed flag. */
  begin(): void;
  /** Camera step confirmed; move on to the first capture step. */
  next(): void;
  /** Abandon the wizard from any step; it will not auto-reopen. */
  skip(): void;
  /** The away step is optional; completing without it is a normal finish. */
  skipAwayStep(): void;
  notifyFramePersisted(label: FrameLabel): void;
  selectCamera(index: number): Promise<void>;
}

const STEP_LABELS: Record<Exclude<OnboardingStep, 'camera'>, FrameLabel> = {
  good: FrameLabel.GOOD,
  bad: FrameLabel.BAD,
  away: FrameLabel.AWAY,
};

/**
 * First-run onboarding wizard state machine: camera selection, then guided
 * good/bad/away captures. Progress counters are session-local (never derived
 * from dataset totals), so re-running setup over an existing dataset still
 * counts from zero.
 */
export function useOnboarding(options: UseOnboardingOptions): OnboardingState {
  let active = $state(false);
  let step = $state<OnboardingStep>('camera');
  let capturedGood = $state(0);
  let capturedBad = $state(0);
  let capturedAway = $state(0);
  // Plain flag: the silent-complete write must fire at most once even if the
  // gate re-evaluates before the settings update propagates back.
  let autoCompleted = false;

  const resetProgress = (): void => {
    capturedGood = 0;
    capturedBad = 0;
    capturedAway = 0;
    step = 'camera';
  };

  const finish = (): void => {
    options.updateSettings({ onboardingCompleted: true });
    active = false;
  };

  // First-run gate. Undecided until settings and stats are both loaded; an
  // existing install (any labeled frames) is completed silently instead of
  // being walked through setup. Never re-decides while a run is in progress.
  $effect(() => {
    if (active) return;
    if (!options.settingsReady()) return;
    if (options.settings().onboardingCompleted) return;
    const stats = options.stats();
    if (!stats) return;
    const labeled = (stats.good ?? 0) + (stats.bad ?? 0) + (stats.away ?? 0);
    if (labeled > 0) {
      if (!autoCompleted) {
        autoCompleted = true;
        options.updateSettings({ onboardingCompleted: true });
      }
      return;
    }
    resetProgress();
    active = true;
  });

  return {
    get active() { return active; },
    get step() { return step; },
    get capturedGood() { return capturedGood; },
    get capturedBad() { return capturedBad; },
    get capturedAway() { return capturedAway; },
    begin() {
      options.updateSettings({ onboardingCompleted: false });
      resetProgress();
      active = true;
    },
    next() {
      if (step === 'camera') step = 'good';
    },
    skip() {
      finish();
    },
    skipAwayStep() {
      finish();
    },
    notifyFramePersisted(label) {
      if (!active || step === 'camera' || STEP_LABELS[step] !== label) return;
      if (step === 'good') {
        capturedGood += 1;
        if (capturedGood >= ONBOARDING_TARGETS.good) step = 'bad';
      } else if (step === 'bad') {
        capturedBad += 1;
        if (capturedBad >= ONBOARDING_TARGETS.bad) step = 'away';
      } else {
        capturedAway += 1;
        if (capturedAway >= ONBOARDING_TARGETS.away) finish();
      }
    },
    async selectCamera(index) {
      if (index === options.settings().cameraIndex) return;
      options.updateSettings({ cameraIndex: index });
      // Flush before restarting so the native camera actor's next Start reads
      // the persisted index instead of racing the settings write.
      await options.flushSettings();
      await options.restartCamera();
    },
  };
}
