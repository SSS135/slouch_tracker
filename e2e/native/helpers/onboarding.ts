import { browser } from '@wdio/globals';

/**
 * First-run onboarding wizard interaction for the packaged native suite.
 *
 * A fresh SLOUCH_APP_DATA_DIR means an unset onboardingCompleted flag and zero
 * labeled frames, so the wizard overlay gates the shell until it is skipped or
 * finished. Skipping it persists the flag through save_ui_settings, so later
 * specs (fresh app processes on the SAME data dir) boot straight into the
 * normal UI.
 *
 * All probes go through execute() DOM operations, never $()/element commands:
 * @wdio/tauri-service's auto window-focus hook intercepts element commands and
 * stalls 5s each on this app (see the note in 10-launch-readiness.e2e.ts).
 */

const OVERLAY_SELECTOR = '[data-testid="onboarding-overlay"]';
const MAIN_UI_SELECTOR =
  'button[aria-label="Open control panel"], button[aria-label="Close control panel"]';

/** Wait until the onboarding overlay is in the DOM (fresh-data-dir boots must show it). */
export async function waitForOnboardingOverlay(timeoutMs = 30_000): Promise<void> {
  await browser.waitUntil(
    async () =>
      browser.execute(
        (selector: string) => document.querySelector(selector) !== null,
        OVERLAY_SELECTOR,
      ),
    {
      timeout: timeoutMs,
      timeoutMsg: 'first-run onboarding overlay never appeared on a fresh data dir',
    },
  );
}

/**
 * Dismiss the visible wizard via its "Skip setup" button and wait for the
 * overlay to leave the DOM. Clicks every poll until the overlay is gone, so a
 * button that mounts or enables late cannot race the dismissal.
 */
export async function skipOnboarding(timeoutMs = 30_000): Promise<void> {
  await browser.waitUntil(
    async () =>
      browser.execute((selector: string) => {
        const overlay = document.querySelector(selector);
        if (!overlay) return true;
        const skip = Array.from(overlay.querySelectorAll('button')).find(
          (button) => button.textContent?.trim() === 'Skip setup',
        );
        if (skip && !(skip as HTMLButtonElement).disabled) {
          (skip as HTMLButtonElement).click();
        }
        return false;
      }, OVERLAY_SELECTOR),
    {
      timeout: timeoutMs,
      timeoutMsg: 'onboarding overlay did not dismiss after clicking "Skip setup"',
    },
  );
}

/**
 * Defensive preamble for DOM-driving specs that can be run standalone (spec
 * filtering) against a data dir where 10-launch-readiness has not completed
 * onboarding yet. Waits until either the wizard overlay or the normal shell is
 * up; skips the wizard when it is the one shown. Returns true when the wizard
 * was seen and skipped.
 */
export async function skipOnboardingIfPresent(timeoutMs = 60_000): Promise<boolean> {
  let sawOverlay = false;
  await browser.waitUntil(
    async () => {
      const state = await browser.execute(
        (overlaySelector: string, mainUiSelector: string) => {
          const overlay = document.querySelector(overlaySelector);
          if (overlay) {
            const skip = Array.from(overlay.querySelectorAll('button')).find(
              (button) => button.textContent?.trim() === 'Skip setup',
            );
            if (skip && !(skip as HTMLButtonElement).disabled) {
              (skip as HTMLButtonElement).click();
            }
            return 'overlay';
          }
          return document.querySelector(mainUiSelector) !== null ? 'main-ui' : 'none';
        },
        OVERLAY_SELECTOR,
        MAIN_UI_SELECTOR,
      );
      if (state === 'overlay') sawOverlay = true;
      return state === 'main-ui';
    },
    {
      timeout: timeoutMs,
      timeoutMsg: 'neither the onboarding overlay nor the normal shell became available',
    },
  );
  return sawOverlay;
}
