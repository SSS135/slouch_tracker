import { browser, expect } from '@wdio/globals';

import { tauriInvoke, waitForNativeReady, expectOk } from './helpers/native.js';
import { skipOnboarding, waitForOnboardingOverlay } from './helpers/onboarding.js';

// DOM probes intentionally use execute() instead of getTitle()/$():
// @wdio/tauri-service's auto window-focus hook intercepts title/element
// commands and probes window states through the guest-JS plugin, which this
// app deliberately does not bundle (src-svelte stays production-clean), so
// each intercepted command would stall for 5s before falling through.
describe('packaged native launch and readiness', () => {
  it('launches the packaged devbuild and reaches full native readiness', async () => {
    const status = await waitForNativeReady();
    expect(status.ready).toBe(true);
    expect(status.inferenceReady).toBe(true);
    expect(typeof status.datasetVersion).toBe('number');
    expect(typeof status.storage.quota).toBe('number');
    expect(status.storage.available).toBeGreaterThanOrEqual(0);
    expect(status.storage.used).toBeGreaterThanOrEqual(0);
  });

  it('renders the Svelte shell inside the packaged webview', async () => {
    const title = await browser.execute(() => document.title);
    expect(title).toContain('Slouch Tracker');

    // This suite runs against a wiped SLOUCH_APP_DATA_DIR (zero labeled frames,
    // onboardingCompleted unset), so the first-run wizard overlay gates the
    // shell. Skip it here: the persisted flag makes every later spec (fresh app
    // processes on the same data dir) boot straight into the normal UI.
    await waitForOnboardingOverlay();
    await skipOnboarding();

    await browser.waitUntil(
      async () =>
        browser.execute(
          () =>
            document.querySelector(
              'button[aria-label="Open control panel"], button[aria-label="Close control panel"]',
            ) !== null,
        ),
      {
        timeout: 30_000,
        timeoutMsg: 'Svelte control-panel toggle never appeared in the packaged webview',
      },
    );
  });

  it('serves the generated registries over IPC', async () => {
    const classifiers = expectOk(
      await tauriInvoke<Array<{ id: string }>>('get_classifier_registry'),
      'get_classifier_registry',
    );
    expect(classifiers.length).toBeGreaterThan(0);
    const features = expectOk(
      await tauriInvoke<Array<{ id: string }>>('get_feature_registry'),
      'get_feature_registry',
    );
    expect(features.length).toBeGreaterThan(0);
  });
});
