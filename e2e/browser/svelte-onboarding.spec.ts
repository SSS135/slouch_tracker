import { expect, test, type Page } from '@playwright/test';

const applicationPath = '/index.svelte-app-harness.html';

// `?onboarding=fresh` (mockTauri.ts) seeds onboardingCompleted: false plus an
// EMPTY dataset — both are required for the wizard gate; with frames present
// the app silently auto-completes onboarding instead of showing the overlay.
const freshPath = `${applicationPath}?onboarding=fresh`;

const readCaptureCalls = (page: Page): Promise<number> =>
  page.evaluate(() => (
    window as typeof window & { __SLOUCH_HARNESS_METRICS__: { captureCalls: number } }
  ).__SLOUCH_HARNESS_METRICS__.captureCalls);

// Captures are async (one-use token + raw-byte save_capture); poll the harness
// metrics between clicks so each click lands after the previous save was issued,
// mirroring how svelte-real-app.spec.ts serializes its capture triggers.
async function captureFrames(page: Page, count: number): Promise<void> {
  const captureButton = page.getByRole('button', { name: 'Capture frame' });
  const before = await readCaptureCalls(page);
  for (let captured = 1; captured <= count; captured += 1) {
    await captureButton.click();
    await expect.poll(() => readCaptureCalls(page), { timeout: 10_000 }).toBe(before + captured);
  }
}

test('first-run wizard walks camera check, good/bad captures and away skip into the app', async ({ page }) => {
  await page.goto(freshPath, { waitUntil: 'commit' });

  const overlay = page.getByTestId('onboarding-overlay');
  await expect(overlay).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('heading', { name: 'Select your camera' })).toBeVisible();

  // The mocked list_cameras device is offered on the camera step.
  await expect(overlay.getByText('Mock Camera').first()).toBeAttached();

  // The harness streams personFound inference results once the camera starts,
  // so both status lines settle to their positive states. 'Person detected'
  // needs exact matching: the default substring match would also accept the
  // negative 'No person detected'.
  await expect(overlay.getByText('Camera OK').first()).toBeVisible({ timeout: 15_000 });
  await expect(overlay.getByText('Person detected', { exact: true }).first()).toBeVisible({ timeout: 15_000 });

  await page.getByRole('button', { name: 'Continue' }).click();
  await expect(page.getByRole('heading', { name: 'Capture good posture' })).toBeVisible({ timeout: 15_000 });
  await captureFrames(page, 5);

  // Five good captures auto-advance to the bad-posture step.
  await expect(page.getByRole('heading', { name: 'Capture bad posture' })).toBeVisible({ timeout: 15_000 });
  await captureFrames(page, 5);

  // Five bad captures auto-advance to the optional away step; skipping it
  // completes onboarding.
  await expect(page.getByRole('heading', { name: 'Capture away frames' })).toBeVisible({ timeout: 15_000 });
  await page.getByRole('button', { name: 'Skip this step' }).click();

  await expect(overlay).toHaveCount(0, { timeout: 15_000 });
  await expect(page.getByRole('button', { name: 'Open control panel' })).toBeVisible({ timeout: 15_000 });
});

test('completed onboarding boots straight into the app without the wizard', async ({ page }) => {
  // Default harness seed: onboardingCompleted true and a labeled frame present.
  await page.goto(applicationPath, { waitUntil: 'commit' });
  await expect(page.getByRole('button', { name: 'Open control panel' })).toBeVisible();

  // Give any late-mounting gate a moment to (wrongly) appear, then require the
  // overlay stayed absent for the whole settle window.
  await page.waitForTimeout(750);
  await expect(page.getByTestId('onboarding-overlay')).toHaveCount(0);
});

test('Skip setup on the camera step dismisses the wizard immediately', async ({ page }) => {
  await page.goto(freshPath, { waitUntil: 'commit' });

  const overlay = page.getByTestId('onboarding-overlay');
  await expect(overlay).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('heading', { name: 'Select your camera' })).toBeVisible();

  await page.getByRole('button', { name: 'Skip setup' }).click();

  await expect(overlay).toHaveCount(0, { timeout: 15_000 });
  await expect(page.getByRole('button', { name: 'Open control panel' })).toBeVisible({ timeout: 15_000 });
});
