import { expect, test, type Locator, type Page } from '@playwright/test';

const applicationPath = '/index.svelte-app-harness.html';

// Drive a real Chromium HTML5 drag-and-drop through the browser's own drag
// pipeline (trusted mouse input), not synthetic dispatchEvent. Chromium only
// promotes a mousedown into a native drag after the pointer travels, so a small
// initial move precedes the travel to the target and a final settling move
// guarantees a dragover on the target before the drop.
async function dragThumbnailToSection(page: Page, source: Locator, target: Locator): Promise<void> {
  await source.scrollIntoViewIfNeeded();
  const sourceBox = await source.boundingBox();
  const targetBox = await target.boundingBox();
  if (!sourceBox || !targetBox) throw new Error('Drag source or target had no bounding box.');

  const startX = sourceBox.x + sourceBox.width / 2;
  const startY = sourceBox.y + sourceBox.height / 2;
  const endX = targetBox.x + targetBox.width / 2;
  const endY = targetBox.y + targetBox.height / 2;

  await page.mouse.move(startX, startY);
  await page.mouse.down();
  await page.mouse.move(startX + 8, startY + 8, { steps: 6 });
  await page.mouse.move(endX, endY, { steps: 16 });
  await page.mouse.move(endX, endY, { steps: 4 });
  await page.mouse.up();
}

test.beforeEach(async ({ page }) => {
  await page.goto(applicationPath, { waitUntil: 'commit' });
  await expect(page.getByRole('button', { name: 'Open control panel' })).toBeVisible();
});

const readCaptureCalls = (page: import('@playwright/test').Page): Promise<number> =>
  page.evaluate(() => (
    window as typeof window & { __SLOUCH_HARNESS_METRICS__: { captureCalls: number } }
  ).__SLOUCH_HARNESS_METRICS__.captureCalls);

test('real app keeps capture buttons live and captures one distinct frame per labelled trigger', async ({ page }) => {
  const good = page.getByRole('button', { name: 'Good' });
  await expect(good).toBeEnabled({ timeout: 15_000 });

  // First labelled trigger captures the current frame and saves it exactly once.
  await good.click();
  await expect(page.getByText('Frame saved.')).toBeVisible();
  expect(await readCaptureCalls(page)).toBe(1);

  // The buttons stay live across the pipeline's continuous token churn - a
  // consumed token no longer blinks them disabled (the port-era regression).
  await expect(good).toBeEnabled();

  // A second labelled trigger during the live pipeline captures a distinct frame
  // (its own one-use token) exactly once - never a double-save of a spent token.
  await page.keyboard.press('b');
  await expect.poll(() => readCaptureCalls(page), { timeout: 10_000 }).toBe(2);
});

test('real app pauses and resumes tracking from the top-center toggle, gating capture while paused', async ({ page }) => {
  const good = page.getByRole('button', { name: 'Good' });
  await expect(good).toBeEnabled({ timeout: 15_000 });

  // Active tracking: the toggle reads "Pause" and is live.
  const pause = page.getByRole('button', { name: 'Pause tracking' });
  await expect(pause).toBeEnabled({ timeout: 15_000 });

  // Pause -> the toggle flips to "Resume" (aria-pressed) and capture is withdrawn
  // as the native stop_camera halts the detection stream.
  await pause.click();
  const resume = page.getByRole('button', { name: 'Resume tracking' });
  await expect(resume).toBeVisible();
  await expect(resume).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Tracking paused')).toBeVisible();
  await expect(good).toBeDisabled({ timeout: 6_000 });

  // Resume -> start_camera restores the stream, the toggle flips back to "Pause"
  // and capture comes live again.
  await resume.click();
  await expect(page.getByRole('button', { name: 'Pause tracking' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('Tracking paused')).toHaveCount(0);
  await expect(good).toBeEnabled({ timeout: 15_000 });
});

const emitTrackingState = (page: Page, paused: boolean): Promise<void> =>
  page.evaluate(
    (value) => (
      window as typeof window & { __SLOUCH_EMIT_TRACKING_STATE__: (p: boolean) => Promise<void> }
    ).__SLOUCH_EMIT_TRACKING_STATE__(value),
    paused,
  );

test('real app reflects a native/tray-initiated pause and resume from the tracking-state event', async ({ page }) => {
  const good = page.getByRole('button', { name: 'Good' });
  await expect(good).toBeEnabled({ timeout: 15_000 });

  // Tracking is live: the top-center toggle reads "Pause".
  await expect(page.getByRole('button', { name: 'Pause tracking' })).toBeEnabled({ timeout: 15_000 });

  // Simulate the native backend (tray menu / global hotkey) pausing tracking by
  // emitting the typed event. The UI button, overlay and capture gate must adopt
  // it as the single source of truth — no click on the frontend control.
  await emitTrackingState(page, true);
  const resume = page.getByRole('button', { name: 'Resume tracking' });
  await expect(resume).toBeVisible();
  await expect(resume).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Tracking paused')).toBeVisible();
  await expect(good).toBeDisabled({ timeout: 6_000 });

  // A duplicate pause echo (as the redundant stop_camera would produce) must not
  // churn the UI: it stays paused, exactly once.
  await emitTrackingState(page, true);
  await expect(resume).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Tracking paused')).toBeVisible();

  // Simulate the native backend resuming: the UI recovers to the live "Pause"
  // state and capture comes back.
  await emitTrackingState(page, false);
  await expect(page.getByRole('button', { name: 'Pause tracking' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('Tracking paused')).toHaveCount(0);
  await expect(good).toBeEnabled({ timeout: 15_000 });
});

test('real app relabels a frame by dragging it onto another section (real Chromium HTML5 DnD)', async ({ page }) => {
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await page.getByRole('tab', { name: 'Training' }).click();

  const source = page.getByRole('button', { name: 'Preview frame frame-1 labeled good' });
  await expect(source).toBeVisible();
  await expect(page.getByRole('button', { name: 'Good Frames (1)' })).toBeVisible();
  const badSection = page.getByRole('group', { name: 'Bad Frames' });

  await dragThumbnailToSection(page, source, badSection);

  // The drop actually relabelled the frame: it now lives under Bad Frames and its
  // accessible name flipped from "labeled good" to "labeled bad".
  await expect(page.getByRole('button', { name: 'Preview frame frame-1 labeled bad' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Preview frame frame-1 labeled good' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Bad Frames (1)' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Good Frames (0)' })).toBeVisible();
});

test('real app relabels and deletes a frame with authoritative native undo', async ({ page }) => {
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await page.getByRole('tab', { name: 'Training' }).click();

  // Relabel through the right-click context menu (the per-thumbnail dropdown is gone).
  const good = page.getByRole('button', { name: 'Preview frame frame-1 labeled good' });
  await expect(good).toBeVisible();
  await good.click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Move to Bad' }).click();
  await expect(page.getByRole('button', { name: 'Preview frame frame-1 labeled bad' })).toBeVisible();

  const undo = page.getByRole('button', { name: 'Undo' });
  await expect(undo).toBeVisible();
  await undo.focus();
  await expect(undo).toHaveAttribute('aria-expanded', 'true');
  await expect(undo).toHaveAttribute('aria-describedby');
  await undo.click();
  await expect(page.getByRole('button', { name: 'Preview frame frame-1 labeled good' })).toBeVisible();

  await page.getByRole('button', { name: 'Delete frame frame-1 labeled good' }).click();
  await expect(page.getByRole('button', { name: 'Preview frame frame-1 labeled good' })).toHaveCount(0);
  await undo.focus();
  await expect(undo).toHaveAttribute('aria-expanded', 'true');
  await expect(undo).toHaveAttribute('aria-describedby');
  await undo.click();
  await expect(page.getByRole('button', { name: 'Preview frame frame-1 labeled good' })).toBeVisible();
});

test('real app resets the dataset and deactivates the stale model while preserving settings', async ({ page }) => {
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await expect(page.getByText('Model Status')).toBeVisible();
  await page.getByRole('tab', { name: 'Training' }).click();

  await page.getByRole('button', { name: 'Reset Dataset', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Reset Dataset', exact: true }).click();
  await expect(page.getByText('Dataset reset complete. Settings preserved; model deactivated.')).toBeVisible();
  await expect(page.getByRole('article', { name: 'Dataset statistics' }).getByText('0', { exact: true }).first()).toBeVisible();

  await page.getByRole('tab', { name: 'Runtime Settings' }).click();
  await expect(page.getByText('No Model Trained')).toBeVisible();
  await expect(page.getByText('engineered_features')).toHaveCount(0);
});

test('real app completes the all-data reset through the production modal', async ({ page }) => {
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await expect(page.getByText('Model Status')).toBeVisible();

  await page.getByRole('button', { name: 'Reset All Data' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Reset', exact: true }).click();
  await expect(page.getByText('Dataset and settings reset.')).toBeVisible();
  await expect(page.getByText('Model Status')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Close control panel' })).toBeVisible();
});

test('real app renders statistics failure without fabricated zero counts and retries', async ({ page }) => {
  await page.goto(`${applicationPath}?failStats=2`, { waitUntil: 'commit' });
  await expect(page.getByRole('button', { name: 'Open control panel' })).toBeVisible();
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await page.getByRole('tab', { name: 'Training' }).click();

  await expect(page.getByRole('alert').filter({ hasText: 'Failed to load dataset statistics: deterministic statistics failure' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('article', { name: 'Dataset statistics' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Retry statistics' }).click();
  await expect(page.getByRole('article', { name: 'Dataset statistics' })).toBeVisible();
  await expect(page.getByRole('article', { name: 'Dataset statistics' }).getByText('1', { exact: true }).first()).toBeVisible();
});

test('real app supports disclosure and context-menu keyboard focus behavior', async ({ page }) => {
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await page.getByRole('tab', { name: 'Training' }).click();

  const disclosure = page.getByRole('button', { name: 'Good Frames (1)' });
  await expect(disclosure).toHaveAttribute('aria-expanded', 'true');
  await disclosure.click();
  await expect(disclosure).toHaveAttribute('aria-expanded', 'false');
  await disclosure.click();

  const preview = page.getByRole('button', { name: 'Preview frame frame-1 labeled good' });
  await preview.click({ button: 'right' });
  const firstMenuItem = page.getByRole('menuitem').first();
  await expect(firstMenuItem).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('menuitem').nth(1)).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('menu')).toHaveCount(0);
  await expect(preview).toBeFocused();
});

test('real app fills the viewport without page scrollbars even with an error banner visible', async ({ page }) => {
  await page.goto(`${applicationPath}?failModelMeta=1`, { waitUntil: 'commit' });
  await expect(page.getByRole('button', { name: 'Open control panel' })).toBeVisible();
  await expect(
    page.getByRole('alert').filter({ hasText: 'Failed to load active model metadata' }),
  ).toBeVisible({ timeout: 15_000 });

  const pageOverflow = async () => page.evaluate(() => {
    const el = document.scrollingElement ?? document.documentElement;
    return { vertical: el.scrollHeight - el.clientHeight, horizontal: el.scrollWidth - el.clientWidth };
  });

  const atDefault = await pageOverflow();
  expect(atDefault.vertical).toBeLessThanOrEqual(0);
  expect(atDefault.horizontal).toBeLessThanOrEqual(0);

  await page.setViewportSize({ width: 800, height: 600 });
  const atMinimum = await pageOverflow();
  expect(atMinimum.vertical).toBeLessThanOrEqual(0);
  expect(atMinimum.horizontal).toBeLessThanOrEqual(0);

  const good = page.getByRole('button', { name: 'Good' });
  await expect(good).toBeVisible();
  const box = await good.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.y + box!.height).toBeLessThanOrEqual(600);
});

test('real app keeps the mobile panel toggle and camera controls inside the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await page.reload({ waitUntil: 'commit' });
  const open = page.getByRole('button', { name: 'Open control panel' });
  await open.focus();
  await page.keyboard.press('Enter');

  const close = page.getByRole('button', { name: 'Close control panel' });
  await expect(close).toBeFocused();
  const box = await close.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(320);

  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Good' })).toBeVisible();
});
