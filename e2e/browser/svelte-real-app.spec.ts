import { expect, test } from '@playwright/test';

const applicationPath = '/index.svelte-app-harness.html';

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

test('real app relabels and deletes through paged metadata with authoritative native undo', async ({ page }) => {
  await page.getByRole('button', { name: 'Open control panel' }).click();
  await page.getByRole('tab', { name: 'Training' }).click();

  const label = page.getByRole('combobox', { name: 'Change label for frame frame-1' });
  await expect(label).toHaveValue('good');
  await label.selectOption('bad');
  await expect(label).toHaveValue('bad');

  const undo = page.getByRole('button', { name: 'Undo' });
  await expect(undo).toBeVisible();
  await undo.focus();
  await expect(undo).toHaveAttribute('aria-expanded', 'true');
  await expect(undo).toHaveAttribute('aria-describedby');
  await undo.click();
  await expect(label).toHaveValue('good');

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
