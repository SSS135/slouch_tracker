import { expect, test } from '@playwright/test';

const harnessPath = '/index.svelte-harness.html';

test.beforeEach(async ({ page }) => {
  await page.goto(harnessPath, { waitUntil: 'commit' });
  await expect(page.getByRole('heading', { name: 'Mocked Tauri plumbing harness' })).toBeVisible();
});

test('initializes native readiness and sends raw capture bytes', async ({ page }) => {
  await page.getByRole('button', { name: 'Initialize' }).click();
  await expect(page.getByTestId('readiness')).toHaveText('ready');

  await page.getByRole('button', { name: 'Capture frame' }).click();
  await expect(page.getByTestId('capture-status')).toHaveText('saved 3 bytes');
  await expect(page.getByTestId('dataset')).toContainText('captured-frame:good');
});

test('mutates paged dataset state and delegates undo to native IPC', async ({ page }) => {
  await page.getByRole('button', { name: 'Refresh dataset' }).click();
  await expect(page.getByTestId('dataset')).toContainText('frame-1:good');

  await page.getByRole('button', { name: 'Relabel frame' }).click();
  await expect(page.getByTestId('dataset')).toContainText('frame-1:bad');

  await page.getByRole('button', { name: 'Delete frame' }).click();
  await expect(page.getByTestId('dataset')).not.toContainText('frame-1');

  await page.getByRole('button', { name: 'Undo dataset change' }).click();
  await expect(page.getByTestId('dataset')).toContainText('frame-1:bad');
});

test('round-trips and resets Rust-owned camera and UI settings', async ({ page }) => {
  await page.getByRole('button', { name: 'Save settings' }).click();
  await expect(page.getByTestId('settings-status')).toHaveText('1280/0.75');

  await page.getByRole('button', { name: 'Reset settings' }).click();
  await expect(page.getByTestId('settings-status')).toHaveText('800/0.3');
});

test('renders tray startup toggles checked by default and persists an unchecked toggle across a reload', async ({ page }) => {
  const minimize = page.getByRole('checkbox', { name: 'Minimize to tray on close' });
  const startHidden = page.getByRole('checkbox', { name: 'Start hidden at login' });

  // Native omits both fields but serde defaults them true, so both toggles load on.
  await expect(minimize).toBeChecked();
  await expect(startHidden).toBeChecked();

  await minimize.uncheck();
  await expect(page.getByTestId('startup-status')).toHaveText('false/true');

  // Re-reading through the native settings pipeline keeps the persisted value:
  // the unchecked toggle survives, the untouched one stays on.
  await page.getByRole('button', { name: 'Reload startup settings' }).click();
  await expect(minimize).not.toBeChecked();
  await expect(startHidden).toBeChecked();
  await expect(page.getByTestId('startup-status')).toHaveText('false/true');
});

test('reports ordered training success, cancellation, and typed failure', async ({ page }) => {
  await page.getByRole('button', { name: 'Train success' }).click();
  await expect(page.getByTestId('training-status')).toHaveText('completed');
  await expect(page.getByTestId('training-events')).toHaveText('started,progress,progress,progress,completed');

  await page.getByRole('button', { name: 'Train cancellable' }).click();
  await expect(page.getByTestId('training-status')).toHaveText('progress');
  await expect(page.getByTestId('training-events')).toHaveText('started,progress');
  await page.getByRole('button', { name: 'Cancel training' }).click();
  await expect(page.getByTestId('training-status')).toHaveText('cancelled');
  await expect(page.getByTestId('training-events')).toHaveText('started,progress,cancelled');

  await page.getByRole('button', { name: 'Train failure' }).click();
  await expect(page.getByTestId('training-status')).toHaveText('failed');
  await expect(page.getByTestId('training-events')).toHaveText('started,progress,failed');
  await expect(page.getByRole('alert')).toHaveText('deterministic training failure');
});
