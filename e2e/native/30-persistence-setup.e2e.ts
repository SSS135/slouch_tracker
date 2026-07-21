import { browser, expect } from '@wdio/globals';

import {
  RGBA_FIXTURES,
  TINY_PNG_BASE64,
  applyNeutralCameraSettings,
  inferFixtureFrame,
  tauriInvoke,
  tauriInvokeRaw,
  waitForNativeReady,
  expectOk,
} from './helpers/native.js';
import {
  PERSISTED_CAMERA_SETTINGS,
  PERSISTED_FRAME_ID,
} from './helpers/persistence-contract.js';

// First half of the SQLite persistence-across-restart pair. Seeds settings and
// a captured frame, then wdio.conf.ts terminates the app process after this
// worker so 31-persistence-restart-verify runs against a genuinely restarted
// process on the same SLOUCH_APP_DATA_DIR.
describe('SQLite persistence across restart: seed', () => {
  before(async () => {
    await waitForNativeReady();
    // Pass-through preprocessing so the silhouette fixture is detected
    // exactly as in the frozen baseline. The distinctive settings are saved
    // AFTER the capture test, in its own test below.
    await applyNeutralCameraSettings();
  });

  it('captures a frame through the raw save_capture path with a real inference token', async () => {
    const inference = expectOk(
      await inferFixtureFrame(RGBA_FIXTURES.personSilhouette, 201),
      'infer_frame(silhouette)',
    );
    expect(inference.personFound).toBe(true);

    // save_capture must reuse the originating infer_frame request ID; the
    // one-time token cache validates the (token, requestId) pair.
    const saved = await tauriInvokeRaw('save_capture', TINY_PNG_BASE64, {
      'x-slouch-ipc-version': '1',
      'x-slouch-request-id': '201',
      'x-slouch-token': String(inference.token),
      'x-slouch-frame-id': PERSISTED_FRAME_ID,
      'x-slouch-timestamp': String(Date.now()),
      'x-slouch-label': 'good',
      'x-slouch-mime-type': 'image/png',
    });
    expect(saved.ok).toBe(true);

    const stats = expectOk(
      await tauriInvoke<{ total: number; good: number }>('get_dataset_stats'),
      'get_dataset_stats',
    );
    expect(stats.total).toBe(1);
    expect(stats.good).toBe(1);
  });

  it('saves distinctive camera settings', async () => {
    expectOk(
      await tauriInvoke<null>('save_camera_settings', { settings: PERSISTED_CAMERA_SETTINGS }),
      'save_camera_settings',
    );
    const readBack = expectOk(
      await tauriInvoke<typeof PERSISTED_CAMERA_SETTINGS>('get_camera_settings'),
      'get_camera_settings',
    );
    expect(readBack).toEqual(PERSISTED_CAMERA_SETTINGS);
  });

  it('marks the live webview so the restart spec can prove a fresh process', async () => {
    await browser.execute(() => {
      (window as unknown as { __SLOUCH_E2E_MARKER__?: string }).__SLOUCH_E2E_MARKER__ =
        'pre-restart';
    });
    const marker = await browser.execute(
      () => (window as unknown as { __SLOUCH_E2E_MARKER__?: string }).__SLOUCH_E2E_MARKER__ ?? null,
    );
    expect(marker).toBe('pre-restart');
  });
});
