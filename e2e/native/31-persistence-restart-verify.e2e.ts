import { browser, expect } from '@wdio/globals';

import {
  tauriInvoke,
  waitForNativeReady,
  expectOk,
} from './helpers/native.js';
import {
  PERSISTED_CAMERA_SETTINGS,
  PERSISTED_FRAME_ID,
} from './helpers/persistence-contract.js';

interface FrameMetadata {
  id: string;
  label: string;
  keypoints: Array<{ x: number | null; y: number | null; score: number | null }>;
  thumbnailMimeType: string;
}

// Second half of the persistence pair: wdio.conf.ts killed the app process
// after the seeding spec, and the tauri-service health check relaunched the
// binary against the same SLOUCH_APP_DATA_DIR before this session opened.
describe('SQLite persistence across restart: verify', () => {
  before(async () => {
    await waitForNativeReady();
  });

  it('runs against a genuinely restarted app process', async () => {
    const marker = await browser.execute(
      () => (window as unknown as { __SLOUCH_E2E_MARKER__?: string }).__SLOUCH_E2E_MARKER__ ?? null,
    );
    expect(marker).toBe(null);
  });

  it('kept the distinctive camera settings', async () => {
    const settings = expectOk(
      await tauriInvoke<typeof PERSISTED_CAMERA_SETTINGS>('get_camera_settings'),
      'get_camera_settings',
    );
    expect(settings).toEqual(PERSISTED_CAMERA_SETTINGS);
  });

  it('kept the captured frame with its keypoints and thumbnail metadata', async () => {
    const stats = expectOk(
      await tauriInvoke<{ total: number; good: number }>('get_dataset_stats'),
      'get_dataset_stats',
    );
    expect(stats.total).toBe(1);
    expect(stats.good).toBe(1);

    const page = expectOk(
      await tauriInvoke<{ frames: FrameMetadata[]; total: number }>('get_dataset_page', {
        offset: 0,
        limit: 10,
      }),
      'get_dataset_page',
    );
    expect(page.total).toBe(1);
    expect(page.frames).toHaveLength(1);
    const frame = page.frames[0];
    expect(frame.id).toBe(PERSISTED_FRAME_ID);
    expect(frame.label).toBe('good');
    expect(frame.keypoints).toHaveLength(17);
    expect(frame.thumbnailMimeType).toBe('image/png');
  });
});
