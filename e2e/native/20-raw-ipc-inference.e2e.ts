import { expect } from '@wdio/globals';

import {
  RGBA_FIXTURES,
  applyNeutralCameraSettings,
  inferFixtureFrame,
  waitForNativeReady,
  expectOk,
  expectApiError,
} from './helpers/native.js';

// Proves the real raw-IPC byte path end to end against the packaged runtime:
// tightly packed RGBA request body + x-slouch-* headers in, MessagePack bytes
// out, produced by the bundled ONNX Runtime running the bundled RTMDet and
// RTMPose models. Fixture frames and their frozen expectations come from
// src-tauri/fixtures/vision/vision-inference-v1.json.
describe('raw IPC byte round-trip and packaged model inference', () => {
  before(async () => {
    await waitForNativeReady();
    // Neutral settings turn the preprocessor into a pass-through; the frozen
    // fixture scores assume unmodified pixels.
    await applyNeutralCameraSettings();
  });

  it('detects the synthetic silhouette through the full RTMDet -> RTMPose cascade', async () => {
    const result = expectOk(
      await inferFixtureFrame(RGBA_FIXTURES.personSilhouette, 101),
      'infer_frame(single-silhouette-square)',
    );
    expect(result.requestId).toBe(101);
    expect(result.personFound).toBe(true);
    expect(result.bbox).not.toBeNull();
    // Frozen ORT-Web baseline score is 0.30486 with a 2e-4 parity tolerance.
    expect(result.bbox!.original.score).toBeGreaterThan(0.3);
    expect(result.bbox!.original.score).toBeLessThan(0.32);
    expect(result.keypoints).toHaveLength(17);
    for (const keypoint of result.keypoints!) {
      expect(Number.isFinite(keypoint.x)).toBe(true);
      expect(Number.isFinite(keypoint.y)).toBe(true);
    }
    expect(result.token).not.toBe(result.requestId);
    expect(result.token).toBeGreaterThan(0);
  });

  it('reports no person for the empty synthetic scene', async () => {
    const result = expectOk(
      await inferFixtureFrame(RGBA_FIXTURES.emptyScene, 102),
      'infer_frame(empty-landscape)',
    );
    expect(result.requestId).toBe(102);
    expect(result.personFound).toBe(false);
    expect(result.bbox).toBeNull();
    expect(result.keypoints).toBeNull();
  });

  it('rejects a stride/width mismatch with the typed invalidRequest envelope', async () => {
    const error = expectApiError(
      await inferFixtureFrame(RGBA_FIXTURES.personSilhouette, 103, { stride: 516 }),
      'infer_frame(bad stride)',
    );
    expect(error.kind).toBe('invalidRequest');
    expect(error.message.toLowerCase()).toContain('stride');
  });
});
