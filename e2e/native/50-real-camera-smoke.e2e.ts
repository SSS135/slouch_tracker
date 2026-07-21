import { browser, expect } from '@wdio/globals';

import {
  RGBA_FIXTURES,
  inferFixtureFrame,
  tauriInvoke,
  tauriInvokeRaw,
  waitForNativeReady,
} from './helpers/native.js';

/**
 * Real-camera capture smoke test.
 *
 * Unlike 20/30 which feed frozen synthetic RGBA fixtures, this spec drives the
 * REAL packaged UI against REAL webcam frames (getUserMedia is auto-granted by
 * allow_camera_for_app_origin). It:
 *  1. Diagnoses real vs fixture keypoint scores and on-screen button gating
 *     (real RTMPose SimCC scores routinely exceed 1, which a stale score-<=1 gate
 *     in CaptureButtonsOverlay wrongly rejected, disabling every capture button).
 *  2. Captures a frame through the on-screen Good button end to end.
 *  3. Verifies frontend warn/error logs forward into the native file log
 *     (tauri-plugin-log invoke permission granted via the log:default capability).
 *  4. When GOOD+BAD frames can be captured (no AWAY), drives training and asserts
 *     a posture-only model deploys successfully (presence falls back to detector
 *     confidence) - the pre-fix "did not produce a presence model" rejection is
 *     now a regression, not a legitimate terminal.
 */

interface LiveFrame {
  width: number;
  height: number;
  rgbaBase64: string;
  videoWidth: number;
  videoHeight: number;
}

interface UiResult {
  requestId: number;
  token: number;
  personFound: boolean;
  bbox: {
    original: { x1: number; y1: number; x2: number; y2: number; score: number; width: number; height: number };
    expanded: { x1: number; y1: number; x2: number; y2: number; score: number; width: number; height: number };
  } | null;
  keypoints: Array<{ x: number; y: number; score: number }> | null;
}

/** Pull the current webcam frame off the live <video> as tightly packed RGBA. */
async function grabLiveFrame(): Promise<LiveFrame | { error: string }> {
  return browser.executeAsync((done: (r: LiveFrame | { error: string }) => void) => {
    const video = document.querySelector('video.camera-video') as HTMLVideoElement | null;
    if (!video) {
      done({ error: 'no <video.camera-video> element in DOM' });
      return;
    }
    const start = Date.now();
    const poll = (): void => {
      if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) {
        // Cap the longest side so the base64 body stays under the embedded
        // WebDriver's ~2 MiB request-body limit. Keypoint SimCC activation
        // scores are largely input-scale-independent, so this stays diagnostic.
        const maxW = 480;
        const maxH = 480;
        const scale = Math.min(1, maxW / video.videoWidth, maxH / video.videoHeight);
        const width = Math.max(1, Math.floor(video.videoWidth * scale));
        const height = Math.max(1, Math.floor(video.videoHeight * scale));
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          done({ error: 'no 2d context' });
          return;
        }
        ctx.drawImage(video, 0, 0, width, height);
        const rgba = ctx.getImageData(0, 0, width, height).data;
        let binary = '';
        const chunk = 0x8000;
        for (let i = 0; i < rgba.length; i += chunk) {
          binary += String.fromCharCode.apply(
            null,
            Array.from(rgba.subarray(i, i + chunk)) as unknown as number[],
          );
        }
        done({
          width,
          height,
          rgbaBase64: btoa(binary),
          videoWidth: video.videoWidth,
          videoHeight: video.videoHeight,
        });
        return;
      }
      if (Date.now() - start > 20000) {
        done({ error: `video never produced frames (readyState=${video.readyState}, vw=${video.videoWidth})` });
        return;
      }
      setTimeout(poll, 250);
    };
    poll();
  });
}

/** Run a live RGBA frame through the real infer_frame command. */
async function inferLiveFrame(frame: LiveFrame, requestId: number): Promise<UiResult> {
  const outcome = await tauriInvokeRaw('infer_frame', frame.rgbaBase64, {
    'x-slouch-ipc-version': '1',
    'x-slouch-pixel-format': 'rgba8',
    'x-slouch-width': String(frame.width),
    'x-slouch-height': String(frame.height),
    'x-slouch-stride': String(frame.width * 4),
    'x-slouch-request-id': String(requestId),
  });
  if (!outcome.ok) {
    throw new Error(`infer_frame(live) failed: ${JSON.stringify(outcome.error)}`);
  }
  const { decode } = await import('@msgpack/msgpack');
  return decode(Uint8Array.from(outcome.value as number[])) as UiResult;
}

async function readNotifications(): Promise<string[]> {
  return browser.execute(() =>
    Array.from(document.querySelectorAll('.notification')).map((el) => el.textContent?.trim() ?? ''),
  );
}

async function dismissNotifications(): Promise<void> {
  await browser.execute(() => {
    document
      .querySelectorAll('.notification-dismiss')
      .forEach((btn) => (btn as HTMLButtonElement).click());
  });
}

/**
 * Click an on-screen capture button until the "Frame saved." toast appears.
 *
 * The buttons stay enabled while the pipeline is live (they no longer blink on
 * per-frame token consumption), but a click can still transiently disable one
 * button during its own capture-feedback loading state, or when no person is in
 * frame. Retrying makes the real end-to-end capture deterministic. A genuine
 * "Failed to save frame" toast is surfaced immediately (that would be the bug).
 */
async function captureViaButton(label: 'Good' | 'Bad' | 'Away', overallTimeoutMs = 45_000): Promise<void> {
  const button = await browser.$(`button[aria-label="${label}"]`);
  const deadline = Date.now() + overallTimeoutMs;
  while (Date.now() < deadline) {
    await dismissNotifications();
    if (await button.isEnabled()) {
      await button.click();
      const outcome = await waitForCaptureOutcome(3_000);
      if (outcome === 'saved') return;
      if (outcome === 'failed') {
        throw new Error(`${label} capture reported a save failure: ${JSON.stringify(await readNotifications())}`);
      }
    }
    await browser.pause(300);
  }
  throw new Error(`could not capture a ${label} frame within ${overallTimeoutMs}ms (no person / persistent flicker)`);
}

/** Poll the notification area for a capture outcome. */
async function waitForCaptureOutcome(timeoutMs: number): Promise<'saved' | 'failed' | 'none'> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const notes = await readNotifications();
    if (notes.some((n) => n.toLowerCase().includes('failed to save frame'))) return 'failed';
    if (notes.some((n) => n.toLowerCase().includes('frame saved'))) return 'saved';
    await browser.pause(250);
  }
  return 'none';
}

describe('real-camera capture smoke', () => {
  before(async () => {
    await waitForNativeReady();
    // Wait for the real UI to mount its capture controls.
    await browser.waitUntil(
      async () => (await browser.$$('button[aria-label="Good"]')).length > 0,
      { timeout: 60_000, interval: 1_000, timeoutMsg: 'capture buttons never rendered' },
    );
  });

  it('DIAGNOSTIC: real vs fixture keypoint scores and on-screen button gating', async () => {
    // 1. Real webcam keypoint scores (downscaled to fit the WebDriver body limit).
    const frame = await grabLiveFrame();
    // eslint-disable-next-line no-console
    console.log('[smoke] grabLiveFrame:', JSON.stringify(
      'error' in frame ? frame : { width: frame.width, height: frame.height, videoWidth: frame.videoWidth, videoHeight: frame.videoHeight },
    ));
    if (!('error' in frame)) {
      try {
        const result = await inferLiveFrame(frame as LiveFrame, 5001);
        const scores = (result.keypoints ?? []).map((k) => k.score);
        // eslint-disable-next-line no-console
        console.log('[smoke] LIVE inference:', JSON.stringify({
          personFound: result.personFound,
          bbox: result.bbox?.original ?? null,
          keypointScoreMin: scores.length ? Math.min(...scores) : null,
          keypointScoreMax: scores.length ? Math.max(...scores) : null,
          keypointScoresOver1: scores.filter((s) => s > 1).length,
          scores,
        }));
      } catch (cause) {
        // eslint-disable-next-line no-console
        console.log('[smoke] LIVE infer_frame error:', String(cause));
      }
    }

    // 2. Fixture keypoint scores (what the existing tests exercise).
    const fixture = await inferFixtureFrame(RGBA_FIXTURES.personSilhouette, 5002);
    if (fixture.ok) {
      const fscores = (fixture.value.keypoints ?? []).map((k) => k.score);
      // eslint-disable-next-line no-console
      console.log('[smoke] FIXTURE inference:', JSON.stringify({
        personFound: fixture.value.personFound,
        keypointScoreMin: fscores.length ? Math.min(...fscores) : null,
        keypointScoreMax: fscores.length ? Math.max(...fscores) : null,
        keypointScoresOver1: fscores.filter((s) => s > 1).length,
      }));
    }

    // 3. On-screen capture-button gating (CaptureButtonsOverlay.hasInferenceData).
    const gating = await browser.execute(() => {
      const labels = ['Good', 'Bad', 'Away'];
      return labels.map((label) => {
        const btn = document.querySelector(`button[aria-label="${label}"]`) as HTMLButtonElement | null;
        return { label, present: Boolean(btn), disabled: btn ? btn.disabled : null };
      });
    });
    // eslint-disable-next-line no-console
    console.log('[smoke] on-screen capture buttons:', JSON.stringify(gating));
  });

  it('captures a frame via the on-screen Good button (must succeed once fixed)', async () => {
    // The button must ENABLE and capture must save for a valid person-found real
    // frame. Before the fix, CaptureButtonsOverlay's keypoint score-<=1 gate kept
    // the button disabled whenever any SimCC activation exceeded 1.
    await captureViaButton('Good');
    const notes = await readNotifications();
    // eslint-disable-next-line no-console
    console.log('[smoke] notifications after Good capture:', JSON.stringify(notes));
    expect(notes.some((n) => n.toLowerCase().includes('frame saved'))).toBe(true);
  });

  it('forwards frontend warn/error logs into the native file log (invoke permission granted)', async () => {
    // The observability fix: logger.ts posts warn/error to tauri-plugin-log's
    // `plugin:log|log` command, which requires the `log:default` capability.
    // A missing capability rejects with a permission error; assert it resolves.
    const outcome = await browser.executeAsync((done: (r: { ok: boolean; error?: string }) => void) => {
      const internals = (window as unknown as {
        __TAURI_INTERNALS__?: { invoke?: (c: string, a?: unknown) => Promise<unknown> };
      }).__TAURI_INTERNALS__;
      if (!internals || typeof internals.invoke !== 'function') {
        done({ ok: false, error: 'no __TAURI_INTERNALS__.invoke' });
        return;
      }
      internals
        .invoke('plugin:log|log', { level: 5, message: '[e2e] real-camera-smoke log-forward permission check' })
        .then(() => done({ ok: true }))
        .catch((cause: unknown) => done({ ok: false, error: String(cause) }));
    });
    // eslint-disable-next-line no-console
    console.log('[smoke] plugin:log|log invoke outcome:', JSON.stringify(outcome));
    expect(outcome.ok).toBe(true);
  });

  it('deploys a posture-only model after capturing GOOD+BAD frames (no AWAY)', async () => {
    // Conditional extension: only meaningful once the dataset has both a GOOD and
    // a BAD frame (has_minimum_frames = good>0 && bad>0). Capturing depends on a
    // person being in the real webcam frame; if that cannot be achieved this exits
    // without asserting rather than failing the suite on an empty room. A single
    // class captured is the legitimate Insufficient-data case (handled by the
    // has_minimum_frames skip below). Seeding GOOD+BAD makes the app auto-retrain
    // (PostureTrackerApp.refreshAndRetrain), so the training lifecycle runs without
    // driving the Training tab's manual Train button.
    try {
      await captureViaButton('Good');
      await captureViaButton('Bad');
    } catch (cause) {
      // eslint-disable-next-line no-console
      console.log('[smoke] could not seed GOOD+BAD frames, skipping train assertion:', String(cause));
      return;
    }

    const stats = await tauriInvoke<{ total: number; good: number; bad: number; hasMinimumFrames: boolean }>(
      'get_dataset_stats',
    );
    // eslint-disable-next-line no-console
    console.log('[smoke] dataset stats after seeding:', JSON.stringify(stats));
    if (!stats.ok || !stats.value.hasMinimumFrames) {
      // eslint-disable-next-line no-console
      console.log('[smoke] dataset lacks GOOD+BAD frames; skipping train assertion (single-class Insufficient-data is legitimate)');
      return;
    }

    // Oracle semantics: GOOD+BAD without any AWAY frames trains a POSTURE-ONLY
    // model and SUCCEEDS, emitting a non-fatal "No AWAY frames collected" warning;
    // presence falls back to the RTMDet detector confidence at runtime. The
    // pre-fix native deploy rejected this with "training did not produce a presence
    // model" - that hard failure must never occur now, and a deployed posture
    // model is the required success terminal.
    let fatalFailure: string | null = null;
    await browser.waitUntil(
      async () => {
        const notes = await readNotifications();
        const fatal = notes.find((n) =>
          /training failed|retraining failed|did not produce|were not activated/i.test(n),
        );
        if (fatal) {
          fatalFailure = fatal;
          return true;
        }
        const meta = await tauriInvoke<{ posture: unknown | null }>('get_active_model_metadata');
        return Boolean(meta.ok && meta.value.posture);
      },
      { timeout: 120_000, interval: 1_000, timeoutMsg: 'posture-only training never deployed for GOOD+BAD data' },
    );

    // A hard training failure for GOOD+BAD data is exactly the regression this
    // spec guards against; posture-only is a valid, successful outcome. (wdio's
    // expect takes a single argument, so surface the context via a thrown error.)
    if (fatalFailure !== null) {
      throw new Error(`training must not fail for GOOD+BAD data (posture-only is valid): ${fatalFailure}`);
    }
    expect(fatalFailure).toBeNull();

    const finalMeta = await tauriInvoke<{ posture: unknown | null; presence: unknown | null }>(
      'get_active_model_metadata',
    );
    // eslint-disable-next-line no-console
    console.log('[smoke] active model metadata after training:', JSON.stringify(finalMeta));
    expect(finalMeta.ok).toBe(true);
    if (finalMeta.ok) {
      // Posture deployed (success); presence is absent (detector-confidence fallback).
      expect(Boolean(finalMeta.value.posture)).toBe(true);
    }
  });
});
