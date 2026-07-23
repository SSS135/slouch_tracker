import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { browser } from '@wdio/globals';
import { decode } from '@msgpack/msgpack';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const framesDir = join(repoRoot, 'src-tauri', 'fixtures', 'vision', 'frames');

/** Typed error envelope returned by every native command (src-tauri/src/errors.rs). */
export interface ApiErrorEnvelope {
  kind: string;
  message: string;
}

export type InvokeOutcome<T> =
  | { ok: true; value: T }
  | { ok: false; error: ApiErrorEnvelope | { jsError: string } };

export interface AppStatusDto {
  ready: boolean;
  inferenceReady: boolean;
  datasetVersion: number;
  storage: { used: number; available: number; quota: number };
}

/** MessagePack DTO returned by the raw `infer_frame` byte command. */
export interface InferenceUiResultDto {
  requestId: number;
  token: number;
  personFound: boolean;
  bbox: {
    original: { x1: number; y1: number; x2: number; y2: number; score: number };
    expanded: { x1: number; y1: number; x2: number; y2: number; score: number };
  } | null;
  keypoints: Array<{ x: number; y: number; score: number }> | null;
  classification: {
    presentProbability: number | null;
    goodProbability: number | null;
  } | null;
}

/** Deterministic synthetic RGBA fixtures with frozen expected outcomes. */
export const RGBA_FIXTURES = {
  personSilhouette: { file: 'single-silhouette-square.rgba', width: 128, height: 128 },
  emptyScene: { file: 'empty-landscape.rgba', width: 160, height: 96 },
} as const;

/** 1x1 opaque PNG used as a syntactically valid capture thumbnail body. */
export const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

/**
 * Camera settings that turn the native preprocessor into an exact
 * pass-through (no CLAHE, single-frame temporal window), so raw
 * fixture frames reach the models byte-identical to the frozen
 * vision-inference-v1 baseline.
 */
export const NEUTRAL_CAMERA_SETTINGS = {
  cameraWidth: 800,
  cameraHeight: 600,
  captureIntervalSeconds: 0.5,
  autoCaptureEnabled: true,
  autoCaptureIntervalSeconds: 2.0,
  privacyMode: true,
  claheStrength: 0,
  smoothingFrames: 1,
  showDetectionOverlay: false,
};

export function loadRgbaFixtureBase64(file: string): string {
  return readFileSync(join(framesDir, file)).toString('base64');
}

/**
 * The wire outcome uses `fault` (never `error`) as the discriminant field:
 * the classic-WebDriver client treats any response whose `value` object owns
 * an `error` key as a W3C error payload and throws, so a page-side outcome of
 * `{ ok: false, error: ... }` would surface as an opaque WebDriverError
 * instead of a value. `fault` is renamed back to `error` on the Node side.
 */
type WireOutcome<T> = { ok: true; value: T } | { ok: false; fault: unknown };

function normalizeError(raw: unknown): ApiErrorEnvelope | { jsError: string } {
  if (raw && typeof raw === 'object' && 'kind' in (raw as Record<string, unknown>)) {
    return raw as ApiErrorEnvelope;
  }
  if (raw && typeof raw === 'object' && 'jsError' in (raw as Record<string, unknown>)) {
    return raw as { jsError: string };
  }
  return { jsError: String(raw) };
}

function fromWire<T>(outcome: WireOutcome<T>): InvokeOutcome<T> {
  if (outcome.ok) {
    return outcome;
  }
  return { ok: false, error: normalizeError(outcome.fault) };
}

/**
 * Invoke a JSON Tauri command from inside the packaged webview.
 *
 * Uses the always-present `window.__TAURI_INTERNALS__.invoke` bridge (the same
 * transport `@tauri-apps/api` delegates to), so it exercises the real IPC layer
 * of the packaged app without requiring `withGlobalTauri` or any test-only
 * frontend shim.
 */
export async function tauriInvoke<T>(
  cmd: string,
  args?: Record<string, unknown>,
): Promise<InvokeOutcome<T>> {
  const outcome = await browser.executeAsync(
    (cmdIn: string, argsIn: Record<string, unknown> | null, done: (result: unknown) => void) => {
      const internals = (window as unknown as {
        __TAURI_INTERNALS__?: { invoke?: (c: string, a?: unknown) => Promise<unknown> };
      }).__TAURI_INTERNALS__;
      if (!internals || typeof internals.invoke !== 'function') {
        done({ ok: false, fault: { jsError: 'window.__TAURI_INTERNALS__.invoke is unavailable' } });
        return;
      }
      internals
        .invoke(cmdIn, argsIn ?? {})
        .then((value) => done({ ok: true, value: value === undefined ? null : value }))
        .catch((rejection: unknown) => done({ ok: false, fault: rejection }));
    },
    cmd,
    args ?? null,
  );
  return fromWire(outcome as WireOutcome<T>);
}

/**
 * Invoke a raw-byte Tauri command (`infer_frame`, `save_capture`): the body is
 * a tightly packed byte payload and the metadata travels in `x-slouch-*`
 * invoke headers. A raw (`tauri::ipc::Response`) reply is returned as a plain
 * number array so the Node side can MessagePack-decode it; JSON replies (e.g.
 * `save_capture`'s unit result) pass through unchanged.
 */
export async function tauriInvokeRaw(
  cmd: string,
  bodyBase64: string,
  headers: Record<string, string>,
): Promise<InvokeOutcome<number[] | null>> {
  const outcome = await browser.executeAsync(
    (
      cmdIn: string,
      b64: string,
      headersIn: Record<string, string>,
      done: (result: unknown) => void,
    ) => {
      const internals = (window as unknown as {
        __TAURI_INTERNALS__?: {
          invoke?: (c: string, a?: unknown, o?: unknown) => Promise<unknown>;
        };
      }).__TAURI_INTERNALS__;
      if (!internals || typeof internals.invoke !== 'function') {
        done({ ok: false, fault: { jsError: 'window.__TAURI_INTERNALS__.invoke is unavailable' } });
        return;
      }
      let bytes: Uint8Array;
      try {
        const binary = atob(b64);
        bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
          bytes[i] = binary.charCodeAt(i);
        }
      } catch (decodeError) {
        done({
          ok: false,
          fault: { jsError: `fixture base64 decode failed: ${String(decodeError)}` },
        });
        return;
      }
      internals
        .invoke(cmdIn, bytes, { headers: headersIn })
        .then((response) => {
          if (response instanceof ArrayBuffer) {
            done({ ok: true, value: Array.from(new Uint8Array(response)) });
          } else if (ArrayBuffer.isView(response)) {
            const view = response as Uint8Array;
            done({
              ok: true,
              value: Array.from(
                new Uint8Array(view.buffer, view.byteOffset, view.byteLength),
              ),
            });
          } else {
            done({ ok: true, value: response === undefined ? null : response });
          }
        })
        .catch((rejection: unknown) => done({ ok: false, fault: rejection }));
    },
    cmd,
    bodyBase64,
    headers,
  );
  return fromWire(outcome as WireOutcome<number[] | null>);
}

export function decodeMsgpack<T>(bytes: number[]): T {
  return decode(Uint8Array.from(bytes)) as T;
}

/** Send a raw RGBA fixture through `infer_frame` and decode the MessagePack reply. */
export async function inferFixtureFrame(
  fixture: { file: string; width: number; height: number },
  requestId: number,
  overrides?: Partial<Record<'stride' | 'width' | 'height', number>>,
): Promise<InvokeOutcome<InferenceUiResultDto>> {
  const width = overrides?.width ?? fixture.width;
  const height = overrides?.height ?? fixture.height;
  const stride = overrides?.stride ?? width * 4;
  const outcome = await tauriInvokeRaw('infer_frame', loadRgbaFixtureBase64(fixture.file), {
    'x-slouch-ipc-version': '1',
    'x-slouch-pixel-format': 'rgba8',
    'x-slouch-width': String(width),
    'x-slouch-height': String(height),
    'x-slouch-stride': String(stride),
    'x-slouch-request-id': String(requestId),
  });
  if (!outcome.ok) {
    return outcome;
  }
  if (!Array.isArray(outcome.value)) {
    return { ok: false, error: { jsError: 'infer_frame returned a non-binary response' } };
  }
  return { ok: true, value: decodeMsgpack<InferenceUiResultDto>(outcome.value) };
}

/**
 * Wait until the packaged native runtime reports full readiness: storage open,
 * both ONNX models loaded, and inference/training actors healthy. Generous
 * timeout because the first launch loads a 54 MiB RTMPose model from disk.
 */
export async function waitForNativeReady(timeoutMs = 240_000): Promise<AppStatusDto> {
  let last: unknown = null;
  let ready: AppStatusDto | null = null;
  await browser.waitUntil(
    async () => {
      const status = await tauriInvoke<AppStatusDto>('app_status');
      last = status;
      if (status.ok && status.value.ready === true && status.value.inferenceReady === true) {
        ready = status.value;
        return true;
      }
      return false;
    },
    {
      timeout: timeoutMs,
      interval: 2_000,
      timeoutMsg: `native runtime did not become ready; last app_status outcome: ${JSON.stringify(last)}`,
    },
  );
  if (!ready) {
    throw new Error('unreachable: readiness poll resolved without a status');
  }
  return ready;
}

/** Make preprocessing a pass-through so fixture expectations hold exactly. */
export async function applyNeutralCameraSettings(): Promise<void> {
  const saved = await tauriInvoke<null>('save_camera_settings', {
    settings: NEUTRAL_CAMERA_SETTINGS,
  });
  expectOk(saved, 'save_camera_settings(neutral)');
}

export function expectOk<T>(outcome: InvokeOutcome<T>, context: string): T {
  if (!outcome.ok) {
    throw new Error(`${context} failed: ${JSON.stringify(outcome.error)}`);
  }
  return outcome.value;
}

export function expectApiError<T>(outcome: InvokeOutcome<T>, context: string): ApiErrorEnvelope {
  if (outcome.ok) {
    throw new Error(`${context}: expected a typed error but got ${JSON.stringify(outcome.value)}`);
  }
  if (!('kind' in outcome.error)) {
    throw new Error(`${context}: expected the ApiError envelope but got ${JSON.stringify(outcome.error)}`);
  }
  return outcome.error;
}
