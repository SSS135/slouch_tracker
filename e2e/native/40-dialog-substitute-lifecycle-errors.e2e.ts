import { expect } from '@wdio/globals';

import {
  RGBA_FIXTURES,
  TINY_PNG_BASE64,
  applyNeutralCameraSettings,
  inferFixtureFrame,
  tauriInvoke,
  tauriInvokeRaw,
  waitForNativeReady,
  expectOk,
  expectApiError,
} from './helpers/native.js';

// SUBSTITUTION NOTE (native dialog cancellation).
// The plan's fifth smoke was "export_dataset, then cancel the native dialog".
// export_dataset/import_dataset call tauri-plugin-dialog's blocking_save_file/
// blocking_pick_file, which runs an OS-modal file dialog pumping its own
// message loop on the app's main thread. The embedded WebDriver executor
// (tauri-plugin-wdio-webdriver) dispatches scripts and synthesized input
// through that same main thread, so while the modal dialog is open no
// WebDriver command can reach the webview to dismiss it - the ESC would have
// to be injected into a focused native dialog, which is inherently
// focus-dependent and non-deterministic on a shared desktop. Per the task
// allowance, this spec substitutes deterministic native-lifecycle contract
// smokes that cover the same risk classes (typed error envelopes, actor
// lifecycle, one-time raw-IPC token consumption) without any OS dialog.
describe('native lifecycle contracts (dialog-cancellation substitution)', () => {
  before(async () => {
    await waitForNativeReady();
    // Pass-through preprocessing so fixture frames behave exactly as in the
    // frozen baseline regardless of what the persistence pair stored.
    await applyNeutralCameraSettings();
  });

  it('cancel_training while idle returns the typed notReady envelope', async () => {
    const error = expectApiError(
      await tauriInvoke<null>('cancel_training'),
      'cancel_training(idle)',
    );
    expect(error.kind).toBe('notReady');
    expect(error.message).toContain('no training is running');

    const status = expectOk(
      await tauriInvoke<{ running: boolean }>('get_training_status'),
      'get_training_status',
    );
    expect(status.running).toBe(false);
  });

  it('save_capture with a no-person token fails typed and restores the token', async () => {
    const statsBefore = expectOk(
      await tauriInvoke<{ total: number }>('get_dataset_stats'),
      'get_dataset_stats(before)',
    );

    const inference = expectOk(
      await inferFixtureFrame(RGBA_FIXTURES.emptyScene, 301),
      'infer_frame(empty scene)',
    );
    expect(inference.personFound).toBe(false);

    // Must reuse the infer_frame request ID: the token cache validates the pair.
    const headers = {
      'x-slouch-ipc-version': '1',
      'x-slouch-request-id': '301',
      'x-slouch-token': String(inference.token),
      'x-slouch-frame-id': 'e2e-native-no-person',
      'x-slouch-timestamp': String(Date.now()),
      'x-slouch-label': 'away',
      'x-slouch-mime-type': 'image/png',
    };

    const first = await tauriInvokeRaw('save_capture', TINY_PNG_BASE64, headers);
    const firstError = expectApiError(first, 'save_capture(no person)');
    expect(firstError.kind).toBe('invalidRequest');
    expect(firstError.message).toContain('no detected person');

    // The failed capture must restore the one-time token, so an identical
    // retry hits the same domain validation instead of an unknown-token error.
    const second = await tauriInvokeRaw('save_capture', TINY_PNG_BASE64, headers);
    const secondError = expectApiError(second, 'save_capture(no person, retry)');
    expect(secondError.kind).toBe('invalidRequest');
    expect(secondError.message).toContain('no detected person');

    const statsAfter = expectOk(
      await tauriInvoke<{ total: number }>('get_dataset_stats'),
      'get_dataset_stats(after)',
    );
    expect(statsAfter.total).toBe(statsBefore.total);
  });

  it('save_capture with an unknown token returns a typed error without writing', async () => {
    const statsBefore = expectOk(
      await tauriInvoke<{ total: number }>('get_dataset_stats'),
      'get_dataset_stats(before)',
    );

    const outcome = await tauriInvokeRaw('save_capture', TINY_PNG_BASE64, {
      'x-slouch-ipc-version': '1',
      'x-slouch-request-id': '303',
      'x-slouch-token': '987654321',
      'x-slouch-frame-id': 'e2e-native-unknown-token',
      'x-slouch-timestamp': String(Date.now()),
      'x-slouch-label': 'good',
      'x-slouch-mime-type': 'image/png',
    });
    const error = expectApiError(outcome, 'save_capture(unknown token)');
    expect(error.kind).not.toBe('internal');
    expect(error.message.length).toBeGreaterThan(0);

    const statsAfter = expectOk(
      await tauriInvoke<{ total: number }>('get_dataset_stats'),
      'get_dataset_stats(after)',
    );
    expect(statsAfter.total).toBe(statsBefore.total);
  });
});
