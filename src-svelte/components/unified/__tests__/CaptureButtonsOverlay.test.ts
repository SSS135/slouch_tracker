import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { InferenceUiResult } from '@generated/bindings';
import {
  createIncompleteNativeInferenceResult,
  createMockNativeInferenceResult,
  createMockNativeKeypoints,
} from '../../../__tests__/utils/mockNativeInferenceResult';
import CaptureButtonsOverlay from '../CaptureButtonsOverlay.svelte';

function inference(overrides: Partial<InferenceUiResult> = {}): InferenceUiResult {
  return { ...createMockNativeInferenceResult(), ...overrides };
}

function props(result?: InferenceUiResult | null) {
  return {
    onCaptureGood: vi.fn().mockResolvedValue(undefined),
    onCaptureBad: vi.fn().mockResolvedValue(undefined),
    onCaptureAway: vi.fn().mockResolvedValue(undefined),
    inferenceResult: arguments.length === 0 ? inference() : result,
  };
}

function expectDisabled(disabled: boolean): void {
  for (const button of screen.getAllByRole('button')) {
    if (disabled) expect(button).toBeDisabled();
    else expect(button).not.toBeDisabled();
  }
}

afterEach(cleanup);

describe('CaptureButtonsOverlay native inference gating', () => {
  it('enables captures for a complete opaque-token result', () => {
    render(CaptureButtonsOverlay, { props: props() });
    expectDisabled(false);
  });

  it.each([
    undefined,
    null,
    createIncompleteNativeInferenceResult('person'),
    createIncompleteNativeInferenceResult('token'),
    createIncompleteNativeInferenceResult('bbox'),
    inference({ keypoints: null }),
    inference({ keypoints: createMockNativeKeypoints().slice(0, 16) }),
    inference({ token: Number.MAX_SAFE_INTEGER + 1 }),
  ])('disables captures without a persistable native result', async (result) => {
    const callbacks = props(result);
    render(CaptureButtonsOverlay, { props: callbacks });
    expectDisabled(true);

    for (const button of screen.getAllByRole('button')) {
      await fireEvent.click(button);
    }
    expect(callbacks.onCaptureGood).not.toHaveBeenCalled();
    expect(callbacks.onCaptureBad).not.toHaveBeenCalled();
    expect(callbacks.onCaptureAway).not.toHaveBeenCalled();
  });

  it('honors the explicit disabled state and suppresses callbacks', async () => {
    const callbacks = { ...props(), disabled: true };
    render(CaptureButtonsOverlay, { props: callbacks });
    expectDisabled(true);
    for (const button of screen.getAllByRole('button')) {
      await fireEvent.click(button);
    }
    expect(callbacks.onCaptureGood).not.toHaveBeenCalled();
    expect(callbacks.onCaptureBad).not.toHaveBeenCalled();
    expect(callbacks.onCaptureAway).not.toHaveBeenCalled();
  });

  it('enables captures when keypoint scores exceed 1 (SimCC activations, not probabilities)', async () => {
    // Regression: real RTMPose SimCC keypoint scores routinely exceed 1. A stale
    // score-<=1 gate disabled every on-screen capture button for well-detected
    // people while keyboard/global-shortcut capture (finiteness-only) still worked.
    const callbacks = props(inference({ keypoints: createMockNativeKeypoints(3.7) }));
    render(CaptureButtonsOverlay, { props: callbacks });
    expectDisabled(false);

    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    expect(callbacks.onCaptureGood).toHaveBeenCalledOnce();
  });

  it('routes each enabled label action', async () => {
    const callbacks = props();
    render(CaptureButtonsOverlay, { props: callbacks });
    await fireEvent.click(screen.getByRole('button', { name: 'Good' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Bad' }));
    await fireEvent.click(screen.getByRole('button', { name: 'Away' }));
    expect(callbacks.onCaptureGood).toHaveBeenCalledOnce();
    expect(callbacks.onCaptureBad).toHaveBeenCalledOnce();
    expect(callbacks.onCaptureAway).toHaveBeenCalledOnce();
  });

  it('tracks rapid null and complete inference transitions', async () => {
    const callbacks = props(null);
    const view = render(CaptureButtonsOverlay, { props: callbacks });
    expectDisabled(true);

    await view.rerender({ ...callbacks, inferenceResult: inference() });
    expectDisabled(false);
    await view.rerender({ ...callbacks, inferenceResult: null });
    expectDisabled(true);
    await view.rerender({ ...callbacks, inferenceResult: inference({ requestId: 8, token: 80 }) });
    expectDisabled(false);
  });

  it('never blinks disabled across successive valid results (token identity churn)', async () => {
    // The overlay gates on liveness/data completeness, not token consumption, so a
    // stream of fresh identities (as during auto-capture at 1-2 fps) must keep the
    // buttons continuously enabled with no per-frame disabled flash.
    const callbacks = props(inference({ requestId: 1, token: 11 }));
    const view = render(CaptureButtonsOverlay, { props: callbacks });
    expectDisabled(false);

    for (const [requestId, token] of [[2, 22], [3, 33], [4, 44]] as const) {
      await view.rerender({ ...callbacks, inferenceResult: inference({ requestId, token }) });
      expectDisabled(false);
    }
  });
});
