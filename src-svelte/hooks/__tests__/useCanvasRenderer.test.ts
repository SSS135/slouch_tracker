import { describe, expect, it } from 'vitest';
import {
  chooseVideoUrl,
  isDebugTilesViewActive,
  isProcessedViewActive,
  type PreviewUrlSelection,
} from '../useCanvasRenderer.svelte';

const RAW = 'slouchcam://localhost/frame';
const PROCESSED = 'slouchcam://localhost/processed';
const DEBUG = 'slouchcam://localhost/debug-tiles';

function selection(overrides: Partial<PreviewUrlSelection> = {}): PreviewUrlSelection {
  return {
    frameUrl: RAW,
    processedFrameUrl: PROCESSED,
    debugTilesFrameUrl: DEBUG,
    privacyMode: false,
    processedView: false,
    preprocessingDebugView: false,
    ...overrides,
  };
}

describe('chooseVideoUrl endpoint selection', () => {
  it('pumps the raw feed when the processed view is off', () => {
    const input = selection();
    expect(isProcessedViewActive(input)).toBe(false);
    expect(chooseVideoUrl(input)).toBe(RAW);
  });

  it('pumps the processed feed when the processed view is on and debug is off', () => {
    const input = selection({ processedView: true });
    expect(isProcessedViewActive(input)).toBe(true);
    expect(isDebugTilesViewActive(input)).toBe(false);
    expect(chooseVideoUrl(input)).toBe(PROCESSED);
  });

  it('pumps the debug-tiles feed when the processed view and debug view are both on', () => {
    const input = selection({ processedView: true, preprocessingDebugView: true });
    expect(isDebugTilesViewActive(input)).toBe(true);
    expect(chooseVideoUrl(input)).toBe(DEBUG);
  });

  it('keeps the raw feed in privacy mode even with processed and debug views on', () => {
    const input = selection({ processedView: true, preprocessingDebugView: true, privacyMode: true });
    expect(isProcessedViewActive(input)).toBe(false);
    expect(isDebugTilesViewActive(input)).toBe(false);
    expect(chooseVideoUrl(input)).toBe(RAW);
  });

  it('ignores the debug view while the processed view is off (debug requires processed active)', () => {
    const input = selection({ processedView: false, preprocessingDebugView: true });
    expect(isDebugTilesViewActive(input)).toBe(false);
    expect(chooseVideoUrl(input)).toBe(RAW);
  });

  it('falls back to the processed feed when the debug URL is missing', () => {
    const input = selection({ processedView: true, preprocessingDebugView: true, debugTilesFrameUrl: undefined });
    expect(isDebugTilesViewActive(input)).toBe(false);
    expect(chooseVideoUrl(input)).toBe(PROCESSED);
  });

  it('falls back to the raw feed when the processed URL is missing', () => {
    const input = selection({ processedView: true, processedFrameUrl: undefined });
    expect(isProcessedViewActive(input)).toBe(false);
    expect(chooseVideoUrl(input)).toBe(RAW);
  });
});
