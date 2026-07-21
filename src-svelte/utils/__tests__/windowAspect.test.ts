import { describe, expect, it } from 'vitest';
import { computeAspectSnap } from '../windowAspect';

const MIN_WIDTH = 800;
const MIN_HEIGHT = 600;

describe('computeAspectSnap', () => {
  it('keeps width and grows height to match a taller camera aspect', () => {
    // 1400x900 window (aspect 1.556) with a 4:3 camera keeps width, height -> 1050.
    const result = computeAspectSnap({ width: 1400, height: 900 }, 800, 600, MIN_WIDTH, MIN_HEIGHT);
    expect(result).toEqual({ width: 1400, height: 1050 });
  });

  it('keeps width and shrinks height for a window taller than the camera aspect', () => {
    // 1000x900 window with a 4:3 camera keeps width, height -> 750.
    const result = computeAspectSnap({ width: 1000, height: 900 }, 800, 600, MIN_WIDTH, MIN_HEIGHT);
    expect(result).toEqual({ width: 1000, height: 750 });
  });

  it('returns null when the size already matches the aspect exactly', () => {
    expect(computeAspectSnap({ width: 1000, height: 750 }, 800, 600, MIN_WIDTH, MIN_HEIGHT)).toBeNull();
  });

  it('returns null when the mismatch is within the default 1px tolerance', () => {
    // 1001 width -> round(1001 * 3/4) = 751, current height 750 differs by 1px.
    expect(computeAspectSnap({ width: 1001, height: 750 }, 800, 600, MIN_WIDTH, MIN_HEIGHT)).toBeNull();
  });

  it('snaps when the mismatch exceeds the tolerance', () => {
    const result = computeAspectSnap({ width: 1000, height: 800 }, 800, 600, MIN_WIDTH, MIN_HEIGHT);
    expect(result).toEqual({ width: 1000, height: 750 });
  });

  it('grows width instead of violating the minimum height for a wide camera', () => {
    // 16:9 camera at min width 800 would need height 450 (< 600), so widen to 1067x600.
    const result = computeAspectSnap({ width: 800, height: 600 }, 1280, 720, MIN_WIDTH, MIN_HEIGHT);
    expect(result).toEqual({ width: 1067, height: 600 });
  });

  it('respects the minimum height for a wide camera at minimum width', () => {
    const result = computeAspectSnap({ width: 800, height: 600 }, 1280, 720, MIN_WIDTH, MIN_HEIGHT);
    expect(result!.height).toBeGreaterThanOrEqual(MIN_HEIGHT);
    expect(result!.width).toBeGreaterThanOrEqual(MIN_WIDTH);
  });

  it('handles a portrait camera by growing height and keeping width', () => {
    const result = computeAspectSnap({ width: 1000, height: 700 }, 480, 800, MIN_WIDTH, MIN_HEIGHT);
    expect(result).toEqual({ width: 1000, height: Math.round((1000 * 800) / 480) });
  });

  it('never produces a size below either minimum for an extreme wide aspect', () => {
    const result = computeAspectSnap({ width: 900, height: 620 }, 3000, 600, MIN_WIDTH, MIN_HEIGHT);
    expect(result!.width).toBeGreaterThanOrEqual(MIN_WIDTH);
    expect(result!.height).toBeGreaterThanOrEqual(MIN_HEIGHT);
  });

  it('returns null for invalid camera dimensions', () => {
    expect(computeAspectSnap({ width: 1000, height: 750 }, 0, 600, MIN_WIDTH, MIN_HEIGHT)).toBeNull();
    expect(computeAspectSnap({ width: 1000, height: 750 }, 800, 0, MIN_WIDTH, MIN_HEIGHT)).toBeNull();
    expect(computeAspectSnap({ width: 1000, height: 750 }, Number.NaN, 600, MIN_WIDTH, MIN_HEIGHT)).toBeNull();
  });

  it('returns null for a degenerate current size', () => {
    expect(computeAspectSnap({ width: 0, height: 0 }, 800, 600, MIN_WIDTH, MIN_HEIGHT)).toBeNull();
  });
});
