import { vi } from 'vitest';
/**
 * Tests for canvas drawing utilities - NLF-L
 * All keypoints use 0-1 normalized coordinates
 */

import {
  drawHumanLikeSkeleton,
  drawDetectionBox,
  drawKeypointOverlay,
} from '../canvasDrawing';
import type { SmoothedKeypoint, Keypoint } from '../canvasDrawing';
import {
  NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
  LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
} from '@/services/posture/keypointIndices';

describe('canvasDrawing', () => {
  describe('boundary detection', () => {
    let mockCtx: CanvasRenderingContext2D;
    const canvasWidth = 640;
    const canvasHeight = 480;

    const createIsolatedKeypoints = (): SmoothedKeypoint[] =>
      Array.from({ length: 17 }, () => ({
        x: 0.5,
        y: 0.5,
        score: 0.9,
        opacity: 0,
      }));

    const activate = (
      keypoints: SmoothedKeypoint[],
      index: number,
      x: number,
      y: number,
    ): void => {
      keypoints[index] = { x, y, score: 0.9, opacity: 1 };
    };

    beforeEach(() => {
      mockCtx = {
        strokeStyle: '',
        fillStyle: '',
        lineWidth: 0,
        globalAlpha: 1,
        font: '',
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        stroke: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        clearRect: vi.fn(),
        rect: vi.fn(),
        setLineDash: vi.fn(),
        strokeText: vi.fn(),
        fillText: vi.fn(),
        save: vi.fn(),
        restore: vi.fn(),
        closePath: vi.fn(),
        translate: vi.fn(),
        rotate: vi.fn(),
      } as unknown as CanvasRenderingContext2D;
    });

    describe('drawHumanLikeSkeleton - limbs', () => {
      it('should hide limb when both endpoints are at boundary', () => {
        const smoothedKeypoints = createIsolatedKeypoints();
        activate(smoothedKeypoints, 5, 0.4, 0.995);
        activate(smoothedKeypoints, 7, 0.35, 0.995);

        drawHumanLikeSkeleton(
          mockCtx,
          smoothedKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        expect(mockCtx.translate).not.toHaveBeenCalled();
        expect(mockCtx.rotate).not.toHaveBeenCalled();
      });

      it('should draw limb when only one endpoint is at boundary', () => {
        const smoothedKeypoints = createIsolatedKeypoints();
        activate(smoothedKeypoints, 5, 0.4, 0.995);
        activate(smoothedKeypoints, 7, 0.35, 0.5);

        drawHumanLikeSkeleton(
          mockCtx,
          smoothedKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        expect(mockCtx.translate).toHaveBeenCalledTimes(1);
        expect(mockCtx.rotate).toHaveBeenCalledTimes(1);
      });

      it('should hide all limbs when all keypoints are at boundaries', () => {
        const smoothedKeypoints: SmoothedKeypoint[] = Array.from(
          { length: 17 },
          (_, idx) => ({
            x: 0.1 + idx * 0.05,
            y: 0.005,
            score: 0.9,
            opacity: 1,
          }),
        );

        drawHumanLikeSkeleton(
          mockCtx,
          smoothedKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        expect(mockCtx.translate).not.toHaveBeenCalled();
        expect(mockCtx.rotate).not.toHaveBeenCalled();
      });

      it('should handle mixed boundary and interior keypoints correctly', () => {
        const smoothedKeypoints = createIsolatedKeypoints();
        activate(smoothedKeypoints, 5, 0.4, 0.995);
        activate(smoothedKeypoints, 7, 0.35, 0.995);
        activate(smoothedKeypoints, 9, 0.3, 0.5);
        activate(smoothedKeypoints, 6, 0.6, 0.4);
        activate(smoothedKeypoints, 8, 0.65, 0.5);
        activate(smoothedKeypoints, 10, 0.7, 0.6);

        drawHumanLikeSkeleton(
          mockCtx,
          smoothedKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        // [5,7] is skipped; [7,9], [6,8], and [8,10] are the only active capsules.
        expect(mockCtx.translate).toHaveBeenCalledTimes(3);
        expect(mockCtx.rotate).toHaveBeenCalledTimes(3);
      });

      it('should draw the torso quad for interior shoulder/hip corners', () => {
        const smoothedKeypoints = createIsolatedKeypoints();
        activate(smoothedKeypoints, 5, 0.4, 0.3);
        activate(smoothedKeypoints, 6, 0.6, 0.3);
        activate(smoothedKeypoints, 11, 0.42, 0.7);
        activate(smoothedKeypoints, 12, 0.58, 0.7);

        drawHumanLikeSkeleton(
          mockCtx,
          smoothedKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        // moveTo/lineTo are only used by the torso quad path.
        expect(mockCtx.moveTo).toHaveBeenCalledTimes(1);
      });

      it('should skip the torso quad when all four corners are at the boundary', () => {
        // Regression: a partial detection clamped to y≈0 painted a full-width bar
        // across the top of the frame. The degenerate torso must be skipped.
        const smoothedKeypoints = createIsolatedKeypoints();
        activate(smoothedKeypoints, 5, 0.4, 0.005);
        activate(smoothedKeypoints, 6, 0.6, 0.005);
        activate(smoothedKeypoints, 11, 0.42, 0.005);
        activate(smoothedKeypoints, 12, 0.58, 0.005);

        drawHumanLikeSkeleton(
          mockCtx,
          smoothedKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        expect(mockCtx.moveTo).not.toHaveBeenCalled();
      });

      it('should respect 0.01 boundary threshold for limbs', () => {
        const atThresholdKeypoints = createIsolatedKeypoints();
        activate(atThresholdKeypoints, 5, 0.4, 0.01);
        activate(atThresholdKeypoints, 7, 0.35, 0.01);

        drawHumanLikeSkeleton(
          mockCtx,
          atThresholdKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        expect(mockCtx.translate).not.toHaveBeenCalled();

        const insideThresholdKeypoints = createIsolatedKeypoints();
        activate(insideThresholdKeypoints, 5, 0.4, 0.02);
        activate(insideThresholdKeypoints, 7, 0.35, 0.02);

        drawHumanLikeSkeleton(
          mockCtx,
          insideThresholdKeypoints,
          canvasWidth,
          canvasHeight,
          canvasWidth,
          canvasHeight,
        );

        expect(mockCtx.translate).toHaveBeenCalledTimes(1);
      });
    });
  });
});

describe('drawHumanLikeSkeleton - head radius', () => {
  const canvasWidth = 640;
  const canvasHeight = 480;

  const makeCtx = (): CanvasRenderingContext2D =>
    ({
      strokeStyle: '',
      fillStyle: '',
      lineWidth: 0,
      globalAlpha: 1,
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      save: vi.fn(),
      restore: vi.fn(),
      closePath: vi.fn(),
      translate: vi.fn(),
      rotate: vi.fn(),
    }) as unknown as CanvasRenderingContext2D;

  const isolated = (): SmoothedKeypoint[] =>
    Array.from({ length: 17 }, () => ({ x: 0.5, y: 0.5, score: 0.9, opacity: 0 }));

  const set = (
    kps: SmoothedKeypoint[],
    index: number,
    x: number,
    y: number,
    opacity = 1,
  ): void => {
    kps[index] = { x, y, score: 0.9, opacity };
  };

  // The head circle is the largest-radius arc drawn by drawFace; when only face
  // and shoulder keypoints are active, every arc comes from drawFace, so the max
  // arc radius is the head radius.
  const headRadiusFrom = (ctx: CanvasRenderingContext2D): number => {
    const calls = vi.mocked(ctx.arc).mock.calls;
    return Math.max(...calls.map((c) => c[2] as number));
  };

  it('(a) frontal face: radius matches the historical ear-span * 0.65 formula', () => {
    const ctx = makeCtx();
    const kps = isolated();
    // Ears level and symmetric; the ear-to-ear pair is the maximum face spread.
    set(kps, LEFT_EAR, 0.42, 0.3);
    set(kps, RIGHT_EAR, 0.58, 0.3);
    set(kps, LEFT_EYE, 0.46, 0.29);
    set(kps, RIGHT_EYE, 0.54, 0.29);
    set(kps, NOSE, 0.5, 0.33);
    set(kps, LEFT_SHOULDER, 0.35, 0.45);
    set(kps, RIGHT_SHOULDER, 0.65, 0.45);

    drawHumanLikeSkeleton(ctx, kps, canvasWidth, canvasHeight, canvasWidth, canvasHeight);

    const earSpanPx = Math.abs(0.58 - 0.42) * canvasWidth;
    const historical = earSpanPx * 0.65;
    const radius = headRadiusFrom(ctx);
    // Within ±10% of the pre-fix frontal appearance (in practice exact here).
    expect(Math.abs(radius - historical) / historical).toBeLessThan(0.1);
    expect(radius).toBeCloseTo(historical, 1);
  });

  it('(b) profile face: ears x-collapse but radius stays sane via nose-ear spread', () => {
    const ctx = makeCtx();
    const kps = isolated();
    // Both ears still confident but their x projections collapse (the bug case):
    // the old |leftEar.x - rightEar.x| * 0.65 formula would give ~4px here.
    set(kps, LEFT_EAR, 0.495, 0.31);
    set(kps, RIGHT_EAR, 0.505, 0.31);
    set(kps, LEFT_EYE, 0.55, 0.3); // near eye
    // RIGHT_EYE (far eye) occluded -> stays opacity 0, filtered out.
    set(kps, NOSE, 0.6, 0.33); // nose well forward of the ears
    set(kps, LEFT_SHOULDER, 0.4, 0.48);
    set(kps, RIGHT_SHOULDER, 0.62, 0.48);

    drawHumanLikeSkeleton(ctx, kps, canvasWidth, canvasHeight, canvasWidth, canvasHeight);

    const collapsedEarSpanPx = Math.abs(0.505 - 0.495) * canvasWidth; // ~6.4px
    const oldBuggyRadius = collapsedEarSpanPx * 0.65; // ~4.16px
    const radius = headRadiusFrom(ctx);
    expect(radius).toBeGreaterThan(30);
    expect(radius).toBeGreaterThan(oldBuggyRadius * 5);
    // nose-to-near-ear diagonal drives the radius: ~67.9px * 0.65 ≈ 44px.
    expect(radius).toBeCloseTo(44.1, 0);
  });

  it('(c) degenerate face spread: falls back to the shoulder-width floor', () => {
    const ctx = makeCtx();
    const kps = isolated();
    // Confident face points all clustered -> spread is ~0; the shoulder-based
    // floor must keep the head from collapsing.
    set(kps, NOSE, 0.5, 0.3);
    set(kps, LEFT_EYE, 0.501, 0.301);
    set(kps, RIGHT_EYE, 0.499, 0.299);
    set(kps, LEFT_SHOULDER, 0.35, 0.45);
    set(kps, RIGHT_SHOULDER, 0.65, 0.45);

    drawHumanLikeSkeleton(ctx, kps, canvasWidth, canvasHeight, canvasWidth, canvasHeight);

    const shoulderWidthPx = Math.abs(0.65 - 0.35) * canvasWidth; // 192
    const floor = shoulderWidthPx * 0.16; // 30.72
    const radius = headRadiusFrom(ctx);
    expect(radius).toBeCloseTo(floor, 1);
  });

  it('(d) missing face entirely: no head circle is drawn and no crash', () => {
    const ctx = makeCtx();
    const kps = isolated();
    // Torso only; all face keypoints stay opacity 0.
    set(kps, LEFT_SHOULDER, 0.35, 0.45);
    set(kps, RIGHT_SHOULDER, 0.65, 0.45);
    set(kps, LEFT_HIP, 0.38, 0.75);
    set(kps, RIGHT_HIP, 0.62, 0.75);

    expect(() =>
      drawHumanLikeSkeleton(ctx, kps, canvasWidth, canvasHeight, canvasWidth, canvasHeight),
    ).not.toThrow();
    // No wrists/limbs/face are active, so drawFace's early return means no arc.
    expect(ctx.arc).not.toHaveBeenCalled();
  });
});

describe('drawDetectionBox', () => {
  let ctx: CanvasRenderingContext2D;

  beforeEach(() => {
    ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn(() => ({ width: 60 }) as TextMetrics),
      strokeStyle: '',
      fillStyle: '',
      lineWidth: 0,
      font: '',
      textBaseline: 'alphabetic' as CanvasTextBaseline,
      textAlign: 'start' as CanvasTextAlign,
    } as unknown as CanvasRenderingContext2D;
  });

  it('strokes the box rectangle scaled to the canvas dimensions', () => {
    drawDetectionBox(ctx, { x1: 0.1, y1: 0.2, x2: 0.8, y2: 0.9, score: 0.87 }, 640, 480);
    // left=0.1*640, top=0.2*480, width=0.7*640, height=0.7*480
    const [left, top, width, height] = vi.mocked(ctx.strokeRect).mock.calls[0];
    expect(left).toBeCloseTo(64);
    expect(top).toBeCloseTo(96);
    expect(width).toBeCloseTo(448);
    expect(height).toBeCloseTo(336);
  });

  it('labels the box with a two-decimal confidence', () => {
    drawDetectionBox(ctx, { x1: 0.1, y1: 0.2, x2: 0.8, y2: 0.9, score: 0.8666 }, 640, 480);
    expect(vi.mocked(ctx.fillText).mock.calls[0][0]).toBe('person 0.87');
  });

  it('draws nothing when any coordinate is null', () => {
    drawDetectionBox(ctx, { x1: null, y1: 0.2, x2: 0.8, y2: 0.9, score: 0.9 }, 640, 480);
    expect(ctx.strokeRect).not.toHaveBeenCalled();
    expect(ctx.fillText).not.toHaveBeenCalled();
  });
});

describe('drawKeypointOverlay', () => {
  let ctx: CanvasRenderingContext2D;

  const keypoints = (score: number): Keypoint[] =>
    Array.from({ length: 17 }, (_, i) => ({ x: 0.1 + i * 0.02, y: 0.1 + i * 0.02, score }));

  beforeEach(() => {
    ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      strokeStyle: '',
      fillStyle: '',
      lineWidth: 0,
      lineCap: 'butt' as CanvasLineCap,
    } as unknown as CanvasRenderingContext2D;
  });

  it('draws a dot per keypoint and a line per COCO edge when all are confident', () => {
    drawKeypointOverlay(ctx, keypoints(0.9), 640, 480);
    // 17 keypoints -> 17 dots (arc); 16 COCO edges -> 16 lines (moveTo/lineTo).
    expect(ctx.arc).toHaveBeenCalledTimes(17);
    expect(ctx.moveTo).toHaveBeenCalledTimes(16);
    expect(ctx.lineTo).toHaveBeenCalledTimes(16);
  });

  it('skips keypoints and edges at or below the score threshold', () => {
    drawKeypointOverlay(ctx, keypoints(0.2), 640, 480);
    expect(ctx.arc).not.toHaveBeenCalled();
    expect(ctx.moveTo).not.toHaveBeenCalled();
  });

  it('draws only the edge whose both endpoints clear the threshold', () => {
    const kps = keypoints(0.1);
    // LEFT_SHOULDER (5) - LEFT_ELBOW (7) is a COCO edge; make only those confident.
    kps[5] = { x: 0.4, y: 0.4, score: 0.9 };
    kps[7] = { x: 0.5, y: 0.6, score: 0.9 };
    drawKeypointOverlay(ctx, kps, 640, 480);
    expect(ctx.arc).toHaveBeenCalledTimes(2); // two dots
    expect(ctx.moveTo).toHaveBeenCalledTimes(1); // one edge
  });
});
