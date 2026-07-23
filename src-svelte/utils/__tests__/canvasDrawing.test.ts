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
