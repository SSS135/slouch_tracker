import { vi } from 'vitest';
/**
 * Tests for canvas drawing utilities - RTMPose-S
 * All keypoints use 0-1 normalized coordinates
 */

import {
  drawHumanLikeSkeleton,
} from '../canvasDrawing';
import type { SmoothedKeypoint } from '../canvasDrawing';

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
