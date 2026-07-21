import { vi } from 'vitest';

import {
  renderCaptureThumbnail,
  estimateThumbnailSize,
  validateThumbnailSize,
  type PreviewFrameSource,
} from '../thumbnailGenerator';
import { createMockCanvasElement, mockDocument, restoreDocument } from '../../../__tests__/utils/mockVideoElement';
import { KEYPOINT_RENDER_MIN_CONFIDENCE } from '@/services/ml/constants';
import * as colorUtils from '@/utils/colorUtils';
import * as bicubicGridRenderer from '@/utils/bicubicGridRenderer';
import * as canvasDrawing from '@/utils/canvasDrawing';

vi.mock('@/utils/colorUtils');
vi.mock('@/utils/bicubicGridRenderer');
vi.mock('@/utils/canvasDrawing');

const mockSampleImageGrid = vi.fn();
const mockRenderBicubicGrid = vi.fn();
const mockDrawHumanLikeSkeleton = vi.fn();

type PresentationKeypoint = { x: number; y: number; score: number };

const createMockKeypoints = (): PresentationKeypoint[] =>
  Array.from({ length: 17 }, (_, i) => ({
    x: 0.1 + i * 0.02,
    y: 0.15 + i * 0.015,
    score: 0.9,
  }));

const previewFrame = (): PreviewFrameSource => ({
  image: {} as CanvasImageSource,
  width: 640,
  height: 480,
});

describe('ThumbnailGenerator', () => {
  beforeEach(() => {
    mockDocument();

    vi.mocked(colorUtils.sampleImageGrid).mockImplementation(mockSampleImageGrid);
    vi.mocked(bicubicGridRenderer.renderBicubicGrid).mockImplementation(mockRenderBicubicGrid);
    vi.mocked(canvasDrawing.drawHumanLikeSkeleton).mockImplementation(mockDrawHumanLikeSkeleton);

    mockSampleImageGrid.mockReturnValue([
      [{ r: 10, g: 20, b: 30 }, { r: 40, g: 50, b: 60 }],
      [{ r: 70, g: 80, b: 90 }, { r: 100, g: 110, b: 120 }],
    ]);
    mockRenderBicubicGrid.mockImplementation(() => {});
    mockDrawHumanLikeSkeleton.mockImplementation(() => {});
  });

  afterEach(() => {
    restoreDocument();
    vi.clearAllMocks();
  });

  describe('renderCaptureThumbnail', () => {
    it('cover-fits the preview frame in non-privacy mode', async () => {
      const canvas = createMockCanvasElement();
      const ctx = canvas.getContext('2d') as unknown as { drawImage: ReturnType<typeof vi.fn> };
      vi.spyOn(document, 'createElement').mockReturnValue(canvas as unknown as HTMLElement);

      const thumbnail = await renderCaptureThumbnail({
        privacyMode: false,
        keypoints: createMockKeypoints(),
        previewFrame: previewFrame(),
      });

      expect(thumbnail).toBeInstanceOf(Blob);
      expect(thumbnail.type).toBe('image/webp');
      expect(ctx.drawImage).toHaveBeenCalled();
      expect(mockDrawHumanLikeSkeleton).not.toHaveBeenCalled();
      expect(mockSampleImageGrid).not.toHaveBeenCalled();
    });

    it('draws the skeleton over a blurred grid sampled from the preview frame in privacy mode', async () => {
      const thumbnail = await renderCaptureThumbnail({
        privacyMode: true,
        keypoints: createMockKeypoints(),
        previewFrame: previewFrame(),
      });

      expect(thumbnail).toBeInstanceOf(Blob);
      expect(mockSampleImageGrid).toHaveBeenCalledWith(expect.anything(), 640, 480, 4);
      expect(mockRenderBicubicGrid).toHaveBeenCalled();
      expect(mockDrawHumanLikeSkeleton).toHaveBeenCalledWith(
        expect.any(Object),
        expect.any(Array),
        expect.any(Number),
        expect.any(Number),
        expect.any(Number),
        expect.any(Number),
        expect.objectContaining({
          threshold: KEYPOINT_RENDER_MIN_CONFIDENCE,
          color: '#4dabf7',
          fillOpacity: 0.8,
          noseColor: '#ffa94d',
          earColor: '#ffa94d',
        }),
      );
      const call = mockDrawHumanLikeSkeleton.mock.calls[0];
      expect(call[1]).toHaveLength(17);
      expect(call[1][0]).toHaveProperty('opacity');
    });

    it('falls back to a skeleton on a dark background when no preview frame is available (privacy)', async () => {
      const canvas = createMockCanvasElement();
      const ctx = canvas.getContext('2d') as unknown as { fillRect: ReturnType<typeof vi.fn> };
      vi.spyOn(document, 'createElement').mockReturnValue(canvas as unknown as HTMLElement);

      await renderCaptureThumbnail({
        privacyMode: true,
        keypoints: createMockKeypoints(),
        previewFrame: null,
      });

      expect(ctx.fillRect).toHaveBeenCalled();
      expect(mockSampleImageGrid).not.toHaveBeenCalled();
      expect(mockDrawHumanLikeSkeleton).toHaveBeenCalled();
    });

    it('falls back to a skeleton on a dark background when no preview frame is available (non-privacy)', async () => {
      const canvas = createMockCanvasElement();
      const ctx = canvas.getContext('2d') as unknown as {
        fillRect: ReturnType<typeof vi.fn>;
        drawImage: ReturnType<typeof vi.fn>;
      };
      vi.spyOn(document, 'createElement').mockReturnValue(canvas as unknown as HTMLElement);

      await renderCaptureThumbnail({
        privacyMode: false,
        keypoints: createMockKeypoints(),
        previewFrame: null,
      });

      expect(ctx.fillRect).toHaveBeenCalled();
      expect(ctx.drawImage).not.toHaveBeenCalled();
      expect(mockDrawHumanLikeSkeleton).toHaveBeenCalled();
    });

    it('draws only the background when no keypoints are supplied in privacy mode', async () => {
      await renderCaptureThumbnail({
        privacyMode: true,
        keypoints: [],
        previewFrame: previewFrame(),
      });

      expect(mockRenderBicubicGrid).toHaveBeenCalled();
      expect(mockDrawHumanLikeSkeleton).not.toHaveBeenCalled();
    });

    it('respects custom quality and dimensions', async () => {
      const canvas = createMockCanvasElement();
      vi.spyOn(document, 'createElement').mockReturnValue(canvas as unknown as HTMLElement);

      await renderCaptureThumbnail({
        privacyMode: false,
        keypoints: createMockKeypoints(),
        previewFrame: previewFrame(),
        width: 320,
        height: 240,
        quality: 0.5,
      });

      expect(canvas.toBlob).toHaveBeenCalledWith(expect.any(Function), 'image/webp', 0.5);
      expect(canvas.width).toBe(320);
      expect(canvas.height).toBe(240);
    });

    it('throws when the canvas context is unavailable', async () => {
      const badCanvas = createMockCanvasElement();
      badCanvas.getContext = vi.fn(() => null) as unknown as HTMLCanvasElement['getContext'];
      vi.spyOn(document, 'createElement').mockReturnValue(badCanvas as unknown as HTMLElement);

      await expect(
        renderCaptureThumbnail({ privacyMode: false, keypoints: [], previewFrame: previewFrame() }),
      ).rejects.toThrow('Failed to get canvas context');
    });

    it('throws when blob generation fails', async () => {
      const canvas = createMockCanvasElement();
      canvas.toBlob = vi.fn((callback: (blob: Blob | null) => void) => callback(null));
      vi.spyOn(document, 'createElement').mockReturnValue(canvas as unknown as HTMLElement);

      await expect(
        renderCaptureThumbnail({ privacyMode: false, keypoints: [], previewFrame: previewFrame() }),
      ).rejects.toThrow('Failed to generate thumbnail blob');
    });
  });

  describe('estimateThumbnailSize', () => {
    it('should return blob size in bytes', () => {
      const blob = new Blob(['test data'], { type: 'image/webp' });

      const size = estimateThumbnailSize(blob);

      expect(size).toBe(blob.size);
      expect(typeof size).toBe('number');
    });

    it('should handle empty blob', () => {
      const blob = new Blob([], { type: 'image/webp' });

      expect(estimateThumbnailSize(blob)).toBe(0);
    });
  });

  describe('validateThumbnailSize', () => {
    it('should validate thumbnail within size limit', () => {
      const smallBlob = new Blob(['A'.repeat(100)], { type: 'image/webp' });

      expect(validateThumbnailSize(smallBlob, 10)).toBe(true);
    });

    it('should reject thumbnail exceeding size limit', () => {
      const largeBlob = new Blob(['A'.repeat(100000)], { type: 'image/webp' });

      expect(validateThumbnailSize(largeBlob, 10)).toBe(false);
    });

    it('should use default 10KB limit', () => {
      const blob = new Blob(['A'.repeat(100)], { type: 'image/webp' });

      expect(validateThumbnailSize(blob)).toBe(true);
    });
  });
});
