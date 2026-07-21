/**
 * Thumbnail Generator
 *
 * Generates WebP thumbnails for dataset storage from the native camera preview
 * frame or, in privacy mode, from the detected skeleton keypoints. The native
 * Rust camera owns capture, so the frontend no longer has a <video> element to
 * snapshot; it works from the streamed preview frame (an ImageBitmap/canvas) and
 * the keypoints pushed with each inference result.
 */

import { THUMBNAIL_RESOLUTION, KEYPOINT_RENDER_MIN_CONFIDENCE } from '../ml/constants';
import { sampleImageGrid } from '@/utils/colorUtils';
import { renderBicubicGrid } from '@/utils/bicubicGridRenderer';
import { drawHumanLikeSkeleton } from '@/utils/canvasDrawing';

interface Keypoint { x: number; y: number; score: number; }

/** A decoded preview frame plus its intrinsic dimensions. */
export interface PreviewFrameSource {
  image: CanvasImageSource;
  width: number;
  height: number;
}

export interface CaptureThumbnailOptions {
  /** When true, render a skeleton visualization instead of the real frame. */
  privacyMode?: boolean;
  /** Keypoints in 0-1 normalized coordinates (skeleton source + privacy overlay). */
  keypoints?: Keypoint[];
  /** Latest decoded preview frame; used for the real-image thumbnail and the privacy blur. */
  previewFrame?: PreviewFrameSource | null;
  width?: number;
  height?: number;
  quality?: number;
}

function encodeWebp(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Failed to generate thumbnail blob'))),
      'image/webp',
      quality,
    );
  });
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  keypoints: Keypoint[],
): void {
  if (keypoints.length === 0) return;
  const smoothed = keypoints.map((kp) => ({
    ...kp,
    opacity: kp.score > KEYPOINT_RENDER_MIN_CONFIDENCE ? 1.0 : 0.0,
  }));
  drawHumanLikeSkeleton(ctx, smoothed, canvas.width, canvas.height, canvas.width, canvas.height, {
    threshold: KEYPOINT_RENDER_MIN_CONFIDENCE,
    color: '#4dabf7',
    fillOpacity: 0.8,
    noseColor: '#ffa94d',
    earColor: '#ffa94d',
  });
}

/** Cover-fit the preview frame onto the thumbnail canvas, centering the crop. */
function drawCoverFit(
  ctx: CanvasRenderingContext2D,
  frame: PreviewFrameSource,
  width: number,
  height: number,
): void {
  const sourceAspect = frame.width / frame.height;
  const targetAspect = width / height;
  let sourceWidth = frame.width;
  let sourceHeight = frame.height;
  let sourceX = 0;
  let sourceY = 0;
  if (sourceAspect > targetAspect) {
    sourceWidth = frame.height * targetAspect;
    sourceX = (frame.width - sourceWidth) / 2;
  } else if (sourceAspect < targetAspect) {
    sourceHeight = frame.width / targetAspect;
    sourceY = (frame.height - sourceHeight) / 2;
  }
  ctx.drawImage(frame.image, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, width, height);
}

/**
 * Render a WebP thumbnail Blob for a capture.
 *
 * Non-privacy: the streamed preview frame, cover-fit to the thumbnail box.
 * Privacy: the skeleton drawn over a blurred colour grid sampled from the
 * preview frame (or a dark background when no preview frame is available, e.g. a
 * background/global-hotkey capture while the preview pipeline is paused).
 *
 * SECURITY: Returns a Blob (not a string) to avoid XSS via malicious URLs.
 */
export async function renderCaptureThumbnail(options: CaptureThumbnailOptions): Promise<Blob> {
  const width = options.width ?? THUMBNAIL_RESOLUTION.width;
  const height = options.height ?? THUMBNAIL_RESOLUTION.height;
  const quality = options.quality ?? 0.8;
  const privacyMode = options.privacyMode ?? false;
  const keypoints = options.keypoints ?? [];
  const previewFrame = options.previewFrame ?? null;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }

  if (privacyMode) {
    if (previewFrame) {
      const colorGrid = sampleImageGrid(previewFrame.image, previewFrame.width, previewFrame.height, 4);
      renderBicubicGrid(ctx, canvas, colorGrid);
    } else {
      ctx.fillStyle = '#10181f';
      ctx.fillRect(0, 0, width, height);
    }
    drawSkeleton(ctx, canvas, keypoints);
  } else if (previewFrame) {
    drawCoverFit(ctx, previewFrame, width, height);
  } else {
    // No preview frame (background capture, preview paused): fall back to the
    // skeleton on a dark background so the thumbnail still conveys the pose.
    ctx.fillStyle = '#10181f';
    ctx.fillRect(0, 0, width, height);
    drawSkeleton(ctx, canvas, keypoints);
  }

  return encodeWebp(canvas, quality);
}

/**
 * Estimate thumbnail size in bytes from Blob
 *
 * @param blob - Thumbnail blob
 * @returns Size in bytes
 */
export function estimateThumbnailSize(blob: Blob): number {
  return blob.size;
}

/**
 * Validate thumbnail meets size requirements
 *
 * @param blob - Thumbnail blob
 * @param maxSizeKB - Maximum size in KB (default: 10KB)
 * @returns true if thumbnail is within size limit
 */
export function validateThumbnailSize(
  blob: Blob,
  maxSizeKB: number = 10
): boolean {
  const sizeKB = blob.size / 1024;
  return sizeKB <= maxSizeKB;
}
