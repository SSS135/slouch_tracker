/**
 * Canvas drawing utilities for pose visualization - NLF-L
 * Pure functions with no dependencies on component state
 *
 * NLF-L supplies 17 COCO body keypoints only (no face/hand keypoints).
 */

import {
  NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
  LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
  LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
  LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
} from '@/services/posture/keypointIndices';

const KEYPOINT_DRAW_THRESHOLD = 0.3;

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface SmoothedKeypoint extends Keypoint {
  opacity: number;
}

function isAtImageBoundary(
  kpt: Keypoint,
  threshold: number = 0.01
): boolean {
  return (
    kpt.x <= threshold ||
    kpt.y <= threshold ||
    kpt.x >= 1 - threshold ||
    kpt.y >= 1 - threshold
  );
}

/**
 * Draw filled capsule (rounded rectangle) between two points
 * Used for drawing limbs (arms, legs)
 */
function drawCapsule(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  width: number,
  fillColor: string,
  strokeColor?: string,
  opacity: number = 1.0,
  baseScale: number = 1.0
): void {
  if (opacity < 0.01) return; // Skip if fully transparent

  const dx = x2 - x1;
  const dy = y2 - y1;
  const angle = Math.atan2(dy, dx);
  const length = Math.sqrt(dx * dx + dy * dy);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.translate(x1, y1);
  ctx.rotate(angle);

  const halfWidth = width / 2;
  ctx.fillStyle = fillColor;
  ctx.beginPath();
  ctx.arc(0, 0, halfWidth, Math.PI / 2, (3 * Math.PI) / 2);
  ctx.arc(length, 0, halfWidth, -Math.PI / 2, Math.PI / 2);
  ctx.closePath();
  ctx.fill();

  if (strokeColor) {
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2 * baseScale;
    ctx.stroke();
  }

  ctx.restore();
}

/**
 * Draw head with facial features
 * Keypoints are expected in 0-1 normalized coordinates
 */
function drawFace(
  ctx: CanvasRenderingContext2D,
  keypoints: SmoothedKeypoint[],
  canvasWidth: number,
  canvasHeight: number,
  threshold: number,
  color: string,
  noseColor?: string,
  earColor?: string,
  baseScale: number = 1.0
): void {
  const nose = keypoints[NOSE];
  const leftEye = keypoints[LEFT_EYE];
  const rightEye = keypoints[RIGHT_EYE];
  const leftEar = keypoints[LEFT_EAR];
  const rightEar = keypoints[RIGHT_EAR];

  const validFacePoints = [nose, leftEye, rightEye, leftEar, rightEar].filter(
    (kp) => kp && kp.opacity > 0.01
  );

  if (validFacePoints.length < 2) return;

  const centerX =
    validFacePoints.reduce((sum, kp) => sum + kp.x, 0) / validFacePoints.length * canvasWidth;
  const centerY =
    validFacePoints.reduce((sum, kp) => sum + kp.y, 0) / validFacePoints.length * canvasHeight;

  const avgOpacity = validFacePoints.reduce((sum, kp) => sum + kp.opacity, 0) / validFacePoints.length;

  let radius = 25 * baseScale; // Default radius (scaled)
  if (leftEar && rightEar && leftEar.opacity > 0.01 && rightEar.opacity > 0.01) {
    const earDistance = Math.abs(leftEar.x - rightEar.x) * canvasWidth;
    radius = earDistance * 0.65; // Head circle slightly larger than ear span
  }

  ctx.save();
  ctx.globalAlpha = avgOpacity;
  ctx.fillStyle = color.replace(')', ', 0.3)').replace('rgb', 'rgba');
  ctx.strokeStyle = 'white';
  ctx.lineWidth = 2 * baseScale;
  ctx.beginPath();
  ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  if (leftEye && leftEye.opacity > 0.01) {
    ctx.save();
    ctx.globalAlpha = leftEye.opacity;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2 * baseScale;
    ctx.beginPath();
    ctx.arc(leftEye.x * canvasWidth, leftEye.y * canvasHeight, radius * 0.14, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  if (rightEye && rightEye.opacity > 0.01) {
    ctx.save();
    ctx.globalAlpha = rightEye.opacity;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2 * baseScale;
    ctx.beginPath();
    ctx.arc(rightEye.x * canvasWidth, rightEye.y * canvasHeight, radius * 0.14, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  if (nose && nose.opacity > 0.01) {
    const noseFillColor = noseColor || '#ffa94d';
    ctx.save();
    ctx.globalAlpha = nose.opacity;
    ctx.fillStyle = noseFillColor.replace(')', ', 0.85)').replace('rgb', 'rgba');
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2 * baseScale;
    ctx.beginPath();
    ctx.arc(nose.x * canvasWidth, nose.y * canvasHeight, radius * 0.09, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  if (leftEar && leftEar.opacity > 0.01) {
    const earFillColor = earColor || '#ffa94d';
    // Position ears 10% further apart (5% each direction from center)
    const leftEarX = rightEar
      ? (leftEar.x * 1.05 - rightEar.x * 0.05) * canvasWidth
      : leftEar.x * canvasWidth;
    ctx.save();
    ctx.globalAlpha = leftEar.opacity;
    ctx.fillStyle = earFillColor.replace(')', ', 0.85)').replace('rgb', 'rgba');
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2 * baseScale;
    ctx.beginPath();
    ctx.arc(leftEarX, leftEar.y * canvasHeight, radius * 0.11, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  if (rightEar && rightEar.opacity > 0.01) {
    const earFillColor = earColor || '#ffa94d';
    const rightEarX = leftEar
      ? (rightEar.x * 1.05 - leftEar.x * 0.05) * canvasWidth
      : rightEar.x * canvasWidth;
    ctx.save();
    ctx.globalAlpha = rightEar.opacity;
    ctx.fillStyle = earFillColor.replace(')', ', 0.85)').replace('rgb', 'rgba');
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2 * baseScale;
    ctx.beginPath();
    ctx.arc(rightEarX, rightEar.y * canvasHeight, radius * 0.11, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

/**
 * Draw human-like skeleton with filled shapes for privacy mode
 * Shows recognizable human figure with body, head, limbs, hands, and facial features
 * Supports smooth opacity fade for keypoints based on confidence
 * Keypoints are expected in 0-1 normalized coordinates
 *
 * @param ctx - Canvas 2D rendering context
 * @param keypoints - Smoothed body keypoints with opacity (COCO format, 17 keypoints, 0-1 normalized)
 * @param canvasWidth - Canvas width in pixels
 * @param canvasHeight - Canvas height in pixels
 * @param videoWidth - Video width in pixels (for coordinate mapping)
 * @param videoHeight - Video height in pixels (for coordinate mapping)
 * @param options - Drawing options
 */
export function drawHumanLikeSkeleton(
  ctx: CanvasRenderingContext2D,
  keypoints: SmoothedKeypoint[],
  canvasWidth: number,
  canvasHeight: number,
  videoWidth: number,
  videoHeight: number,
  options?: {
    threshold?: number;
    color?: string;
    fillOpacity?: number;
    noseColor?: string;
    earColor?: string;
  }
): void {
  const threshold = options?.threshold ?? KEYPOINT_DRAW_THRESHOLD;
  const color = options?.color ?? '#4dabf7';
  const noseColor = options?.noseColor;
  const earColor = options?.earColor;
  const fillOpacity = options?.fillOpacity ?? 0.8;

  const baseScale = canvasWidth / 640; // 640px reference width for resolution-independent sizing

  const leftShoulder = keypoints[LEFT_SHOULDER];
  const rightShoulder = keypoints[RIGHT_SHOULDER];
  const leftHip = keypoints[LEFT_HIP];
  const rightHip = keypoints[RIGHT_HIP];

  let shoulderWidth = 50;
  if (
    leftShoulder &&
    rightShoulder &&
    leftShoulder.opacity > 0.01 &&
    rightShoulder.opacity > 0.01
  ) {
    shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x) * canvasWidth;
  }
  const limbWidth = shoulderWidth * 0.25;

  const isValid = (kp: SmoothedKeypoint | undefined): boolean =>
    kp !== undefined && kp.opacity > 0.01;

  const fillColor = color.replace(')', `, ${fillOpacity})`).replace('rgb', 'rgba');
  const strokeColor = 'white';

  // A torso whose four corners are all clamped to the image boundary is a
  // degenerate detection (keypoints pinned to y≈0 / x≈0 when a person is only
  // partially in frame). Filling that quad paints a full-width bar across the
  // frame edge, so skip it — mirroring the per-limb boundary guard below. A
  // legitimate torso keeps at least one corner in the interior.
  const torsoDegenerate =
    isValid(leftShoulder) &&
    isValid(rightShoulder) &&
    isValid(leftHip) &&
    isValid(rightHip) &&
    isAtImageBoundary(leftShoulder) &&
    isAtImageBoundary(rightShoulder) &&
    isAtImageBoundary(leftHip) &&
    isAtImageBoundary(rightHip);

  if (
    !torsoDegenerate &&
    isValid(leftShoulder) &&
    isValid(rightShoulder) &&
    isValid(leftHip) &&
    isValid(rightHip)
  ) {
    const torsoOpacity = (leftShoulder.opacity + rightShoulder.opacity + leftHip.opacity + rightHip.opacity) / 4;

    ctx.save();
    ctx.globalAlpha = torsoOpacity;
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2 * baseScale;
    ctx.beginPath();
    ctx.moveTo(leftShoulder.x * canvasWidth, leftShoulder.y * canvasHeight);
    ctx.lineTo(rightShoulder.x * canvasWidth, rightShoulder.y * canvasHeight);
    ctx.lineTo(rightHip.x * canvasWidth, rightHip.y * canvasHeight);
    ctx.lineTo(leftHip.x * canvasWidth, leftHip.y * canvasHeight);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  const limbs: Array<[number, number]> = [
    [LEFT_SHOULDER, LEFT_ELBOW],
    [LEFT_ELBOW, LEFT_WRIST],
    [RIGHT_SHOULDER, RIGHT_ELBOW],
    [RIGHT_ELBOW, RIGHT_WRIST],
    [LEFT_HIP, LEFT_KNEE],
    [LEFT_KNEE, LEFT_ANKLE],
    [RIGHT_HIP, RIGHT_KNEE],
    [RIGHT_KNEE, RIGHT_ANKLE],
  ];

  limbs.forEach(([idx1, idx2]) => {
    const kp1 = keypoints[idx1];
    const kp2 = keypoints[idx2];
    if (isValid(kp1) && isValid(kp2)) {
      // Skip if both endpoints at image boundary (clamped invalid keypoints)
      if (isAtImageBoundary(kp1) && isAtImageBoundary(kp2)) {
        return;
      }

      const limbOpacity = (kp1.opacity + kp2.opacity) / 2;

      drawCapsule(
        ctx,
        kp1.x * canvasWidth,
        kp1.y * canvasHeight,
        kp2.x * canvasWidth,
        kp2.y * canvasHeight,
        limbWidth,
        fillColor,
        strokeColor,
        limbOpacity,
        baseScale
      );
    }
  });

  [LEFT_WRIST, RIGHT_WRIST].forEach((idx) => {
    const wrist = keypoints[idx];
    if (isValid(wrist)) {
      ctx.save();
      ctx.globalAlpha = wrist.opacity;
      ctx.fillStyle = fillColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2 * baseScale;
      ctx.beginPath();
      ctx.arc(wrist.x * canvasWidth, wrist.y * canvasHeight, limbWidth * 0.8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  });

  drawFace(ctx, keypoints, canvasWidth, canvasHeight, threshold, color, noseColor, earColor, baseScale);
}
