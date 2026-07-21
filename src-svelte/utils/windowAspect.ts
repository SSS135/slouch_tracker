/**
 * Pure geometry helper for matching the window content area to the camera
 * aspect ratio. Free of Tauri/DOM dependencies so the math stays unit-testable.
 */

export interface WindowSize {
  width: number;
  height: number;
}

/**
 * Compute the logical window size that matches the camera aspect ratio.
 *
 * Strategy: keep the current width and derive the height from the camera
 * aspect. When that height would fall below `minHeight`, grow the width instead
 * so the aspect ratio is preserved without letterbox/pillarbox bars.
 *
 * Returns `null` when the current size already matches within `tolerancePx`.
 * That no-op result is also the primary guard against a
 * setSize -> onResized -> setSize feedback loop: once the window matches the
 * aspect, every subsequent resize snap resolves to `null`.
 */
export function computeAspectSnap(
  current: WindowSize,
  cameraWidth: number,
  cameraHeight: number,
  minWidth: number,
  minHeight: number,
  tolerancePx = 1,
): WindowSize | null {
  if (
    !Number.isFinite(cameraWidth) ||
    !Number.isFinite(cameraHeight) ||
    cameraWidth <= 0 ||
    cameraHeight <= 0 ||
    !Number.isFinite(current.width) ||
    !Number.isFinite(current.height) ||
    current.width <= 0 ||
    current.height <= 0
  ) {
    return null;
  }

  let width = current.width;
  let height = Math.round((width * cameraHeight) / cameraWidth);

  if (height < minHeight) {
    // Height is clamped at the minimum, so widen to keep the aspect ratio.
    height = minHeight;
    width = Math.round((minHeight * cameraWidth) / cameraHeight);
  }

  if (width < minWidth) {
    width = minWidth;
    height = Math.round((minWidth * cameraHeight) / cameraWidth);
    // Extreme aspect ratios cannot fit the min box exactly; satisfy both mins.
    if (height < minHeight) height = minHeight;
  }

  if (
    Math.abs(width - current.width) <= tolerancePx &&
    Math.abs(height - current.height) <= tolerancePx
  ) {
    return null;
  }

  return { width, height };
}
