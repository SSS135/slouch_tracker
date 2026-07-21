/**
 * Bicubic grid rendering utilities for privacy mode backgrounds.
 * Renders low-resolution color grids (e.g., 4×4) to full canvas size
 * using high-quality bicubic interpolation via GPU-accelerated canvas scaling.
 */

import { RGB, smoothGridColors } from './colorUtils';

/**
 * Render a color grid to full canvas size using bicubic interpolation.
 * Creates a small intermediate canvas with the grid colors, then uses
 * GPU-accelerated bicubic upsampling to scale to target size.
 *
 * Canvas imageSmoothingQuality='high' uses bicubic-like interpolation:
 * - Chrome/Edge: Lanczos3 (similar quality to bicubic)
 * - Firefox: Bicubic
 * - Safari: Bicubic
 *
 * @param ctx - Canvas 2D context to render to
 * @param canvas - Target canvas element
 * @param colorGrid - 2D array of RGB colors (e.g., 4×4 grid)
 */
export function renderBicubicGrid(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  colorGrid: RGB[][]
): void {
  if (!colorGrid || colorGrid.length === 0 || colorGrid[0].length === 0) {
    return;
  }

  const gridHeight = colorGrid.length;
  const gridWidth = colorGrid[0].length;

  // Create small canvas with grid colors (1 pixel per color)
  const gridCanvas = document.createElement('canvas');
  gridCanvas.width = gridWidth;
  gridCanvas.height = gridHeight;

  const gridCtx = gridCanvas.getContext('2d', { willReadFrequently: true });
  if (!gridCtx) return;

  // Fill grid canvas with colors (1 pixel per grid cell)
  for (let row = 0; row < gridHeight; row++) {
    for (let col = 0; col < gridWidth; col++) {
      const color = colorGrid[row][col];
      gridCtx.fillStyle = `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
      gridCtx.fillRect(col, row, 1, 1);
    }
  }

  // Upscale to full canvas size using bicubic interpolation
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high'; // Bicubic-like interpolation
  ctx.drawImage(gridCanvas, 0, 0, canvas.width, canvas.height);
}

/**
 * Render a color grid with temporal smoothing (exponential moving average).
 * Smooths transitions between current and target grids to prevent jarring changes.
 *
 * @param ctx - Canvas 2D context to render to
 * @param canvas - Target canvas element
 * @param currentGrid - Current smoothed color grid (state), or null for initialization
 * @param targetGrid - New target color grid
 * @param alpha - Smoothing factor (0.1 = 10% new, 90% old)
 * @returns Updated smoothed color grid (save this for next frame)
 */
export function renderSmoothedBicubicGrid(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  currentGrid: RGB[][] | null,
  targetGrid: RGB[][],
  alpha: number = 0.1
): RGB[][] {
  // Initialize current grid on first frame
  if (!currentGrid) {
    renderBicubicGrid(ctx, canvas, targetGrid);
    return targetGrid;
  }

  // Apply temporal smoothing (exponential moving average)
  const smoothedGrid = smoothGridColors(currentGrid, targetGrid, alpha);

  // Render smoothed grid
  renderBicubicGrid(ctx, canvas, smoothedGrid);

  return smoothedGrid;
}
