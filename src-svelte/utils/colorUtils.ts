export interface RGB {
  r: number;
  g: number;
  b: number;
}

function parseRGB(cssColor: string): RGB {
  const fallback = { r: 26, g: 27, b: 30 }; // #1a1b1e

  try {
    // Handle rgb(r, g, b) format
    if (cssColor.startsWith('rgb')) {
      const match = cssColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
      if (match) {
        return {
          r: parseInt(match[1], 10),
          g: parseInt(match[2], 10),
          b: parseInt(match[3], 10),
        };
      }
    }

    // Handle #rrggbb format
    if (cssColor.startsWith('#')) {
      const hex = cssColor.slice(1);
      if (hex.length === 6) {
        return {
          r: parseInt(hex.slice(0, 2), 16),
          g: parseInt(hex.slice(2, 4), 16),
          b: parseInt(hex.slice(4, 6), 16),
        };
      }
    }

    return fallback;
  } catch (error) {
    return fallback;
  }
}

/**
 * Format an RGB object as a CSS color string.
 *
 * @param color - RGB object
 * @returns CSS color string in rgb(r, g, b) format
 */
export function formatRGB(color: RGB): string {
  const r = Math.max(0, Math.min(255, Math.round(color.r)));
  const g = Math.max(0, Math.min(255, Math.round(color.g)));
  const b = Math.max(0, Math.min(255, Math.round(color.b)));
  return `rgb(${r}, ${g}, ${b})`;
}

function lerpRGB(color1: RGB, color2: RGB, alpha: number): RGB {
  const clampedAlpha = Math.max(0, Math.min(1, alpha));
  return {
    r: color1.r * (1 - clampedAlpha) + color2.r * clampedAlpha,
    g: color1.g * (1 - clampedAlpha) + color2.g * clampedAlpha,
    b: color1.b * (1 - clampedAlpha) + color2.b * clampedAlpha,
  };
}

function smoothColor(
  currentColor: RGB,
  newColor: RGB,
  alpha: number = 0.1
): RGB {
  return lerpRGB(currentColor, newColor, alpha);
}

function progressiveDownsample2x(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  targetSize: number
): HTMLCanvasElement | null {
  if (!sourceWidth || !sourceHeight) {
    return null;
  }

  try {
    // Start with source dimensions (use square aspect ratio)
    const size = Math.min(sourceWidth, sourceHeight);
    let currentWidth = size;
    let currentHeight = size;

    // Create initial canvas with the source frame
    let currentCanvas = document.createElement('canvas');
    currentCanvas.width = currentWidth;
    currentCanvas.height = currentHeight;

    let ctx = currentCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;

    // Draw the source frame (crop to square)
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium'; // Bilinear with 2×2 averaging
    ctx.drawImage(source, 0, 0, currentWidth, currentHeight);

    // Progressive 2× reductions until we reach target size
    while (currentWidth > targetSize * 2 && currentHeight > targetSize * 2) {
      // Calculate next size (half of current)
      const nextWidth = Math.floor(currentWidth / 2);
      const nextHeight = Math.floor(currentHeight / 2);

      // Create next canvas
      const nextCanvas = document.createElement('canvas');
      nextCanvas.width = nextWidth;
      nextCanvas.height = nextHeight;

      const nextCtx = nextCanvas.getContext('2d', { willReadFrequently: true });
      if (!nextCtx) break;

      // Downsample by 2× with bilinear (GPU 2×2 pixel averaging)
      nextCtx.imageSmoothingEnabled = true;
      nextCtx.imageSmoothingQuality = 'medium';
      nextCtx.drawImage(currentCanvas, 0, 0, nextWidth, nextHeight);

      // Move to next iteration
      currentCanvas = nextCanvas;
      currentWidth = nextWidth;
      currentHeight = nextHeight;
    }

    // Final resize to exact target size if needed
    if (currentWidth !== targetSize || currentHeight !== targetSize) {
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = targetSize;
      finalCanvas.height = targetSize;

      const finalCtx = finalCanvas.getContext('2d', { willReadFrequently: true });
      if (!finalCtx) return currentCanvas;

      finalCtx.imageSmoothingEnabled = true;
      finalCtx.imageSmoothingQuality = 'medium';
      finalCtx.drawImage(currentCanvas, 0, 0, targetSize, targetSize);

      return finalCanvas;
    }

    return currentCanvas;
  } catch (error) {
    return null;
  }
}

/**
 * Sample a decoded image frame at grid points to create a color grid.
 * Downscales the source to 64×64 and averages all pixels in each grid cell.
 *
 * Performance: Processes 4,096 pixels (64×64) instead of millions (e.g., 1920×1080)
 * Quality: Cell averaging provides more representative colors and reduces noise
 *
 * @param source - Decoded image source (ImageBitmap/canvas) to sample from
 * @param sourceWidth - Intrinsic width of the source in pixels
 * @param sourceHeight - Intrinsic height of the source in pixels
 * @param gridSize - Size of grid (3 for 3×3, 4 for 4×4)
 * @returns 2D array of RGB colors representing the grid
 */
export function sampleImageGrid(
  source: CanvasImageSource | null,
  sourceWidth: number,
  sourceHeight: number,
  gridSize: number = 3
): RGB[][] {
  // Fallback color grid if the source is unavailable
  const fallbackColor = parseRGB('#1a1b1e');
  const fallbackGrid: RGB[][] = [];
  for (let i = 0; i < gridSize; i++) {
    fallbackGrid[i] = [];
    for (let j = 0; j < gridSize; j++) {
      fallbackGrid[i][j] = { ...fallbackColor };
    }
  }

  if (!source || !sourceWidth || !sourceHeight) {
    return fallbackGrid;
  }

  try {
    // Use progressive downsampling to 64×64 for better quality
    const downscaleSize = 64;
    const downscaledCanvas = progressiveDownsample2x(source, sourceWidth, sourceHeight, downscaleSize);

    if (!downscaledCanvas) {
      return fallbackGrid;
    }

    const ctx = downscaledCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      return fallbackGrid;
    }

    // Get pixel data for downscaled image
    const imageData = ctx.getImageData(0, 0, downscaleSize, downscaleSize);
    const data = imageData.data;

    // Calculate cell dimensions in downscaled image
    const cellPixelSize = downscaleSize / gridSize; // e.g., 64/4 = 16 pixels per cell

    // Average all pixels in each grid cell
    const colors: RGB[][] = [];
    for (let row = 0; row < gridSize; row++) {
      colors[row] = [];
      for (let col = 0; col < gridSize; col++) {
        // Pixel bounds for this cell
        const x0 = Math.floor(col * cellPixelSize);
        const y0 = Math.floor(row * cellPixelSize);
        const x1 = Math.floor((col + 1) * cellPixelSize);
        const y1 = Math.floor((row + 1) * cellPixelSize);

        // Average all pixels in cell
        let r = 0,
          g = 0,
          b = 0,
          count = 0;

        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const idx = (y * downscaleSize + x) * 4;
            r += data[idx];
            g += data[idx + 1];
            b += data[idx + 2];
            count++;
          }
        }

        colors[row][col] = {
          r: Math.round(r / count),
          g: Math.round(g / count),
          b: Math.round(b / count),
        };
      }
    }

    return colors;
  } catch (error) {
    // Return fallback on any error
    return fallbackGrid;
  }
}

/**
 * Apply exponential moving average to smooth grid color transitions.
 * Each grid cell is smoothed independently.
 *
 * @param currentGrid - Current smoothed color grid
 * @param targetGrid - New target color grid
 * @param alpha - Smoothing factor (higher = faster convergence, typical: 0.1)
 * @returns Smoothed color grid
 */
export function smoothGridColors(
  currentGrid: RGB[][],
  targetGrid: RGB[][],
  alpha: number = 0.1
): RGB[][] {
  if (!currentGrid || !targetGrid || currentGrid.length === 0 || targetGrid.length === 0) {
    return targetGrid;
  }

  const gridSize = Math.min(currentGrid.length, targetGrid.length);
  const smoothed: RGB[][] = [];

  for (let row = 0; row < gridSize; row++) {
    smoothed[row] = [];
    const rowSize = Math.min(
      currentGrid[row]?.length || 0,
      targetGrid[row]?.length || 0
    );

    for (let col = 0; col < rowSize; col++) {
      smoothed[row][col] = smoothColor(
        currentGrid[row][col],
        targetGrid[row][col],
        alpha
      );
    }
  }

  return smoothed;
}
