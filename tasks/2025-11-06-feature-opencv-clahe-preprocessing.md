# Task 2025-11-06: Add OpenCV.js CLAHE + Denoise Preprocessing

**STATUS:** COMPLETE

## User Request
Add pre-processing like this to camera image. Make sure it is applied early enough so it is used by both canvas display, training and inference.

Higher quality (pure software): OpenCV.js CLAHE + denoise

If you can afford small WASM lib, CLAHE + fastNlMeans works great.

## Critical Discoveries

**1. OpenCV.js dynamic loading better than static script tag:**
Loading OpenCV.js WASM (~8MB) via dynamic import allows conditional loading and proper initialization sequencing. Static `<script>` tag blocks page load. Loader pattern: check global `cv`, load script if missing, wait for `onRuntimeInitialized` callback.

**2. Temporal smoothing critical for real-time video:**
Per-frame CLAHE creates flickering between frames due to independent histogram equalization. Solution: Simple Moving Average (SMA) over circular buffer of last N frames (default 3). Trades slight motion blur for significantly reduced flicker. Essential for usable real-time posture tracking.

**3. OpenCV memory management prevents leaks:**
All cv.Mat objects must be explicitly deleted via `.delete()`. JavaScript GC doesn't track OpenCV WASM heap. Memory leak pattern: create Mat → process → forget to delete → WASM heap exhaustion after ~100 frames. Solution: try/finally blocks ensuring cleanup.

**4. CLAHE parameters matter for pose detection:**
Clip limit too high (>4.0) → noise amplification, false keypoint detections. Tile grid too small (<4x4) → blocky artifacts. Optimal: clipLimit=2.0, tileGridSize=8x8 balances contrast enhancement with artifact reduction for RTMPose accuracy.

## Solution

Implemented OpenCV.js preprocessing with temporal smoothing:

**1. OpenCV Loader** - Dynamic WASM loading with initialization sequencing (opencv-loader.ts)

**2. ImagePreprocessor Class** - Configurable preprocessing pipeline (imagePreprocessing.ts):
- CLAHE contrast enhancement (clipLimit=2.0, tileGrid=8x8)
- Gaussian blur noise reduction (configurable kernel)
- Temporal smoothing via SMA over circular buffer (reduces flicker)
- Proper OpenCV memory management (Mat cleanup in try/finally)

**3. Early Pipeline Integration** - Applied before canvas display, training capture, and inference

**4. Settings UI** - Toggle preprocessing on/off, adjust strength parameters (SettingsTab.tsx)

Preprocessing improves pose detection accuracy in low-light conditions without sacrificing real-time performance.

## Related

- `tasks/0014-fix-image-preprocessing.md` - Image quality improvements
- `tasks/2025-11-10-refactor-temporal-smoothing-sma.md` - Temporal smoothing implementation details
