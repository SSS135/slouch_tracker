# Task 2025-11-06: Add Preprocessing Sliders (CLAHE, Gaussian Blur, EMA)

**STATUS:** COMPLETED

## User Request

Add CLAHE preprocessing strength slider from 0 to whatever, remove checkbox. Also add some noise reduction with strength slider from 0 to whatever. Also add averaging between frames, 0-0.9 strength, using exponential moving average. frame = avg_frame * c + new_frame * (1 - c). But no marks at sliders except start and end. Reuse existing sliders used in app. Remove intermediate mark support, leave only start and end.

**Clarification**: Temporal smoothing should run at detection FPS, but FIRST in the pipeline (right after capturing camera frame, before blur/CLAHE).

## Critical Discoveries

**1. EMA Buffer Detachment Bug**
Returning `avgFrameBuffer` directly causes buffer detachment when transferred to worker. ImageData buffer becomes detached (length 0) on next frame, resulting in black screen. Solution: Always return fresh ImageData copy, update avgFrameBuffer in-place with `.set()`.

**2. Preprocessing Order Matters**
User wants EMA first (right after camera capture), then blur, then CLAHE. All at detection FPS. Original implementation applied EMA last, which was incorrect.

**3. Gaussian Blur Kernel Validation**
OpenCV requires odd kernel sizes. Validation in settings hook rounds even numbers down to nearest odd (4→3, 6→5). Kernel size 0 or 1 means disabled.

**4. Settings Migration Pattern**
`usePersistedState` merges stored settings with defaults using object spread. Migration runs before validation. Old `claheEnabled: boolean` converted to `claheStrength: number` (true→2.0, false→0).

## Implementation Details

**Processing Order**: **EMA → Blur → CLAHE** (all at detection FPS)

**Settings (`useCameraSettings.ts`):**
- Replaced `claheEnabled: boolean` with `claheStrength: number` (0-10, default 0)
- Added `gaussianBlurKernel: number` (0-15, default 0, odd only)
- Added `emaStrength: number` (0-0.9, default 0.5)
- Validation: clamps ranges, rounds claheStrength to 1 decimal, ensures odd kernel
- Migration: converts old `claheEnabled` to new `claheStrength`

**Preprocessing (`imagePreprocessing.ts`):**
- Extracted `applyEMA()` as public method (no OpenCV required, pure JS)
- `preprocess()` handles Blur + CLAHE only (removed EMA parameter)
- EMA buffer persists across frames, reset when video element changes
- **Bug fix**: Return fresh ImageData copies to prevent buffer detachment

**Detection Pipeline (`useFrameProcessor.ts`):**
- Capture raw video frame from camera
- Apply EMA first if `emaStrength > 0`
- Then apply blur/CLAHE if enabled
- Processing order: Capture → EMA → Blur → CLAHE → Worker
- All preprocessing happens at detection FPS (configured interval, e.g., 0.5s)

**UI (`SettingsTab.tsx`):**
- Removed CLAHE checkbox
- Added 3 sliders using custom Slider component:
  - **CLAHE Strength**: 0-10, step 0.1, "Off" when 0
  - **Gaussian Blur**: 0-15, step 2 (ensures odd), "Off" when 0
  - **Temporal Smoothing**: 0-0.9, step 0.05, "Off" when 0
- Grouped under "Image Preprocessing" section
- Custom Slider component auto-shows min/max labels only (no intermediate marks)

**Component Integration (`PostureCamera.tsx`):**
- Pass `claheStrength`, `gaussianBlurKernel`, `emaStrength` to `useFrameProcessor`
- All preprocessing handled at detection FPS in frame processor
- Display shows preprocessed frozen frame from detection pipeline

## Related

- `tasks/2025-11-06-feature-opencv-clahe-preprocessing.md` - Original CLAHE implementation
