# Task 2025-11-10: Refactor Temporal Smoothing to Simple Moving Average

**STATUS:** COMPLETED

## User Request
Rework how temporal smoothing works. Make it evenly average N last frames (N 1-10 in UI). Also ensure that CLAHE, blur, smoothing are disabled when on minimum settings to save processing power.

## Critical Discoveries

**1. Temporal smoothing MUST run at camera FPS, not detection FPS**
Initial implementation ran SMA in main capture loop (detection FPS: ~2fps at 0.5s interval). This meant averaging only 1-3 frames captured 0.5s apart, not 30fps frames. Fix: Restore RAF (requestAnimationFrame) loop running at camera FPS (~30fps) to continuously feed frames to circular buffer.

**Correct architecture:**
```
RAF Loop (30fps): raw frame → circular buffer → compute SMA → store in smoothedFrameBuffer
Main Loop (detection FPS): read smoothedFrameBuffer → blur → CLAHE → worker
```

**2. Split preprocessing responsibilities**
`ImagePreprocessor` now has two distinct modes:
- `addFrameToBuffer()`: Called at camera FPS (RAF loop), maintains circular buffer and computes running SMA
- `preprocess()`: Called at detection FPS, applies only blur/CLAHE (smoothing already done)

**3. No migration code**
Migration code was removed for simplicity. Users with old `emaStrength` settings will get default value (1).

## Implementation Details

**Settings Schema Change:**
- `emaStrength: number (0-0.9)` → `smoothingFrames: number (1-10)`
- Default: `0.8` → `1` (no smoothing by default)
- No migration code (users get default if old setting exists)

**ImagePreprocessor Refactor:**
- Added `smoothedFrameBuffer: ImageData | null` - stores latest SMA result
- Added `addFrameToBuffer(frame, smoothingFrames)` - updates circular buffer and computes SMA
- Added `getSmoothedFrame()` - returns pre-computed smoothed frame
- Removed `smoothingFrames` param from `preprocess()` - now only handles blur/CLAHE
- Circular buffer resets on: dimension change, N change, video element change

**useFrameProcessor RAF Loop:**
- Runs continuously when `smoothingFrames > 1` and `enabled`
- Captures raw frame at camera FPS
- Calls `globalPreprocessor.addFrameToBuffer()`
- Disabled when `smoothingFrames === 1` (optimization)

**Main Capture Loop:**
- Reads from `globalPreprocessor.getSmoothedFrame()` when smoothing enabled
- Falls back to raw frame capture when smoothing disabled
- Applies blur/CLAHE via `preprocess()` (no smoothing param)
- Sends to worker at detection interval

**Processing Optimizations:**
- Skip temporal smoothing when `smoothingFrames === 1`
- Skip blur when `gaussianBlurKernel === 0`
- Skip CLAHE when `claheStrength === 0`

## Files Modified
- `src/hooks/useCameraSettings.ts` - schema, migration, validation
- `src/utils/imagePreprocessing.ts` - SMA algorithm, circular buffer, split methods
- `src/hooks/useFrameProcessor.ts` - RAF loop restoration, pipeline split
- `src/components/PostureCamera.tsx` - prop rename
- `src/components/unified/SettingsTab.tsx` - UI slider (1-10 with marks)
- `src/pages/PostureTrackerApp.tsx` - prop passthrough

## Related Tasks
- `tasks/2025-11-06-feature-preprocessing-sliders.md` - Original EMA implementation with buffer detachment bug
- `tasks/0010-fix-camera-detection-sync.md` - Frame snapshot synchronization pattern
