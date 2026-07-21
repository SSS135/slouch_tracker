# Task 2025-11-03: Add Privacy Mode Feature
**STATUS:** COMPLETED

## User Request
add a privacy mode, on by default, changed in settings. in privacty mode fill the screen with neutral color and draw just skeleton, but make it look good, not ugly, without numbers and with complimentary colors.

**Enhanced Requirements:**
- Privacy mode for ALL images (live view, thumbnails, saved frames)
- No real camera feed saved anywhere (ML models process in-memory only)
- Human-like skeleton with body, hands, head, eyes, nose, ears
- Smooth animations with fade in/out
- Consistent appearance across all resolutions
- Eyes too large, nose/ears need different colors
- Skeleton should fade smoothly when person leaves frame
- Background should use average color of original frame instead of hardcoded dark color

## Critical Discoveries (Non-Obvious)

**1. SmoothedKeypoint type mismatch in thumbnails:**
`drawHumanLikeSkeleton()` expects `SmoothedKeypoint[]` (with `opacity` field), but `generateThumbnail()` passed plain `Keypoint[]`. Without opacity, `kp.opacity > 0.01` check returned undefined → nothing drawn (black thumbnails). Fix: Map keypoints to add `opacity: 1.0` before drawing.

**2. Score checks prevented smooth fading:**
Drawing functions checked BOTH `score > threshold` AND `opacity > 0.01`. When confidence dropped, parts disappeared instantly despite opacity being interpolated. Fix: Remove all `score > threshold` checks from drawing, let opacity-only checks control visibility.

**3. Empty target array caused instant disappearance:**
`interpolateKeypoints()` used `newKeypoints.map(...)` which returned `[]` when no person detected. Fix: Add special case to fade out existing keypoints: `prevSmoothed.map(kp => ({ ...kp, opacity: kp.opacity * (1 - alpha) }))`.

**4. Exponential smoothing felt wrong:**
`1 - exp(-deltaTime / smoothTime)` caused fast-then-slow motion (asymptotic). User expected constant speed. Fix: Linear interpolation `Math.min(1, deltaTime / smoothTime)`.

**5. Resolution-dependent rendering:**
Hardcoded pixel values (14px eyes, 9px nose, 2px strokes) looked different on 640px thumbnails vs 1920px camera canvas. Eyes appeared 3× larger on thumbnails. Fix: Introduce `baseScale = canvasWidth / 640` to scale all absolute sizes.

**6. Facial features too large:**
Even with baseScale, features overwhelmed the face. Fix: Make features proportional to head radius instead of baseScale. Then reduce by 4× (eyes: 0.56→0.14, nose: 0.36→0.09, ears: 0.44→0.11 of head radius).

**7. Average frame color for adaptive background:**
Hardcoded dark background (#1a1b1e) felt disconnected from video lighting. Better UX: calculate average RGB from downsampled video frame (10×10 pixels), apply exponential moving average for smooth transitions. Background now adapts to ambient lighting while maintaining privacy.

**8. Single color too uniform for varied scenes:**
Single average color lost spatial lighting variation (e.g., bright window left, dark wall right). Upgraded to 4×4 color grid with bicubic interpolation: progressive 2× downsampling to 64×64, sample 16 cell averages, render with GPU-accelerated bicubic upsampling. Result: smooth gradients showing spatial lighting while preserving privacy.

**9. Progressive 2× downsampling quality benefits:**
Single-pass resize (1920×1080 → 64×64) used browser's default bilinear interpolation with potential aliasing. Progressive 2× chained reductions (→960×540→480×270→240×135→128×72→64×64) with `imageSmoothingQuality='medium'` ensured proper 2×2 pixel averaging at each step, preserving quality through multiple passes.

**10. Blurhash DCT performance bottleneck:**
Initial blurhash implementation decoded at full canvas resolution (1920×1080 = 2M pixels), causing ~1 fps. Fixed by decoding at 128×128 (16K pixels) then GPU-scaling to full size. However, 4×4 grid + bicubic approach proved ~5× faster (~0.3ms vs ~1.4ms per frame) with simpler implementation and no external dependency.

## Solution

**Privacy Mode Infrastructure:**
- Added `privacyMode: boolean` setting (default: true) in `useCameraSettings`
- Privacy Settings UI in `SettingsTab` with toggle and help text
- `useCanvasRenderer` calculates average frame color and uses it as background when privacy mode ON
- All privacy props threaded through: PostureTrackerApp → PostureCamera → useFrameSampler

**Bicubic Grid Background:**
- Created `bicubicGridRenderer.ts` with GPU-accelerated bicubic interpolation:
  - `renderBicubicGrid()`: Renders 4×4 color grid to full canvas using `imageSmoothingQuality='high'` (bicubic)
  - `renderSmoothedBicubicGrid()`: Adds temporal smoothing (exponential moving average, alpha=0.1)
- Updated `colorUtils.ts` with progressive downsampling:
  - `progressiveDownsample2x()`: Chains 2× reductions (e.g., 1920×1080→960×540→480×270→...→64×64)
  - `sampleVideoGrid()`: Uses progressive downsampling before 4×4 cell averaging
- Live view: `useCanvasRenderer` samples 4×4 grid → renders with bicubic + smoothing per frame (~60fps)
- Thumbnails: `thumbnailGenerator` samples and renders grid directly (no temporal smoothing)
- Performance: Grid sampling + bicubic upsampling = ~0.3ms per frame (~5× faster than blurhash)
- Quality: Browser-native bicubic (Chrome: Lanczos3, Firefox/Safari: bicubic) provides smooth gradients
- Fallback to solid `#1a1b1e` when video not ready or on error

**Human-Like Skeleton Rendering:**
- Created `drawHumanLikeSkeleton()` in `canvasDrawing.ts` with filled shapes:
  - Head: Semi-transparent circle with facial features
  - Eyes: Black circles (14% of head radius)
  - Nose: Orange circle (9% of head radius)
  - Ears: Orange circles (11% of head radius, positioned 10% further apart)
  - Torso: Filled polygon connecting shoulders and hips
  - Limbs: Rounded capsules (arms/legs)
  - Hands: Circles at wrists
- All parts have white 2px outlines (scaled with `baseScale`) for visibility when overlapping
- Contrasting colors: blue body (#4dabf7), orange nose/ears (#ffa94d), black eyes

**Smooth Animations:**
- Added `SmoothedKeypoint` interface extending `Keypoint` with `opacity: number`
- `calculateSmoothingAlpha()`: Linear interpolation based on detection interval (0.5-2s)
- `interpolateKeypoints()`: Smoothly transitions positions and opacity
  - Keypoints with confidence > 0.2 fade in (opacity → 1.0)
  - Keypoints with confidence ≤ 0.2 fade out (opacity → 0.0)
  - When no person detected, existing keypoints fade while maintaining position
- PostureCamera maintains smoothing state refs and interpolates on every render frame (~60fps)

**Resolution-Independent Sizing:**
- `baseScale = canvasWidth / 640` scales all stroke widths and default sizes
- Facial features scale with calculated head radius (proportional to face size)
- Ensures consistent appearance: 640px thumbnail matches 1920px camera view

**Privacy-Safe Thumbnails:**
- Modified `generateThumbnail()` to accept `{ privacyMode, keypoints }` options
- When privacy mode ON: draws dark background + skeleton on offscreen canvas
- Converts plain keypoints to SmoothedKeypoint (add `opacity: 1.0`) before drawing
- Manual captures (G/B/A buttons) now pass privacy mode settings
- Auto-captures already used privacy mode from `useFrameSampler` config

**Files Modified:**
- `src/utils/canvasDrawing.ts` - Drawing functions with smoothing, scaling, opacity support
- `src/utils/colorUtils.ts` - Progressive 2× downsampling, updated grid sampling (MODIFIED)
- `src/utils/bicubicGridRenderer.ts` - Bicubic grid rendering with temporal smoothing (NEW)
- `src/services/dataset/thumbnailGenerator.ts` - Privacy mode thumbnail generation with bicubic grid
- `src/hooks/useFrameSampler.ts` - Privacy mode config for auto-captures
- `src/hooks/useCameraSettings.ts` - Privacy mode setting
- `src/hooks/useCanvasRenderer.ts` - Bicubic grid background when privacy mode ON
- `src/components/unified/SettingsTab.tsx` - Privacy Settings UI
- `src/components/PostureCamera.tsx` - Smoothing state, privacy mode rendering
- `src/pages/PostureTrackerApp.tsx` - Privacy mode for manual captures

## Related
- `tasks/2025-11-03-feature-rtmpose-raw-spatial-features.md` - Keypoint data structure context
- `tasks/2025-10-31-refactor-collapsible-right-panel-overlay.md` - Settings UI patterns
