# Task 2025-11-07: Simplify Thumbnail Generation by Reusing Display Canvas

**STATUS:** COMPLETED

## User Request

how does thumbnail capture works? is it re-uses current canvas image and extracted features or something more complicated?

can we simplify it by making manual and auto-capture thumbs reuse current canvas image?

## Critical Discoveries

**1. Display canvas already has everything:**
The `useCanvasRenderer` renders video/grid, skeleton overlays, and all visual elements. Thumbnails were duplicating this work by calling `sampleVideoGrid()`, `renderBicubicGrid()`, and `drawHumanLikeSkeleton()` again - completely unnecessary.

**2. Canvas ref must be ready before capture:**
Added `isCanvasReady` state tracking to prevent capturing thumbnails before canvas dimensions are initialized (width/height = 0). Canvas dimensions are set asynchronously during first render loop, so keyboard shortcuts must wait for ready state.

**3. TypeScript HTMLCanvasElement not in CanvasImageSource union:**
TypeScript's `CanvasImageSource` type doesn't include `HTMLCanvasElement` (only `HTMLImageElement | SVGImageElement | HTMLVideoElement | ImageBitmap | OffscreenCanvas`). Required `as unknown as CanvasImageSource` cast despite runtime support.

## Solution

**Refactored thumbnail generation to snapshot display canvas instead of re-rendering:**

1. **thumbnailGenerator.ts** - Added `sourceCanvas` option, removed duplicate privacy mode rendering (grid sampling, bicubic interpolation, skeleton drawing). Falls back to video element if no canvas provided.

2. **useFrameSampler.ts** - Added `displayCanvasRef` parameter, removed `privacyMode` config (no longer needed). Passes canvas to `generateThumbnail()`.

3. **useCanvasRenderer.ts** - Added optional `canvasRef` parameter for external ref injection, added `isCanvasReady` state to track when canvas dimensions are initialized.

4. **PostureCamera.tsx** - Added `canvasRef` prop and `onCanvasReady` callback, passed canvas ref to `useCanvasRenderer`, notified parent of canvas ready state changes.

5. **PostureTrackerApp.tsx** - Created `canvasRef` alongside `videoRef`, added `isCanvasReady` state, gated keyboard shortcuts on canvas ready state, passed canvas ref to both `PostureCamera` and `useFrameSampler`, updated manual capture to use display canvas.

6. **Test updates** - Updated all test signatures to match new function parameters (`useFrameSampler` now requires `displayCanvasRef`), fixed thumbnailGenerator tests to use options object syntax.

**Benefits:** Removed ~80 lines of duplicate rendering logic, guaranteed thumbnail consistency with display, single render path eliminates performance overhead, works automatically in all modes (normal/privacy/debug).

## Related

- `tasks/0017-fix-engineered-features-training-errors.md` - Feature extraction pipeline that thumbnails are part of
