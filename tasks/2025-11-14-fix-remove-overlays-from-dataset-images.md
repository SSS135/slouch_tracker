# Task 2025-11-14: Remove Visual Overlays from Dataset Images
**STATUS:** COMPLETED

## User Request
"images are recorded to dataset with rtmdet boxes and rtmpose points, they should not include them (but still include clahe / blur / smoothing). same for image list icons"

## Critical Discoveries

**1. Display Canvas Contains All Overlays**
Task 2025-11-07 refactored thumbnails to reuse display canvas (avoiding re-rendering), but inadvertently captured all visual overlays (RTMDet boxes, RTMPose keypoints, z-values, skeleton). Display canvas includes overlays added by `useCanvasRenderer.onDraw` callback.

**2. Preprocessed ImageData Already Exists**
The `displayFrame` state in `PostureCamera` contains preprocessed ImageData (with CLAHE, blur, temporal smoothing) BEFORE overlays are drawn. Using this as thumbnail source provides clean images without duplicating preprocessing.

**3. Fallback Chain is Critical**
Must maintain backward compatibility: ImageData → Canvas → Video. Canvas fallback needed for cases where ImageData not yet available (early initialization).

## Solution

**Architecture Change:**
```
Before: Display Canvas (with overlays) → Thumbnail
After:  Preprocessed ImageData (no overlays) → Thumbnail
```

**Implementation:**
1. `thumbnailGenerator.ts` - Added `sourceImageData` option (highest priority)
2. `useFrameSampler.ts` - Accept `displayFrameRef` parameter, pass ImageData to thumbnail generator
3. `PostureCamera.tsx` - Expose `displayFrame` via ref prop, sync with state
4. `PostureTrackerApp.tsx` - Wire `displayFrameRef` through all components

**Benefits:**
- Thumbnails contain preprocessing (CLAHE/blur/smoothing) but exclude overlays
- No preprocessing duplication (reuses existing ImageData)
- Backward compatible (fallback to canvas if ImageData unavailable)
- Fixes both dataset images and image list icons

## Related

- `tasks/2025-11-07-refactor-simplify-thumbnail-generation.md` - Introduced bug by using display canvas
- `tasks/2025-11-06-feature-opencv-clahe-preprocessing.md` - Preprocessing pipeline architecture
- `tasks/2025-11-10-refactor-temporal-smoothing-sma.md` - Preprocessing flow understanding
