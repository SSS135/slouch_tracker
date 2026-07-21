# Task 2025-11-04: Fix Skeleton Clustering at Image Edges

**STATUS:** COMPLETED

## User Request
Check if keypoints are clamped to image size or can go outside of it. Problem is, in skeleton drawing there are some body parts that cluster around bottom of it. Like they clamped to bottom. If there is no clamping, hide parts that start and end at bottom.

## Critical Discoveries

**1. Bounding boxes clamped, keypoints not:**
Bounding boxes ARE clamped to image boundaries in `inference-worker.ts:994-997`, but keypoints themselves are NOT clamped in `transformKeypoints()`. When body parts extend outside the frame, keypoints can have coordinates that fall outside [0, width/height], causing clustering at edges when drawn.

**2. 5-pixel threshold effective:**
Using a 5-pixel margin from edges (x/y ≤ 5 or x/y ≥ width/height - 5) reliably detects clamped keypoints without false positives for valid near-edge poses.

**3. Both endpoints must be at boundary:**
Only hiding connections where BOTH endpoints are at boundaries prevents hiding valid partial poses (e.g., arm extending toward edge with shoulder inside frame).

## Solution

Added boundary detection to hide skeleton connections and limbs where both endpoints are at/near image edges (within 5px threshold):

1. **Added `isAtImageBoundary()` helper** (`canvasDrawing.ts:103-115`)
   - Checks if keypoint is within 5px of any edge (top, bottom, left, right)
   - Used by both normal and privacy mode skeleton drawing

2. **Updated `drawConnections()`** (`canvasDrawing.ts:120-155`)
   - Added `videoWidth` and `videoHeight` parameters
   - Skips connections where both endpoints at boundaries
   - Keeps connections visible when only one endpoint at boundary

3. **Updated `drawAllKeypoints()`** (`canvasDrawing.ts:290-300`)
   - Passes `videoWidth` and `videoHeight` to `drawConnections()`

4. **Updated `drawHumanLikeSkeleton()`** (`canvasDrawing.ts:667-695`)
   - Applied same boundary detection to privacy mode limbs (capsules)
   - Skips limbs where both endpoints at boundaries

5. **Added comprehensive tests** (`canvasDrawing.test.ts`)
   - 11 new tests covering boundary detection for both drawing modes
   - Tests all four edges, exact threshold (5px vs 6px), partial vs full boundary
   - All 26 tests in suite pass

## Related

- `tasks/2025-11-03-feature-privacy-mode.md` - Introduced `drawHumanLikeSkeleton()` function
- `tasks/2025-10-24-refactor-replace-rtmw3d-with-rtmpose-s.md` - RTMPose-S model integration and keypoint handling
