# Task 2025-10-24: Add Performance API Timing to Unified Pose Worker

**STATUS:** COMPLETED

## User Request

Add Performance API (performance.mark/measure) timing measurements to unified-pose-worker.ts to track:
- Total frame processing time
- RTMDet inference time
- RTMW3D inference time
- Preprocessing time (crop/transform)
- ML classification time (if classifier loaded)

Requirements:
- Use Performance API (performance.mark/measure) for browser DevTools visibility
- Measurements should appear in Chrome/Firefox Performance profiler
- Track all major pipeline stages
- Clean up old marks periodically to prevent memory buildup
- Handle both "person found" and "no person" cases

## Critical Discoveries (Non-Obvious)

**1. Inlining required for granular timing:**
Original `detectPerson()` helper abstracted away preprocessing/inference steps. Inlining into `processFrame()` enabled separate marks for RTMDet preprocess/inference/postprocess without code duplication.

**2. Frame-based mark naming prevents collisions:**
Using `frame-{frameCounter}-{stage}-{start|end}` pattern creates unique marks per frame, enabling DevTools filtering (e.g., "Show frame 42") and simplifying cleanup logic.

**3. Cleanup strategy:**
Performance entries consume memory if never cleared. Cleanup every 100 frames keeps last 200 frames for analysis while preventing unbounded growth. Must clear measures when clearing marks (measures reference marks by name).

## Solution

Instrumented `unified-pose-worker.ts` with Performance API marks at all major pipeline stages:

**State management:** Added `frameCounter` and `MARK_CLEANUP_INTERVAL` (100 frames) for tracking and periodic cleanup.

**Helper functions:** Created `perfMark()`, `perfMeasure()`, `cleanupOldMarks()` utilities. Marks use pattern `frame-{N}-{stage}-{start|end}` for unique naming.

**Instrumentation points:**
- RTMDet: preprocess, inference, postprocess, total
- Crop: bbox expansion and cropping
- RTMW3D: preprocess, inference, postprocess, total
- Transform: keypoint coordinate transformation
- Classifier: ML inference (conditional, only when loaded)
- Frame: total processing time

**Cleanup:** Every 100 frames, removes marks older than 200 frames and clears all measures to prevent memory buildup.

## Files Modified

- `src/workers/unified-pose-worker.ts` (+305/-64 lines) - Added performance instrumentation throughout processFrame() pipeline, inlined detectPerson() for granular timing, added cleanup logic

## Impact

Developers can now profile worker performance in browser DevTools (Performance tab → User Timing track). Timing data reveals bottlenecks across RTMDet vs RTMW3D, preprocessing overhead, and classifier inference cost. Enables correlation with browser events (GC, layout) and export to JSON for offline analysis.
