# Task 0010: Synchronize Camera Frame Display with Detection State

**STATUS:** ✅ COMPLETED

## User Request
synchronize displayed camera frame, keypoints drawn on it and overall detection state. Right now video is displayed at camera fps and keypoints are drawn at detection fps on current (unrelated to detection) frame.

## General Description
The current implementation has a desynchronization issue between video display and detection results. The canvas rendering loop (useCanvasRenderer) runs at camera FPS via requestAnimationFrame, drawing every video frame. Meanwhile, detection results arrive at a different rate (detection FPS, controlled by captureIntervalSeconds). This causes keypoints to be drawn on frames that weren't actually analyzed, creating visual misalignment.

## Action Plan
1. Modify useCanvasRenderer to freeze the displayed frame when detection completes
2. Capture the analyzed ImageData snapshot alongside inference results
3. Update canvas rendering to draw the frozen detection frame instead of live video
4. Only update displayed frame when new detection results arrive
5. Ensure detection state (classification, keypoints) matches the displayed frame

## Rationale
**Current Architecture Problem:**
- useCanvasRenderer draws live video at ~30 FPS (requestAnimationFrame)
- useFrameProcessor captures frames at detection interval (e.g., 1 FPS)
- Worker returns results asynchronously at detection rate
- lastResultRef updates with keypoints, but canvas draws current live frame
- Result: Keypoints from frame N drawn on frame N+30 (1 second later)

**Solution Approach:**
Instead of continuously rendering live video, we should render the last analyzed frame as a still image when detection results are available. This ensures perfect synchronization between displayed frame, keypoints, and classification state.

**Why Not Alternatives:**
- Synchronizing detection to camera FPS: Too computationally expensive (30 FPS detection)
- Buffering frames: Memory intensive, complex queue management
- Timestamping: Doesn't solve fundamental issue of keypoints on wrong frame

## Alternative Approaches Considered
1. **Timestamp-based matching**: Tag frames with timestamps and match detection results to buffered frames. Rejected due to complexity and memory overhead.
2. **Reduce camera FPS to match detection**: Modify video constraints to match detection rate. Rejected as it impacts user experience during collection (users want smooth preview).
3. **Display interpolated keypoints**: Smooth keypoint motion between detections. Rejected as it shows incorrect data (predictions on unanalyzed frames).

## Files to Modify
- `src/hooks/useCanvasRenderer.ts` - Switch from live video to frozen frame rendering
- `src/hooks/useFrameProcessor.ts` - Capture analyzed ImageData snapshot
- `src/hooks/useWebWorkerInference.ts` - Return ImageData with detection results
- `src/components/RTMW3DCameraWeb.tsx` - Pass frozen frame to renderer
- `src/workers/unified-pose-worker.ts` - Transfer ImageData back from worker (optional optimization)

## Related Code References
- `useCanvasRenderer` (line 87-88): Currently draws live video every frame
- `useFrameProcessor` (line 82-88): Captures frame and sends to worker
- `handleWorkerResult` in RTMW3DCameraWeb (line 61-114): Processes detection results
- `lastResultRef` in RTMW3DCameraWeb (line 39, 99-105): Stores keypoints for rendering

## Implementation Details

Successfully implemented frame synchronization between camera display and detection state, ensuring keypoints are always drawn on the exact frame that was analyzed.

**Core Changes:**

1. **Frame Capture Strategy (useFrameProcessor.ts)**
   - Implemented dual ImageData capture: one for worker processing (transferred via zero-copy), one retained as snapshot for display
   - Added `frameSnapshot` to returned tuple, providing access to analyzed frame after worker transfer
   - Preserved existing performance optimizations (temp canvas reuse, transferable worker messages)

2. **Result Synchronization (useWebWorkerInference.ts)**
   - Modified worker result callback to accept and pass through frame snapshot
   - Updated state management to store `analyzedFrame` alongside detection results
   - Ensures frame snapshot is available when detection results arrive asynchronously

3. **Rendering Pipeline (useCanvasRenderer.ts)**
   - Switched from continuous live video rendering to frozen frame display
   - Implemented intelligent fallback: frozen frame (when available) → live video (when not)
   - Optimized temp canvas reuse for ImageData-to-canvas conversion
   - Proper cleanup of canvas resources in useEffect

4. **Component Integration (RTMW3DCameraWeb.tsx)**
   - Wired frame snapshots through the processing pipeline
   - Updated result handlers to store analyzed frames
   - Passed frozen frames to canvas renderer for synchronized display

5. **Test Coverage**
   - Updated all test files with new API signatures (frameSnapshot parameter)
   - Verified frozen frame rendering logic with comprehensive test cases
   - All 51 test suites passing with 1118+ individual tests

**Performance Impact:**
- Memory overhead: ~1.2 MB per detection interval (single ImageData snapshot)
- No impact on worker zero-copy transfer (dual capture approach)
- Rendering optimized with temp canvas reuse (no per-frame allocation)

**Visual Result:**
- Perfect synchronization between displayed frame, drawn keypoints, and detection classification
- No more keypoint drift or misalignment issues
- Smooth user experience with graceful fallback to live video when needed
