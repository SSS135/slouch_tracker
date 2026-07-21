# Task 2025-11-03: Add Undo Button for Frame Captures
**STATUS:** COMPLETED

## User Request
add undo button. when I capture frame either by buttons in auto-capture list or bottom buttons manual capture, it should be saved to action history (len=5, reset on page reload). Undo button should appear on screen at top near auto-capture list. When I hover over button it should show thumb, original and user assigned classification statuses (as good -> bad change or something). When i click undo, this sample is removed from dataset and history. Then model retrains. Undo button displays previous sample if any remain.

## Critical Discoveries

**1. React Query mutation API quirk:**
`datasetOps.deleteFrame` is a mutation object, not a function. Must use `.mutateAsync(frameId)` not `deleteFrame(frameId)`.

**2. Mantine Popover controlled state required for hover:**
Default `disabled` prop doesn't work for hover triggers. Need controlled `opened` state with `onMouseEnter`/`onMouseLeave`.

**3. Hidden vs disabled button:**
Button should be completely hidden when `canUndo=false` (not just disabled) for cleaner UI. Conditional rendering in CameraViewport.

**4. Blob to dataURL conversion needed:**
Action history stores thumbnails as base64 dataURLs (strings) not Blobs to avoid memory leaks and simplify popover rendering.

## Solution

**Architecture**: Ref-based circular buffer (max 5 actions, FIFO eviction) matching existing training queue pattern. No persistence (resets on page reload).

**Implementation**:
1. Added `CaptureAction` interface with frameId, timestamp, label, thumbnailUrl (base64), captureSource
2. Created `useActionHistory` hook: ref-based storage with reactive `canUndo`/`lastAction` state for UI updates
3. Modified 3 capture handlers (`handleCaptureWithLabel`, `handleSaveAll`, `handleSaveFrameWithLabel`) to convert Blob→dataURL and push to history
4. Created `handleUndo`: pops action, calls `deleteFrame.mutateAsync()`, invalidates stats, triggers auto-training
5. Built `UndoButton` component: hover popover (thumbnail + label + capture source), click to undo, shows "Press U key" hint
6. Integrated in `CameraViewport`: positioned left:184px (right of frame list), only renders when `canUndo=true`
7. Added 'U' keyboard shortcut to `useHotkeys` array

**Tests**: Added `useActionHistory.test.ts` (20 tests: push/undo/clear/max size/integration) and `UndoButton.test.tsx` (23 tests: render states/click/hover popover/async rendering with waitFor).

## Related
- `tasks/2025-10-31-feature-capture-buttons-keyboard-shortcuts.md` - Manual capture handlers pattern
- `tasks/2025-11-01-feature-auto-training-on-capture.md` - Auto-training trigger usage
- `tasks/2025-10-31-feature-frame-list-smart-queueing.md` - Frozen frames synchronization pattern
