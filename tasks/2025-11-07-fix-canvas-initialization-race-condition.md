# Task 2025-11-07: Fix Canvas Initialization Race Condition
**STATUS:** COMPLETED

## User Request
i occasionally have this error,not sure if it breaks anything

```
[Capture] Failed to capture frame: InvalidStateError: Failed to execute 'drawImage' on 'CanvasRenderingContext2D': The image argument is a canvas element with a width or height of 0.
    at generateThumbnail (thumbnailGenerator.ts:48:9)
    at useFrameSampler.ts:134:33
    at PostureTrackerApp.tsx:217:12
```

## Critical Discoveries (Non-Obvious)

**1. Canvas sizing happens AFTER video ready:**
Canvas starts at 0×0 and only gets sized when render loop runs AND video has valid dimensions. Keyboard shortcuts and capture buttons are enabled immediately on mount, creating a ~100-500ms window where captures fail.

**2. Error causes data loss:**
User confirmed that early captures are silently lost - not just console noise. This is a functional issue requiring UI blocking, not just error handling.

**3. Multiple capture trigger points:**
- Keyboard shortcuts (G/B/A) - registered immediately on mount
- Manual capture buttons - clickable immediately
- Auto-capture timer - starts immediately if enabled
- Posture change detector - runs immediately if model loaded

All trigger before canvas ready, all need guarding.

**4. Canvas cleanup resets to 0×0:**
`useCanvasRenderer` cleanup sets `canvas.width = 0; canvas.height = 0` which can trigger error if capture happens during unmount/remount cycle.

## Solution

**Validation Layer (thumbnailGenerator.ts)**
- Added canvas dimension validation before `drawImage()` call
- Throws descriptive error instead of cryptic browser error
- Validates `sourceCanvas.width > 0 && sourceCanvas.height > 0`

**State Tracking (useCanvasRenderer.ts)**
- Added `isCanvasReady` boolean state
- Set to `true` when canvas first sized in render loop
- Reset to `false` on cleanup
- Exposed via hook return value

**Propagation Chain**
- PostureCamera: Added `onCanvasReady` callback prop, notifies parent of ready state changes
- PostureTrackerApp: Tracks `isCanvasReady` state, passes to CameraViewport
- CameraViewport: Accepts `isSystemReady` prop, disables capture buttons when not ready

**Keyboard Shortcuts Guard (PostureTrackerApp.tsx)**
- Wrapped G/B/A keyboard handlers with `if (isCanvasReady)` check
- Shortcuts silently no-op until canvas sized
- C (clear) and U (undo) remain available (don't need canvas)

**Error Handling (useFrameSampler.ts)**
- Catch canvas initialization errors specifically
- Log as warning instead of error for "not ready yet" cases
- Prevents confusing error messages during normal initialization

## Related
- `2025-11-04-fix-skeleton-clustering-at-edges.md` - Similar initialization timing issue with canvas rendering
- `2025-11-03-fix-manual-capture-blocked-during-training.md` - Related capture blocking logic patterns
