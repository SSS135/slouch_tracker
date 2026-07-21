# Task 2025-10-31: Create Capture Buttons Overlay & Keyboard Shortcuts
**STATUS:** COMPLETED

## User Request
Add manual capture buttons at bottom center of video and implement keyboard shortcuts (G, B, A, C) for capture and clear actions. Fix capture buttons to prevent size changes during animation—buttons should maintain consistent width and keep original label text.

## Critical Discoveries

**1. Mantine Button loading state causes layout shifts:**
Loading spinner + text changes ("Good" → "Captured!") create size fluctuations during animation. Fixed with `width: 90px` + opacity feedback instead of text replacement.

**2. Mantine useHotkeys auto-ignores form elements:**
No manual filtering needed—hook automatically ignores keyboard events from input/textarea/select elements. Simplifies implementation.

**3. Z-index hierarchy prevents button obstruction:**
Buttons at z-index 40, frame preview at 50. Ensures capture buttons hide when hovering frame list (avoiding obstruction of preview image).

## Solution

Created CaptureButtonsOverlay component positioned at bottom center (semi-transparent background rgba(0, 0, 0, 0.75), blur(6px)). Reused existing AnimatedCaptureButton for visual consistency. Positioned with `bottom: 16, left: '50%', transform: 'translateX(-50%)'` at z-index 40.

Fixed AnimatedCaptureButton sizing: added fixed width (90px) with minWidth, removed text change to "Captured!", added opacity feedback (0.7) during success state with 200ms transition. Retained Mantine's loading spinner for visual feedback. All three buttons now have uniform width without layout shifts.

Integrated overlay into VideoSection via optional props (`onCaptureGood`, `onCaptureBad`, `onCaptureAway`, `isSaving`). Conditional rendering ensures overlay only appears when all handlers provided.

Implemented global keyboard shortcuts using Mantine's `useHotkeys` hook: G (Good), B (Bad), A (Away), C (Clear). All shortcuts use existing handlers (`handleCaptureWithLabel`, `clearFrames`).

```typescript
// AnimatedCaptureButton.tsx (button sizing fix)
style={{
  width: 90,
  minWidth: 90,
  opacity: state === 'success' ? 0.7 : 1,
  transition: 'opacity 200ms ease-in-out',
}}

// CaptureButtonsOverlay.tsx (bottom center positioning)
position: 'absolute',
bottom: 16,
left: '50%',
transform: 'translateX(-50%)',
zIndex: 40  // Below frame preview (50) to hide during preview
```

## Bug Fix #1: Frame List Animation Flicker

**Problem:**
Clicking capture buttons (G/B/A) caused visible flicker in left panel frame list—frames appeared to be added and removed simultaneously (entry + exit animations within <200ms).

**Root Cause:**
Manual button captures incorrectly used the auto-capture buffer (`recentFrames`):
1. `captureFrame('manual')` → added frame to buffer → entry animation triggered
2. Frame saved to IndexedDB
3. `removeFrame()` called → exit animation triggered
4. Both animations visible in rapid succession = flicker

**Fix:**
Modified `handleCaptureWithLabel` in `UnifiedPosturePage.tsx` to bypass buffer entirely for manual captures. Inlined frame creation logic directly in handler—frames now save to IndexedDB without touching `recentFrames` buffer.

```typescript
// Before: used buffer (caused flicker)
const frameId = captureFrame('manual');
await persistCapturedFrame(...);
removeFrame(frameId);

// After: direct save (no buffer interaction)
const newFrame: PostureFrame = {
  id: `manual_${Date.now()}_${Math.random()}`,
  timestamp: Date.now(),
  features: { ...inferenceResult.features },
  thumbnail: await generateThumbnail(videoRef.current),
  label: postureLabel,
};
await persistCapturedFrame(newFrame);
```

**Architecture Clarification:**
- Left panel frame list = `recentFrames` buffer for auto-captured, unlabeled frames only
- Manual captures (buttons/shortcuts) = Direct IndexedDB save (bypass buffer)
- Auto-captures = Continue using buffer normally

**Result:**
- Manual captures: No frame list animation (silent save)
- Auto-capture: Works normally with buffer animations
- All tests pass (useFrameSampler: 19/19 passed)

## Bug Fix #2: Frozen Frames Synchronization

**Problem:**
Saving frames from auto-capture list during hover didn't animate out properly. Frames only disappeared when user unhovered the list.

**Root Cause:**
Frozen frames mechanism creates snapshot of `recentFrames` when hovering to keep list stable during preview/interaction:
1. `handleSaveFrameWithLabel` and `handleSaveAll` called `removeFrame()` → updated `recentFrames` state
2. UI still displayed `frozenFrames` (frozen snapshot) → No visual update
3. Only when unhovering, `frozenFrames` cleared → Frame finally disappeared

**Fix:**
Modified both save handlers in `UnifiedPosturePage.tsx` to synchronize `frozenFrames` when removing frames during hover:

```typescript
// handleSaveFrameWithLabel (lines ~389-413)
removeFrame(frameId);
if (frozenFrames) {
  setFrozenFrames(prev => prev ? prev.filter(f => f.id !== frameId) : null);
}
// Added frozenFrames to dependency array

// handleSaveAll (lines ~366-392)
for (const frame of frames) {
  await persistCapturedFrame(frame, label);
  removeFrame(frame.id);
  if (frozenFrames) {
    setFrozenFrames(prev => prev ? prev.filter(f => f.id !== frame.id) : null);
  }
}
// Added frozenFrames to dependency array
```

**Result:**
- Frames now animate out immediately when clicking save buttons (even during hover)
- Hover stability maintained for other frames in the list
- Auto-capture continues adding frames to frozen snapshot during hover
- All tests pass (useFrameSampler: 19/19 passed)

**Important Note:**
This issue was NOT caused by Bug Fix #1. It was a pre-existing problem with the frozen frames mechanism that became more noticeable after the flicker fix.

## Lessons
- Reuse over recreation: AnimatedCaptureButton already had animation—maintained consistency across CollectTab and overlay
- Fixed sizing prevents layout shifts: explicit width + opacity feedback better than text changes
- Mantine hooks handle edge cases: `useHotkeys` superior to manual window.addEventListener
- Buffer separation: Auto-capture and manual capture have different UX requirements—auto needs buffer preview, manual needs silent save

## Related
- `tasks/0009-feature-hover-preview-frames.md` - Overlay positioning patterns
- `tasks/2025-10-31-feature-frame-list-overlay-component.md` - Z-index hierarchy

## Files Modified
- `src/components/unified/CaptureButtonsOverlay.tsx` - New overlay component
- `src/components/unified/AnimatedCaptureButton.tsx` - Fixed button sizing, removed text change, added opacity feedback
- `src/components/unified/VideoSection.tsx` - Added capture props and conditional overlay rendering
- `src/pages/UnifiedPosturePage.tsx` - Added useHotkeys, passed capture handlers to VideoSection

## Impact
Users can manually capture frames from video feed without navigating to CollectTab. Keyboard shortcuts (G/B/A/C) enable faster data collection workflow. Capture buttons maintain consistent size during animations, improving visual stability. Overlay positioned at bottom center with automatic hiding behind frame preview (z-index management).
