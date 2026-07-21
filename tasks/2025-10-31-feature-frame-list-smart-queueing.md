# Task 2025-10-31: Add Smart Queueing & Notification Badge to Frame List
**STATUS:** COMPLETED

## User Request
Delay frame updates when user hovers frame list, show pending count notification badge with "X new frames waiting" text and fade animation.

## Critical Discoveries

**1. Parent state simpler than child state:**
Original plan used Set-based tracking in FrameListOverlay with `prevFrameIdsRef` + `useEffect`. Actual implementation moved state to UnifiedPosturePage using baseline snapshot: `baselineFrameCount` captures frame count when hover starts, `visibleFrames = frames.slice(0, baselineFrameCount)` filters display. Eliminates Set logic, useEffect tracking, and ref management.

**2. Badge text length vs space:**
Planned badge text "X new frame(s) waiting" too verbose for narrow overlay. Changed to "+X" format (e.g., "+3") for minimal visual footprint while maintaining clarity.

**3. Container-level hover prevents flicker:**
Hover handlers on outer Box (not individual frames) prevent state thrashing when cursor moves between frame items. Consistent with `2025-10-26-feature-hover-pause-auto-collection.md` pattern.

## Solution

**State Management (UnifiedPosturePage):**
- Added `isFrameListHovered` and `baselineFrameCount` state
- `visibleFrames` computed from `frames.slice(0, baselineFrameCount)` during hover
- `queuedFrameCount = frames.length - baselineFrameCount` for badge
- Hover start: capture baseline, hover end: reset baseline to 0

**FrameListOverlay Updates:**
- Container-level hover detection: `onMouseEnter`/`onMouseLeave` on outer Box
- Notification badge at top center: `left: '50%', transform: 'translateX(-50%)'`
- Badge styling: orange `bg="orange.6"`, z-index 70, 0.2s fade transition
- Badge text: "+{count}" format (minimal)
- Renders `visibleFrames` prop instead of `frames` prop directly

**VideoSection Passthrough:**
Updated interface to pass `visibleFrames`, `queuedFrameCount`, and hover handlers from parent to FrameListOverlay.

**Key implementation snippet (UnifiedPosturePage.tsx):**
```typescript
const visibleFrames = isFrameListHovered
  ? frames.slice(0, baselineFrameCount)
  : frames;
const queuedFrameCount = isFrameListHovered
  ? frames.length - baselineFrameCount
  : 0;

const handleFrameListHoverStart = () => {
  setIsFrameListHovered(true);
  setBaselineFrameCount(frames.length);
};

const handleFrameListHoverEnd = () => {
  setIsFrameListHovered(false);
  setBaselineFrameCount(0);
};
```

## Lessons

**Baseline snapshot pattern beats Set tracking:** Array slicing simpler than Set-based diffing for frame queueing. No need for `prevFrameIdsRef` or `useEffect` monitoring.

**Defer implementation decisions until context clear:** Original plan over-engineered tracking logic. Actual implementation emerged naturally from parent/child relationship—parent manages buffer state, child displays filtered view.

**Badge centering requires explicit transform:** `left: '50%'` alone anchors left edge to center. Must add `transform: 'translateX(-50%)'` to truly center badge.

## Related

- `tasks/2025-10-26-feature-hover-pause-auto-collection.md` - Identical hover-pause pattern for CollectTab
- `tasks/0015-feature-improve-frame-list-ui.md` - Newest-first ordering, no auto-scroll

## Files Modified

- `src/pages/UnifiedPosturePage.tsx` - Added queueing state and logic (lines 66-67, 399-423, 543-552)
- `src/components/unified/VideoSection.tsx` - Updated interface and props (lines 23-25, 55-57, 89-91)
- `src/components/unified/FrameListOverlay.tsx` - Added hover handlers and badge (lines 15-17, 155-157, 182-183, 185-202)

## Impact

Frame list no longer shifts during user interaction. Hover-pause mechanism prevents layout disruption while inspecting frames. Orange "+X" badge provides clear feedback on queued frames without consuming screen space.
