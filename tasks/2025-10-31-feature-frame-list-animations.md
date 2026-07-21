# Task 2025-10-31: Frame List Animations (Preview Fade, Scroll-to-Top, Layout Animation)
**STATUS:** COMPLETED

## User Requests
1. "make frame preview appear and disappear over 0.2s. When new frames added to list scroll to top."
2. "add smooth animation on new frame appearing to this list."
3. "only top button appearance is animated, I want all buttons to move down freeing space for it, not just teleport"

## Critical Discoveries

**Conditional rendering breaks CSS transitions:** `{previewFrame && <Box>}` mounts/unmounts instantly, bypassing transitions. Solution: Keep element mounted, control opacity via delayed state change (10ms `setTimeout`).

**Scroll-to-top needs addition detection:** Simple `useEffect` on `frames.length` triggers on ANY change. Used `prevFrameCountRef` to detect only increases: `currentCount > prevCount`.

**Layout animations can't use pure CSS:** CSS transitions don't animate implicit flexbox reflow when items added. Options: (1) Manual position tracking with refs (complex), (2) Animation library. Chose Framer Motion—`layout` prop automatically animates position changes, declarative API, handles edge cases.

## Solution

**Preview Fade (VideoSection.tsx):** Added `isPreviewVisible` state + useEffect with 10ms delay. Element stays mounted while `previewFrame` exists, opacity transitions via state: `opacity: isPreviewVisible ? 1 : 0`.

**Scroll-to-Top (FrameListOverlay.tsx):** Added `scrollAreaRef` + `prevFrameCountRef`. useEffect compares counts, only scrolls when frames ADDED: `if (currentCount > prevCount) scrollAreaRef.current?.scrollTo({ top: 0, behavior: 'smooth' })`.

**Frame Layout Animation (FrameListOverlay.tsx):**
- Initial approach (CSS `@keyframes`): Animated new frame only, existing frames teleported
- Final approach (Framer Motion): Wrapped frames in `<motion.div layout>` + `<AnimatePresence mode="popLayout">`
- New frames: `initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}`
- Existing frames: `layout` prop animates position changes automatically
- Duration: 0.2s `easeOut` (consistent with other animations)
- Result: New frame fades in at top, existing frames smoothly slide down

## Lessons

**State-based animation for conditional elements:** Two-level control—(1) prop controls mount/unmount, (2) state controls transition. Allows CSS transition to work while element stays in DOM.

**Layout animations need libraries or FLIP:** CSS can't animate implicit layout changes. Framer Motion's `layout` prop handles automatically—worth 30KB for UX improvement.

## Related

- `tasks/2025-10-31-feature-frame-list-overlay-component.md` - Initial frame list with container slide-in
- `tasks/2025-10-31-feature-frame-list-smart-queueing.md` - Hover-pause with queueing

## Files Modified

- `package.json` - Added framer-motion
- `src/components/unified/VideoSection.tsx` - State + useEffect for preview (lines 72-84), state-based opacity (line 149)
- `src/components/unified/FrameListOverlay.tsx` - Framer Motion layout animation (lines 230-250), scroll refs + effect (lines 160-174)

## Impact

Preview fades in/out smoothly (0.2s). Auto-scroll shows newest frames without manual intervention. Frame appearance creates fluid, choreographed animation—new frame fades in, existing frames slide down to make room. Layout transitions maintain spatial continuity, eliminate jarring "teleport" effect. Works seamlessly with smart queueing.
