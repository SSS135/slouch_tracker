# Task 0006: Replace Browser Popup with Subtle Capture Feedback
**STATUS:** COMPLETED

## User Request
when I capture good / bad frame, browser default popup appears. I'd like to use something less verbose, like small notification at the corner or button animation. What is simpler to implement and maintain?

## Critical Discoveries

**1. Button animation is significantly simpler than corner notifications:**
Button animation requires no dependencies, no state management, no positioning logic, and no cleanup. Self-contained in component with pure CSS/React Native animations. Corner notifications would require toast library (react-hot-toast, react-toastify), portal/overlay state, z-index positioning, stacking logic, and manual cleanup - over-engineered for simple success feedback.

**2. Browser alert() interrupts workflow:**
Native browser popups block interaction, require dismissal, and break user flow. Need non-blocking feedback mechanism that provides confirmation without interrupting capture workflow.

## Solution

**Button animation feedback (CollectTab.tsx):** Added scale + background flash animation on capture button press. Green pulse for Good button, red pulse for Bad button. 300-500ms duration with auto-reset. Optional checkmark/x icon displays briefly. Animations trigger on press, providing immediate visual feedback at interaction point without blocking other actions.

**Removed popup calls (app/index.tsx or handlers):** Removed `showSuccess()` calls from handleCaptureGood and handleCaptureBad handlers (previously lines ~310, 352). Kept `showError()` for actual failure cases where user needs explicit error message. Success feedback now purely visual via button animation.

**Result:** Non-blocking capture feedback that doesn't interrupt workflow. No external dependencies. Self-contained component logic. Better UX with immediate feedback at click location. Simpler to test and maintain than toast notifications or status badges.

## Related

- No related tasks (isolated UX improvement)
