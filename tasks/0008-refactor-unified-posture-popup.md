# Task 0008: Unify Posture Notification Popups

**STATUS:** DONE

## User Request
unify popups with good bad posture, no person, no trained model. make it have one of these 4 states. play sound only on bad posture state.

## General Description
PostureStatusBadge component currently displays 4 states (good posture, bad posture, no person, no model) but lacks visual consistency - each state appears to come from different components with inconsistent styling. Additionally, a critical sound bug exists where sound continues playing when transitioning from bad posture to no person detected state.

## Action Plan
1. **Fix critical sound bug in app/index.tsx** - postureData construction uses stale classification state, causing sound to continue when person leaves frame
2. Redesign PostureStatusBadge component for consistent visual appearance across all 4 states
3. Ensure all states use unified layout structure (same padding, borders, sizing)
4. Update tests for PostureStatusBadge to verify unified styling and sound behavior

## Rationale
**Critical Sound Bug:** Lines 202-210 in `app/index.tsx` construct `postureData` based on `classification` state, but `classification` can be stale when person leaves frame. The worker stops sending classification updates when no person is detected, but the old classification state (with `prediction: 'bad'`) persists. This causes `postureData.slouching` to remain `true` even when no person is in frame, triggering continuous sound playback.

**Root Cause:** `postureData` uses `classification ? { person_found: true, ... }` logic, assuming classification existence means person presence. This is incorrect - classification can linger from previous frames.

**Correct Fix:** Check `inferenceResult` (current frame data) to determine if person is actually present, not just whether `classification` state exists. When `inferenceResult` is null or person not detected, `postureData` should be null regardless of stale classification state.

**Visual Consistency:** Current implementation has different header structures (mlHeader vs simpleHeader), different icon systems (emoji vs text symbols), and inconsistent spacing between states. Users perceive this as fragmented UI rather than a unified state indicator.

**Component Architecture:** PostureStatusBadge is already the single source of truth for state visualization (lines 12-110 in PostureStatusBadge.tsx). Refactoring this component maintains single responsibility while improving UX.

## Alternative Approaches Considered
1. **Modify usePostureSound to add timeout logic:** Rejected - doesn't fix root cause (stale state), adds complexity
2. **Clear classification state on no-person detection:** Rejected - loses useful data, doesn't address state synchronization
3. **Add explicit person detection flag to classification:** Rejected - redundant with inferenceResult data already available
4. **Create separate popup/modal component:** Rejected - adds UI complexity and conflicts with user preference for always-visible status badge
5. **Completely redesign with new layout:** Rejected - good/bad posture states work well, only no-person/no-model need alignment

## Files to Modify
- `app/index.tsx` - Fix postureData construction to check inferenceResult for person presence instead of relying on stale classification state (lines 199-213)
- `src/components/PostureStatusBadge.tsx` - Unify visual design across all 4 states
- `src/components/__tests__/PostureStatusBadge.test.tsx` - Update tests for new unified styling
- `app/__tests__/index.test.tsx` - Add test for sound bug fix (verify sound stops when inferenceResult becomes null)

## Related Code References
- Current state priority logic: PostureStatusBadge.tsx lines 13-41 (no model > no person > ML classification)
- Sound triggering logic: usePostureSound.ts lines 34-47 (already filters for bad posture only)
- State styling: PostureStatusBadge.tsx lines 124-135 (goodPosture, badPosture, noModel, noPerson styles)

## Implementation Details

**Sound Bug Fix (app/index.tsx):**
- Changed postureData construction from checking `classification` state to `inferenceResult && classification` (line 205)
- RTMW3DCameraWeb sets `inferenceResult` to null when no person detected, but `classification` state can linger from previous frames
- Now sound stops immediately when person leaves frame instead of continuing with stale classification
- Added detailed comment explaining the fix (lines 199-203)

**Visual Unification (PostureStatusBadge.tsx):**
- Replaced separate style variants (mlHeader/simpleHeader, mlIcon/warningIcon/infoIcon) with unified `header` and `icon` styles (lines 137-149)
- All 4 states now use consistent layout: icon + title in flexDirection row, 8px marginBottom
- Standardized icon system: "!" (no model), "i" (no person), "OK" (good posture), "X" (bad posture)
- All icons rendered as white, bold, 20px text instead of emojis

**Test Coverage:**
- Added 3 comprehensive tests in app/__tests__/index.test.tsx (lines 521-665):
  1. Verifies postureData resets to null when person leaves frame (bad posture → no person)
  2. Verifies postureData resets to null when person detection becomes false (good posture → no person)
  3. Verifies postureData remains null when inferenceResult exists but no classification
- Updated PostureStatusBadge.test.tsx with visual unification tests (lines 281-375):
  1. Verifies unified text icons across all 4 states
  2. Verifies consistent header structure (icon + title) for all states
- All 1,168 tests passing across 51 test suites

**STATUS:** ✅ COMPLETED
