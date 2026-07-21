# Task 2025-10-31: Per-Posture Cooldown Auto-Capture
**STATUS:** COMPLETED

## User Request
improve auto-capture. It should work as follows 1. capture every 5 seconds (resets at any capture); 2. capture after each posture status change (2s reset interval, individual for each posture type. e.g. if we had good->bad, then bad->good after 0.1s, both (1-bad, 2-good) should be captured. if we did again good->bad after 0.1s, no new bad state is captured. if instead we did good->away, then away state will be captured.

## Critical Discoveries (Non-Obvious)

**1. React render cycle timing issue:**
Hook returned `shouldCapture/captureLabel` but effect never saw `shouldCapture: true` due to render cycle. Computed values in hooks don't persist to effects running in same render pass. **Solution:** Changed to callback pattern - hook calls `onCapture(label)` directly via `useEffect`.

**2. Dependency array with high-frequency updates causes interval recreation:**
Timer effect had `inferenceResult` in dependency array, causing `setInterval` to be recreated constantly. **Solution:** Used ref pattern (`inferenceResultRef`) to break dependency cycle. Same fix as task 2025-10-27 for `useFrameSampler`.

**3. Per-posture cooldown independence:**
Cooldown only blocks capturing the SAME posture type, not different ones. GOOD→BAD→GOOD in 0.2s captures both (independent cooldowns), but GOOD→BAD→BAD blocks second BAD (same posture cooldown active).

## Solution

**Per-Posture Cooldown (usePostureChangeDetector):**
- Changed default cooldown from 2500ms to 2000ms
- Replaced `lastCaptureTimeRef: number` with `Map<FrameLabel, number>` for per-posture tracking
- Each posture type (GOOD, BAD, AWAY) has independent cooldown
- Cooldown check uses `lastCaptureTimeMapRef.get(predictionValue)` instead of single timestamp
- Changed to callback pattern: hook calls `onCapture(label)` directly instead of returning computed values

**Timer Captures Include AWAY (UnifiedPosturePage):**
- Removed check that skipped AWAY states (`if (goodProbability === null) return`)
- Timer now captures all three states: GOOD, BAD, AWAY
- Fixed timer recreation issue using `inferenceResultRef` instead of dependency array
- Callback integration: passes `handlePostureChangeCapture` to hook

**Debug Logging:**
- State checks, transition detection, cooldown checks, capture triggers
- Helps debug with `?log=detection:debug`

**Tests:**
- All 20 tests passing in `usePostureChangeDetector.test.ts`
- Added 2 new tests for per-posture cooldown behavior
- Updated default cooldown from 2500ms to 2000ms across all tests

## Lessons

- Callback pattern essential when hooks need to trigger actions in parent components
- Ref pattern breaks dependency cycles in timer effects with high-frequency updates
- Per-posture cooldowns allow capturing rapid legitimate transitions while preventing duplicate captures

## Related
- tasks/2025-10-27-fix-auto-capture-complete.md - Ref pattern for interval effects
- tasks/0002-feature-posture-change-based-capture.md - Original shared cooldown implementation

## Files Modified
- `src/hooks/usePostureChangeDetector.ts` - Per-posture cooldown + callback pattern + logging
- `src/pages/UnifiedPosturePage.tsx` - AWAY in timer + callback integration + ref pattern
- `src/hooks/__tests__/usePostureChangeDetector.test.ts` - Updated and new tests

## Impact

**Behavior improvements:**
- Rapid GOOD→BAD→GOOD transitions: Both captured (per-posture cooldowns independent)
- Rapid GOOD→BAD→BAD transitions: Second BAD blocked (same posture cooldown)
- Timer fires every 5s, resets on any capture (posture-change or timer)
- AWAY states captured by both mechanisms (timer and posture-change)

**Dataset quality:**
- Captures all transition types without duplicates
- Balanced training data across all three states
- Responsive 2s cooldown per posture type (reduced from 2.5s)
