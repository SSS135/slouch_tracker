# Task 2025-10-27: Fix Auto-Capture System (AWAY States + Timer Intervals)

**STATUS:** COMPLETED

## User Request

**Initial issue:** "frame auto-captured only on good / bad state change, fix it. it should capture on good / bad / away changes."

**Follow-up issue:** "that fixed, but I found that auto capture every 5 seconds since last frame doesn't work, fix it"

**Third issue:** "when no model trained, auto-capture happens every frame, not every 5 seconds, fix it"

**Final issue (2025-11-01):** "fix auto-capture, use existing task. Problem is when no model trained it auto captures every frame or so"

## Critical Discoveries

**1. AWAY state treated as null instead of valid state:**
`predictionValue` returned `null` when `goodProbability === null` (person away) instead of `FrameLabel.AWAY`. This bypassed transition detection logic entirely - GOOD→AWAY and AWAY→GOOD transitions were silently ignored.

**2. Event-driven timing breaks with low inference frequency:**
Timer implementation using `useEffect(() => {...}, [inferenceResult, ...])` only checked timing when inference updated. If inference frequency was low, timer never fired. Needed independent setInterval.

**3. fullConfig object recreation caused cascading instability:**
Object spread `{ ...DEFAULT_CONFIG, ...config }` created new object every render → `captureFrame` recreated every render → `onCapture` callback recreated → interval timer constantly reset → captures every frame instead of every 5 seconds.

**4. Wrong interval setting used for auto-capture:**
Auto-capture used `settings.captureIntervalSeconds` (0.5s, meant for manual collection) instead of intended 5-second interval for automated data collection. This caused captures every 500ms instead of 5s.

## Solution

**Event-driven posture-change detection:**
Extended `usePostureChangeDetector` to handle three-state transitions (GOOD/BAD/AWAY). Changed predictionValue computation to return `FrameLabel.AWAY` for null goodProbability using IIFE pattern. Added test coverage for all away transitions.

**Timer-driven interval capture:**
Replaced event-driven useEffect with setInterval-based timer running every 500ms independently. Both event-driven and timer-driven mechanisms share `lastAutoRecordTimeRef` for "since ANY captured frame" behavior.

**Stability improvements (regression fixes):**
- Memoized `fullConfig` object to prevent recreation on every render
- Added `isCapturingRef` pattern to break dependency chain (alongside existing `inferenceResultRef`)
- Changed auto-capture to use configurable `autoCaptureIntervalSeconds` setting (default 5s, range 1-15s)

**Configurable interval:**
Added Developer Settings slider control for auto-capture interval (1-15 seconds). Applies to both no-model scenario and backup timer when model exists.

## Lessons

- Complete the ref pattern: When breaking dependency chains with refs, ensure ALL frequently-updating state variables are covered (both `inferenceResult` and `isCapturing`)
- Memoize configuration objects: Object spreads create new references every render, breaking dependency chains even when values haven't changed
- Use independent timers for periodic operations: Event-driven approaches fail when trigger frequency is unpredictable
- Wrong settings can look like bugs: Using manual collection interval (0.5s) for auto-capture (5s) caused captures every frame

## Related

- tasks/2025-10-26-feature-add-away-presence-detection.md - Added AWAY state infrastructure
- tasks/2025-10-23-feature-event-driven-5s-auto-recording.md - Original flawed timer implementation
- tasks/0002-feature-posture-change-based-capture.md - Posture-change detection

## Files Modified

**Initial fixes:**
- src/hooks/usePostureChangeDetector.ts - Three-state transition detection
- src/hooks/__tests__/usePostureChangeDetector.test.ts - Away transition tests
- app/index.tsx (lines 242-283) - setInterval-based timer
- src/hooks/useFrameSampler.ts - Ref pattern for stable captureFrame

**Regression fixes (2025-11-01):**
- src/hooks/useFrameSampler.ts - Memoized fullConfig, added isCapturingRef
- src/hooks/useCameraSettings.ts - Added autoCaptureIntervalSeconds setting
- src/components/unified/RuntimeTab.tsx - Added slider control (1-15s)
- src/pages/UnifiedPosturePage.tsx - Use autoCaptureIntervalSeconds setting
- src/hooks/__tests__/useCameraSettings.test.ts - Updated test expectations

## Impact

Complete auto-capture system now functional with configurable intervals (1-15s, default 5s):

1. **Event-driven** - Captures on any state transition (GOOD↔BAD↔AWAY)
2. **Timer-driven with model** - Captures every N seconds regardless of inference frequency
3. **Interval-driven without model** - Captures every N seconds when no ML model exists

Enables hands-free dataset collection for both posture and presence detection training. Users can adjust capture frequency via Developer Settings based on their needs (rapid training vs less intrusive collection).
