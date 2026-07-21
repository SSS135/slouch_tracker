# Task 2025-10-23: Event-Driven 5-Second Auto-Recording

**STATUS:** COMPLETED

## User Request

record frames if auto recording is on every 5 seconds since last recorder frame, be it interval- or posture change- recorded last

## Critical Discoveries

**1. Event-driven timing beats timer-based:**
`useEffect` watching `inferenceResult` eliminates race conditions vs `setInterval`. Check fires immediately after each detection cycle completes, not on wall-clock intervals.

**2. Shared timestamp prevents double-capture:**
Single `lastAutoRecordTimeRef` tracks last capture from ANY source. When posture-change fires at 4.9s, interval check at 5.0s sees elapsed time < 5s and skips capture.

**3. Graceful initialization pattern:**
`lastAutoRecordTimeRef = -Infinity` ensures first inference ALWAYS passes 5s threshold without special-case logic.

**4. Label assignment divergence:**
Posture-change captures use `captureLabel` (NEW posture after transition). Interval captures use `inferenceResult.classification.prediction` (CURRENT stable posture).

## Solution

**Core Implementation:**

Added event-driven 5-second interval check in `app/index.tsx`:
```typescript
// Shared timestamp for auto-recording (both modes)
const lastAutoRecordTimeRef = useRef<number>(-Infinity);

// Event-driven 5-second check (fires after each inference)
useEffect(() => {
  if (!settings.autoCaptureEnabled || !trainedModel || !inferenceResult?.classification) return;

  const now = Date.now();
  if (now - lastAutoRecordTimeRef.current >= 5000) {
    captureFrame('interval', inferenceResult.classification.prediction);
    lastAutoRecordTimeRef.current = now;
  }
}, [inferenceResult, settings.autoCaptureEnabled, trainedModel, captureFrame]);
```

Updated posture-change effect to update shared timestamp:
```typescript
useEffect(() => {
  if (shouldCapture && captureLabel) {
    captureFrame('interval', captureLabel);
    lastAutoRecordTimeRef.current = Date.now(); // Reset interval timer
  }
}, [shouldCapture, captureLabel, captureFrame]);
```

Removed `cooldownMs: 2500` from posture-change detector config (line 147).

**Testing:**
- 8 new tests added (5-second interval + shared timestamp behavior)
- All 29 tests pass
- Coverage: No capture when disabled, model requirement, label accuracy, no duplicates

## Lessons

**Architecture:** Event-driven checks aligned with detection cycles prevent race conditions and provide more responsive behavior than timer-based polling.

**Shared state:** Single timestamp ref across multiple capture modes simplifies rate limiting logic and prevents edge cases (simultaneous triggers).

**Initialization patterns:** Using `-Infinity` for timestamp initialization eliminates need for special-case "first capture" logic.

## Related

- None (new feature, no coupled tasks)

## Files Modified

1. **`app/index.tsx`** (lines 117, 148-192)
   - Added `lastAutoRecordTimeRef` shared timestamp ref
   - Removed `cooldownMs: 2500` from posture-change detector
   - Updated posture-change effect to update shared timestamp
   - Added new 5-second interval check effect (event-driven)

2. **`app/__tests__/index.test.tsx`** (new test suites)
   - Added "5-Second Interval Auto-Recording" suite (6 tests)
   - Added "Shared Timestamp Behavior" suite (2 tests)

## Impact

**User Experience:** Auto-recording now captures frames every 5 seconds during stable postures, providing balanced training data (transitions + stable states) without manual intervention.

**Performance:** Event-driven approach eliminates independent timer overhead and aligns with detection cadence (~10-30 FPS depending on device).

**Data Quality:** Interval captures with current classification labels provide training examples for stable postures, complementing posture-change captures (transitions).
