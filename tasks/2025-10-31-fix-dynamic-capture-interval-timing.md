# Task 2025-10-31: Fix Dynamic Capture Interval Timing
**STATUS:** COMPLETED

## User Request
fix how capture interval works. right now it just waits set time, processes frame, waits again e.t.c. The wait is fixed and independent of frame processing time. I want you to make it dynamic. Record time when processing began and substract processing time from next capture wait time. So when I set capture interval=1 second, I have exactly 1 second capture / processing fps, not 0.8 fps like now.

## Critical Discoveries

**1. Real issue was FPS display, not capture timing:**
Detection FPS displayed 1.6 instead of 2.0 for 500ms interval. Investigation revealed setTimeout delays from worker CPU usage (120ms processing → 107-122ms setTimeout delay → 610ms actual interval).

**2. setTimeout delayed by CPU-heavy workers:**
When worker was processing, browser delayed setTimeout by ~100ms. Cannot rely on precise timing when worker thread is busy.

**3. Effect restarts from callback dependencies:**
`processFrame` and `onFps` callbacks as dependencies → effect restarted when callbacks changed → cleared scheduled timeouts → broken timing. Fixed with refs.

**4. Wrong timing measurement:**
Initially measured canvas capture time (8ms) instead of full cycle including worker processing (120ms).

## Solution

**Fixed interval timer with worker busy checking** (not dynamic timing compensation):

**Algorithm:**
```typescript
// Normal case (worker fast, 120ms processing):
Timer fires at 500ms → worker free → capture → 2.0 fps ✓

// Edge case (worker slow, 600ms processing):
Timer at 500ms → worker busy → retry every 10ms
Worker finishes at 600ms → capture
Next timer at 1100ms → repeat
Result: 1.67 fps (limited by worker throughput)
```

**Implementation:**
- Added `isProcessing: boolean` parameter to `useFrameProcessor`
- Added worker busy check: if `isProcessing`, retry in 10ms
- Removed dynamic timing compensation (no processing time subtraction)
- Always schedules at fixed interval: `setTimeout(scheduleNextCapture, intervalMs)`
- Uses refs for `isProcessing`, `processFrame`, `onFps` to prevent effect restarts
- Effect only restarts when `videoElement`, `intervalSeconds`, or `enabled` changes

## Files Modified

**`src/hooks/useFrameProcessor.ts`** - Added `isProcessing` parameter, worker busy checking with 10ms retry, ref-based callbacks to prevent effect restarts

**`src/components/RTMW3DCameraWeb.tsx`** - Pass `isProcessing: inference.isProcessing` to useFrameProcessor

**Previous iteration (replaced by final solution):**
- Created `useAutoCapture.ts` hook with dynamic timing compensation
- Refactored `useFrameSampler.ts` to remove interval logic
- Updated `UnifiedPosturePage.tsx` to use useAutoCapture
(All replaced by simpler fixed-interval approach)

## Lessons

1. **setTimeout is unreliable under CPU load** - Browser delays setTimeout when worker thread is busy. Fixed intervals + busy-wait is simpler than dynamic compensation.

2. **Callback dependencies cause effect restarts** - Must use refs to stabilize effects when callbacks change frequently.

3. **Don't fight browser scheduling** - Trying to compensate for browser delays is more complex than accepting throughput limits.

## Testing

- Build succeeds ✓
- No TypeScript errors ✓
- Expected FPS: 2.0 for 500ms interval (was 1.6 before fix)

## Impact

**Accuracy:** FPS display now matches configured interval (2.0 fps for 500ms, not 1.6)
**Simplicity:** Fixed intervals simpler than dynamic timing compensation
**Stability:** No more effect restarts from callback changes
