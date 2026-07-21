# Task 0012: Consecutive Bad Frames Sound Alert

**STATUS:** ✅ COMPLETED

## User Request
Sound plays only after N consecutive bad frames (default N=2). Non-looping, plays on every bad frame after threshold.

## Critical Discoveries (Non-Obvious)

**1. expo-audio restart quirk:**
Calling `player.play()` with `loop=false` doesn't restart if sound already playing/finished. Need explicit reset:
```typescript
player.pause();   // Stop current
player.seekTo(0); // Reset position
player.play();    // Start fresh
```

**2. Split state → duplicate triggers:**
Worker sends ONE message with both `inferenceResult` + `classification`, but they were split into separate React state variables. Result:
- 1 worker message → 2 sequential state updates → 2 re-renders per frame
- Each re-render creates new `postureData` object → sound plays 2-4x per frame

**3. Effect dependency anti-pattern:**
Original deps: `[postureData, volume, threshold, paused]` meant effect ran on settings changes, not just new frames. Counter incremented on wrong events.

## Solution

Added consecutive frame counter (`useRef`) with configurable threshold (1/2/3/5 frames, persisted in localStorage).

**Fixed duplicate triggers (3 parts):**
1. Merged `classification` into `InferenceResult` type (single state update per frame)
2. Memoized `postureData`: `useMemo(() => {...}, [inferenceResult])`
3. Fixed effect deps: `}, [postureData]);` (trigger ONLY on new frames)

**Key principle:** Effects depend on DATA that triggers them (frames), READ settings as configuration (don't depend on settings).

## Files Modified
- `src/hooks/useCameraSettings.ts` - Added `consecutiveBadFramesThreshold` setting
- `src/hooks/usePostureSound.ts` - Frame counter + effect deps fixed
- `src/components/unified/RuntimeTab.tsx` - UI control
- `app/index.tsx` - Memoized postureData, removed separate classification state
- `src/services/onnx/rtmw3dInference.ts` - Added classification to InferenceResult type
- `src/components/RTMW3DCameraWeb.tsx` - Include classification in inferenceResult

## Impact
- Sound plays exactly once per frame (was 2-4x)
- 50-75% fewer re-renders per frame
- Settings changes don't interfere with frame-based logic

## Related
See Task 0013 for full state consolidation refactoring details.
