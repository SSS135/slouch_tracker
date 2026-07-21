# Task 0013: Consolidate Classification State

**STATUS:** ✅ COMPLETED

## User Request
Eliminate duplicate sound triggers and clean up unused variables/outdated comments in `app/index.tsx` and related files.

## Critical Discoveries (Non-Obvious)

**State separation anti-pattern:**
Splitting data that arrives together (`inferenceResult` + `classification` in one worker message) into separate React state causes:
- Multiple re-renders per logical update (1 message → 2 state updates → 2 renders)
- Timing bugs between updates (stale state between first and second update)
- Difficult-to-debug performance issues (2-4x re-renders per frame)

**Effect dependency best practice:**
Effects should depend on DATA that triggers them (frames), READ settings as configuration (don't depend on settings). Wrong pattern:
```typescript
}, [postureData, volume, threshold, paused]);  // ❌ Runs on settings changes
```
Correct pattern:
```typescript
}, [postureData]);  // ✅ Runs only on new frames, reads settings
```

## Solution

**Phase 1 - Consolidate State:**
Merged `classification` into `InferenceResult` type. Removed `onClassification` callback chain. Classification now flows: Worker → InferenceResult → Components (no split/recombine).

**Phase 2 - Memoization:**
```typescript
const postureData = useMemo(() => {
  if (!inferenceResult?.classification) return null;
  return { /* ... */ };
}, [inferenceResult]);  // Only recreate when data changes
```

**Phase 3 - Clean Up:**
Removed unused `workerClassifierReady`, redundant `classification` from CameraContext, incorrect dependency arrays (`showSuccess` unused in 3 callbacks).

## Files Modified (7 total)
1. `src/services/onnx/rtmw3dInference.ts` - Added classification to type
2. `src/components/RTMW3DCameraWeb.tsx` - Removed onClassification callback
3. `app/index.tsx` - Removed separate classification state, memoized postureData, cleaned up
4. `src/hooks/usePostureSound.ts` - Fixed effect deps to `[postureData]` only
5. `src/contexts/CameraContext.tsx` - Removed redundant classification field
6. `src/components/unified/VideoSection.tsx` - Access classification via inferenceResult

## Impact
- **50-75% fewer re-renders** per frame (2-4 renders → 1 render)
- **Sound plays exactly once** per frame (no duplicate/stuttering)
- **Single source of truth** for classification
- **Simpler data flow:** Worker → InferenceResult → Components

## Architecture Lesson
Keep related data in single state object. Split only when legitimately different update sources or lifecycles. Data arriving together should stay together.
