# Task 2025-10-28: Fix Missing neck_p5 Feature Extraction Call
**STATUS:** COMPLETED

## User Request
Fix error: "[featureExtractors] Failed to extract neck_p5 features for concatenation" and "[Unified Worker] Failed to extract features for presence classification"

## Critical Discoveries (Non-Obvious)

**1. Incomplete refactoring:**
Commit 9c182af refactored presence detection from 49-dimensional RTMDet bbox features to 192-dimensional P5 neck features. The refactoring:
- Added `extractRtmDetP5Features()` function (lines 484-519)
- Added scaling constants (C_AVG_P5, C_STD_P5, C_MAX_P5)
- Added output tensor name constant (RTMDET_NECK_P5_OUTPUT_NAME)
- Changed field name from `rtmDetFeatures` to `rtmDetP5Features`
- Declared variable `let rtmDetP5Features: Float32Array | undefined;`
- **BUT forgot to add the extraction call itself**

**2. Silent failure during worker initialization:**
Worker didn't crash, but presence detection silently failed. The `rtmDetP5Features` variable remained `undefined`, causing feature extraction to fail when the presence classifier attempted to use neck_p5 features. Error only appeared in logs when classifier tried to extract features.

**3. All infrastructure was present:**
- Function definition: `extractRtmDetP5Features()` existed and was correct
- Constants: C_AVG_P5, C_STD_P5, C_MAX_P5 properly defined
- Output name: RTMDET_NECK_P5_OUTPUT_NAME correctly identified ONNX tensor
- Feature registry: neck_p5 registered with 192 dimensions
- Only missing: 4-line extraction call

## Solution

**Added P5 feature extraction call** (src/workers/unified-pose-worker.ts, lines 1027-1030):

```typescript
if (rtmdetResults[RTMDET_NECK_P5_OUTPUT_NAME]) {
  const p5Tensor = rtmdetResults[RTMDET_NECK_P5_OUTPUT_NAME].data as Float32Array;
  rtmDetP5Features = extractRtmDetP5Features(p5Tensor);
}
```

**How it works:**
1. Check if RTMDet model output includes P5 neck tensor
2. Extract Float32Array from ONNX tensor result
3. Pass to `extractRtmDetP5Features()` which pools 64 channels × (10×10 spatial) into 192 features:
   - 64 avg-pooled values × C_AVG_P5
   - 64 std-pooled values × C_STD_P5
   - 64 max-pooled values × C_MAX_P5
4. Scaling constants normalize features to RMS = 1.0

**Verification:**
- Worker compiles successfully with no TypeScript errors
- P5 features now available for presence classification
- Feature extraction no longer fails

## Lessons

- **Refactoring checklist:** When replacing features, verify ALL steps: (1) function definition, (2) constants, (3) variable declaration, (4) **extraction call**, (5) field assignments
- **Test coverage gap:** No integration test caught missing extraction call. Worker tests should validate feature extraction pipeline end-to-end
- **Defensive coding:** Feature extractors should throw errors when required features are undefined (fail-fast vs silent failure)
- **Code review:** Complex refactorings like feature type changes need careful review of data flow paths

## Files Modified

- `src/workers/unified-pose-worker.ts` (lines 1027-1030) - Added P5 neck feature extraction call

## Impact

- **Severity:** HIGH - Presence detection (PRESENT vs AWAY classification) was completely non-functional
- **Root cause:** Incomplete refactoring in commit 9c182af (2025-10-28)
- **Fix complexity:** Minimal - single 4-line addition
- **User impact:** Presence detection now works correctly with 192-dimensional P5 neck features
- **Performance:** No change - extraction function is fast (simple pooling operations)
