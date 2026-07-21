# Task 2025-11-03: Fix PLS-DA Validation and KNN Edge Case
**STATUS:** COMPLETED

## Issues Found

### Issue 1: Type Guard Rejecting PLS-DA Models
**Problem**: Models trained with PLS-DA dimensionality reduction failed validation when saving to IndexedDB.

**Root Cause**: The `isTrainedModel()` type guard in `guards.ts:37` only accepted `type === 'random_projection'`, rejecting PLS-DA models with `type === 'pls-da'`.

**Error Message**:
```
[Storage] Validation failed in savePresenceModel
Failed to save presence model: Error: Invalid model data: validation failed
```

### Issue 2: KNN Crashes When k > Training Samples
**Problem**: During cross-validation with small datasets, KNN crashed when k (number of neighbors) exceeded the number of training samples in a CV fold.

**Root Cause**: `tf.topk()` was called with k larger than the tensor dimension, causing error: `'k' passed to topk() must be <= the last dimension (4) but got 5`

**Scenario**:
- Dataset: 6 samples (3 good, 3 bad)
- CV strategy: 3-fold (limited from 5 due to class size)
- Training samples per fold: 4
- Configured k: 5
- Result: k > 4, topk() error

## Solutions Implemented

### Fix 1: Update Type Guard for PLS-DA
**File**: `src/services/validation/guards.ts`
**Line**: 37

**Before**:
```typescript
(model.dimReductionTransformer === null || (
  typeof model.dimReductionTransformer === 'object' &&
  model.dimReductionTransformer.type === 'random_projection' &&
  model.dimReductionTransformer.data !== undefined
))
```

**After**:
```typescript
(model.dimReductionTransformer === null || (
  typeof model.dimReductionTransformer === 'object' &&
  (model.dimReductionTransformer.type === 'random_projection' || model.dimReductionTransformer.type === 'pls-da') &&
  model.dimReductionTransformer.data !== undefined
))
```

### Fix 2: Clamp k to Available Training Samples
**File**: `src/services/ml/knnClassifier.ts`

**Changes**:

1. **Single prediction** (line 170-174):
```typescript
// Get indices of k-nearest neighbors (smallest distances)
// Clamp k to actual number of training samples to prevent topk() error
const nTrainingSamples = training.shape[0];
const effectiveK = Math.min(this.params.k, nTrainingSamples);
const topK = tf.topk(tf.neg(distances as tf.Tensor1D), effectiveK);
```

2. **Update probability calculation** (line 199):
```typescript
return count0 / effectiveK;  // Use effectiveK instead of this.params.k
```

3. **Batch prediction** (line 410-412):
```typescript
// Clamp k to actual number of training samples to prevent topk() error
const effectiveK = Math.min(model.k, nTraining);
const topK = tf.topk(tf.neg(distances), effectiveK, true);
```

### Fix 3: Add Test Coverage
**File**: `src/services/ml/__tests__/knnClassifier.test.ts`

**Added test** (lines 408-435):
```typescript
describe('Edge Cases', () => {
  it('should handle k > training samples by clamping to available samples', () => {
    // Create a very small dataset (4 samples total)
    const config: ClassifierConfig = {
      classifierId: 'knn',
      params: { k: 10 },  // k=10 but only 4 samples
    };
    const classifier = new KNNClassifier(
      config,
      { method: 'none', components: 64 },
      [FEATURE_GAU]
    );

    const dataset = createMockDataset(2, 2);  // Only 4 samples total

    // Training should succeed even though k > n_samples
    const result = classifier.train(dataset);
    expect(result.success).toBe(true);

    // Prediction should work by automatically clamping k to n_samples
    const features = createGAUFeatures(150);
    const prob = classifier.predictProba(features);

    // Should return valid probability
    expect(prob).toBeGreaterThanOrEqual(0);
    expect(prob).toBeLessThanOrEqual(1);
    expect(isNaN(prob)).toBe(false);
  });
});
```

## Testing

All tests pass:
- ✅ PLS-DA tests: 17/17 passed
- ✅ Validation guards tests: 58/58 passed
- ✅ Validation schemas tests: 41/41 passed
- ✅ KNN tests: 21/21 passed (including new edge case test)

## Impact

### Before Fixes:
- PLS-DA models failed to save to IndexedDB (validation error)
- KNN crashed during CV with small datasets when k > fold size
- User could not train models with PLS-DA reduction

### After Fixes:
- PLS-DA models save and load correctly
- KNN gracefully handles k > training samples by clamping
- Cross-validation works with any dataset size
- No user-facing errors during training

## Files Modified
1. `src/services/validation/guards.ts` - Type guard for PLS-DA
2. `src/services/ml/knnClassifier.ts` - k clamping in prediction methods
3. `src/services/ml/__tests__/knnClassifier.test.ts` - Edge case test coverage

## Backward Compatibility
✅ No breaking changes
✅ Existing models continue to work
✅ All existing tests pass
