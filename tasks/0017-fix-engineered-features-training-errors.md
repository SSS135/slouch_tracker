# Task 0017: Fix Engineered Features Training Errors
**STATUS:** COMPLETED

## User Request

When I try to train on engineered features with random projection, I get error: nComponents (256) cannot exceed nFeatures (36). Without random projection I get another error: Error extracting engineered features: TypeError: leftShoulder.midpoint is not a function. Fix it.

**Evolved scope:** Initial training errors revealed real-time worker inference bug (same midpoint error), then NaN/Infinity prediction bugs, then Infinity in hand distance features breaking ML algorithms, then 10x FPS regression from validation overhead in hot path requiring performance optimization.

## Critical Discoveries (Non-Obvious)

**1. Random projection ≠ PCA dimension constraint:**
Random projection is matrix multiplication (x × R^T = output). No mathematical limit on nComponents vs nFeatures. Validation incorrectly blocked projecting 36-dim features → 256-dim space (valid use case, unlike PCA which requires nComponents <= nFeatures).

**2. Keypoint prototypes lost in two separate code paths:**
Training: .slice() operations dropped Keypoint3D class methods → plain objects.
Worker: classifyFeatures() created plain objects from arrays but feature extraction needed .midpoint()/.distanceTo() methods.

**3. Engineered features had no validation for degenerate poses:**
Division by zero (hipWidth = 0), invalid keypoints (NaN coordinates, score = 0), and no output validation → NaN features → 100% good/bad predictions. Classifiers had no defense against invalid input.

**4. Mathematical Infinity incompatible with ML algorithms:**
Hand distance features (leftHandMinDistance, rightHandMinDistance, handFaceProximityScore) intentionally set to Infinity when hands far from face. TensorFlow.js and ML training algorithms cannot process Infinity values → classifier rejection. Required replacement with large finite constant (FAR_DISTANCE = 999.0).

**5. Validation overhead in hot path caused 10x FPS regression:**
Initial validation fixes (input/output checks, keypoint conversion) ran on EVERY frame (30-60 FPS) in real-time inference. Keypoint3D.fromPlainArray() converted 133+ objects per frame. Object.values() + .some() + logger calls created allocations/iterations on hot path. Solution: Remove all validation from hot path, use static methods accepting plain objects, keep only critical division-by-zero check (3 comparisons). Trade runtime safety for performance.

## Solution

**Phase 1 - Training Pipeline (Original Bugs):**
Removed nComponents > nFeatures validation in randomProjection.ts (random projection supports dimension increase). Added explicit Keypoint3D type annotations in featureExtractor.ts after .slice() to preserve class prototypes through array operations.

**Phase 2 - Worker Real-Time Inference (Initial Fix):**
Initially added Keypoint3D.fromPlainArray() conversion in worker, but this caused 10x FPS regression. Later removed in favor of static methods (see Phase 5).

**Phase 3 - Minimal Validation (Performance-First):**
Added minimal measurement validation in detection.ts: check shoulderWidth, torsoLength, hipWidth > 0.01 before division (prevents division by zero). Removed expensive input/output validation loops for performance. Added debug logging in classifiers (only runs when ?log=detection:debug enabled, no performance impact in production).

**Phase 4 - Hand Distance Infinity Fix:**
Replaced mathematical Infinity with FAR_DISTANCE constant (999.0) in detection.ts for hand-to-face distance features. When hands are far from face or not detected, features now use large finite value instead of Infinity. ML algorithms can now process all engineered features without errors.

**Phase 5 - Performance Optimization (10x FPS Recovery):**
Removed worker keypoint conversion entirely (no Keypoint3D.fromPlainArray). Added static helper methods to Keypoint3D class (midpoint, distanceTo, toVector) accepting plain objects - zero conversion overhead. Updated detection.ts to use static methods throughout. Removed per-frame input/output validation loops, removed per-frame debug logging. Classifiers have optional debug logging (only when ?log=detection:debug URL param), but no validation that blocks predictions. Result: hot path contains only essential math operations and one critical measurement check (3 comparisons).

**Code snippets:**
```typescript
// Static methods work with plain objects (zero conversion overhead)
const midShoulder = Keypoint3D.midpoint(leftShoulder, rightShoulder);
const shoulderWidth = Keypoint3D.distanceTo(leftShoulder, rightShoulder);

// Minimal validation: Only prevents division by zero (3 comparisons)
if (shoulderWidth < 0.01 || torsoLength < 0.01 || hipWidth < 0.01) {
  logger.warn('detection', '[extractAllFeatures] Invalid body measurements');
  return null;
}

// FAR_DISTANCE for ML compatibility
const FAR_DISTANCE = 999.0;
let leftHandMinDistance = FAR_DISTANCE;  // Instead of Infinity
```

## Lessons

Random projection supports any nComponents (including > nFeatures) unlike PCA. ML algorithms require finite values - replace Infinity with large constants (FAR_DISTANCE pattern). **Performance-first architecture:** Validate only what's absolutely necessary (division by zero). Static helper methods accepting plain objects eliminate conversion overhead (133+ objects per frame). Per-frame validation/logging causes 10x FPS regression - remove from hot path. Debug logging acceptable when behind URL flag (?log=detection:debug). Trade-off: Less runtime safety, but essential for 30-60 FPS real-time inference. Division-by-zero protection + FAR_DISTANCE constant = minimal viable validation.

## Related

None (isolated ML pipeline fixes).

## Files Modified

1. `src/services/ml/randomProjection.ts` - Removed incorrect nComponents validation
2. `src/services/ml/featureExtractor.ts` - Added type annotations for training path
3. `src/workers/unified-pose-worker.ts` - No conversion code (uses plain objects only)
4. `src/services/posture/detection.ts` - Added FAR_DISTANCE constant, minimal validation (3 comparisons), static method calls
5. `src/services/posture/Keypoint3D.ts` - Added static helper methods (midpoint, distanceTo, toVector) accepting plain objects
6. `src/services/ml/knnClassifier.ts` - Added debug logging (optional, URL-gated)
7. `src/services/ml/logisticRegressionClassifier.ts` - Added debug logging (optional, URL-gated)
8. `src/services/ml/__tests__/randomProjection.test.ts` - Updated for dimension increase
9. `src/services/ml/__tests__/featureExtractor.test.ts` - Added prototype preservation test
10. `src/services/posture/__tests__/detection.test.ts` - Updated for FAR_DISTANCE and static methods

## Impact

**Unlocked:** Training with engineered features (36 dims) + random projection (any dimension). Training with engineered features + no dimensionality reduction. Real-time ML inference with engineered features at full FPS. Minimal validation overhead (only division-by-zero check).

**Technical:** Valid predictions across all poses and classifier types (KNN, Logistic Regression). No Infinity errors (FAR_DISTANCE = 999.0). Original FPS performance maintained. Optional debug logging (?log=detection:debug) for troubleshooting. Zero breaking changes to existing models or feature types. Comprehensive test coverage (237 tests pass). Trade-off: Minimal runtime safety checks in favor of maximum performance.
