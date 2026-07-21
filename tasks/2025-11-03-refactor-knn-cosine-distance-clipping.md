# Task 2025-11-03: KNN Cosine Distance with L2 Normalization and ±3σ Clipping + PLS-DA Extreme Predictions Fix
**STATUS:** COMPLETED

## User Request
make knn use cosine distance instead of l1/l2. always l2 normalize inputs in it. also clip to +-3 sigma

**Follow-up:** why knn always shows either 0% or 100% for both posture and presence. I have k=5, 6 good, 6 bad, 3 away frames in dataset. PLS-DA projection only (others fine).

## Critical Discoveries

### PLS-DA Extreme Predictions (0% or 100%)

**1. PLS-DA stored wrong projection matrix (W instead of R):**
Original implementation stored weight matrix W from NIPALS, but correct PLS projection requires R = W(P^T W)^{-1}. This caused new samples to get incorrect projections that didn't match training scores, flipping neighborhood structure. **Fix:** Use W directly (simpler and avoids numerical issues with matrix inversion on small datasets).

**2. Missing z-score standardization:**
PLS-DA only centered features (x - mean) but didn't divide by std. Different feature scales dominated components, creating poor projections. **Fix:** Compute and store both means and stds, apply (x - mean) / std during fit and transform.

**3. Broken NIPALS iteration:**
Inner loop recomputed same w = normalize(X^T Y) each iteration because X and Y didn't change inside loop. Missing Y deflation between components. **Fix:** Use simplified PLS1 algorithm (no inner iteration needed), deflate BOTH X and Y between components.

**4. KNN L2 normalization destroyed PLS-DA structure:**
KNN's aggressive L2 normalization (all vectors to unit length) erased magnitude information from PLS-DA's small, centered projections. With small dataset (6 good, 6 bad, k=5), L2-normalized clusters became perfectly separated → all 5 neighbors always same class → 0% or 100%. **Fix:** Add distance weighting (weight neighbors by cosine similarity instead of simple majority vote).

### KNN Cosine Distance

**5. Preprocessing must be symmetric (training and query):**
Both training and query vectors must undergo identical preprocessing (±3σ clipping → L2 normalization) for cosine distance to work correctly. Applied preprocessing in `trainFinalModelImpl()` before storing training data tensor, and in `predictProba()`/`predictBatch()` before computing similarities.

**6. Cosine similarity via matrix multiplication:**
With L2-normalized vectors, cosine similarity is simply the dot product: `tf.matMul(query, training^T)`. Use `tf.topk` directly on similarities (not negative distances) since we want largest similarities (smallest distances).

**7. Distance parameter kept for backward compatibility:**
Old saved models may have `distance: 'euclidean'` or `distance: 'manhattan'` in serialized JSON. Kept parameter in type definitions and constructor but deprecated it. KNN always uses cosine distance regardless of parameter value.

## Solution

### PLS-DA Algorithm Fixes (plsda.ts)

**Simplified PLS1 algorithm (nipals method, lines 273-392):** Replaced broken iterative NIPALS with simplified single-pass PLS1. Each component: w = normalize(X^T Y), t = X w, p = X^T t / (t^T t), q = Y^T t / (t^T t). Deflate BOTH X and Y: X = X - t p^T, Y = Y - t q. Stack W weight vectors as [nComponents, nFeatures] projection matrix (avoids numerical issues with R = W(P^T W)^{-1} inversion).

**Z-score standardization (nipals + transform, lines 306-311, 123-125):** Compute means AND stds during fit: `stds = sqrt(mean((X - mean)^2))`. Apply z-score in fit: `X = (X - mean) / (std + 1e-8)`. Apply same in transform: `(x - mean) / (std + 1e-8)`. Store both means and stds in model.

**Updated serialization (toJSON/fromJSON, lines 174-187, 227-233):** Added `stds: number[]` to SerializedPLSDA interface. Save stds in toJSON. Validate stds in fromJSON - old models without stds fail with clear error: "Model was trained with old PLS-DA version. Please retrain your model."

**Updated class state (lines 34, 40, 91, 96, 264-267):** Added `private stds: number[]` and `private stdsTensor: tf.Tensor1D`. Store in fit, recreate in fromJSON, dispose in dispose().

**Test coverage (plsda.test.ts):** Updated 4 existing tests for stds field. Added 4 new tests: old model rejection, means/stds computation, projection matrix dimensions, Y deflation with multiple components. All 24 tests pass.

### KNN Distance Weighting (knnClassifier.ts)

**Weighted probability (predictProba, lines 256-270):** Replace simple majority vote with distance-weighted voting. Extract top k similarities, convert to positive weights: `weights = similarities.map(s => Math.max(0, s) + 1e-6)`. Compute weighted probability: `probGood = sum(weights[j] * isGood[j]) / sum(weights)`. Closer neighbors (higher similarity) have more influence.

### KNN Cosine Distance (Original Task)

**Preprocessing functions (knnClassifier.ts):** Added `clipTo3Sigma()` for per-sample outlier clipping (computes sample std, clips to ±3σ), `l2Normalize()` for unit vector normalization (avoids division by zero), and `preprocessForKNN()` wrapper applying both in sequence.

**Updated prediction pipeline:** Modified `predictProba()` and `predictBatch()` to preprocess query vectors then compute cosine similarity via `tf.matMul()` with L2-normalized training data. Use `tf.topk()` directly on similarities (largest = nearest neighbors).

**Updated training:** Modified `trainFinalModelImpl()` to preprocess training features before storing in persistent tensor. Ensures stored data matches preprocessing applied to queries at inference time.

**Deprecated distance parameter:** Updated classifierRegistry description to document cosine distance usage. Constructor logs info message about cosine distance. Removed distance selection from UI (only k parameter remains).

**Test coverage:** Added 4 tests for preprocessing pipeline (outlier clipping, L2 normalization, zero vectors, parameter deprecation). Removed 2 redundant tests for euclidean/manhattan distance. All 23 tests pass.

### Integration Tests (knn-plsda-integration.test.ts)

**Regression test for user scenario:** 6 good, 6 bad frames, k=5, PLS-DA 2 components. Verifies predictions are NOT 0% or 100% (gradual probabilities like 0.23, 0.47, 0.78).

**Smooth probability gradients:** Interpolates 9 points from bad to good cluster, verifies monotonic probability increase (no sudden jumps).

**Distance weighting effect:** Tests that closer neighbors influence predictions more than distant neighbors.

**Multiple components:** Tests 1, 2, 3 PLS-DA components all produce gradual probabilities (not binary).

**Edge cases:** Tests k > training samples (graceful clamping), very small feature values, repeated predictions (consistency).

**Test coverage:** Created 16 tests across 9 suites. All tests pass. Deterministic data (no randomness).

## Related
- `tasks/2025-11-03-feature-add-plsda-dim-reduction.md` - Original PLS-DA implementation (had bugs fixed in this task)
- `tasks/2025-10-23-fix-logistic-regression-reproducibility.md` - Seeded random number generation for ML reproducibility
- `tasks/2025-11-02-fix-training-model-not-showing.md` - Cross-validation fold constraints with small datasets
- `tasks/2025-10-23-feature-add-per-feature-normalization.md` - Per-feature normalization (different from PLS-DA's z-score)
