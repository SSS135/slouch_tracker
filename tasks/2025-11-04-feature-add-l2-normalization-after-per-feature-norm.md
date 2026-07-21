# Task 2025-11-04: Add L2 Normalization After Per-Feature Normalization

**STATUS:** COMPLETED

## User Request
Add L2 normalization after per-feature normalization, always enabled when per-feature normalization is enabled. Refactor batch normalization functions to use batched TensorFlow.js operations to avoid repeated Float32Array ↔ Tensor conversions.

## Critical Discoveries (Non-Obvious)

**1. TensorFlow.js tensor type inference issue:**
After `tf.div()`, TypeScript loses the `Tensor2D` type. Need explicit cast: `as tf.Tensor2D` when passing result to methods expecting 2D tensors.

**2. Float32Array vs Float64 precision in tests:**
`toBeCloseTo()` default precision (10 digits) fails for Float32Array. Must use precision=7 for unit tests working with Float32Array (32-bit precision limit).

**3. Batched TensorFlow.js performance optimization:**
Converting Float32Array ↔ Tensor per-sample creates N conversions. Batched approach: single conversion for entire batch → ~10-100× speedup. Critical for training with large datasets (100-1000+ samples).

**4. Training vs inference strategy:**
Training benefits from batched TensorFlow.js (amortizes conversion cost). Inference on single samples faster with plain Float32Array (avoids TensorFlow.js overhead). Different implementations for same operation.

**5. L2 normalization is parameter-free:**
Unlike per-feature normalization (stores mean/std), L2 norm has no parameters. Only requires applying same operation at inference. Simplifies model serialization.

## Solution

**Implementation** (4 files modified):

**1. Added L2 normalization utilities** (`layerNorm.ts`):
- `applyL2Norm()`: Single-sample L2 norm using Float32Array math
- Handles zero-norm edge case (< 1e-10 returns original vector)

**2. Added batched L2 norm** (`baseClassifier.ts:applyL2NormBatch`):
- TensorFlow.js implementation using `tf.norm()` for row-wise norms
- Uses `tf.where()` to handle zero norms conditionally
- Returns normalized 2D tensor

**3. Refactored per-feature normalization** (`baseClassifier.ts:normalizePerFeature`):
- Replaced per-sample loops with batched TensorFlow.js operations
- Single Float32Array → Tensor conversion at start
- Compute mean/std using `tf.mean()` and `tf.moments()`
- Apply per-feature norm then L2 norm in tensor form
- Single Tensor → Float32Array conversion at end

**4. Updated inference** (`baseClassifier.ts:applyPerFeatureNorm`):
- Added L2 norm step after per-feature standardization
- Uses Float32Array-based `applyL2Norm()` for efficiency

**5. Refactored layer normalization** (`layerNorm.ts:applyLayerNormBatch`):
- Replaced `map()` per-sample processing with batched TensorFlow.js
- Computes mean/variance per sample (axis=1) in single operation

**6. Updated UI** (`TrainingTab.tsx`):
- Label: "Per-Feature Normalization" → "Per-Feature + L2 Normalization"
- Help text updated to reflect automatic L2 norm

**7. Documentation** (`specs.md`):
- Added "Feature Normalization" section explaining all three modes
- Documented batched TensorFlow.js performance optimization
- Explained when to use per-feature + L2 normalization

**Tests** (`layerNorm.test.ts` - 11 new tests):
- Unit length verification (L2 norm = 1.0)
- Known vector test ([3,4] → [0.6, 0.8])
- Edge cases: zero vectors, near-zero vectors, single element
- Sign preservation and scale invariance
- Large vectors (1536 dims - backbone features)
- Immutability verification

## Related
- `tasks/2025-10-23-feature-add-per-feature-normalization.md` - Original per-feature norm implementation (foundation for this task)
- `tasks/2025-11-03-feature-add-plsda-classifier.md` - Contains reference L2 norm implementation
