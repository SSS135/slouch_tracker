# Task 2025-11-04: Add Max and Std Pooling for Backbone and GAU Features

**STATUS:** COMPLETED

## User Request
Right now we do only avg pooling of raw features. I want you to add another two types - max pooling and std pooling. For both backbone and gau features. So it will add 4 new feature types to enable.

## Critical Discoveries

**1. Epsilon required for std pooling numerical stability:**
Std pooling uses `tf.sqrt(tf.add(variance, 1e-5))` to prevent numerical issues with zero variance (constant inputs). This epsilon matches layer normalization implementation for consistency.

**2. Population std vs sample std:**
Used population std (tf.moments default) rather than sample std (N-1 correction). Consistent with feature extraction practices where unbiased estimation is unnecessary.

**3. Feature registry auto-updates validation schemas:**
Adding new feature types to `FEATURE_TYPES` constant automatically updates Zod schemas in `validation/schemas.ts` - no manual schema changes needed.

**4. Test quality focus on behavior over implementation:**
Tests verify output dimensions, layer normalization properties (mean≈0, variance≈1), and edge cases (null inputs, constant values) rather than internal TensorFlow.js implementation details.

**5. On-demand computation pattern scales well:**
Computing pooled features on-demand (storageCost=0) adds negligible overhead (~1ms per feature type) while saving significant storage (2KB+ per frame for 4 new features).

## Solution

Added 4 new on-demand computed feature types using max/std pooling for backbone (512 dims) and GAU (256 dims) features. All features compute from existing raw features with zero storage overhead.

**Implementation:**
- Added 4 pooling functions in `rtmposeFeatures.ts`: `poolBackboneFeaturesMax/Std`, `poolGAUFeaturesMax/Std`
- Each function: Reshape raw → Apply pooling (max or std with epsilon) → Layer normalize → Return Float32Array
- All wrapped in `tf.tidy()` for automatic memory management
- Max pooling uses `tf.max()` over spatial/keypoint dimensions
- Std pooling uses `tf.moments()` + `tf.sqrt(variance + 1e-5)` for numerical stability

**Registry:**
- Added 4 constants to `featureRegistry.ts`: `FEATURE_BACKBONE_MAX/STD`, `FEATURE_GAU_MAX/STD`
- Updated `FEATURE_TYPES` array from 5 to 9 feature types
- Registered 4 features with `computed: true, storageCost: 0, recommended: true`
- Extract functions compute on-demand from raw features with null handling

**Testing:**
- Created `rtmposeFeatures.test.ts` with 20 tests (5 per pooling function)
- Tests verify: dimensions, layer normalization, pooling behavior, constant inputs, memory management
- Updated `featureRegistry.test.ts` from 20 to 39 tests
- Tests verify: feature count, dimensions, storage costs, computed flags, extract functions
- All 59 tests passing, no regressions

**Documentation:**
- Updated `specs.md` "Available Feature Types" section with 9 feature types
- Added usage guidance: max pooling for salient features, std pooling for variation detection
- Corrected outdated dimensions (was 1536/768, now 512/256)

**User-facing changes:**
- Training tab dropdown shows 9 feature types (was 5)
- New options: "Backbone Features (Max Pool)", "Backbone Features (Std Pool)", "GAU Features (Max Pool)", "GAU Features (Std Pool)"
- All marked as recommended in UI
- Zero storage impact, backward compatible with existing datasets

## Related
- `tasks/2025-11-03-refactor-computed-pooled-features.md` - Established on-demand pooling pattern
- `tasks/2025-11-03-feature-rtmpose-raw-spatial-features.md` - Added raw features for pooling
- `tasks/2025-10-28-fix-neck-p5-extraction.md` - RTMDet P5 multi-pooling reference (avg/std/max)
