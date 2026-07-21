# Task 2025-11-03: Store Only Raw Features, Compute Pooled On-Demand

**STATUS:** COMPLETED

## User Request

Rework to store only raw features and calculate pooled when needed for inference or training.

## Critical Discoveries (Non-Obvious)

**1. TensorFlow.js pooling requires explicit reshaping:**
Can't directly pool flat arrays - must reshape to [batch, channels, height, width] first. Backbone: [24576] → [1, 512, 8, 6] → pool → [512]. GAU: [4352] → [1, 17, 256] → pool → [256].

**2. Layer normalization needed after pooling:**
Pooling alone produces unstable feature distributions. Applied layer norm (mean centering + variance scaling) to match training expectations. Formula: `(x - mean) / sqrt(variance + epsilon)`.

**3. Computed features need zero storage cost:**
Feature registry uses `storageCost` metadata for quota tracking. Marking pooled features with `storageCost: 0` prevents double-counting (raw features already tracked). UI automatically excludes zero-cost features from storage estimates.

**4. Mock utilities must filter computed features:**
Test mocks were creating ALL feature types, including computed ones. This caused test failures when code tried to re-compute from non-existent raw features. Solution: Filter `FEATURE_TYPES.filter(type => !FEATURE_REGISTRY[type].computed)` in `createMockFeatures()`.

**5. Unused estimatePose() function contained inconsistent code:**
Dead helper function still stored pooled features despite refactor. No production impact (never called), but violated architecture. Removed pooled storage to maintain consistency.

## Solution

**Architecture change:** Eliminated redundant storage of pooled features (backbone_features [512], gau_features [256]) by storing only raw spatial features and computing pooled on-demand. Saves ~3 KB per frame (768 dims × 4 bytes).

**Feature registry (featureRegistry.ts):** Added TensorFlow.js pooling functions (layerNormTF, computeBackbonePooled, computeGAUPooled). Marked pooled features as `computed: true` with `storageCost: 0`. Extract functions compute from raw features using reshape → avg pool → layer norm pipeline.

**Worker (unified-pose-worker.ts):** Main `processFrame()` stores only raw features (FEATURE_BACKBONE_RAW [24576], FEATURE_GAU_RAW [4352]). Removed pooled feature storage. Fixed dead `estimatePose()` helper to match architecture.

**Storage (storage.ts):** Bumped STORAGE_VERSION to 5 with comment explaining removal of stored pooled features.

**Tests:** Updated `createMockFeatures()` to filter computed features. Updated 3 test files (featureRegistry, featureExtractors, featureExtractor) to verify computed vs stored behavior. All 78 feature tests passing.

**Performance:** Computation cost <1ms per feature type (TensorFlow.js accelerated). Storage savings: 3,072 bytes per frame. No in-memory caching (user decision - computation fast enough).

## Related

- tasks/2025-11-03-feature-rtmpose-raw-spatial-features.md (added raw features in v4)
- tasks/2025-10-25-refactor-generic-feature-system.md (generic feature architecture enabling this refactor)
- tasks/2025-10-27-feature-update-worker-model-outputs.md (worker model outputs)
- tasks/2025-10-23-feature-extraction-alignment.md (feature extraction patterns)
