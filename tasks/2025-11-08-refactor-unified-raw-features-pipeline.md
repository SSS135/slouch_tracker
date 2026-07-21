# Task 2025-11-08: Refactor Pipeline with Unified Raw Features Architecture

**STATUS:** COMPLETE

## User Request

Redo training/inference/data capture pipeline structure. Desired pipeline:

Data Loading and Capture produce identical Raw Features. Raw features is `dict[str, FloatArray]` features and sometimes 'good' | 'bad' | 'away' class, possibly with metadata irrelevant to workers.

Raw Features passed as-is to Model via predict or fit methods accepting batched Raw Features.

In predict/fit methods they utilize Feature Extractor first to process features, then call predict/fit in Classifier.

Model consists of Extractor and Classifier.

## Critical Discoveries

**1. Manual serialization simpler than decorators:**
Explicit `toJSON()`/`fromJSON()` methods with Zod validation provide type safety, debuggability, and compile-time checks. No metaprogramming magic, clear stack traces. Net -170 lines vs decorator system.

**2. Zod validation at boundaries prevents corruption:**
IndexedDB can store anything. Validating with Zod schemas during load/save catches corrupt data early with clear error messages. Found issues: missing fields, type mismatches, stale schemas.

**3. FeatureExtractor must fit before transform:**
Normalization (per-feature mode) and dimensionality reduction (PLS-DA, Random Projection) require fitting on training data to compute parameters (mean, std, projection matrices). Transform uses fitted params. Attempting transform before fit throws clear error.

**4. Nested serialization requires composition pattern:**
Model serializes both FeatureExtractor and Classifier states. Each component handles own serialization via `toJSON()`, Model orchestrates composition. Reconstruction via `fromJSON()` rebuilds component graph with proper validation.

**5. Clean break faster than migration:**
Force dataset deletion instead of backward compatibility. No dual code paths, no migration logic. Implementation: 7-10 days vs 11-15 days with migration. Users retrain models anyway during development.

## Solution

Restructured entire ML pipeline into 3-layer architecture:

**1. Unified Features (Phase 1)**
- Single `features: Record<string, Float32Array>` dictionary replaces split `presenceFeatures`/`postureFeatures`
- Both `InferenceResult` and `PostureFrame` extend `FeatureContainer` interface
- Worker merges RTMDet and RTMPose features into unified dict
- Storage schema v5→v6, old data detection with clear deletion error

**2. FeatureExtractor Class (Phase 2)**
- Extracts selected features from raw dict, concatenates, normalizes (per-feature/layer/none+L2), applies dim reduction (Random Projection/PLS-DA/Linear NCA/none)
- API: `fit(rawFeatures[], labels[])`, `transform(rawFeatures)`, `transformBatch()`
- Manual serialization with `SerializedFeatureExtractorSchema` Zod validation
- Proper memory management with `dispose()`

**3. Simplified Classifiers (Phase 3)**
- Removed preprocessing from KNN, LogisticRegression, SVM
- Accept pre-processed features only: `train(features[], labels[])`
- Manual serialization per classifier with Zod schemas
- Pure ML algorithm focus, no feature engineering

**4. Model Composition (Phase 4)**
- Model = FeatureExtractor + Classifier composition
- API: `fit(rawFeatures[], labels[])`, `predict(rawFeatures)`
- Nested serialization with `SerializedModelSchema`
- IndexedDB stores serialized Model directly (no conversion)

**5. Pipeline Integration (Phase 5)**
- Training worker creates Model, calls `fit()`, serializes to IndexedDB
- Inference worker loads Model via `fromJSON()`, calls `predict()` with raw features
- Dual model training (posture + presence) with unified features
- UI shows model metadata (extractor + classifier config)

**6. Cleanup (Phase 6)**
- Removed all `presenceFeatures`/`postureFeatures` references
- Removed old extraction logic from baseClassifier
- No backward compatibility code

## Related

- `tasks/2025-10-25-refactor-generic-feature-system.md` - Previous feature system refactor
- `tasks/2025-11-03-refactor-computed-pooled-features.md` - On-demand feature computation
- `tasks/2025-11-06-refactor-classifier-hierarchy.md` - Classifier code deduplication
- `tasks/2025-11-07-refactor-critical-architecture-fixes.md` - Feature cloning, worker validation patterns
