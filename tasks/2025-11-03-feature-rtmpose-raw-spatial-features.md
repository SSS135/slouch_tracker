# Task 2025-11-03: RTMPose Raw Spatial Features Integration

**STATUS:** COMPLETED

## User Request

I've updated rtmpose onnx model in project, docs above. Adjust for changed outputs. Add raw backbone and gau features to training, implement pooling and scaling how it was before (similar to rtmdet) to restore pooled and scaled features.

## General Description

RTMPose ONNX model (renamed to `rtmpose-s.onnx`) now exports raw spatial features instead of pre-pooled features:
- **backbone_features**: [B, 512, 8, 6] raw spatial (was [B, 1536] pooled)
- **gau_features**: [B, 17, 256] raw per-keypoint (was [B, 768] pooled)

Implement double layer normalization preprocessing:
1. Flatten raw outputs → Layer norm → Raw features
2. Reshape normalized → Avg pool → Layer norm → Pooled features

Support both raw (for advanced spatial modeling) and pooled (for general use) variants.

## Action Plan

1. **Update worker preprocessing** (unified-pose-worker.ts):
   - Implement double layer normalization pipeline using TensorFlow.js
   - Extract both raw and pooled features from new model outputs
   - Update model path to `rtmpose-s.onnx`

2. **Update feature registry** (featureRegistry.ts):
   - Add 4 feature types: backbone_features (512 dims), backbone_features_raw (24576 dims), gau_features (256 dims), gau_features_raw (4352 dims)
   - Mark raw features as advanced (`recommended: false`)

3. **Update types and storage**:
   - Update PostureFrame comments
   - Bump STORAGE_VERSION to 4

4. **Update UI** (TrainingTab.tsx):
   - Remove storage warnings for raw features (user request)

5. **Tests** (via unit-test-engineer):
   - Test double normalization correctness
   - Test dimensions match registry
   - Update mock data

## Rationale

**Double normalization approach:**
- First normalization: Standardizes raw spatial features before pooling
- Second normalization: Standardizes pooled statistics for consistent training
- No pre-calculated constants needed (adaptive to data)

**Supporting both raw and pooled:**
- Raw: Preserves spatial structure for CNNs/attention models (24KB/frame)
- Pooled: General purpose, compatible with existing approach (2KB/frame)
- Users choose based on dataset size and model requirements

## What Was Done

**Files Modified:**

1. **unified-pose-worker.ts** (src/workers/):
   - Added TensorFlow.js import for layer normalization
   - Created `layerNormTF()` helper: `(x - mean) / sqrt(variance + epsilon)`
   - Created `extractRTMPoseBackboneFeatures()`: [512, 8, 6] → raw [24576] + pooled [512]
   - Created `extractRTMPoseGAUFeatures()`: [17, 256] → raw [4352] + pooled [256]
   - Updated `processFrame()` to call extraction functions and store 5 feature types
   - Added feature constant imports (FEATURE_BACKBONE_RAW, FEATURE_GAU_RAW, etc.)
   - Updated model path to `rtmpose-s.onnx`

2. **featureRegistry.ts** (src/services/dataset/):
   - Added 5 constants: FEATURE_BACKBONE, FEATURE_BACKBONE_RAW, FEATURE_GAU, FEATURE_GAU_RAW, FEATURE_NECK_P5
   - Updated FEATURE_TYPES array with all 5 types
   - Updated registry with new dimensions and metadata:
     - backbone_features: 512 dims (recommended, pooled)
     - backbone_features_raw: 24,576 dims (advanced, raw spatial)
     - gau_features: 256 dims (recommended, pooled)
     - gau_features_raw: 4,352 dims (advanced, raw per-keypoint)
     - neck_p5: 192 dims (presence detection, auto-configured)

3. **storage.ts** (src/services/dataset/):
   - Bumped STORAGE_VERSION to 4
   - Added migration notes for raw spatial features

4. **PostureCamera.tsx, useWebWorkerInference.ts** (src/components/, src/hooks/):
   - Updated rtmw3dPath to `rtmpose-s.onnx`

5. **TrainingTab.tsx** (src/components/unified/):
   - Removed storage warnings (user request)

**Bugs Fixed:**

1. **Dimension mismatch error (28,928 vs 768)**:
   - Root cause: Worker was storing raw ONNX outputs instead of extracted features
   - Fix: Replaced direct output storage with extraction function calls in `processFrame()`
   - Result: Frames now save with correct 5 feature types and proper dimensions

2. **Missing feature constants**:
   - Worker couldn't find FEATURE_BACKBONE_RAW, FEATURE_GAU_RAW
   - Fix: Added missing imports to unified-pose-worker.ts

**Tests Updated (via unit-test-engineer):**

Updated 8 test files with new dimensions (1536→512, 768→256):
- featureExtractor.test.ts
- useDatasetOperations.test.tsx
- useModelTraining.test.ts
- operations.test.ts
- storage.test.ts
- guards.test.ts
- schemas.test.ts
- usePostureClassifier.test.ts

All 246 feature/dimension tests passing ✅

## Related

- tasks/2025-10-27-feature-update-worker-model-outputs.md
- tasks/2025-10-28-fix-neck-p5-extraction.md
- tasks/2025-10-25-refactor-generic-feature-system.md
- tasks/2025-10-24-refactor-replace-rtmw3d-with-rtmpose-s.md
