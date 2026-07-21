# Task 2025-10-27: Update Worker for New Model Outputs
**STATUS:** COMPLETED

## User Request

Update the worker to use new model outputs:

**RTMDet Model Changes:**
- Now exports P5 neck features: `neck_p5` [batch, 64, 10, 10]
- Still outputs: `dets` [batch, 100, 5] and `labels` [batch, 100]
- P5 features need pooling (avg/std/max over spatial dims) and scaling
- Scaling constants: C_AVG_P5 = 4.663780369242740, C_STD_P5 = 1.795139758069046, C_MAX_P5 = 0.422883943415846
- Results in 192 dimensions (64 channels × 3 pooling methods)

**RTMPose Model Changes:**
- Reduced from 6 to 4 outputs (removed MLP input and GAU per-keypoint pooling)
- Still outputs: `simcc_x` [batch, 17, 384] and `simcc_y` [batch, 17, 512]
- Now outputs PRE-SCALED features:
  - `backbone_features` [batch, 1536] - already scaled
  - `gau_features` [batch, 768] - already scaled
- No client-side scaling needed for these outputs

## Critical Discoveries

**P5 pooling reduces spatial [10, 10] to 192 dims:** 64 channels × 3 methods (avg/std/max). Each method captures different statistics (central tendency, variability, salient activations).

**Pre-scaled RTMPose outputs eliminate client-side scaling:** Model now does normalization internally, simplifying extraction code and reducing computation.

**Storage version 3 invalidates old datasets:** Feature name changes (backbone_concat → backbone_features) require clean break. No migration logic needed per user.

**Generic feature pattern maintained zero-copy transfers:** All features use Float32Array with transferables list, avoiding serialization overhead for large vectors.

## Solution

**Worker (unified-pose-worker.ts):**
- Added P5 scaling constants (L288-291)
- Implemented `extractRtmDetP5Features()` with pooling over [10, 10] grid (L619-670)
- Updated ProcessResult interface with `rtmDetP5Features` field (L156)
- Added P5 extraction from RTMDet results (L1189-1197)
- Included P5 in both "no person" and "person detected" results (L1242, L1326)
- Added P5 to transferables list for zero-copy transfer (L1372-1375)

**Feature Registry (featureRegistry.ts):**
- Updated FEATURE_TYPES with new features: backbone_features, gau_features, neck_p5 (L18-25)
- Added feature definitions with proper dimensions and metadata (L68-142)
- Marked legacy features (mlp_input_concat, gau_per_kpt) as not recommended

**Storage (storage.ts):**
- Bumped STORAGE_VERSION from 2 to 3 (L47-50)
- Added `frameRtmDetP5Key` generator and P5 save/load logic (L66, L206-212, L298-302)
- Included P5 when building frames and in deleteFrame (L340, L408)

**Type Definitions (types.ts):**
- Added `rtmDetP5Features` to InferenceResult and PostureFrame (L49-51, L106-107)

**Tests Updated:**
- TrainingConfigContext.test.tsx - Uses gau_features instead of mlp_input_concat
- featureExtractor.test.ts - Updated mock data and assertions
- featureRegistry.test.ts - Updated dimension checks

## Lessons

Pre-scaling features in model reduces client complexity. Pooling spatial features (avg/std/max) captures complementary information in compact form.

## Files Modified

- `src/workers/unified-pose-worker.ts` - P5 extraction, ProcessResult update, transferables
- `src/services/dataset/featureRegistry.ts` - New feature definitions, updated FEATURE_TYPES
- `src/services/dataset/storage.ts` - Version 3, P5 save/load logic
- `src/services/dataset/types.ts` - Added rtmDetP5Features to interfaces
- `src/contexts/TrainingConfigContext.test.tsx` - Updated to gau_features
- `src/services/ml/featureExtractor.test.ts` - Updated mock data
- `src/services/dataset/featureRegistry.test.ts` - Updated dimension checks

## Impact

**Feature dimensions:** neck_p5 (192), backbone_features (1536), gau_features (768)

**Storage:** ~200 bytes reduction per frame (essentially neutral). 100-frame dataset still ~1 MB.

**Breaking change:** STORAGE_VERSION 3 invalidates old datasets. Users must re-collect with new models.

**Performance:** Zero-copy transfers maintained for all features via transferables.

## Follow-up Fix: neck_p5 Output Name Mismatch

**Issue:** After implementation, the worker logged a warning:
```
[RTMDet] No neck_p5 output found in RTMDet model - P5 features unavailable
```

**Root Cause:** The updated RTMDet model exports the P5 output with the internal ONNX tensor name `/neck/out_convs.2/pointwise_conv/activate/Mul_output_0` instead of the alias `neck_p5`. The worker code was checking for `rtmdetResults.neck_p5`, which didn't exist.

**Solution:** Updated `src/workers/unified-pose-worker.ts`:
- Added constant for ONNX output name (line 295)
- Updated P5 extraction to use correct output name (line 1194-1196)

**Result:** P5 features now extract successfully. Warning no longer appears.

## Follow-up Fix #2: Training Error - features[0] Undefined

**Issue:** After fixing neck_p5 output name, users encountered training error:
```
[useModelTraining] Presence model training failed: can't access property "length", features[0] is undefined
```

**Root Cause:** The feature matrix contained undefined elements when training reached normalization step. Error occurred in `baseClassifier.ts:204` when `normalizePerFeature()` tried to access `features[0].length`.

**Solution:** Added comprehensive validation and diagnostic logging to catch and diagnose the issue:

1. **Feature matrix validation** (`baseClassifier.ts:295-313`):
   - Validates integrity after building feature matrix
   - Catches undefined/null rows before normalization
   - Throws clear error with sample frame IDs

2. **Improved normalization error** (`baseClassifier.ts:203-210`):
   - Enhanced validation for `features[0]`
   - Diagnostic logging for corrupted matrices
   - Detailed error messages

3. **Pre-training diagnostics** (`useModelTraining.ts:185-210`):
   - Logs frame counts and feature availability
   - Identifies frames missing rtmDetFeatures
   - Provides detailed frame structure for debugging

**Result:** Training failures now provide clear, actionable error messages showing exactly which frames are missing required features, enabling users to diagnose and fix data collection issues.

**Files Changed:**
- `src/services/ml/baseClassifier.ts` - Added validation and improved error messages
- `src/hooks/useModelTraining.ts` - Added diagnostic logging before training

## Follow-up Fix #3: Complete Migration to P5 Features (Breaking Change)

**Issue:** After fixing neck_p5 extraction, discovered training was still using old 49-dim handcrafted RTMDet features instead of new 192-dim P5 deep learned features.

**Evidence:** Training logs showed:
```
[RTMDet] Saving RTMDet features for frame: 49 dims
[FEATURE_EXTRACT] Feature types: rtmdet
[FEATURE_EXTRACT] Expected concatenated dimensions: 49
```

**Root Cause:**
- Training was hardcoded to use `FEATURE_RTMDET` (old 49-dim geometric features)
- Should have been using `FEATURE_NECK_P5` (new 192-dim deep features)
- Both features were being extracted and stored, but only old one was used for training

**Solution:** Complete removal of old features and migration to P5:

1. **Training** (`useModelTraining.ts:173`):
   - Changed: `const presenceFeatureTypes: FeatureType[] = [FEATURE_RTMDET];`
   - To: `const presenceFeatureTypes: FeatureType[] = [FEATURE_NECK_P5];`

2. **Feature Registry** (`featureRegistry.ts`):
   - Removed FEATURE_RTMDET constant and 49-dim definition
   - Kept only FEATURE_NECK_P5 (192-dim) for presence detection

3. **Worker** (`unified-pose-worker.ts`):
   - Removed `extractRtmDetFeatures()` function (~110 lines)
   - Removed extraction call and rtmDetFeatures variable
   - Updated ProcessResult interface

4. **Type Definitions** (`types.ts`):
   - Removed `rtmDetFeatures?: Float32Array;` from InferenceResult
   - Removed `rtmDetFeatures?: Float32Array;` from PostureFrame
   - Kept `rtmDetP5Features?: Float32Array;` in both

5. **Storage** (`storage.ts`):
   - Removed `frameRtmDetKey()` helper function
   - Removed rtmDetFeatures save/load logic
   - Removed from frame deletion logic

6. **Feature Extractor** (`featureExtractor.ts`):
   - Removed special case handling for FEATURE_RTMDET
   - Streamlined to use registry's extract function consistently

7. **Validation** (`schemas.ts`, `guards.ts`):
   - Renamed `RtmDetFeaturesSchema` to `RtmDetP5FeaturesSchema` (192-dim)
   - Updated all schema references
   - Renamed validation guard functions

8. **Components**:
   - Updated useFrameSampler, RTMW3DCameraWeb, app/index.tsx
   - Removed rtmDetFeatures references, kept rtmDetP5Features

9. **Tests** (20+ files):
   - Removed FEATURE_RTMDET imports
   - Updated mocks: `new Float32Array(49)` → `new Float32Array(192)`
   - Updated assertions and expectations

**Result:**
- Presence training now uses 192-dim deep learned P5 features (expected to perform better)
- ~150 lines of code removed (extraction function + supporting code)
- Simpler, cleaner codebase with single presence feature type
- All tests passing (1148 tests)

**Breaking Change:** Users must re-collect datasets with new features. However, STORAGE_VERSION 3 already required dataset recollection, so no additional user impact.

**Files Changed:**
- `src/hooks/useModelTraining.ts` - Switch to FEATURE_NECK_P5
- `src/services/dataset/featureRegistry.ts` - Remove old feature
- `src/workers/unified-pose-worker.ts` - Remove extraction function
- `src/services/dataset/types.ts` - Remove rtmDetFeatures fields
- `src/services/dataset/storage.ts` - Remove save/load logic
- `src/services/ml/featureExtractor.ts` - Streamline extraction
- `src/services/validation/schemas.ts` - Update validation
- `src/services/validation/guards.ts` - Update guards
- Plus 20+ test files updated
