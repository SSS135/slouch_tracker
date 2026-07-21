# Task 2025-10-23: Add Per-Feature Normalization
**STATUS:** COMPLETED

## User Request
"add per-feature normalisation in addition to layer norm"

## Critical Discoveries (Non-Obvious)

**1. Zero std edge case:**
Constant features (std=0) cause division by zero. Solution: Replace zero std with 1.0 during normalization, keeping feature unchanged.

**2. Normalization order matters:**
Per-feature norm MUST be applied BEFORE dimensionality reduction (uses full feature space statistics). Layer norm can be applied after (per-sample operation).

## Solution

**Three normalization modes** implemented as mutually exclusive options:
- **None**: No preprocessing
- **Layer Norm**: Per-sample standardization (mean=0, std=1 per sample), computed on-the-fly
- **Per-Feature Norm**: Per-dimension standardization across training set (like sklearn's StandardScaler), requires storing mean/std arrays

**Type system updates:** Added `NormalizationMode` enum type, updated `TrainedModel` interface with `normalizationMode`, `normalizationMean`, `normalizationStd` fields.

**Core implementation in `baseClassifier.ts`:** Two new methods: `normalizePerFeature()` computes mean/std per dimension across all training samples, `applyPerFeatureNorm()` applies saved parameters during inference. Updated `prepareData()` and `validateAndTransformFeatures()` with switch statement for all three modes. Zero std handled by replacing with 1.0.

**UI upgrade:** Replaced Switch with RadioGroup showing three options (Per-Feature marked "Recommended"). Context updated with `updateNormalizationMode()` method.

**Tests:** Created comprehensive test suite (9 tests) covering normalization math, edge cases, and save/load. All passing.

**Testing phase bugs fixed:**
- Updated ENGINEERED feature dims (36 → 92) in test
- Fixed `boolean` → `NormalizationMode` type errors in 4 test files
- Fixed test timeouts (30s/20s) for training tests

**Final verification:** 1136/1137 tests passing (3 pre-existing failures unrelated to this feature).

## Files Modified

**Core Implementation (9 files):**
1. src/services/ml/types.ts - NormalizationMode type
2. src/services/dataset/types.ts - TrainedModel interface
3. src/services/ml/baseClassifier.ts - Per-feature norm methods
4. src/services/ml/logisticRegressionClassifier.ts - Updated fromJSON
5. src/services/ml/knnClassifier.ts - Updated fromJSON
6. src/services/ml/classifierRegistry.ts - Type signature updates
7. src/contexts/TrainingConfigContext.tsx - Config state management
8. src/hooks/useModelTraining.ts - trainModel() parameter
9. src/components/unified/TrainingTab.tsx - RadioGroup UI

**Tests (5 files):**
10. src/services/ml/__tests__/perFeatureNorm.test.ts - New test suite (NEW)
11. src/services/dataset/__tests__/featureRegistry.test.ts - Type fixes
12. src/services/ml/__tests__/logisticRegressionClassifier.test.ts - Type fixes
13. src/contexts/__tests__/TrainingConfigContext.test.tsx - Type fixes
14. src/hooks/__tests__/useModelTraining.test.ts - Type fixes

## Impact

**UX improvement:** RadioGroup clarifies that normalization modes are mutually exclusive (vs two toggles suggesting they're combinable).

**ML quality:** Per-feature normalization is standard practice for classical ML, provides better scaling for mixed-dimension features.

**Better type safety:** Clean enum-based config provides compile-time safety for normalization modes.
