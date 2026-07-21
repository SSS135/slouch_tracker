# Task 2025-11-03: Add PLS-DA Dimensionality Reduction
**STATUS:** COMPLETED

## User Request
also add PLS-DA/OPLS, 1–2 components. plan.
[After clarification: just pls-da. is it a good idea to add it to dim reduction so we can select either off / random proj / pls-da. also 1-5 pls-da dim chooser like for random projection.]

## Solution Summary

Successfully added PLS-DA (Partial Least Squares Discriminant Analysis) as a supervised dimensionality reduction option. PLS-DA is now available alongside Random Projection and None in the training UI, supporting 1-5 components and working with all classifiers (KNN, Logistic Regression).

**Key Implementation:**
- NIPALS algorithm for supervised component extraction
- Supervised dimensionality reduction (passes labels during fit)
- UI with conditional component selectors (1-5 for PLS-DA, 64/256/1024 for Random Projection)
- Full serialization support
- Comprehensive test coverage (17 tests, all passing)

## Implementation Details

### 1. Type System Updates
- Added `'pls-da'` to `DimensionalityReductionConfig.method` union type
- Created `SerializedPLSDA` interface with projection matrix, means, and class labels
- Added `PLSDATransformerWrapper` type for serialization
- Added type guard `isSerializedPLSDA()` for runtime validation

### 2. PLS-DA Transformer (plsda.ts)
- Implemented NIPALS (Non-linear Iterative Partial Least Squares) algorithm
- Supervised fitting using label information to find discriminative features
- Stores projection matrix (nComponents × nFeatures) and feature means
- TensorFlow.js tensor optimization for fast matrix operations
- Supports 1-5 components (validated at constructor)
- Convergence criteria: max 100 iterations, tolerance 1e-6

### 3. Base Classifier Modifications
- Updated `applyDimReduction()` to accept `yTrain` labels parameter
- Added PLS-DA case handling in both CV and final training
- Updated `crossValidate()` to pass labels to dim reduction
- Updated `trainFinalModel()` to handle supervised fitting
- Union type updated: `RandomProjectionTransformer | PLSDATransformer | null`

### 4. Classifier Serialization Updates
**Logistic Regression:**
- Added PLSDATransformer import
- Updated `toJSON()` to serialize PLS-DA transformers
- Updated `fromJSON()` to deserialize PLS-DA transformers

**KNN:**
- Added PLSDATransformer import
- Updated `toJSON()` to serialize PLS-DA transformers
- Updated `fromJSON()` to deserialize PLS-DA transformers

### 5. UI Updates (TrainingTab.tsx)
- Added PLS-DA option to dimensionality reduction radio group with "Recommended" badge
- Implemented conditional component selectors:
  - PLS-DA: 1-5 components with help text "2-3 components recommended"
  - Random Projection: 64/256/1024 dimensions (unchanged)
- Updated handlers to manage different component ranges
- Default: 2 components for PLS-DA, 256 for Random Projection

### 6. Testing
Created comprehensive test suite (17 tests, all passing):
- Constructor validation (valid/invalid component counts)
- Fit/transform correctness on separable data
- Input validation (empty dataset, mismatched lengths, single class, insufficient samples)
- Dimension mismatch detection
- Serialization/deserialization round-trip
- Batch transformation
- Memory management (tensor disposal)
- Component range (1 and 5 components)

### 7. Type Fixes
Fixed TypeScript type errors with `.squeeze()` by adding explicit `as tf.Tensor1D` type assertions in three locations (lines 344, 362, 391).

## Architecture Change

**Before:**
```
Features → DimReduction (unsupervised) → Classifier → Prediction
```

**After:**
```
Features + Labels → DimReduction (supervised or unsupervised) → Classifier → Prediction
```

## Files Created
- `src/services/ml/plsda.ts` (410 lines)
- `src/services/ml/__tests__/plsda.test.ts` (315 lines)

## Files Modified
- `src/services/dataset/types.ts` - Added 'pls-da' to method union, updated TrainingResult
- `src/services/ml/types.ts` - Added SerializedPLSDA, PLSDATransformerWrapper, type guard
- `src/services/ml/baseClassifier.ts` - Added supervised dim reduction support
- `src/services/ml/logisticRegressionClassifier.ts` - Added PLS-DA serialization
- `src/services/ml/knnClassifier.ts` - Added PLS-DA serialization
- `src/components/unified/TrainingTab.tsx` - Added PLS-DA UI with conditional selectors

## Backward Compatibility
- ✅ No migration needed - existing models with 'random_projection' or 'none' continue to work
- ✅ PLS-DA is opt-in via UI selection
- ✅ Default remains 'random_projection' for existing users

## Performance
- **Training**: ~100-500ms (iterative NIPALS vs ~10ms Random Projection)
- **Inference**: ~1ms (same as Random Projection)
- **Memory**: Similar to Random Projection (stores projection matrix, not training data)

## Usage
1. Navigate to Training tab
2. Select PLS-DA from Dimensionality Reduction methods
3. Choose 2-3 components (recommended for posture detection)
4. Train with any classifier (KNN or Logistic Regression)
5. Model automatically serializes PLS-DA state for inference
