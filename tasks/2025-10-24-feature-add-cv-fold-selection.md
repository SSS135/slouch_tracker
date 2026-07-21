# Task 2025-10-24: Add CV Fold Selection to Developer Settings
**STATUS:** COMPLETED

## User Request
Add N fold selection in developer settings, right now we use only 3 folds, make 5 default, fold slider range 2-20.

## Critical Discoveries

**Auto-limiting prevents training failures:**
Dataset with 10 samples + user selects 20 folds → auto-limited to 10 folds. Prevents `createStratifiedKFold` errors when `k > numSamples`.

**Data flow spans 8 files:**
RuntimeTab → useCameraSettings → TrainingTab → useModelTraining → classifierRegistry → baseClassifier → crossValidation. CV folds must thread through entire pipeline.

## Solution

Added configurable cross-validation fold count to Developer Settings with automatic dataset-size-based limiting.

**UI Implementation:**
- Slider in Developer Settings section (collapsed by default)
- Range 2-20, step 1, default 5
- Help text explains auto-limiting behavior
- Persisted via localStorage (useCameraSettings hook)

**Auto-Limiting Logic (baseClassifier.ts:503-515):**
```typescript
const actualFolds = Math.min(this.cvFolds, features.length);
if (actualFolds < this.cvFolds) {
  logger.warn('training', `Fold count limited from ${this.cvFolds} to ${actualFolds}`);
}
logger.info('training', `Using ${actualFolds}-fold cross-validation`);
```

**Pipeline Integration:**
- useCameraSettings stores `cvFolds: 5` (default)
- TrainingTab passes `settings.cvFolds` to `trainModel()`
- useModelTraining logs user-selected count, passes to classifier factory
- classifierRegistry accepts `cvFolds` parameter in all factory functions
- baseClassifier constructor stores and auto-limits before calling `createStratifiedKFold`

## Lessons

**Default 5 folds** is industry standard (better variance reduction than 3, less computation than 10).

**Range 2-20** chosen: minimum 2 for valid CV, maximum 20 prevents excessive computation while allowing near-LOOCV for small datasets.

**Console logging** provides transparency when auto-limiting occurs (helps users understand why fold count differs from setting).

## Files Modified

1. `src/hooks/useCameraSettings.ts` - Added `cvFolds: 5` to interface and defaults
2. `src/components/unified/RuntimeTab.tsx` - Added slider UI in Developer Settings
3. `src/hooks/useModelTraining.ts` - Added cvFolds parameter and logging
4. `src/services/ml/classifierRegistry.ts` - Updated factory functions to accept cvFolds
5. `src/services/ml/baseClassifier.ts` - Added cvFolds constructor param, auto-limiting, logging
6. `src/services/ml/logisticRegressionClassifier.ts` - Added cvFolds to constructor
7. `src/services/ml/knnClassifier.ts` - Added cvFolds to constructor
8. `src/components/unified/TrainingTab.tsx` - Added useCameraSettings hook integration

## Impact

Users can now adjust CV fold count based on dataset size and desired validation rigor. Auto-limiting prevents training failures from excessive fold counts. Console logs provide feedback when auto-limiting occurs.
