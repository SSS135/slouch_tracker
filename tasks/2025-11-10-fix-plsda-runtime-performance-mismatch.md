# Task 2025-11-10: Fix Cross-Validation Data Leakage Causing Biased Metrics
**STATUS:** COMPLETED

## User Request
pls-da shows way better f1 score than no reduction, but in practice works not any better at runtime. find out why.

## General Description
Cross-validation metrics (F1 scores) are optimistically biased because preprocessing (normalization + dimensionality reduction) is fit on the full dataset before CV split. This causes test fold data to "leak" into preprocessing parameters, especially affecting supervised methods like PLS-DA which uses labels to find discriminative directions. CV shows inflated F1 scores while runtime performance with truly unseen data is much lower.

## Root Cause Analysis

**Current Buggy Flow** (`model.ts:96-104`):
```
1. Fit preprocessing on FULL dataset (including future test folds)
   - FeatureExtractor.fit(allFeatures, allLabels)
   - PLS-DA sees ALL labels before CV split
2. Transform ALL samples with fitted parameters
3. Split transformed features into CV folds
4. Evaluate on test folds (already "seen" during preprocessing)
```

**Impact by Method:**
- **PLS-DA/Linear NCA**: HIGH impact (supervised, uses labels from test folds)
- **Per-feature normalization**: Medium impact (test fold statistics leak into mean/std)
- **Random Projection**: Low impact (unsupervised, no labels)
- **Layer norm / L2 only**: No impact (no fitted parameters)

**Why PLS-DA shows inflated CV metrics:**
- PLS-DA fits on all samples with all labels before CV
- Learns to perfectly separate those specific samples (including test folds)
- CV test folds were "seen" during PLS-DA fitting
- Runtime data is truly unseen → performance drops

**Why "no reduction" is more honest:**
- Only normalization leakage (lower impact)
- Less opportunity to overfit to leaked information
- CV metrics closer to true generalization

## Solution Implemented

Extracted cross-validation from Model class into separate scikit-learn-style utility (`evaluation.ts`). This fixes data leakage by fitting FeatureExtractor independently per CV fold.

**Files Created:**
1. `src/services/ml/evaluation.ts` (~200 lines)
   - `crossValidate()` function creates fresh Model instances per fold
   - Per-fold FeatureExtractor fitting prevents test data leakage
   - `CVMetrics` interface (same structure as before)
   - `getEmptyMetrics()` helper

2. `src/services/ml/__tests__/evaluation.test.ts` (~350 lines)
   - 13 comprehensive tests for crossValidate()
   - Tests different classifiers, feature types, normalization modes
   - Tests data leakage prevention
   - All tests passing

**Files Modified:**
1. `src/services/ml/model.ts` (~150 lines removed)
   - Removed all CV methods
   - `fit()` simplified: no cvFolds param, no return value, throws on error
   - Single responsibility: just trains the model

2. `src/workers/training-worker.ts` (~30 lines modified)
   - Calls `crossValidate()` before `model.fit()`
   - Constructs TrainingResult from CV metrics
   - No breaking changes to worker protocol

3. `src/services/ml/__tests__/model.test.ts` (updated)
   - Removed CV tests (moved to evaluation.test.ts)
   - Updated fit() tests for new signature
   - 12 tests covering Model functionality
   - All tests passing

**Test Results:**
- 25 tests passing (13 evaluation + 12 model)
- No breaking changes to UI or storage

## Critical Discoveries

**1. CV Data Leakage (HIGH IMPACT)**
Preprocessing fitted on full dataset before CV split caused test fold information to leak:
- Normalization parameters (mean/std) included test fold statistics
- PLS-DA/Linear NCA used test fold labels to find discriminative directions
- CV metrics optimistically biased (overestimated generalization)
- Runtime performance with truly unseen data was lower

**2. Supervised Dimensionality Reduction Most Affected**
- PLS-DA: HIGH impact (supervised, uses labels)
- Linear NCA: HIGH impact (supervised, uses labels)
- Random Projection: LOW impact (unsupervised)
- Per-feature normalization: MEDIUM impact (test fold statistics leak)

**3. Scikit-learn Style API Better Architecture**
Separating CV from Model improves:
- Code organization (separation of concerns)
- Testability (independent unit tests)
- Flexibility (CV is optional)
- Maintainability (single-responsibility principle)

## Related
- `tasks/2025-11-08-refactor-unified-raw-features-pipeline.md` - Recent refactoring that introduced FeatureExtractor pipeline
- `tasks/2025-11-03-feature-add-plsda-dim-reduction.md` - PLS-DA implementation
- `tasks/2025-11-06-feature-linear-nca-reduction.md` - Linear NCA (also affected by same issue)
- `tasks/0007-refactor-cv-strategy.md` - Current CV implementation (3-fold stratified)
