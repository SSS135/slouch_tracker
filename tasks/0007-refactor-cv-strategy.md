# Task 0007: Switch to 3-Fold Cross-Validation

**STATUS:** ✅ COMPLETED

## User Request
use 3-fold cv instead of loo or whatever we use now

## General Description
The system had both LOO CV and K-fold CV logic, switching between them based on dataset size. This task simplified the cross-validation strategy by removing LOO CV entirely and using 3-fold stratified CV for all dataset sizes.

## Action Plan
1. Update CV configuration to always use 3-fold CV
2. Remove LOO CV implementation and conditional logic
3. Simplify baseClassifier CV strategy selection
4. Update tests to reflect new CV behavior
5. Fix KNN tests to work with smaller training sets

## Rationale
**Why this approach:**
- LOO CV was overly complex and slow for small datasets
- 3-fold CV provides sufficient validation even for small datasets (9-10 frames minimum)
- Simplifies codebase by removing conditional CV logic
- Faster training for all dataset sizes
- Consistent behavior regardless of dataset size

**Project conventions:**
- All ML config centralized in `config.ts`
- Cross-validation logic in `crossValidation.ts` (pure functions)
- BaseClassifier coordinates training and CV

## Alternative Approaches Considered
1. **Keep LOO CV for very small datasets** - Rejected for complexity and performance
2. **Use 5-fold CV** - Rejected; 3-fold is sufficient and requires fewer samples
3. **Dynamic fold count** - Rejected; fixed 3-fold is simpler and more predictable

## Files to Modify
- `src/services/ml/config.ts` - Update CV configuration
- `src/services/ml/baseClassifier.ts` - Remove LOO CV logic
- `src/services/ml/crossValidation.ts` - Delete LOO CV implementation
- `src/services/ml/__tests__/crossValidation.test.ts` - Remove LOO CV tests
- `src/services/ml/__tests__/knnClassifier.test.ts` - Fix for smaller training sets

## Related Code References
- `src/services/ml/config.ts` - `TRAINING_CONFIG.cvFolds` (authoritative source)
- `src/services/ml/baseClassifier.ts` - CV strategy selection
- `src/services/ml/crossValidation.ts` - CV implementations

## Implementation Details

### 1. Updated CV Configuration
**File:** `src/services/ml/config.ts`
- Changed `cvFolds: 5` → `cvFolds: 3`
- Removed `loocvThreshold: 25` (no longer needed)
- Updated comment to "always 3-fold stratified CV"

### 2. Simplified CV Logic
**File:** `src/services/ml/baseClassifier.ts`
- Removed `createLOOCV` import
- Deleted conditional LOO/K-fold decision logic (12 lines)
- Now always uses stratified 3-fold CV regardless of dataset size
- Updated strategy string to always report "3-fold CV"

### 3. Removed LOO CV Code
**File:** `src/services/ml/crossValidation.ts`
- Updated file header comment (removed LOO CV reference)
- Deleted `createLOOCV()` function completely (26 lines)
- Cleaned up all LOO-specific code and comments

### 4. Updated Tests
**File:** `src/services/ml/__tests__/crossValidation.test.ts`
- Removed `createLOOCV` import
- Deleted entire "createLOOCV" describe block (22 lines)
- Updated "createStratifiedKFold" tests to use 3-fold
- Added test for class balance validation

### 5. Fixed KNN Tests
**File:** `src/services/ml/__tests__/knnClassifier.test.ts`
- Changed `k: 5` → `k: 3` (works with smaller training sets)
- Changed dataset size from `(5, 5)` → `(6, 6)` (12 frames total)
- Updated probability expectations from 1/5 to 1/3

## Test Results
✅ All 189 ML tests pass (11 test suites, 0 failures)

## Impact
- **Code deletion:** 48 lines removed (LOO CV implementation + tests)
- **Performance:** Faster training for small datasets (3 folds vs up to 25 with LOOCV)
- **Simplicity:** Single CV strategy for all dataset sizes
- **Maintainability:** Cleaner, more predictable codebase
