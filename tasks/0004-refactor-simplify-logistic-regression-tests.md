## User Request
simplify logistic regression classifier tests to bare-bones since mocking tensorflow js in infeasable for performance and complexity reasons. remove tensorflow js mocks.

## General Description
The current logistic regression classifier tests (`logisticRegressionClassifier.test.ts`) use `jest.unmock('@tensorflow/tfjs')` to bypass the global TensorFlow.js mock defined in `jest.setup.js`. This approach has become problematic because:

1. The global TensorFlow.js mock in `jest.setup.js` (lines 4-236) is extremely complex, attempting to simulate tensor operations, matrix multiplication, and other TF operations
2. Maintaining this mock is difficult and error-prone as it tries to replicate TensorFlow.js behavior
3. The tests already unmock TF.js anyway, making the mock unnecessary for these tests
4. The mock adds performance overhead and testing complexity without providing value

The solution is to strip the logistic regression tests down to bare-bones integration tests that use the real TensorFlow.js library, testing only the essential classifier contract without detailed assertions about internal TF.js operations.

## Action Plan
1. Review current test file to identify which tests rely on TF.js mocking behavior
2. Simplify tests to focus on the classifier's public API contract:
   - Constructor initialization
   - Dataset validation (minimum samples per class)
   - Training success/failure (basic)
   - Prediction returns valid class labels
   - Serialization/deserialization (toJSON/fromJSON)
3. Remove or simplify tests that make detailed assertions about TensorFlow.js internals
4. Remove the TensorFlow.js mock from `jest.setup.js` (lines 4-236)
5. Ensure remaining tests still provide adequate coverage of the classifier interface

## Rationale
**Why this approach:**

1. **Separation of Concerns**: We should test the classifier's behavior, not TensorFlow.js internals. The current mock attempts to simulate TF.js operations, which is redundant since TF.js itself is well-tested.

2. **Maintainability**: The 230+ line TensorFlow.js mock in `jest.setup.js` is a maintenance burden. Every TF.js API change or new operation requires mock updates.

3. **Test Philosophy**: Integration tests using the real library are more valuable than unit tests with complex mocks. The classifier's correctness depends on proper TF.js integration, which mocks cannot validate.

4. **Already Unmocked**: The logistic regression tests already call `jest.unmock('@tensorflow/tfjs')` on line 6, indicating the tests were designed to use the real library anyway.

5. **Project Pattern**: The codebase already uses real dependencies in tests where feasible (e.g., fake-indexeddb for storage tests). This is consistent with that pattern.

**Why NOT keep the mock:**
- The mock doesn't provide isolation benefits (tests unmock it)
- It doesn't improve test speed (real TF.js on CPU backend is fast enough)
- It creates false confidence (passing tests with mock might fail with real TF.js)
- It blocks testing actual TF.js integration issues

## Alternative Approaches Considered

### Alternative 1: Keep Mock, Remove jest.unmock()
Use the TensorFlow.js mock for all tests to improve test isolation and speed.

**Rejected because:**
- The mock is too complex to maintain accurately
- Mock behavior may diverge from real TF.js, causing false positives
- Integration bugs with real TF.js would not be caught
- Performance gains are minimal (TF.js CPU backend is already fast for small test datasets)

### Alternative 2: Partial Mock (Mock Only Backend Initialization)
Keep a minimal mock that only suppresses TensorFlow.js backend initialization logs, while using real TF.js for operations.

**Rejected because:**
- Adds complexity without clear benefit
- `jest.unmock()` already achieves silent backend initialization
- Half-mock, half-real approach is confusing and error-prone

### Alternative 3: Separate Test Suite for Integration Tests
Keep mocked unit tests AND add separate integration tests with real TF.js.

**Rejected because:**
- Doubles test maintenance burden
- Classifier is already a thin wrapper around TF.js; mocking provides little value
- Test duplication without corresponding value

## Files to Modify

### Primary Changes
1. **`src/services/ml/__tests__/logisticRegressionClassifier.test.ts`**
   - Simplify to bare-bones integration tests
   - Keep: constructor, validation, train success/failure, predict returns valid classes, serialization
   - Remove/simplify: Detailed assertions about weights, bias, L2 regularization effects, probability calibration details
   - Focus on testing the classifier contract, not TensorFlow.js internals

2. **`jest.setup.js`**
   - Remove entire TensorFlow.js mock (lines 4-236)
   - Keep other mocks (IndexedDB, logger, React Native, etc.)

### Verification
- **`src/services/ml/logisticRegressionClassifier.ts`** - No changes needed, only test changes

## Related Code References

**Testing Philosophy Precedent:**
- `src/services/dataset/__tests__/storage.test.ts` - Uses real fake-indexeddb library instead of mocking IndexedDB
- This follows the same pattern: use real implementations when feasible, mock only when necessary

**Other Classifiers:**
- If KNN classifier tests exist, they should follow similar simplification approach
- Pattern should be consistent across all ML classifier tests

## Implementation Details

**Bare-Bones Test Coverage Should Include:**

1. **Constructor**: Accepts config, creates instance
2. **Validation**: Rejects datasets with <5 samples per class
3. **Training**:
   - Returns success=true with valid balanced dataset
   - Returns success=false with insufficient data
4. **Prediction**:
   - predict() returns 'good' or 'bad' (not testing which, just valid output)
   - predictProba() returns object with 'good' and 'bad' keys, probabilities sum to ~1.0
   - Throws error when model not trained
5. **Serialization**:
   - toJSON() returns object with expected keys (classifier, params, nFeatures, etc.)
   - fromJSON() reconstructs model that can make predictions
   - Throws error for incompatible old formats

**Tests to Remove or Simplify:**

1. **L2 Regularization test** (lines 260-293):
   - Removes detailed assertions about weight magnitudes
   - TensorFlow.js implementation correctness is TF.js's responsibility

2. **Probability Calibration test** (lines 295-316):
   - Removes assertions about confidence ranges with specific C values
   - Too dependent on TF.js internals and dataset characteristics

3. **nFeatures validation test** (lines 213-224):
   - Simplify to just verify nFeatures is set, don't assert specific value matching reduced dimensions
   - Internal consistency check, not core contract

**Expected Test File Size:** ~150-200 lines (down from 354 lines)

## Open Questions
None. The approach is straightforward: remove TF.js mock, simplify tests to focus on classifier contract rather than TF.js internals.

---

## Implementation Results

**Completed**: Task successfully implemented with the following changes:

### Changes Made

1. **jest.setup.js** - Removed TensorFlow.js mock
   - Reduced from 457 to 245 lines (removed 212 lines)
   - Eliminated entire TF.js mock section (previously lines 4-236)
   - Kept other mocks (IndexedDB, logger, React Native, etc.)

2. **logisticRegressionClassifier.test.ts** - Simplified to integration tests
   - Reduced from 354 to 265 lines (removed 89 lines)
   - Added header comment clarifying these are integration tests using real TensorFlow.js
   - Removed 2 tests that were testing TF.js internals:
     - L2 Regularization test (was verifying TF.js weight magnitudes)
     - Probability Calibration test (was testing TF.js behavior with specific C values)
   - Kept all essential public API contract tests:
     - Constructor initialization
     - Dataset validation (minimum samples per class)
     - Training success/failure paths
     - Prediction returns valid class labels
     - Serialization/deserialization (toJSON/fromJSON)

### Test Results

- All 1121 tests pass (100% success rate)
- Logistic regression classifier: 13 tests passing
- Test suite execution time: 19.47s
- No regressions introduced

### Impact

**Code Reduction:**
- Total lines removed: 301 lines (212 from jest.setup.js + 89 from test file)
- Test file complexity reduction: 25% smaller, more focused

**Maintainability Improvements:**
- Eliminated 212-line TensorFlow.js mock that was difficult to maintain
- Tests now focus on classifier contract, not TF.js internals
- Integration tests provide higher confidence in real-world behavior
- Consistent with project pattern of using real implementations where feasible

**Test Quality:**
- More realistic tests using actual TensorFlow.js library
- Better coverage of integration issues
- Simpler, more readable test code
- Faster to understand and modify

### Lessons Learned

1. **Bare-bones integration testing approach works well** - Keeping only essential contract tests provides adequate coverage without excessive complexity
2. **Mock removal was straightforward** - No dependencies on the mock outside of the tests that already unmocked it
3. **Test count reduction justified** - The 2 removed tests (L2 regularization, probability calibration) were testing TF.js internals, not classifier behavior
4. **Performance acceptable** - Real TF.js with CPU backend is fast enough for test suite (19.47s total for all 1121 tests)
