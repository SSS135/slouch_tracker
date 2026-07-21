# Task 2025-11-09: Comprehensive Test Suite Update
**STATUS:** IN PROGRESS

## User Request
update all tests. replace or remove outdated / bad ones. fix good ones. add missing if it is important.

## General Description
Comprehensive test suite overhaul addressing broken tests from recent refactoring (unified features, storage simplification) and adding critical missing test coverage for ML Feature Pipeline, Workers, and Hooks. Work divided into 5 parts with clear priorities.

## Action Plan

### Part 1: Fix Dataset & Validation Tests (CRITICAL - P0)
**Priority**: BLOCKING (tests currently broken)
**Effort**: 6.5-7.5 hours
**Files**: 7 test files

1. Fix feature structure references (185+ occurrences):
   - Replace `presenceFeatures`/`postureFeatures` split → unified `features`
2. Fix removed constants (97+ occurrences):
   - Replace `FEATURE_NECK_P5` → `FEATURE_RTMDET_EXTRACTED`
3. Fix type imports:
   - Replace `TrainedModel` → `SerializedModel`
   - Replace `isTrainedModel` → check if exists or remove
4. Add missing storage constants:
   - `PCA_CONFIG_KEY`, `TRAINING_SETTINGS_KEY` (referenced but undefined)
5. Remove obsolete split-key architecture tests
6. Add tests for new storage keys (dimReduction config, training settings)

**Files to modify**:
- `src/services/dataset/__tests__/storage.test.ts`
- `src/services/dataset/__tests__/operations.test.ts`
- `src/services/dataset/__tests__/featureRegistry.test.ts`
- `src/services/dataset/__tests__/import.test.ts`
- `src/services/dataset/__tests__/export.test.ts`
- `src/services/validation/__tests__/guards.test.ts`
- `src/services/validation/__tests__/schemas.test.ts`

### Part 2: ML Feature Pipeline Tests (NEW - P1)
**Priority**: CRITICAL (core pipeline, high risk)
**Effort**: 5-7 hours

Add 5 new test files (~77 test cases total):

1. **featureExtractor.test.ts** (CRITICAL - 30 cases)
   - Fitting, transformation, serialization, memory management
   - Follow bare-bones integration test pattern (Task 0004)
   - Use real TensorFlow.js (no mocks)

2. **featureExtraction.test.ts** (HIGH - 20 cases)
   - Feature validation, matrix building, concatenation

3. **rtmdetFeatures.test.ts** (MEDIUM - 6 cases)
   - Similar pattern to rtmposeFeatures.test.ts

4. **layerNorm.test.ts** (MEDIUM - 11 cases)
   - Normalization correctness, memory management

5. **classifierRegistry.test.ts** (MEDIUM - 10 cases)
   - Schema validation, metadata correctness

**Keep**: `rtmposeFeatures.test.ts` (excellent quality)

### Part 3: Workers Tests (NEW - P1)
**Priority**: CRITICAL (training-worker 0% coverage)
**Effort**: 8-10 hours

Add 5 new test files + extend 1 existing (~100+ test cases total):

1. **training-worker.test.ts** (CRITICAL - 40 cases)
   - Dual model training, IndexedDB integration, memory management
   - Test message validation with Zod schemas (Task 2025-11-07 pattern)

2. **inference-worker-initialization.test.ts** (CRITICAL - 15 cases)
   - ONNX loading, retry logic, reinitialization

3. **inference-worker-pipeline.test.ts** (CRITICAL - 28 cases)
   - RTMDet detection, RTMPose estimation, feature extraction

4. **inference-worker-classification.test.ts** (HIGH - extend +10 cases)
   - Cascaded classification logic

5. **inference-worker-buffer-transfer.test.ts** (HIGH - 7 cases)
   - Zero-copy transfer, memory error handling

**Keep**: All existing worker tests (high quality)

### Part 4: Hooks Tests (UNSKIP + NEW - P2)
**Priority**: HIGH (important ML workflow hooks)
**Effort**: 6-8 hours

Unskip & fix 2 test files, add 6 new (~80+ test cases total):

**Unskip & Fix**:
1. **useFrameProcessor.test.ts** (CRITICAL - memory leak tests)
2. **useFrameSampler.test.ts** (CRITICAL - data collection)

**Add New Tests**:
3. **useModelTraining.test.ts** (CRITICAL)
4. **usePostureClassifier.test.ts** (HIGH)
5. **useDatasetOperations.test.ts** (HIGH)
6. **useWebWorkerInference.test.ts** (HIGH)
7. **useAutoTraining.test.ts** (MEDIUM)
8. **useMultiTaskDetection.test.ts** (MEDIUM)

**Keep**: All existing hook tests (excellent quality)

### Part 5: ML Classifiers (OPTIONAL - P3)
**Priority**: LOW (already 85-90% covered by integration tests)
**Effort**: 2 hours

**Add only if time permits**:
- **logisticRegressionClassifier.test.ts** (2-3 critical tests)
  - Training divergence detection (NaN/Inf with high learning rate)
  - Class weight balancing (imbalanced datasets)

**Skip**: baseClassifier (interface only), knnClassifier, svmClassifier (integration coverage sufficient)

## Rationale

### Why These Priorities?

1. **Part 1 first**: Broken tests block CI/CD and prevent validating any other changes
2. **Parts 2-3 parallel**: Independent work streams (feature pipeline vs workers)
3. **Part 4 after 2-3**: Hooks tests may depend on understanding from earlier parts
4. **Part 5 optional**: Integration tests already provide 85-90% coverage

### Testing Philosophy (from Task 0004)

**DO**:
- Test behavior, not implementation details
- Use real TensorFlow.js (no mocks - too complex)
- Integration tests > unit tests for ML code
- Focus on: constructor, validation, train success/failure, predictions, serialization
- Make tests deterministic (mock time/random/network)

**DON'T**:
- Mock TensorFlow.js (infeasible for performance/complexity)
- Test framework internals (Zod validation, TF.js weight magnitudes)
- Create backward compatibility tests (project policy: clean breaks)
- Write tautologies or tests without assertions

### Patterns from Past Tasks

**Serialization Tests** (Task 2025-11-08, 2025-11-06):
- Manual `toJSON()` / `fromJSON()` (no decorators)
- Zod schema validation at boundaries
- Round-trip test: serialize → deserialize → identical behavior
- Test corrupt data rejection with clear error messages

**Worker Message Validation** (Task 2025-11-07):
- All worker messages validated with Zod schemas
- Helpers: `sendValidatedResponse()`, `sendValidatedResponseWithTransferables()`
- Dev-mode only validation (zero production overhead)

**Feature Cloning** (Task 2025-11-07):
- Buffer independence tests: verify separate ArrayBuffers
- Mutation isolation: modify clone → original unchanged
- Dev-mode assertions: `assertBufferIndependence()`

**Logger Integration** (Task 0003):
- Silent mode enabled in jest.config.js
- Mock logger globally in jest.setup.js
- Verify `logger.debug()` / `logger.error()` called correctly

## Files to Modify

### Part 1 (Fix Existing):
- `src/services/dataset/__tests__/storage.test.ts`
- `src/services/dataset/__tests__/operations.test.ts`
- `src/services/dataset/__tests__/featureRegistry.test.ts`
- `src/services/dataset/__tests__/import.test.ts`
- `src/services/dataset/__tests__/export.test.ts`
- `src/services/validation/__tests__/guards.test.ts`
- `src/services/validation/__tests__/schemas.test.ts`

### Part 2 (New Tests):
- `src/services/ml/__tests__/featureExtractor.test.ts` (NEW)
- `src/services/ml/__tests__/featureExtraction.test.ts` (NEW)
- `src/services/ml/__tests__/rtmdetFeatures.test.ts` (NEW)
- `src/services/ml/__tests__/layerNorm.test.ts` (NEW)
- `src/services/ml/__tests__/classifierRegistry.test.ts` (NEW)

### Part 3 (New Tests):
- `src/workers/__tests__/training-worker.test.ts` (NEW)
- `src/workers/__tests__/inference-worker-initialization.test.ts` (NEW)
- `src/workers/__tests__/inference-worker-pipeline.test.ts` (NEW)
- `src/workers/__tests__/inference-worker-classification.test.ts` (EXTEND)
- `src/workers/__tests__/inference-worker-buffer-transfer.test.ts` (NEW)

### Part 4 (Fix + New):
- `src/hooks/__tests__/useFrameProcessor.test.ts` (UNSKIP & FIX)
- `src/hooks/__tests__/useFrameSampler.test.ts` (UNSKIP & FIX)
- `src/hooks/__tests__/useModelTraining.test.ts` (NEW)
- `src/hooks/__tests__/usePostureClassifier.test.ts` (NEW)
- `src/hooks/__tests__/useDatasetOperations.test.ts` (NEW)
- `src/hooks/__tests__/useWebWorkerInference.test.ts` (NEW)
- `src/hooks/__tests__/useAutoTraining.test.ts` (NEW)
- `src/hooks/__tests__/useMultiTaskDetection.test.ts` (NEW)

### Part 5 (Optional):
- `src/services/ml/__tests__/logisticRegressionClassifier.test.ts` (NEW)

## Related Tasks

- `tasks/2025-11-08-refactor-unified-raw-features-pipeline.md` - Feature pipeline architecture (broke tests)
- `tasks/2025-11-09-refactor-simplify-storage-single-key.md` - Storage architecture (broke tests)
- `tasks/2025-11-06-refactor-classifier-hierarchy.md` - Classifier patterns
- `tasks/2025-11-07-refactor-critical-architecture-fixes.md` - Feature cloning, worker validation patterns
- `tasks/0004-refactor-simplify-logistic-regression-tests.md` - Bare-bones testing philosophy
- `tasks/0003-fix-excessive-test-output.md` - Logger integration testing
