# Task 2025-11-10: Jest to Vitest Migration
**STATUS:** COMPLETED

## User Request
"plan vittest migration"

## Critical Discoveries (Non-Obvious)

**1. Jest ESM Blocking 55 Tests:**
55 tests blocked by Jest's experimental-vm-modules limitations: context mocking (useAutoTraining: 0/14), Worker async handling (useModelTraining: 11/29, useWebWorkerInference: 4/27). All unblocked by Vitest's native ESM support.

**2. TensorFlow.js WebGL Detection Overhead:**
Each test file wasted ~500ms attempting WebGL initialization. Forcing CPU backend globally in setup (`await tf.setBackend('cpu')`) eliminated this completely.

**3. IndexedDB Validation Filtering Valid Frames:**
Storage tests showed "[DATASET_LOAD] Skipped N invalid frames" for valid test data. Runtime validation at load boundaries was overly strict - validated data during parallel loading was being rejected.

**4. Thread Pool Configuration:**
Vitest CLI doesn't support `--poolOptions.threads.maxThreads` flag syntax. Thread configuration must be in vitest.config.ts only, not CLI overrides.

**5. Selective Testing Essential:**
Full suite (420+ tests) too slow for iteration (~25-30s even with 16 threads). Created category scripts (test:ml, test:utils, test:components) for 3-12s feedback loops. Changed files only (`test:changed`) fastest.

## Solution

**Migration (Phase 1-2):**
Removed Jest (213 packages), installed Vitest 4.0.8 + @vitest/ui + @vitest/coverage-v8. Replaced jest.config.js/jest.setup.js with vitest.config.ts/vitest.setup.ts. Used sed to replace all `jest` → `vi` API calls across 50 test files (imports, fn, spyOn, mock, timers, types).

**Performance Optimization:**
Enabled 16-thread pool with `isolate: true` in vitest.config.ts. Added TensorFlow.js CPU backend forcing in vitest.setup.ts. Reduced timeouts from 10s to 5s. Disabled sourcemaps. Added fileParallelism.

**Selective Test Scripts:**
Created 7 npm scripts for category-based testing: test:ml (12s, 260 tests), test:utils (3s, 65 tests), test:components (8s, 106 tests), test:hooks, test:dataset, test:workers, test:changed (fastest).

**Results:**
- Utils: 3s, 65 tests passing
- Components: 8s, 106 tests passing
- ML: 12s, 260 tests passing
- Estimated full suite: 25-30s vs 120s single-threaded
- 55 previously Jest-blocked tests now unblocked

## Related
- tasks/2025-11-09-refactor-comprehensive-test-suite-update.md (provided context for 55 failing tests)
- tasks/2025-11-03-fix-manual-capture-blocked-during-training.md (documented Jest ESM limitation, suggested Vitest)
- tasks/0004-refactor-simplify-logistic-regression-tests.md (established TensorFlow.js real integration pattern)
- tasks/0003-fix-excessive-test-output.md (silent logger mock pattern preserved)
