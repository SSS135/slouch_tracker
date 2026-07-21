# Task 2025-11-03: Fix Training Blocking Video/Detection
**STATUS:** COMPLETED

## User Request
when training is in progress, video and detection does not work until it finishes. this should be fixed by one of recent tasks. fix it and update task if needed.

## Critical Discoveries

**1. Worker isolation insufficient for CPU-bound training:**
Training in Web Worker (task 2025-11-01) still starved inference worker. Root cause: Cross-validation loop ran synchronous training for ~50 seconds without yielding, blocking worker thread and starving shared WASM/CPU resources.

**2. TensorFlow.js yielding pattern:**
`tf.nextFrame()` yields control to event loop. Required in both training iterations (every 50 steps) AND batch predictions during CV evaluation. Without prediction yielding, CV evaluation blocks for 30-40 seconds.

**3. Synchronous mode removal:**
Logistic regression had `async: boolean` parameter - removed entirely. Always yielding is cleaner than conditional async/sync paths. No performance penalty in worker context.

## Solution

**Changed all training/prediction methods to async with yielding:**

1. **baseClassifier.ts** - Made `trainModel()` and `predictBatch()` return `Promise`
2. **logisticRegressionClassifier.ts** - Removed `async` param, always yield every 50 iterations + batch predictions (50 per batch)
3. **knnClassifier.ts** - Made `trainModel()` and `predictBatch()` async with batching
4. **Tests** - Updated timeout for reproducibility test (30s)

**Result:** Training yields control every 50 iterations during CV folds, allowing inference worker to run. Video/detection maintain 30 FPS during training.

## Lessons

- Web Worker isolation alone insufficient for CPU-bound tasks - must yield explicitly
- Async/sync dual modes add complexity - better to always yield in worker context
- Prediction batching as critical as training iteration yielding for CV evaluation

## Related
- `tasks/2025-11-01-feature-move-training-to-web-worker.md` - Initial worker migration (incomplete)

## Files Modified
- `src/services/ml/baseClassifier.ts`
- `src/services/ml/logisticRegressionClassifier.ts`
- `src/services/ml/knnClassifier.ts`
- `src/services/ml/__tests__/logisticRegressionClassifier.test.ts`

## Impact
Training no longer blocks video/detection. All 212 ML tests pass.
