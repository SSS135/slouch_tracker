# Task 2025-10-28: Fix TensorFlow.js Memory Leaks in Training Code
**STATUS:** COMPLETED

## User Request
"there is memory leak in training code. analyze it, make sure it is wrapped with tf.tidy at the root"

## Critical Discoveries

**1. Loss function creates tensors 1000+ times without cleanup:**
Loss function (logisticRegressionClassifier.ts:490-494) called on every training iteration creates intermediate tensors (`logits`, `crossEntropy`) without `tf.tidy()` wrapper. With default 1000 iterations, this accumulates hundreds of MB in GPU memory.

**2. Training tensors not guaranteed disposal on error paths:**
X, y, sampleWeights tensors only disposed on successful completion (line 523-525). If training throws error (NaN loss, validation failure), tensors leak.

**3. Cross-validation runs 5 trainings without GC opportunity:**
Cross-validation loop runs training 5 times sequentially. Even with proper disposal, TensorFlow.js doesn't immediately free GPU memory without yielding to event loop.

**4. Weight decay operation leaks temporary tensors:**
`.mul()` operation (line 507) creates temporary tensor on each iteration without cleanup.

## Solution

Wrapped entire `trainModelCore()` in `tf.engine().startScope()`/`endScope()` for comprehensive tensor cleanup. Wrapped loss function body in `tf.tidy()` to clean up intermediate tensors created on each iteration. Wrapped weight decay operation in `tf.tidy()` to clean up `.mul()` temporary tensor. Added explicit `dispose()` calls for optimizer and training tensors. Wrapped entire `crossValidate()` in `startScope()`/`endScope()`. Added `tf.nextFrame()` between CV folds to allow TensorFlow.js GC.

**Key code changes:**
```typescript
// logisticRegressionClassifier.ts:475-542 - Root scope
tf.engine().startScope();
// Line 497-501 - Loss function cleanup
const loss = () => tf.tidy(() => { /* intermediate tensors auto-disposed */ });
// Line 515-519 - Weight decay cleanup
tf.tidy(() => { this.weights!.assign(decayed); });
// Line 535-538 - Explicit disposal
optimizer.dispose(); X.dispose(); y.dispose(); sampleWeights.dispose();
tf.engine().endScope();

// baseClassifier.ts:405-472 - CV root scope + GC yield
tf.engine().startScope();
// Line 464 - Allow GC between folds
await tf.nextFrame();
tf.engine().endScope();
```

Added comprehensive memory leak regression test verifying only 2 tensors remain after training (weights + bias).

## Lessons

TensorFlow.js requires multi-level cleanup strategy: root scope for overall cleanup, `tf.tidy()` for functions called repeatedly, explicit `dispose()` for immediate cleanup, `tf.nextFrame()` for GC between sequential operations. Before fix: leaked 50-500+ tensors per training run. After fix: stable 2 tensors (weights + bias).

## Related

- `tasks/0005-fix-memory-leak.md` - Established disposal patterns
- `tasks/2025-10-28-refactor-adamw-weight-decay.md` - Created trainModelCore helper
- `tasks/2025-10-27-feature-class-weight-balancing.md` - Added class weights to loss

## Files Modified

- `src/services/ml/logisticRegressionClassifier.ts` (4 changes: root scope, loss tidy, weight decay tidy, explicit dispose)
- `src/services/ml/baseClassifier.ts` (2 changes: root scope, nextFrame between folds)
- `src/services/ml/__tests__/logisticRegressionClassifier.test.ts` (new regression test)

## Impact

Eliminates memory leaks causing browser crashes during training with high-dimensional features. Users can now train models repeatedly without memory exhaustion. Test results: all 1144 tests pass, regression test verifies stable memory usage.
