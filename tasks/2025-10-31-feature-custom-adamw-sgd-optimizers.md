# Task 2025-10-31: Add Custom AdamW and SGD Optimizers with Momentum and Weight Decay
**STATUS:** COMPLETED

## User Request
Add custom AdamW and SGD optimizers with momentum and weight decay to replace TensorFlow.js built-in optimizers. Integrate SGD into LogisticRegressionClassifier.

## Critical Discoveries

**1. TensorFlow.js lacks proper weight decay in built-in optimizers:**
TensorFlow.js Adam optimizer has no weight decay parameter. Manual weight decay post-optimizer step was added in `tasks/2025-10-28-refactor-adamw-weight-decay.md`, but proper AdamW algorithm requires decoupled weight decay (applied before gradient update, not after). Built-in SGD optimizer exists but lacks momentum and weight decay parameters needed for modern training.

**2. Decoupled weight decay timing is critical:**
AdamW applies weight decay BEFORE gradient update: `θ = (θ * (1 - lr * decay)) - lr * grad_update`. Previous manual implementation applied AFTER: `θ = θ - lr * grad_update; θ = θ * (1 - lr * decay)`. Order matters for adaptive optimizers - decoupled weight decay prevents interaction between adaptive learning rates and regularization.

**3. Momentum requires velocity buffer management:**
SGD with momentum maintains per-parameter velocity buffers. Must initialize lazily on first gradient application (like Adam's moment buffers). TensorFlow.js disposal system requires explicit cleanup of all velocity variables in `dispose()` method to prevent memory leaks.

**4. Nesterov momentum requires lookahead gradient:**
Regular momentum: `θ = θ - lr * v`. Nesterov: `θ = θ - lr * (grad + momentum * v)`. Nesterov uses "lookahead" by adding current gradient to momentum-scaled velocity, providing better convergence for convex objectives.

**5. PyTorch vs TensorFlow.js defaults differ:**
PyTorch SGD defaults: lr=0.01, momentum=0.9. TensorFlow.js SGD defaults: lr=0.01, momentum=0.0 (no momentum). Custom implementation uses PyTorch defaults for better out-of-box performance.

## Solution

**Created AdamW Optimizer (`src/services/ml/adamw.ts`)**

Implements decoupled weight decay following Loshchilov & Hutter (2019):
- Parameters: learningRate (0.001), beta1 (0.9), beta2 (0.999), epsilon (1e-7), weightDecay (0.0)
- Maintains first/second moment accumulators with bias correction
- Weight decay applied before gradient update: `θ = θ * (1 - lr * decay)` then `θ = θ - lr * m_hat / (sqrt(v_hat) + ε)`
- Proper memory management with `dispose()`, serialization support
- Factory function `adamw()` for convenient instantiation

**Key implementation (lines 113-130):**
```typescript
// Decoupled weight decay (before gradient update)
let newValue = value;
if (this.weightDecay !== 0) {
  newValue = tf.mul(value, 1 - this.learningRate * this.weightDecay);
}

// Gradient update with bias-corrected moments
newValue = tf.add(
  tf.mul(
    tf.div(
      biasCorrectedFirstMoment,
      tf.add(tf.sqrt(biasCorrectedSecondMoment), this.epsilon)
    ),
    -this.learningRate
  ),
  newValue
);
```

**Created SGD Optimizer (`src/services/ml/sgd.ts`)**

Implements momentum (regular + Nesterov) with decoupled weight decay following PyTorch:
- Parameters: learningRate (0.01), momentum (0.9), weightDecay (0.0), nesterov (false)
- Maintains velocity buffers (lazy initialization)
- Three modes: vanilla SGD (momentum=0), momentum SGD, Nesterov momentum
- Decoupled weight decay (same timing as AdamW)
- Proper memory management and serialization

**Momentum variants (lines 79-101):**
```typescript
if (this.momentum !== 0) {
  // v = momentum * v + grad
  const newVelocity = tf.add(tf.mul(velocity, this.momentum), gradient);
  velocity.assign(newVelocity);

  if (this.nesterov) {
    // Nesterov: θ = θ - lr * (grad + momentum * v)
    const nesterovUpdate = tf.add(gradient, tf.mul(newVelocity, this.momentum));
    newValue = tf.sub(newValue, tf.mul(nesterovUpdate, this.learningRate));
  } else {
    // Regular: θ = θ - lr * v
    newValue = tf.sub(newValue, tf.mul(newVelocity, this.learningRate));
  }
} else {
  // Vanilla SGD: θ = θ - lr * grad
  newValue = tf.sub(newValue, tf.mul(gradient, this.learningRate));
}
```

**Integrated SGD into LogisticRegressionClassifier**

Replaced manual weight decay with SGD optimizer:
- Uses momentum=0.9 (hardcoded for now, can be parameterized later)
- Uses nesterov=false (regular momentum)
- Weight decay parameter still configurable via UI
- Removed manual `weights.assign(weights.mul(1 - lr * decay))` after optimizer step
- Updated documentation to reflect SGD usage

**Comprehensive Test Coverage**

AdamW tests (19 tests, 437 lines):
- Constructor and factory function validation
- Convergence on simple quadratic problem
- Weight decay verification (compares with/without decay)
- Memory management (no tensor leaks)
- Parameter validation (negative learning rate errors)
- Bias correction for first/second moments
- Serialization round-trip (getConfig, getWeights, setWeights)
- Comparison with manual weight decay approach

SGD tests (23 tests, 505 lines):
- Constructor and factory function validation
- Convergence on quadratic problem
- Momentum variants (vanilla, momentum, Nesterov)
- Weight decay verification
- Memory management (velocity buffer disposal)
- Parameter validation
- Serialization round-trip
- Comparison with TensorFlow.js built-in optimizer

All existing LogisticRegressionClassifier tests pass (24 tests) with new SGD optimizer.

## Lessons

**Decoupled Weight Decay Superiority:** Weight decay applied separately from gradient update prevents interaction between adaptive learning rates and regularization. AdamW converges more reliably than Adam + L2 penalty. Same principle applies to SGD - decoupled weight decay cleaner than L2 penalty in loss.

**Momentum Accelerates Convergence:** SGD with momentum (0.9) provides significant acceleration over vanilla SGD, especially for ill-conditioned problems. Nesterov variant provides additional benefits for convex objectives but minimal difference on logistic regression.

**Custom Optimizers Worth Effort:** Despite 400+ lines of implementation + tests, custom optimizers provide full control over training dynamics. TensorFlow.js built-in optimizers lack modern features (AdamW, momentum SGD). Custom implementation allows PyTorch-style defaults and future enhancements.

**Lazy Initialization Pattern:** Velocity/moment buffers initialized on first gradient application prevents coupling optimizer construction to model structure. Allows single optimizer instance to handle variable parameter counts.

## Related

- `tasks/2025-10-28-refactor-adamw-weight-decay.md` - Added manual weight decay (replaced by proper AdamW)
- `tasks/2025-10-28-fix-tensorflow-memory-leaks.md` - Established disposal patterns for optimizers
- `tasks/2025-10-27-feature-class-weight-balancing.md` - Added class weights to loss function

## Files Created

- `src/services/ml/adamw.ts` (238 lines) - AdamW optimizer implementation
- `src/services/ml/__tests__/adamw.test.ts` (437 lines) - AdamW test suite
- `src/services/ml/sgd.ts` (182 lines) - SGD optimizer implementation
- `src/services/ml/__tests__/sgd.test.ts` (505 lines) - SGD test suite

## Files Modified

- `src/services/ml/logisticRegressionClassifier.ts` - Replaced manual weight decay with SGD optimizer (lines 4-8, 29, 495-508, 544)

## Impact

**Code Quality:**
- Net addition: ~1,362 lines (942 lines implementation + tests for new optimizers)
- Modern optimizer implementations matching PyTorch quality
- Comprehensive test coverage (42 new tests)
- No breaking changes (existing models still work)

**Training Performance:**
- SGD momentum (0.9) provides faster convergence than vanilla SGD
- Proper AdamW implementation available for future use
- Decoupled weight decay more predictable than L2 penalty

**Maintainability:**
- Self-contained optimizer implementations (no external dependencies)
- Full serialization support for model checkpointing
- Proper memory management prevents leaks
- PyTorch-style API familiar to ML practitioners

**Testing:** All 1,186 tests passing (24 LogisticRegression + 19 AdamW + 23 SGD + 1,120 other tests)
