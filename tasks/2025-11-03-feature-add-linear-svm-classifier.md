# Task 2025-11-03: Add Linear SVM Classifier

**STATUS:** COMPLETED

## User Request

add linear (possibly configurable kernel in future if it wont compicate things) svm classifier. are there different versions? which should we use? is it hard to implement with tfjs?

## Critical Discoveries (Non-Obvious)

**1. TensorFlow.js hinge loss implementation:**
Labels must be {-1, +1} for SVM (not {0, 1} like Logistic Regression). Decision threshold at 0: `decision > 0 → bad, decision ≤ 0 → good`. This is opposite to typical binary classification conventions.

**2. Regularization parameter C semantics:**
C has **inverse** relationship to weight decay. Lower C = more regularization (wider margin). This is opposite to Logistic Regression's weightDecay parameter where higher = more regularization.

**3. Class weight formula consistency:**
Reused scikit-learn balanced formula `n_samples / (n_classes * bincount)` from Logistic Regression. Critical for maintaining consistent behavior across classifiers with imbalanced datasets.

**4. No native TensorFlow.js SVM:**
No production-ready TensorFlow.js SVM libraries exist. ml-svm library is standalone JavaScript (no GPU acceleration, different API paradigm). From-scratch implementation using existing SGD optimizer was more consistent with codebase architecture.

**5. Probability calibration skipped:**
Raw `sigmoid(decision)` provides uncalibrated probabilities. Platt scaling requires additional validation data and sigmoid fitting. Skipped for initial implementation simplicity - can add later if needed.

## Solution

**Implementation:**
- Extended `AbstractClassifier` with hinge loss: `L = C * Σ w_i * max(0, 1 - y_i(w·x_i + b)) + ||w||²`
- Used existing SGD optimizer (momentum=0.9, learning_rate=0.01)
- Added 3 parameters: C (regularization), maxIterations, useClassWeights
- Implemented class weight balancing using scikit-learn formula
- Serialization with weights, bias, classWeights
- Full dimensionality reduction support (Random Projection, PLS-DA)
- Proper TensorFlow.js memory management (tf.tidy, dispose)

**Type system:**
- Added `SerializedSVM` interface (weights, bias, classWeights)
- Added `SVMParams` interface (C, maxIterations, useClassWeights)
- Added type guards: `isSVMParams`, `isSerializedSVM`
- Updated `SerializedClassifierState` and `ClassifierParams` unions

**Registry:**
- Schema-driven UI auto-generation with 3 parameters
- C: exponential scale [0.01, 100], default 1.0
- maxIterations: linear scale [100, 10000], default 1000
- useClassWeights: boolean toggle, default false

**Testing:**
- 33 comprehensive tests across 9 categories
- Training convergence, class weights, serialization, CV integration
- Memory management, edge cases, dim reduction, normalization
- All tests pass deterministically using TRAINING_CONFIG.randomSeed

**Files:**
- `src/services/ml/svmClassifier.ts` (550 lines)
- `src/services/ml/__tests__/svmClassifier.test.ts` (870 lines, 33 tests)
- `src/services/ml/types.ts` (+42 lines)
- `src/services/ml/classifierRegistry.ts` (+38 lines)

## Related

- `tasks/2025-10-27-feature-class-weight-balancing.md` - Pattern for useClassWeights parameter
- `tasks/2025-10-31-feature-custom-adamw-sgd-optimizers.md` - Reused SGD optimizer implementation
- `tasks/2025-11-03-feature-add-plsda-dim-reduction.md` - Integration pattern with dim reduction
