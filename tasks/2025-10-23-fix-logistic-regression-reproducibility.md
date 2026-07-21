# Task 2025-10-23: Fix Logistic Regression Training Reproducibility
**STATUS:** COMPLETED

## User Request
"each time I click train I get different accuracy. Like seed is not used or forgotten in some place."

## Critical Discoveries

**1. TensorFlow.js global RNG unseeded:**
Weight initialization used `tf.randomNormal()` without seeding TensorFlow.js's global RNG. K-Fold CV and Random Projection already used seeded custom RNG (seed=42), but TensorFlow.js operations required separate seeding via separate mechanism.

**2. Seed must be passed per-tensor creation:**
Cannot rely on `tf.setRandomSeed()` global state. Instead, pass seed directly to `tf.randomNormal()` via 5th parameter: `tf.randomNormal([n, m], 0, 0.01, 'float32', TRAINING_CONFIG.randomSeed)`.

## Solution

**Seed initialization:** Updated weight initialization in `logisticRegressionClassifier.ts` to pass `TRAINING_CONFIG.randomSeed` directly to `tf.randomNormal()` fifth parameter. Ensures deterministic weight initialization without relying on global state.

**Reproducibility test:** Added test verifying two models trained on identical data produce element-wise equal weights and identical predictions. Catches future regressions in reproducibility.

## Related
- `tasks/2025-10-24-fix-training-validation-and-buffer-size.md` - Training validation improvements
- `tasks/2025-10-22-feature-add-logistic-regression.md` - Original implementation
