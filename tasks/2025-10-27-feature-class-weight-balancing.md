# Task 2025-10-27: Add Class Weight Balancing to Logistic Regression
**STATUS:** COMPLETED

## User Request
"scale logistic regression weight for classes so it works better with imbalanced dataset. scikit-learn does something like that."

**Follow-up:** "we did training loss re-scaling for unbalanced classes, add a way to disable it in training settings ui (disabled by default)"

## Critical Discoveries (Non-Obvious)

**1. TensorFlow.js loss weight application:**
`tf.losses.sigmoidCrossEntropy()` accepts per-sample weights (not class weights). Must manually map class weights to sample weights based on labels before passing to loss function.

**2. Class weights only affect training, not inference:**
Saved class weights don't change predictions - model uses trained parameters. Stored for transparency only. This means old models work perfectly for inference even when backward compatibility logic triggers for retraining.

**3. Backward compatibility without breaking changes:**
New parameter `useClassWeights` defaults to `false` for new models but `true` when loading old models (missing parameter). Old models remain functional for inference, maintain original behavior if retrained.

## Solution

Implemented automatic class weight balancing with optional UI toggle. Uses scikit-learn formula: `n_samples / (n_classes * bincount)`. Disabled by default (user opt-in).

**Core Implementation:**
- `useClassWeights` parameter in classifier registry (schema-driven UI auto-generates toggle)
- `trainModelCore()`: Conditionally computes weights - balanced if enabled, `[1.0, 1.0]` if disabled
- `fromJSON()`: Backward compatibility - old models default to `useClassWeights=true`
- UI: "Use Class Weight Balancing" toggle in Training Tab parameters section

**Formula example (20 good, 5 bad):**
- Good weight: `25 / (2 * 20) = 0.625`, Bad weight: `25 / (2 * 5) = 2.5`
- Result: Bad frames get 4x more weight during training

**Tests added (28 total):**
- useClassWeights=true produces balanced weights, useClassWeights=false uses equal weights
- Parameter serialization/deserialization, backward compatibility (old models default to true)
- Training behavior differs between settings

## Files Modified

- `src/services/ml/classifierRegistry.ts` - Added `useClassWeights` boolean parameter
- `src/services/ml/logisticRegressionClassifier.ts` - Conditional weight computation, backward compatibility
- `src/services/ml/types.ts` - Updated `LogisticRegressionParams` interface
- `src/services/ml/__tests__/logisticRegressionClassifier.test.ts` - 5 new tests for toggle feature

## Impact

**User-facing:**
- Default: Class weight balancing disabled (user must opt-in)
- Existing models continue working for inference (no breaking changes)
- UI toggle auto-generated from schema (no manual UI code)

**Technical:**
- Backward compatible: Old models default to balanced weights when loaded
- 28/28 tests passing (Logistic Regression)
- No performance impact (weight computation negligible)
