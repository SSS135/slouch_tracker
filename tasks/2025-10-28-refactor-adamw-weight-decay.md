# Task 2025-10-28: Refactor Logistic Regression - AdamW Weight Decay & Training Method Duplication
**STATUS:** COMPLETED

## User Request

**Phase 1:** "use adamw like weight decay there"

**Phase 2:** "can you add to the same task document refactoring of trainModel and trainFinalModelImpl code duplication"

**Phase 3:** Fix weight decay parameter slider displaying "NaN" in UI (parameter migration issue)

## Critical Discoveries

**1. Training Method Duplication (~90% identical code):**
`trainModel()` and `trainFinalModelImpl()` differed only in: async/await, setting nFeatures, logging prefixes. ~140 lines duplicated made Phase 1 changes error-prone (must update both methods identically).

**2. AdamW Implementation Gotcha:**
TensorFlow.js Adam optimizer has no built-in weight decay support. Manual approach: `w = w * (1 - lr * decay)` after optimizer step. Must apply to weights tensor directly after `optimizer.minimize(loss)`.

**3. Breaking Change Acceptable:**
Replacing C → weightDecay changes serialization format. No conversion possible (L2 penalty ≠ weight decay mathematically). User confirmed old models can be discarded.

**4. Parameter Migration Gotcha (Phase 3):**
Saved configs had old `C: 1.0` parameter, new code expected `weightDecay: 0.01`. When UI loaded `config.params['weightDecay']` it got `undefined` → exponential slider scale converted `undefined` to `NaN`. Required three-layer defense: update defaults, add migration logic, defensive fallback in slider.

## Solution

**Phase 1: AdamW Weight Decay Implementation**
- Replaced `C` parameter with `weightDecay` in types, registry, classifier
- Removed L2 penalty from loss function (was: `crossEntropy + (1/(2*C)) * ||w||²`)
- Added manual weight decay after optimizer step: `weights.assign(weights.mul(1 - lr * decay))`
- Updated default: weightDecay = 0.01 (moderate decay), range: 0.0001 - 1.0
- Updated all tests to use weightDecay instead of C

**Phase 2: Duplication Refactoring**
Created three helper methods to eliminate ~140 lines of duplication:

1. `validateFeatures(features, context)` - Feature validation (95% duplication eliminated)
2. `disposeModel()` - Model disposal logic (100% duplication eliminated)
3. `trainModelCore(features, labels, options)` - Core training logic (~115 lines extracted)
   - Options: `{ async: boolean, setNFeatures: boolean, logPrefix: string }`
   - Conditionally yields to main thread if async=true
   - 85-90% duplication eliminated between trainModel/trainFinalModelImpl

Simplified methods:
- `trainModel` → 4 lines (calls trainModelCore with async=false)
- `trainFinalModelImpl` → 4 lines (calls trainModelCore with async=true)
- `predictBatch` → uses validateFeatures helper

**Phase 3: Parameter Migration & NaN Fix**
- Updated DEFAULT_CONFIG in TrainingConfigContext: `C: 1.0` → `weightDecay: 0.01`, `learningRate: 0.01` → `0.001`
- Added parameter migration logic (line 124-127): merge saved params with classifier registry defaults before setting config
- Added defensive fallback in ClassifierSelector slider: `value={config.params[name] ?? paramDef.default}`
- Ensures all parameters exist, handles C → weightDecay migration automatically, prevents NaN in UI
- Updated all TrainingConfigContext tests: replaced C references with weightDecay, updated defaults (15 tests passing)

## Lessons

**AdamW vs L2 Regularization:** AdamW decouples weight decay from adaptive learning rate mechanism. Better convergence properties, clearer semantics (higher = more decay), more predictable tuning across learning rates.

**Refactor Before Feature Work:** Phase 2 refactoring made Phase 1 implementation safer (weight decay added in one place instead of two). Always extract duplication before modifying duplicated code.

**Parameter Migration Strategy:** Breaking parameter changes need three-layer defense: (1) update defaults in context, (2) merge saved params with registry defaults (handles missing params), (3) defensive fallback in UI (last resort). Without all three, edge cases like NaN can slip through.

## Files Modified

1. `src/services/ml/types.ts` - Changed C → weightDecay in LogisticRegressionParams
2. `src/services/ml/classifierRegistry.ts` - Updated parameter schema (weightDecay: 0.01 default, 0.0001-1.0 range)
3. `src/services/ml/logisticRegressionClassifier.ts` - Refactored training methods + AdamW implementation
4. `src/services/ml/__tests__/logisticRegressionClassifier.test.ts` - Updated all test references
5. `src/services/ml/__tests__/types.test.ts` - Fixed parameter references
6. `src/services/ml/__tests__/classifierRegistry.test.ts` - Fixed parameter references
7. `src/contexts/TrainingConfigContext.tsx` - Updated defaults, added migration logic (Phase 3)
8. `src/components/dataset/ClassifierSelector.tsx` - Added defensive fallback (Phase 3)
9. `src/contexts/__tests__/TrainingConfigContext.test.tsx` - Updated parameter references (Phase 3)

## Impact

**Breaking Changes:**
- Saved models with C parameter cannot load (deserialization fails)
- Users must retrain models after upgrade
- UI shows "Weight Decay" instead of "C (Regularization Strength)"
- Phase 3 added backward compatibility: old configs with C parameter auto-migrate to weightDecay

**Code Quality:**
- Net reduction: ~130 lines removed
- Single source of truth for training logic
- Cleaner loss function (no L2 penalty computation)
- Modern regularization (AdamW standard in PyTorch/Transformers)
- Reduced maintenance burden for future enhancements

**User Experience:**
- Phase 3 fixes: New users get correct defaults, existing users auto-migrate, no NaN in UI
- Backward compatible with saved training configurations

**Testing:** All 179 tests passing (164 ML tests + 15 context tests, ~20 seconds execution time)
