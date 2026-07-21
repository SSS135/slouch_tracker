# Task 2025-10-31: Set Default Optimizer to Per-Feature and No Random Projection

**STATUS:** COMPLETED

## User Request
"set default optimizer to per-feature and no random projection"

**Clarifications:**
- "per-feature optimizer" means "per-feature normalization mode"
- Keep alternatives available in UI (just change defaults)
- classifierRegistry should be single source of truth for classifier parameter defaults

## Critical Discoveries

**1. Mismatched defaults between context and registry:**
TrainingConfigContext had hardcoded defaults (`weightDecay: 0.01`, `maxIterations: 1000`, `learningRate: 0.001`) that differed from classifierRegistry defaults (`weightDecay: 30.0`, `maxIterations: 100`, `learningRate: 0.01`). New users got context defaults, but UI showed registry defaults - confusing UX.

**2. classifierRegistry as single source of truth:**
Using `getDefaultParams('logistic_regression')` in DEFAULT_CONFIG eliminates duplication. Context now pulls defaults from registry automatically. Future classifier changes propagate to context without manual sync.

**3. Backward compatibility maintained:**
Existing users' saved configurations unaffected. Only DEFAULT_CONFIG changes (for new users or reset scenarios). LocalStorage preserved, no migration needed.

## Solution

**Updated TrainingConfigContext Defaults** (`src/contexts/TrainingConfigContext.tsx`, lines 51-62):
- Changed `normalizationMode: 'layer'` → `'per_feature'`
- Changed `dimReductionConfig.method: 'random_projection'` → `'none'`
- Replaced hardcoded `params` with `getDefaultParams('logistic_regression')` call
- Updated fallback (line 93): Changed hardcoded `'layer'` to `DEFAULT_CONFIG.normalizationMode`

**Before:**
```typescript
const DEFAULT_CONFIG: TrainingConfig = {
  // ...
  normalizationMode: 'layer',
  dimReductionConfig: { method: 'random_projection', params: { targetDims: 100 } },
  params: { weightDecay: 0.01, maxIterations: 1000, learningRate: 0.001 }
};
```

**After:**
```typescript
const DEFAULT_CONFIG: TrainingConfig = {
  // ...
  normalizationMode: 'per_feature',
  dimReductionConfig: { method: 'none', params: {} },
  params: getDefaultParams('logistic_regression') // now: { weightDecay: 30.0, maxIterations: 100, learningRate: 0.01 }
};
```

**Updated Tests** (`src/contexts/__tests__/TrainingConfigContext.test.tsx`):
Updated expectations to match new defaults:
- `weightDecay: 0.01` → `30.0`
- `maxIterations: 1000` → `100`
- `learningRate: 0.001` → `0.01`
- `dimReductionConfig.method: 'random_projection'` → `'none'`
- Added assertion for `normalizationMode: 'per_feature'`

All 15 tests passing.

## Lessons

**Registry as Single Source of Truth:** Using `getDefaultParams()` in context eliminates dual maintenance burden. Classifier registry changes automatically propagate to new user defaults. No manual sync needed.

**Per-Feature Normalization Preferred:** Layer normalization normalizes across features (makes different feature types comparable). Per-feature normalization normalizes each feature independently (preserves feature-specific scales). Per-feature generally better for diverse feature types (keypoints, geometric, backbone).

**Dimensionality Reduction Often Unnecessary:** Random projection adds computational overhead and complexity. For low-dimensional features (keypoints, geometric) it's unnecessary. For high-dimensional features (backbone, neck), users can enable it explicitly when needed.

## Related

- `tasks/2025-10-23-feature-add-per-feature-normalization.md` - Added per-feature normalization mode
- `tasks/2025-10-25-refactor-generic-feature-system.md` - Established feature type flexibility
- `tasks/2025-10-28-refactor-adamw-weight-decay.md` - Changed weightDecay defaults in registry

## Files Modified

- `src/contexts/TrainingConfigContext.tsx` (lines 51-62, 93)
- `src/contexts/__tests__/TrainingConfigContext.test.tsx` (updated test expectations)

## Impact

**New User Experience:**
- Get recommended defaults immediately (per-feature normalization, no dim reduction)
- Faster training (no random projection overhead)
- Better defaults aligned with registry values

**Maintainability:**
- Single source of truth for classifier parameters
- No more manual sync between context and registry
- Future classifier changes automatically propagate

**Backward Compatibility:**
- Existing users unaffected (saved configs preserved)
- No breaking changes or migrations needed

**Testing:** All 15 TrainingConfigContext tests passing
