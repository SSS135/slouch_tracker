# Task 2025-10-24: Fix Training Validation and Buffer Size Constants
**STATUS:** COMPLETED

## User Request
1. Reduce minimum frames per class requirement from 5 to 1 for training
2. Fix auto-capture counter that stops at 20 instead of expected 30 limit

## Critical Discoveries

**1. DRY violation in validation:**
`storage.ts` hardcoded `good >= 5 && bad >= 5` instead of importing `TRAINING_CONFIG.minFramesPerClass`. Two sources of truth → maintenance burden.

**2. Incomplete refactoring from Task 0015:**
Task 0015 increased buffer 20→30 but only updated 2 of 5 locations:
- ✅ Local constants in CollectTab.tsx, UnifiedFrameGrid.tsx
- ❌ Missed: app/index.tsx (actual limit), DataCollectionControls.tsx (UI display), UnifiedFrameGrid.tsx default param

Result: Auto-capture stopped at 20, UI showed "20/30" implying 30 limit.

## Solution

**Issue 1 - Unified minimum frames validation:**
Changed `minFramesPerClass: 5 → 1` in config.ts. Fixed storage.ts to import and use `TRAINING_CONFIG.minFramesPerClass` instead of hardcoded 5. Updated UI text in TrainingTab and DataCollectionControls to reflect "1 frame" requirement.

**Issue 2 - Centralized buffer size constant:**
Created `src/services/dataset/constants.ts` with `MAX_BUFFER_SIZE = 30` as single source of truth. Updated all 5 locations to import and use this constant: app/index.tsx (useFrameSampler maxBufferSize), CollectTab.tsx (replaced local constant), DataCollectionControls.tsx (StatItem max), UnifiedFrameGrid.tsx (default param and local constant).

## Files Modified

**Configuration & Storage (2):**
- `src/services/ml/config.ts` - minFramesPerClass: 5→1
- `src/services/dataset/storage.ts` - Import TRAINING_CONFIG, replace hardcoded validation

**New Constant Module (1):**
- `src/services/dataset/constants.ts` - NEW: Export MAX_BUFFER_SIZE = 30 with JSDoc

**Buffer Size Updates (4):**
- `app/index.tsx` - Use MAX_BUFFER_SIZE in useFrameSampler
- `src/components/unified/CollectTab.tsx` - Import shared constant
- `src/components/dataset/DataCollectionControls.tsx` - Use constant in StatItem max
- `src/components/dataset/UnifiedFrameGrid.tsx` - Use constant in default param

**UI Text Updates (2):**
- `src/components/unified/TrainingTab.tsx` - "5 frames"→"1 frame"
- `src/components/dataset/DataCollectionControls.tsx` - "5 frames"→"1 frame"

**Tests (3):**
- `src/services/dataset/__tests__/constants.test.ts` - NEW: 6 tests validating constant
- `src/components/dataset/__tests__/DataCollectionControls.test.tsx` - Updated assertion
- `src/components/unified/__tests__/TrainingTab.test.tsx` - Updated assertion

## Impact

**Training workflow:** Users can now train with 1 frame per class (faster experimentation during development). Cross-validation still provides meaningful feedback even with minimal data.

**Auto-capture consistency:** Counter correctly shows "X/30" everywhere. Buffer properly fills to 30 frames. Single source of truth prevents future drift.

**Architecture improvement:** New constants.ts follows pattern from ml/config.ts, provides clear home for dataset configuration values.
