# Task 2025-11-02: Fix Training Model Not Showing After Training
**STATUS:** COMPLETED

## User Request
I've reset all app data, captured 3 frames in each category. It ran auto training, then I clicked training to be sure. It still says no model training. See logs.

## Critical Discoveries

**1. Cross-validation fold constraint with small datasets:**
With 3 AWAY frames and cvFolds=5, stratified split creates degenerate folds with 0 training samples for minority class (each fold reserves 1 for testing, leaving only 2). Fixed: `actualFolds = Math.min(nFolds, Math.floor(minClassSize / 2))` guarantees >=1 training sample per class per fold.

**2. UI model state doesn't auto-reload after training:**
Models saved to IndexedDB by worker, but main thread never reloaded them. UI continued showing `hasModel=false` despite successful training. Required custom event dispatch to trigger `usePostureClassifier` reload from IndexedDB.

**3. Dual-model training had no graceful degradation:**
Single model failure caused entire training to fail. With 3 AWAY frames, presence model couldn't train but posture model had sufficient data (6 frames).

## Solution

**Cross-validation fix:** Updated `crossValidation.ts` fold calculation to `Math.floor(minClassSize / 2)`. With 3 AWAY frames, limits to 1 fold instead of 3 degenerate folds.

**Graceful degradation:** Added independent pre-training validation per model in `training-worker.ts`. Presence requires >=2 AWAY + >=2 PRESENT, posture requires >=2 GOOD + >=2 BAD. Skip models with insufficient data (warn user), continue training others.

**Auto-reload models:** Implemented `loadModelsIntoUnifiedWorker()` in `useModelTraining.ts` dispatching `modelsUpdated` custom event after training. `usePostureClassifier` listens for event and reloads from IndexedDB, triggering UI state refresh without page reload.

**UI validation:** Added pre-training checks in `TrainingTab.tsx` displaying warning badges for insufficient data per model. Shows which models will train vs skip before starting.

## Related
- `tasks/2025-11-01-feature-move-training-to-web-worker.md` - Web Worker training architecture
- `tasks/2025-10-26-feature-add-away-presence-detection.md` - Dual-model architecture
- `tasks/2025-10-24-fix-training-validation-and-buffer-size.md` - Previous validation fixes
