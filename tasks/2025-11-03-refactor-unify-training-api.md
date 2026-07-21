# Task 2025-11-03: Unify Training API
**STATUS:** COMPLETED

## User Request
why have separate trainDualModels and queueTraining? all training should use queue. Unify them, add do CV parameter or something like that. Propose how to improve. Plan.

train should not receive a dataset, it should send message to worker, worker should load latest dataset. plan

I prefer moving cvFolds to training settings. source: 'manual' - why do we need this parameter at all? also I prefer approach A. continue planning

create a task. use do cv parameter instead of skip cv. do not create any migration or backward compatibility code, we do not need it. do not implement it for now, just task creation.

## Critical Discoveries

**1. RTMW3DCameraWeb feature mapping bug (BLOCKING):**
Component used outdated field names from previous refactoring: `rtmDetP5Features` and `intermediateFeatures`. Worker actually returns `presenceFeatures` and `postureFeatures`. Caused all frame captures to fail with "No features available in inference result" error. Fixed by using spread operator `...result` to preserve correct field names from worker ProcessResult. Must fix before any training can work.

**2. Dataset transfer overhead:**
Training transferred 10+ MB dataset + config parameters on every training request despite worker having direct IndexedDB access. Worker never used its storage capability. Solution: Worker loads everything from IndexedDB autonomously, eliminating parameter passing and data transfer.

**3. Config debounce race condition:**
TrainingConfig auto-saves with 300ms debounce. Training could start before config flush completed, causing worker to load stale config from IndexedDB. Added `flushToStorage()` to force immediate save before training starts, bypassing debounce.

**4. trainingSource parameter only for UI semantics:**
Parameter `source: 'manual' | 'auto'` passed through multiple layers (TrainingTab → useAutoTraining → train() → worker → state) only to control UI behavior (show blocking spinner vs badge). Replaced with local `isAutoTrainingActive` flag in useAutoTraining hook. Training logic doesn't need to know about UI presentation.

## Solution

**API unification:** Replaced 3 functions (trainModel, trainDualModels, queueTraining) with single `train(options?: { doCV?: boolean })`. Simplified from 6+ parameters to optional doCV flag (defaults to true). Auto-queuing built-in with last-wins semantics (replace pending request, don't stack).

**Worker autonomy:** Worker loads dataset and config from IndexedDB instead of receiving as parameters. On `train` message, worker calls `datasetStorage.loadDataset()` and `datasetStorage.getTrainingDefaults()` to get fresh data. Eliminates 10+ MB transfer and ensures atomic snapshot of training data.

**Config organization:** Moved `cvFolds` from CameraSettings to TrainingDefaults alongside classifier/dimReduction/features for logical grouping. Added `updateCvFolds()` and `flushToStorage()` to TrainingConfigContext. UI moved cvFolds selector from RuntimeTab to TrainingTab.

**UI separation:** Removed `trainingSource` from TrainingState. Added `isAutoTrainingActive` boolean flag to useAutoTraining hook. VideoSection reads flag directly from useAutoTraining instead of training state. Training API no longer couples to UI presentation layer.

**Message protocol:** Simplified training message to `{ type: 'train', payload: { doCV?: boolean } }`. Removed dataset, classifierConfig, dimReductionConfig, featureTypes, normalizationMode, cvFolds, and source parameters. Worker handles all data loading internally.

**Files updated:** RTMW3DCameraWeb.tsx (feature mapping fix), useModelTraining.ts (unified API), training-worker.ts (load from IndexedDB), TrainingContext.tsx (expose train()), types.ts (add cvFolds to TrainingDefaults), TrainingConfigContext.tsx (cvFolds + flush), useCameraSettings.ts (remove cvFolds), useAutoTraining.ts (isAutoTrainingActive flag), TrainingTab.tsx (use config.cvFolds, call train()), VideoSection.tsx (use isAutoTrainingActive), RuntimeTab.tsx (remove cvFolds selector).

**Result:** ~300 lines deleted, no data transfer, single source of truth (IndexedDB), simpler API (`train()` or `train({ doCV: false })`), separation of training logic from UI concerns.

## Related

- `tasks/2025-11-01-feature-move-training-to-web-worker.md` - Web Worker training implementation
- `tasks/2025-11-01-feature-auto-training-on-capture.md` - Queue system and trainingSource pattern
- `tasks/2025-11-03-fix-training-blocks-video-detection.md` - Worker yielding patterns
- `tasks/0013-refactor-consolidate-classification-state.md` - Anti-pattern: splitting data from single source
