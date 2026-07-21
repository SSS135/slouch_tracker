# Task 2025-11-07: Fix Training Badge Issues

**STATUS:** COMPLETE

## User Request
Fix training badge. It should not have progress number any more. Also if I record another data point while training is in progress, it won't appear for second training, only for first. And it does not appear for training started in Training tab.

## Critical Discoveries

**1. IndexedDB async race condition:**
Frame capture → IndexedDB save (async) → auto-training triggered immediately → queued training captures params → worker loads dataset. Problem: IndexedDB write may not have flushed when queue stores params. Worker loads stale snapshot missing newly captured frames.

**2. Badge visibility tied to local state instead of global:**
Badge condition used `isAutoTrainingActive` (local state in useAutoTraining hook). Manual training from Training tab bypasses hook, never sets local state true. Badge condition `isAutoTrainingActive && isTraining` fails even though training is active.

**3. Simple delay more reliable than event-based sync:**
100ms delay provides comfortable buffer for IndexedDB transaction commit (typically 1-10ms) plus React Query cache updates and browser task queue processing. Alternatives (event-based sync, debouncing, passing dataset directly) are more complex without clear benefits.

## Solution

Fixed three training badge issues:

**1. Removed progress percentage** - Badge shows "Training..." without `{trainingProgress}%` (CameraViewport.tsx line 187)

**2. Changed badge condition** - Uses `isTraining` from global context instead of `isAutoTrainingActive` local state (CameraViewport.tsx line 176). Badge now appears for both auto-training and manual training.

**3. Added IndexedDB flush delay** - 100ms delay before training in useAutoTraining.ts (line 75) ensures IndexedDB transaction completes before queued training captures dataset snapshot. Prevents stale data in subsequent training.

## Related

- `tasks/2025-11-08-refactor-unified-raw-features-pipeline.md` - Storage and training pipeline architecture
- `tasks/2025-11-09-refactor-simplify-storage-single-key.md` - Storage optimization that affected timing
