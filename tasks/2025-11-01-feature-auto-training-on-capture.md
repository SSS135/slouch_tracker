# Task 2025-11-01: Auto-Training on Frame Capture
**STATUS:** COMPLETED

## User Request
When user add frame to dataset via capture button or auto-capture list, it starts training right away without user prompt.

## Critical Discoveries

**Queue System:** Training queue uses ref to store pending params. If training in progress, queue stores latest config and auto-starts after completion. Only ONE queued training (no multi-level stack).

**Training Source Tracking:** Added `trainingSource: 'manual' | 'auto' | null` to differentiate Training tab (blocking spinner) from auto-triggered (subtle badge only).

**Three Capture Paths:** Auto-training integrated into `handleCaptureWithLabel()` (manual G/B/A), `handleSaveAll()` (batch), `handleSaveFrameWithLabel()` (single).

## Solution

**Training Queue** (`useModelTraining.ts`): Added `trainingQueued`, `trainingSource` state; queue ref stores pending params; `queueTraining()` function; auto-starts after completion.

**Auto-Training Hook** (`useAutoTraining.ts`): Checks minimum frames, calls `trainModel(source='auto')`, queues if busy, graceful error handling.

**Frame Capture** (`UnifiedPosturePage.tsx`): `await autoTraining.triggerTraining()` after all 3 capture paths; `TrainingBlockingSpinner` only for `trainingSource === 'manual'`.

**Visual Indicator** (`VideoSection.tsx`): Orange training badge "Training... X%" below PostureStatusBadge; only shows for `trainingSource === 'auto'`.

**Context Updates** (`TrainingContext.tsx`): Exposed `trainingQueued`, `trainingSource`, `queueTraining()`.

**Why Queue (Not Debounce):** Ensures all captures result in training (no lost updates). Simple queue prevents excessive runs while using latest dataset.

**Why Always-On:** Training completes in seconds (~1-60s), making per-capture training viable. Matches user expectation: "I captured bad posture → model learns immediately."

## Related

- `tasks/2025-11-01-feature-move-training-to-web-worker.md` - Web Worker training (dependency)
- `tasks/2025-10-26-feature-hash-based-model-staleness-tracking.md` - Model staleness tracking
- `tasks/2025-10-31-feature-frame-list-smart-queueing.md` - Queueing pattern and badge UI

## Files Modified

- `src/hooks/useModelTraining.ts` - Training queue system
- `src/hooks/useAutoTraining.ts` - NEW: Auto-training hook
- `src/pages/UnifiedPosturePage.tsx` - Integration with capture handlers
- `src/components/unified/VideoSection.tsx` - Training badge indicator
- `src/contexts/TrainingContext.tsx` - Queue state and trainingSource

## Impact

- Training starts automatically after every frame capture (G/B/A keys, auto-capture batch, single frame save)
- Models stay up-to-date with latest dataset without manual "Train Model" clicks
- Queue prevents duplicate concurrent training runs
- Non-blocking UI during auto-training (badge indicator only)
- Manual training from Training tab still shows blocking spinner (preserved UX)
