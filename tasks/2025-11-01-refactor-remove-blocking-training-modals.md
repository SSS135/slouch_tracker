# Task 2025-11-01: Remove Blocking Training Modals and Model Outdated Warnings
**STATUS:** COMPLETE

## User Request
After that we need to remove model outdated popup, replacing it with model training popup. Remove modal blocking window on training. During training detection must still work, then switch to new model automatically when training is done.

## Critical Discoveries

**Auto-Training Eliminates Staleness:** With auto-training (Task 2), models retrain on EVERY frame capture. Staleness detection becomes impossible by design - models are always current. The "Model Outdated" warning contradicts auto-training behavior (warning appears while retraining is already queued).

**Web Worker Enables Non-Blocking UI:** Training moved to Web Worker (Task 1) ensures detection continues at full FPS during training. Blocking modals contradict this architecture - they prevent interaction during background operations that don't actually block anything.

**Unified Removal Pattern:** Three UI layers all checking staleness/loading: UnifiedPosturePage (modal control), VideoSection (prop passing), PostureStatusBadge (StalenessNotice rendering). Removed from all three for clean architecture.

## Solution

Removed all blocking UI elements and staleness detection from codebase:

**UnifiedPosturePage.tsx:**
- Removed `TrainingBlockingSpinner`, `useModelStaleness` imports
- Removed `shouldShowLoadingPopup` state, `loadingPopupStartTimeRef` ref, loading popup useEffect
- Removed `handleQuickTrain` callback (used by "Train Now" button)
- Removed both TrainingBlockingSpinner instances (manual training + model loading popups)
- Removed `isLoadingModel`, `modelIsStale`, `onTrainNow` props from VideoSection

**VideoSection.tsx:**
- Removed `isLoadingModel`, `modelIsStale`, `onTrainNow` from interface and props
- Removed prop passes to PostureStatusBadge
- Kept orange training badge for auto-training (non-blocking indicator from Task 2)

**PostureStatusBadge.tsx:**
- Removed unused Mantine imports (Alert, Badge, Button, ThemeIcon)
- Removed `isLoadingModel`, `modelIsStale`, `onTrainNow` from interface
- Removed entire `StalenessNotice` component (22 lines)
- Removed `shouldShowStaleness` logic and all StalenessNotice renders (3 locations)
- Removed "Loading model..." and "Updating model..." badges (2 locations)
- Simplified to single Paper components (removed Stack wrappers)

## Files Modified

**Modified (3 files):**
- `src/pages/UnifiedPosturePage.tsx` - Removed blocking modals, staleness detection, loading popup
- `src/components/unified/VideoSection.tsx` - Removed staleness props
- `src/components/PostureStatusBadge.tsx` - Removed StalenessNotice, loading badges

## Build Verification

✅ Build succeeded in 39.77s
- No compilation errors
- Training worker: 1,738.89 kB
- Unified pose worker: 2,052.29 kB

## Testing

**Tests Fixed:**
- PostureStatusBadge.test.tsx: Removed 85 lines of staleness tests (24 tests passing)
  - Removed "Loading States" section (2 tests)
  - Removed "Model Staleness" section (4 tests)

**Files Deleted:**
- `src/hooks/useModelStaleness.ts` - Staleness detection hook (no longer used)
- `src/services/dataset/modelStalenessDetector.ts` - Hash computation service (no longer used)

**Hash Computation Removed:**
- training-worker.ts: Removed hash computation for presence + posture models
- storage.ts: Removed contentHash computation for frames
- Hash fields remain optional in TrainedModel type (backward compatible)

**Test Results:** 39 suites passed, 797 tests passed (3 pre-existing failures unrelated to this task)

**User Experience:** Detection continues during training (Web Worker), auto-training badge shows progress (non-blocking), no staleness warnings, models auto-load after training.

## Related

- `tasks/2025-11-01-feature-move-training-to-web-worker.md` - Web Worker architecture (dependency)
- `tasks/2025-11-01-feature-auto-training-on-capture.md` - Auto-training and badge indicator (dependency)
- `tasks/0001-feature-add-training-blocking-modal.md` - Original blocking modal implementation
- `tasks/2025-10-26-feature-hash-based-model-staleness-tracking.md` - Staleness detection system
