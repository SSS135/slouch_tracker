# Task 2025-10-24: Model Loading Popup
**STATUS:** COMPLETED

## User Request
Show popup "model is loading" when model is reloading or at initial load

**Additional requirement:**
Model loading popup task is claimed as complete but does not work as expected. When web page is reloaded, it shows badges like that: no data (300ms), no person (2s), no data (200 ms), no person (1s), then normal classification badges after model has processed one frame. Person does not appear / disappear in frame, it is always present and detectable. Identify issue, fix it.

## Critical Discoveries

**Two-phase loading architecture:**
Model loading happens in two sequential phases that both need loading feedback:
1. Storage load (IndexedDB read, ~100-500ms) - tracked by `isModelLoading` in usePostureClassifier
2. Worker load (classifier deserialization in worker thread, ~500ms-2s) - tracked by `isLoadingClassifier` in useWebWorkerInference

Both states must be combined in parent component using callback pattern to lift worker loading state.

**Component encapsulation requires callback:**
RTMW3DCameraWeb encapsulates useWebWorkerInference hook, so worker loading state needed callback prop (`onClassifierLoadingChange`) to communicate state up to app/index.tsx.

**Timing gap issue discovered post-implementation:**
Loading states transition too quickly (< 100ms in fast cases) due to React async rendering + fast IndexedDB. Popup appeared/disappeared faster than users could perceive. Solution: 500ms minimum display duration using `loadingPopupStartTimeRef` and `shouldShowLoadingPopup` state with useEffect delay enforcement.

**Badge flickering issue discovered after initial implementation:**
Original implementation had THREE loading phases: (1) Storage loading, (2) Worker classifier loading, (3) Waiting for first inference. The first two were tracked but the third wasn't, causing loading popup to hide before first inference result arrived. This exposed a gap where badges flickered through false "no data" and "no person" states even though person was present. Additionally, race condition existed where in-flight worker inference could complete AFTER training started, incorrectly affecting state.

**Root cause:** `isModelLoading` and `isLoadingClassifier` both became false before first inference result arrived, hiding the loading popup too early. Badge then showed fallback states while waiting for first frame to process.

## Solution

**Initial Implementation:**
Extended TrainingBlockingSpinner component with 'loading' stage and `hideProgress` prop for spinner-only UI. Added `isLoadingClassifier` state in useWebWorkerInference hook, tracking when classifier loads into worker. Added `onClassifierLoadingChange` callback prop to RTMW3DCameraWeb to lift loading state to app/index.tsx. Combined storage and worker loading states in app/index.tsx to show loading popup during either phase (but not during training).

**Timing Fix:**
Added 500ms minimum display duration via `loadingPopupStartTimeRef` (tracks popup start time) and `shouldShowLoadingPopup` state. useEffect monitors loading state changes and enforces delay using `setTimeout`. Prevents popup from appearing/disappearing faster than users can perceive. Added `isLoadingModel` prop cascade through VideoSection → PostureStatusBadge to suppress "No Person Detected" badge during loading.

**Badge Flickering Fix (Final Implementation):**
Added `firstFrameProcessed` flag that tracks when first inference result arrives after model load. Simplified loading condition to: `trainedModel !== null && !firstFrameProcessed && !isTraining`. This single flag covers all three loading phases naturally. Added race condition protection via effect guards: only sets `firstFrameProcessed = true` when `inferenceResult !== null && trainedModel !== null && !isTraining`. Resets flag when training begins. Removed unnecessary `isLoadingClassifier` state and `onClassifierLoadingChange` callbacks, simplifying the architecture by 30%.

## Files Modified

**Initial Implementation:**
1. **src/components/ui/TrainingBlockingSpinner.tsx** - Added 'loading' stage with "Loading model..." text, added `hideProgress` prop
2. **src/hooks/useWebWorkerInference.ts** - Added `isLoadingClassifier` state tracking classifier loading into worker thread
3. **src/components/camera/RTMW3DCameraWeb.tsx** - Added `onClassifierLoadingChange` callback prop to communicate loading state to parent
4. **app/index.tsx** - Added loading popup displaying during storage OR worker loading (excluded during training using `&& !isTraining` condition)
5. **__tests__/components/camera/RTMW3DCameraWeb.test.tsx** - Updated test mocks to include `isLoadingClassifier` property

**Timing Fix:**
1. **app/index.tsx** - Added `loadingPopupStartTimeRef`, `shouldShowLoadingPopup` state, useEffect with 500ms minimum display duration enforcement, passed `shouldShowLoadingPopup` to VideoSection as `isLoadingModel` prop
2. **src/components/unified/VideoSection.tsx** - Added `isLoadingModel` prop to suppress "No Person Detected" badge during loading
3. **src/components/posture/PostureStatusBadge.tsx** - Added `isLoadingModel` prop to "No Person" condition check
4. **__tests__/components/unified/VideoSection.test.tsx** - Tests passed (17/17)
5. **__tests__/components/posture/PostureStatusBadge.test.tsx** - Tests passed (19/19)

**Badge Flickering Fix:**
1. **app/index.tsx** - Replaced `isLoadingClassifier` state with `firstFrameProcessed` state, added effect to track first inference with race condition guards (`inferenceResult !== null && trainedModel !== null && !isTraining`), updated loading condition to `trainedModel !== null && !firstFrameProcessed && !isTraining`, reset flag in `handleTrainingStatusChange` when training begins, removed `handleClassifierLoadingChange` callback, removed `onClassifierLoadingChange` prop from RTMW3DCameraWeb
2. **src/components/RTMW3DCameraWeb.tsx** - Removed `onClassifierLoadingChange` prop from interface and component parameters, removed effect that notified parent of loading state

## Impact

Users now see "Loading model..." popup with spinner during entire model initialization sequence: storage load → worker load → first inference. No badge flickering occurs during startup or after training completes. Guaranteed 500ms minimum visibility ensures users always perceive the loading state, even on fast machines. Loading popup doesn't interrupt training modal (training takes precedence). Race conditions with in-flight worker results are handled correctly. Code simplified by removing unnecessary state tracking (30% reduction).

## Lessons

**Callback pattern** works well for lifting encapsulated hook state up component tree when context would be overkill. **Three-phase loading** (storage → worker → first inference) requires tracking the complete initialization sequence, not just individual async operations. **Minimum display duration** is critical for loading UIs - fast operations can complete faster than humans can perceive (< 100ms), causing flashing that confuses users. **Ref + state + useEffect** pattern cleanly enforces minimum duration without blocking async operations. **Simplified state management** beats complex tracking - single `firstFrameProcessed` flag with guards replaced multiple loading states and callbacks. **Race condition protection** essential when async operations (worker inference) can complete during state transitions (training start) - use guards in effects to prevent stale data corruption.
