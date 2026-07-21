# Task 2025-10-24: Add Detection Thresholds to Runtime Settings
**STATUS:** FIXED (3 Critical Bugs)

## User Request
Add person and slouch detection thresholds to runtime settings. Both sliders should have 0-1 range, placed in Developer Settings section, with immediate updates and percentage display. Visual indicator in PostureStatusBadge when ML confidence below threshold.

## Critical Discoveries (Bugs Found Post-Completion)

**1. Person Detection Threshold Never Wired Up:**
Threshold slider was implemented but `useMultiTaskDetection` never called with settings parameter. Person detection always used hardcoded 0.5 default. Fixed by passing settings from CameraContext.

**2. ML Confidence Threshold Visual-Only:**
Threshold only added orange warning border - didn't act as decision boundary. Classifier still predicted "good" below threshold. Fixed by applying threshold in `PostureStatusBadge` logic: `isGood = goodProbability > threshold`.

**3. ClassificationResult Over-Engineering:**
Type had redundant fields: `prediction` (computed from probabilities), `confidence` (meaningless per user), `probabilities: {good, bad}` (bad = 1 - good). Refactored to single `goodProbability: number`. Removed `predict()` method entirely, changed `predictProba()` to return scalar.

**4. Debug Logging Performance:**
Classifiers logged expensive serialization (`JSON.stringify(distances)`) unconditionally. Added conditional guards: `if (logger.isDebugEnabled('worker'))` before expensive operations.

**5. Orange Warning Pattern:**
Visual indicator uses orange border + "!" icon (not red) to signal low confidence without implying error. Appends "(Low)" text for accessibility.

**6. Dependency Array Sensitivity:**
`useMultiTaskDetection` hook requires `personDetectionConfidence` in dependency array to trigger re-computation when threshold changes. Without it, slider moves but detection uses stale value.

## Solution

**Phase 1 (Original Implementation):**
Added two threshold sliders to Developer Settings: Person Detection (RTMDet confidence floor) and ML Confidence (classifier trust boundary). Settings persist to localStorage with 0.5 defaults, auto-upgrade existing saves, use 0.05 step increments.

**Phase 2 (Refactoring):**
Eliminated 3-layer prop drilling via `DetectionSettings` interface in CameraContext. Changed scalar params to settings object pattern: `detectMultiTask(result, settings?)` and `useMultiTaskDetection(inferenceResult, settings?)`.

**Phase 3 (Bug Fixes):**
1. Wired person detection threshold through `useMultiTaskDetection` (was unused)
2. Applied ML threshold as decision boundary in `PostureStatusBadge`
3. Simplified `ClassificationResult` type (removed prediction, confidence, probabilities object)
4. Changed classifiers to return single `goodProbability` number
5. Added logging guards for expensive debug operations

**Breaking Changes:** No backward compatibility for serialized models (old format incompatible). Users must retrain.

## Files Modified

**Core (Phase 1 + 2):** `useCameraSettings.ts`, `detection.ts`, `useMultiTaskDetection.ts`, `PostureStatusBadge.tsx`, `RuntimeTab.tsx`, `VideoSection.tsx`, `CameraContext.tsx`, `app/index.tsx`

**Bug Fixes (Phase 3):** `types.ts` (ClassificationResult), `baseClassifier.ts`, `logisticRegressionClassifier.ts`, `knnClassifier.ts`, `unified-pose-worker.ts`, `logger.ts` (isDebugEnabled), `guards.ts`, `schemas.ts`, `jest.setup.js`

**Tests:** `useCameraSettings.test.ts`, `useMultiTaskDetection.test.ts`, `RuntimeTab.test.ts`, `VideoSection.test.tsx`, `PostureStatusBadge.test.tsx`, classifier tests, worker tests, validation tests

## Impact

**Functionality:** Users can now fine-tune detection sensitivity without code changes. Thresholds properly applied to both person detection and ML predictions.

**Architecture:** Eliminated prop drilling, reduced coupling, improved extensibility via settings object pattern. Better adherence to KISS/DRY/React Context best practices.

**Performance:** Debug logging no longer serializes large data structures unconditionally.

**Code Quality:** Simplified ML types by removing redundant/computed fields. Single source of truth for predictions.
