# Task 2025-10-26: Add AWAY Presence Detection State

**STATUS:** COMPLETE

## User Request

"add new detection result - not present. So now it is good / bad / not present. Add another detector that detects present / not present, trained with same params as good / bad. Run good bad detection only if person present. This replaces 0.5 person presence score from rtmdet. Add collect no person to collect tab and frames in it. Did i miss anyting? is there better name for no person state? I want to use it when im not working on pc and want detection to pause."

## Development Journey & Struggles

The implementation revealed a cascade of interconnected bugs that required systematic debugging:

**Initial Struggle: "Missing rtmdet features" error during training**
- Symptom: Training failed even though logs showed worker extracting RTMDet features
- Investigation: Traced data flow from worker → main thread → storage
- Discovery: `handleWorkerResult` manually reconstructed `InferenceResult` field-by-field, omitting `rtmDetFeatures`
- Breakthrough: Realized brittle pattern - any new field would be missed
- Solution: Refactored to spread operator pattern for robustness

**Second Issue: "Model outdated" popup immediately after training**
- Symptom: Model showed stale despite just being trained
- Investigation: Compared hashes computed during training vs staleness check
- Discovery: Training used filtered feature types (excludes rtmdet), staleness check used unfiltered (includes rtmdet)
- Breakthrough: Model stores which features it was trained with - use those, not UI config
- Solution: Changed staleness check to use `model.featureTypes` instead of `config.featureTypes`

**Third Challenge: "Person Away" shown without classifier score**
- Symptom: App always showed "Person Away, Good/Bad N/A" even when present
- Investigation: Added logging at every step of classification pipeline
- Discovery #1: Worker only checked `loadedClassifier` (legacy), not dual-model classifiers
- Discovery #2: When 0 persons detected, worker exited before classification
- Discovery #3: Main thread discarded classification by setting `inferenceResult = null`
- Breakthrough: Presence classification doesn't need person detection - that's its purpose!
- Solution: Three fixes - classification gate, run classification before early return, preserve results

**Fourth Problem: Auto-collect disabled during away state**
- Symptom: Couldn't capture "away" frames for training
- Investigation: Traced frame capture validation logic
- Discovery: Validation rejected frames with empty `features` dict, but away frames only have `rtmDetFeatures`
- Breakthrough: Realized validation was too strict - rtmDetFeatures alone are valid
- Solution: Allow capture if either pose features OR rtmDetFeatures present

**Fifth Obstacle: Loading spinner never ends**
- Symptom: Spinner indefinitely visible when reloading with no person present
- Investigation: Checked loading state dependencies
- Discovery: `firstFrameProcessed` required `inferenceResult !== null`, but no-person frames set it to null
- Breakthrough: Loading complete when model loads, not when first frame processes
- Solution: Removed `inferenceResult` dependency from loading check

**Sixth Confusion: rtmdet in UI but ignored during training**
- Symptom: Users could select rtmdet but it had no effect (auto-configured)
- Investigation: Reviewed feature selection and training logic
- Discovery: No declarative way to mark features as "internal use only"
- Breakthrough: Use feature properties instead of hardcoded filtering by name
- Solution: Added `userSelectable: false` property and helper function

**Seventh Bug: Both good and bad showed "100%"** (FIXED 2025-10-31)
- Symptom: Good posture showed "Good 100%", bad showed "Bad 100%"
- Investigation: Traced probability calculations through badge component
- Discovery: Code inverted probability for bad posture: `(1 - goodProbability)`
- Breakthrough: User wanted consistent metric - always show good probability
- Solution: Removed inversion, always display `goodProbability` directly
- **Note:** This fix was documented but never applied. Bug resurfaced and was fixed on 2025-10-31 by changing PostureStatusBadge.tsx lines 159-160 to always show `goodProbability` with label "Good"

**Key Lessons:**
1. Brittle field-by-field object construction causes bugs → Use spread operator
2. Hash inputs must match between training and validation → Use model metadata
3. Early returns bypass critical logic → Run classification before return
4. Assumptions about "person required" broke presence detection → Challenge assumptions
5. Hardcoded string filtering is fragile → Use declarative properties
6. Debug systematically: trace data flow, add logging, challenge assumptions

**Pattern Recognized:** Most bugs stemmed from implicit assumptions (person required for classification, features dict required for capture, etc.) that broke when adding presence detection. The fix was making these assumptions explicit and conditional.

## Critical Discoveries (Non-Obvious)

**1. RTMDet features not flowing to InferenceResult:**
Manual field reconstruction in `handleWorkerResult` missed new `rtmDetFeatures` → training failed. Fixed with spread operator (`...result`) to auto-include all fields.

**2. Model staleness false positives:**
Training filters features (excludes 'rtmdet' for posture), staleness check used unfiltered config → hash mismatch. Fixed: use `model.featureTypes` (trained) not `config.featureTypes` (UI selection).

**3. Worker classification never runs for dual-model:**
Worker only checked `loadedClassifier` (legacy), not `loadedPostureClassifier`/`loadedPresenceClassifier` → "Person Away" showed without scores.

**4. No classification when 0 persons detected:**
Worker exited early when `!bbox`, never reaching classification. Main thread set `inferenceResult = null` → "Away" without score.

**5. Auto-collect blocked during away:**
Validation required non-empty `features` dict, but away frames only have `rtmDetFeatures` → capture blocked.

**6. Loading spinner waits for person:**
`firstFrameProcessed` required `inferenceResult !== null`, but no-person frames set it to null → spinner stuck until person appeared.

**7. rtmdet shown in Training UI:**
No mechanism for "internal use only" features → rtmdet displayed in selector but ignored during training (UX confusion).

**8. Good/Bad probability display inverted:** (FIXED 2025-10-31)
PostureStatusBadge inverted for bad: `(1 - goodProbability)` → both good and bad showed 100%. Fix was documented but never applied; finally fixed on 2025-10-31.

**9. RTMDet feature vector (49 dims):**
24 per-person (top 2: confidence, bbox, area, aspect ratio), 25 spatial grid (5×5 confidence). Always returns features (zeros if no people). Storage: 196 bytes/frame.

**10. Zero-copy transfer via transferables:**
Worker extracts Float32Array, transfers ownership to main thread via postMessage transferables (no serialization).

## Solution

Extended binary (GOOD/BAD) to three-state (GOOD/BAD/AWAY) via dual-classifier architecture:
- **Data model:** Added `AWAY` to FrameLabel enum, `rtmDetFeatures?: Float32Array` to PostureFrame/InferenceResult
- **Feature extraction:** Worker extracts 49-dim RTMDet features, zero-copy transfer via transferables
- **Feature registry:** RTMDet type with `presenceOnly: true` + `userSelectable: false`, UI badges/filtering prevent invalid selections
- **Collection:** "Capture Away" button (blue, keyboard `A`), blue border styling, stats show 4 categories
- **Training:** `trainDualModels()` trains presence (RTMDet, GOOD+BAD vs AWAY) and posture (pose, GOOD vs BAD) with shared config
- **Inference:** Cascaded - presence first (if away → exit with `goodProbability: null`), then posture
- **Storage:** Parallel save/load RTMDet features, `countFramesByLabel()` handles AWAY
- **Backward compatible:** Optional `rtmDetFeatures`, gracefully handles old frames

**Bugs Fixed:**
1. RTMDet features not flowing → spread operator refactor (RTMW3DCameraWeb.tsx)
2. Staleness false positives → use `model.featureTypes` for hash (modelStalenessDetector.ts)
3. Dual-model classification gate → check all three classifiers (unified-pose-worker.ts line 1335)
4. No-person classification → run before early return, preserve in main thread (unified-pose-worker.ts lines 1232-1250, RTMW3DCameraWeb.tsx lines 88-100)
5. Auto-collect blocked → relaxed validation for rtmDetFeatures (useFrameSampler.ts lines 96-103)
6. Loading spinner stuck → removed `inferenceResult` dependency (app/index.tsx lines 314-320)
7. rtmdet in UI → added `userSelectable` property, `getUserSelectableFeatureTypes()` helper (featureRegistry.ts, FeatureMultiSelector.tsx, TrainingTab.tsx)
8. Probability inversion → always show good probability directly (PostureStatusBadge.tsx line 111)