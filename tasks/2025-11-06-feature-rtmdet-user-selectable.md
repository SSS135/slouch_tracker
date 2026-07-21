# Task 2025-11-06: Make RTMDet P5 Features User-Selectable

**STATUS:** COMPLETED

## User Request
add rtmdet features to features selectable for posture detection training

## Critical Discoveries

**1. Two-line change enables full functionality:**
Only `userSelectable: false` → `true` and `modelType: 'presence'` → `undefined` needed. All infrastructure (extraction, concatenation, storage) already supported mixed feature types via `FeatureContainer` interface.

**2. Presence training remains auto-configured:**
Training worker hard-codes `presenceFeatureTypes = [FEATURE_NECK_P5]` for PRESENT vs AWAY detection. User selection only applies to posture quality training, not presence detection. This is correct - presence needs specific features.

**3. Feature dictionary separation is organizational, not restrictive:**
P5 stored in `presenceFeatures` (RTMDet output), posture features in `postureFeatures` (RTMPose output). Extract functions handle both dictionaries transparently - no special logic needed for cross-dictionary concatenation.

## Solution

**Code Changes:**
1. `src/services/dataset/featureRegistry.ts` (lines 248-249):
   - `modelType: 'presence'` → `modelType: undefined` (removes restriction badge)
   - `userSelectable: false` → `userSelectable: true` (shows in UI)

2. `specs.md`:
   - Updated "presence detection only" → "user-selectable for both posture and presence detection"
   - Added P5 to dimensionality reduction recommendations

**Tests Added (11 new tests):**
- `featureRegistry.test.ts`: 6 tests for user-selectability, modelType validation, P5 extraction
- `featureExtractors.test.ts`: 5 tests for P5 concatenation with posture features (GAU, Backbone)
- All 70 tests passing

**Verification:**
- P5 now appears in `getUserSelectableFeatureTypes()`
- Training tab UI displays "RTMDet P5 Neck (192 dims)" without model restriction badge
- Multi-feature concatenation works: P5 (192) + GAU (256) + Backbone (512) = 960 dims

## Related
- `tasks/2025-10-26-feature-add-away-presence-detection.md` - Introduced `userSelectable` property pattern
- `tasks/2025-10-27-feature-update-worker-model-outputs.md` - Added RTMDet P5 features initially
- `tasks/2025-10-25-refactor-generic-feature-system.md` - Established flexible `FeatureContainer` interface
