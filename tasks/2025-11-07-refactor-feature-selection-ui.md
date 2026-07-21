# Task 2025-11-07: Simplify Feature Selection UI
**STATUS:** COMPLETED

## User Request
improve feature selection ui. remove recommended, posture only, presence only, storage infos. remove dim info from description text. make default features GAU avg, max, std.

## Critical Discoveries

**1. Recommended field was unused except for UI badges:**
The `recommended` field in FeatureDefinition was only used for displaying badges in the UI. No business logic depended on it. Removal was completely safe and cleanup was straightforward.

**2. Test expectations needed updating:**
TrainingConfigContext tests expected old default features (FEATURE_BACKBONE, FEATURE_GAU) and had to be updated to new defaults (FEATURE_GAU, FEATURE_GAU_MAX, FEATURE_GAU_STD). Tests also had outdated weightDecay expectation (30.0 vs actual default 1.0).

## Solution

**Removed recommended field system:**
- Deleted `recommended: boolean` from FeatureDefinition interface (featureRegistry.ts:90)
- Removed `recommended` property from all 9 feature definitions
- Deleted `getRecommendedFeatureTypes()` helper function
- Removed badge logic from FeatureMultiSelector.tsx (lines 107, 147-161)
- Removed badge logic from FeatureTypeSelector.tsx (React Native component)
- Deleted "Feature recommendations" test suite from featureRegistry.test.ts

**Cleaned UI components:**
- FeatureMultiSelector.tsx: Removed Recommended/Posture Only/Presence Only badges, removed Dims/Storage metadata row. Now shows only icon, name, description per feature.
- FeatureTypeSelector.tsx: Same cleanup for React Native version. Removed badge styles, simplified header layout.

**Updated feature descriptions:**
All 9 feature descriptions cleaned to remove dimension counts. Examples:
- Before: "Average pooled backbone features from RTMPose (avg pooling, 512 dims, computed on-demand)"
- After: "Average pooled backbone features from RTMPose - Computed on-demand"

**Changed default features:**
- TrainingConfigContext.tsx: Updated DEFAULT_CONFIG.featureTypes from `[FEATURE_BACKBONE, FEATURE_GAU]` to `[FEATURE_GAU, FEATURE_GAU_MAX, FEATURE_GAU_STD]`
- Added imports for FEATURE_GAU_MAX and FEATURE_GAU_STD
- Updated comment to reflect new defaults

**Test fixes:**
- featureRegistry.test.ts: Removed `recommended` property assertion, deleted two test suites (recommendations + getRecommendedFeatureTypes)
- TrainingConfigContext.test.tsx: Fixed default feature types expectations (unit-test-engineer agent)

## Related
None - First comprehensive UI cleanup for feature selection system.
