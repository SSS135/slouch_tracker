# Task 2025-11-04: Refactor Feature Registry with Safe Getter Function

**STATUS:** COMPLETED

## User Request

"refactor feature registry, create a function to get feature from registry that throws error when this feature not found. use it instead of manual index access and checks. assume it is a very broken behaviour when feature not found and we should throw error"

## Critical Discoveries

**1. Fail-fast error pattern prevents downstream bugs:**
`requireFeatureDefinition()` throws immediately with feature type and available types list. This catches programming errors at the source instead of silently propagating bad data (returning `0` dimensions or `false` for computed checks).

**2. Helper functions must also throw:**
`getFeatureDimensions()` and `isComputedFeature()` previously returned fallback values (`0`, `false`) for missing features. These fallbacks masked bugs. Both now delegate to `requireFeatureDefinition()` and throw on invalid types.

**3. Test utilities need graceful fallbacks:**
`mockPostureFrame.ts` and `mockInferenceResult.ts` use try-catch around `requireFeatureDefinition()` to allow custom/unknown feature types in tests (fallback to 100 dimensions).

## Solution

**Core Changes:**
- Added `requireFeatureDefinition(type)` to `featureRegistry.ts` - throws `Error` with available types list
- Updated `getFeatureDimensions()` - now throws instead of returning `0`
- Updated `isComputedFeature()` - now throws instead of returning `false`

**Callsite Updates (8 files):**
- `featureExtractors.ts` - removed null checks, throws on invalid types
- `featureExtractor.ts` - removed null checks, throws on invalid types
- `FeatureTypeSelector.tsx` - uses throwing version (safe, iterates registry keys)
- `FeatureMultiSelector.tsx` - uses throwing version (safe, iterates registry keys)
- `mockPostureFrame.ts` - try-catch for test flexibility with unknown types
- `mockInferenceResult.ts` - try-catch for test flexibility with unknown types
- `storage.ts` - throws on corrupted IndexedDB data (user preference: fail hard)

**Test Updates:**
- Added 5 comprehensive tests for `requireFeatureDefinition()` throwing behavior
- Updated `getFeatureDimensions()` test - expects throw instead of `0`
- Updated `isComputedFeature()` test - expects throw instead of `false`
- All feature registry tests pass (34 tests)
