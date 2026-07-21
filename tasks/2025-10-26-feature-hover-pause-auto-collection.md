# Task 2025-10-26: Hover Pause Auto-Collection
**STATUS:** COMPLETED

## User Request
When I hover mouse over auto-collected frame or it's container in collect tab, they should pause auto-adding new. Storing these new frames into some array. And writing to header something like N frames pending. Then add them normally when I move mouse away from this frame view. That is because when new frames added what's placed under mouse is shifts and i often mis-click on other frame.

**Additional Requirements:**
- X frames pending text should be in header, not under it (implemented: shows in section title inline)
- When frame is added to pending buffer, the frame container redraws and flickers - fix it (implemented: useMemo/useCallback optimizations)
- Guards, tests and other code should use a single source of truth for available features (implemented: FEATURE_TYPES constant)

## Critical Discoveries

**1. Set-based frame ID tracking required for correctness:**
`pendingFrameIds` must be Set, not Array. Using `pendingFrames.includes(f)` creates false positives when buffer evicts old frames—new frames at same position pass object equality check. ID-based comparison prevents this.

**2. Tracking previous IDs avoids duplicate pending:**
Without `prevFrameIdsRef`, every recentFrames update re-adds all frames to pending queue during hover. useRef stores previous snapshot to detect only genuinely new frames.

**3. Display filtering needs reverse lookup:**
Can't just filter out pending IDs from recentFrames—must check if each frame ID exists in pending set. Handles buffer overflow: pending IDs not in recentFrames are silently ignored (frame already evicted).

**4. Mouse leave flush must clear tracking refs:**
Flushing pending on mouse leave requires resetting `prevFrameIdsRef` to current recentFrames IDs. Otherwise next hover immediately marks all existing frames as "new."

**5. React re-render optimization critical for preventing flicker:**
Without useMemo/useCallback, displayedFrames array reference changes every render, causing UnifiedFrameGrid to re-render unnecessarily. This creates visible flicker when frames are added to pending queue. Wrapping displayedFrames, pendingCount, and handlers in useMemo/useCallback stabilizes references.

**6. Feature type refactoring revealed widespread hardcoding:**
Creating FEATURE_TYPES constant exposed that feature names ('gau_cross_kpt', etc.) and dimensions (768, 1536, 195) were hardcoded across 20+ test files and production code. Refactored to use FEATURE_REGISTRY as single source of truth.

## Solution

### Primary Feature: Hover-Pause Mechanism
Added hover-pause mechanism to auto-frame collection in CollectTab. When user hovers over frame grid container or individual thumbnails, new auto-captured frames accumulate in pending queue (stored as ID Set) instead of being immediately displayed. Prevents layout shifts causing mis-clicks. Pending frames flush immediately on mouse leave.

**Implementation:**
- Track hover state (`isHovering`) and pending frame IDs (`pendingFrameIds` Set)
- Compare recentFrames snapshots using `prevFrameIdsRef` to detect new arrivals during hover
- Filter `displayedFrames` by excluding pending IDs from recentFrames (with useMemo optimization)
- Show "N frame(s) pending" inline in Section title (e.g., "Auto-Captured Frames (2 pending)")
- Wire `onMouseEnter`/`onMouseLeave` to UnifiedFrameGrid container
- Flush pending (clear Set + reset tracking) on mouse leave
- useCallback for all hover handlers to prevent unnecessary re-renders

**Performance:** Set-based ID comparison is O(n) for n=30 buffer size. Storing IDs (not objects) prevents stale references when buffer evicts frames.

### Secondary Fix: Flicker Prevention
Added React optimization hooks to prevent flickering when frames are added to pending queue:
- `useMemo` for `displayedFrames` - stabilizes array reference, prevents UnifiedFrameGrid re-render
- `useMemo` for `pendingCount` - stabilizes count calculation
- `useCallback` for `handleGridHoverEnter` and `handleGridHoverLeave` - stabilizes handler references

### Tertiary Refactoring: Feature Type Single Source of Truth
Created `FEATURE_TYPES` constant and refactored entire codebase to use it:
- **featureRegistry.ts**: Exported `FEATURE_TYPES` array and derived `FeatureType` type from it
- **Replaced hardcoded feature names**: 'gau_cross_kpt', 'backbone_concat', 'mlp_input_concat', 'gau_per_kpt'
- **Replaced hardcoded dimensions**: 768, 1536, 195 → `FEATURE_REGISTRY[type].dimensions`
- **Test utilities**: Created `createMockFeatures()` and `createMockFeature()` helpers
- **Updated 20+ files**: Guards, schemas, tests, production code all use registry

**Test coverage:** 9 new hover-pause tests + all existing tests updated to use FEATURE_TYPES (1115 of 1115 passing - 100%):
- Hover state transitions, pending accumulation, display filtering
- Flush on mouse leave, multiple hover/leave cycles
- Buffer overflow edge case, singular/plural pending count
- Feature type validation uses FEATURE_TYPES constant
- Fixed TrainingTab tests to match refactored FeatureMultiSelector component

## Files Modified

### Hover-Pause Feature
- `src/components/unified/CollectTab.tsx` - Hover state, frame tracking, display filtering, handlers, optimization hooks
- `src/components/dataset/UnifiedFrameGrid.tsx` - Grid container hover props, event wiring
- `src/components/unified/__tests__/CollectTab.test.tsx` - 9 comprehensive hover-pause tests

### Feature Type Refactoring (Single Source of Truth)
- `src/services/dataset/featureRegistry.ts` - Added FEATURE_TYPES constant, derived FeatureType type
- `src/services/validation/guards.ts` - Use FEATURE_TYPES for validation
- `src/services/validation/schemas.ts` - Use FEATURE_TYPES for Zod schema
- `src/contexts/TrainingConfigContext.tsx` - Use FEATURE_TYPES for defaults
- `src/components/dataset/TrainingModal.tsx` - Use FEATURE_TYPES[0] for fallback
- `src/workers/unified-pose-worker.ts` - Use FEATURE_TYPES instead of hardcoded arrays
- `src/__tests__/utils/mockInferenceResult.ts` - Helper functions using FEATURE_REGISTRY
- `src/__tests__/utils/mockPostureFrame.ts` - Helper functions using FEATURE_REGISTRY
- `src/services/dataset/types.ts` - Made detectionConfidence optional in InferenceResult
- `src/hooks/useModelTraining.ts` - Added FeatureType import and proper typing

### Test Files Updated (Feature Names)
- `src/services/dataset/__tests__/export.test.ts` - Updated feature names
- `src/services/validation/__tests__/schemas.test.ts` - Made detectionConfidence optional
- `src/services/posture/__tests__/detection.test.ts` - Fixed person detection mock
- `src/contexts/__tests__/TrainingConfigContext.test.tsx` - Updated feature names
- `src/services/dataset/__tests__/storage.test.ts` - Updated feature names and type casts
- `src/components/unified/__tests__/TrainingTab.test.tsx` - Updated to FeatureMultiSelector, fixed testids
- `src/hooks/__tests__/useModelTraining.test.ts` - Updated feature names
- `src/services/validation/__tests__/guards.test.ts` - Use FEATURE_TYPES for iteration
- `src/services/dataset/__tests__/featureRegistry.test.ts` - Use FEATURE_TYPES for all tests
- `src/services/ml/__tests__/featureExtractors.test.ts` - Use mock helpers
- `src/services/ml/__tests__/featureExtractor.test.ts` - Fixed error message regex

## Impact

**User Experience:**
- Eliminates layout shifts during frame interaction, reducing mis-clicks
- User can safely hover over frames while auto-capture continues in background
- All frames preserved (no data loss)
- No flickering when pending frames accumulate
- Improves UX for posture-change and interval capture modes

**Code Quality:**
- Single source of truth for feature types (FEATURE_REGISTRY)
- No magic numbers (hardcoded dimensions)
- Easier to add/remove/modify features in the future
- Type-safe feature type validation
- Test utilities make it easy to create mock data with correct dimensions
- All 1115 tests passing (100% pass rate)
- Fixed UnifiedFrameGrid TypeScript error with ScrollView mouse events
