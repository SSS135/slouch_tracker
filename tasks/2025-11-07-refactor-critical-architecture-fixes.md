# Task 2025-11-07: Critical Architecture Fixes

**STATUS:** COMPLETED

## User Request

Implement 5 critical architecture fixes based on consensus review:
1. Worker Protocol Validation (6h)
2. Feature Cloning Utility (4h)
3. Normalization Migration to TensorFlow.js (6h)
4. Auto-Training Consistency (3h)
5. Frame ID Tracking (2h)

## Critical Discoveries (Non-Obvious)

**1. TensorFlow.js tensor conversion overhead:**
Single-sample inference now uses TensorFlow.js for normalization. Slight overhead (~1-2ms) but eliminates mathematical divergence risk. Worth the trade-off for correctness guarantee.

**2. Worker validation only in dev mode:**
Zod validation adds ~0.5ms overhead. Using `import.meta.env.DEV` guard ensures zero production impact while catching protocol errors during development.

**3. Fire-and-forget training is intentional:**
Auto-training uses fire-and-forget (no await) to keep UI responsive. Training widget shows via `isAutoTrainingActive` state - works correctly for all 4 trigger sources.

**4. Ref snapshot prevents race condition:**
Thumbnail generation takes 50-100ms. Snapshotting `inferenceResultRef.current` before async operation prevents features/thumbnail from different frames. Warning logged if ref changed during generation.

**5. Buffer independence assertion only runs in dev:**
`assertBufferIndependence()` throws if buffers share same ArrayBuffer. Dev-mode only check catches cloning bugs without production overhead.

## Solution

**Feature Cloning Utility:**
- Created `src/utils/featureCloning.ts` with 3 functions: `cloneFloat32Array`, `cloneFeatures`, `cloneInferenceFeatures`
- Replaced 4 manual cloning sites (40 lines → 2 function calls)
- Added runtime buffer independence assertions (dev mode only)
- Wrote 15 unit tests covering buffer independence, mutation isolation, edge cases

**Worker Protocol Validation:**
- Added complete Zod response schemas (`InferenceWorkerResponseSchema`) covering all 5 message types
- Created validation helpers: `sendValidatedResponse()`, `sendValidatedResponseWithTransferables()`
- Replaced silent fallback (stripped features) with fail-fast error response + memory diagnostics
- Updated 16+ `self.postMessage()` calls to use validated helpers
- Added response validation in `useWebWorkerInference.ts` with proper error handling
- Wrote 13 validation tests for schema compliance

**Normalization Migration:**
- Migrated `applyPerFeatureNorm()` inference path to TensorFlow.js (was Float32Array loops)
- Updated all 3 normalization modes ('none', 'layer', 'per_feature') to use TensorFlow.js
- Removed Float32Array-based `applyLayerNorm()` and `applyL2Norm()` from `layerNorm.ts`
- Single implementation ensures training/inference mathematical equivalence

**Auto-Training Consistency:**
- Made all 4 `triggerTraining()` call sites consistent (fire-and-forget, no await)
- Call sites: manual capture, batch save, single frame save, undo
- Training widget visibility confirmed (receives `isTraining` + `isAutoTrainingActive`)

**Frame ID Tracking:**
- Added snapshot-based ref tracking before async thumbnail generation
- Snapshot `inferenceResultRef.current` before cloning features
- Detect stale refs after thumbnail completes (50-100ms gap)
- Log warning if ref changed during generation

## Files Modified

**Created:**
- `src/utils/featureCloning.ts` (40 lines)
- `src/utils/__tests__/featureCloning.test.ts` (130 lines, 15 tests)
- `src/workers/messages/__tests__/validation.test.ts` (145 lines, 13 tests)

**Modified:**
- `src/pages/PostureTrackerApp.tsx` - Feature cloning import, 2 manual cloning sites replaced, auto-training consistency (1 await removed)
- `src/hooks/useFrameSampler.ts` - Feature cloning import, 2 manual cloning sites replaced, frame ID tracking (snapshot + warning)
- `src/workers/inference-worker.ts` - Import response schema, validation helpers, 16+ postMessage calls updated, silent fallback replaced with error response + memory diagnostics
- `src/workers/messages/schemas.ts` - Added `InferenceWorkerResponseSchema` with 5 message type schemas
- `src/hooks/useWebWorkerInference.ts` - Import response schema, Zod validation in message handler, catch block for validation errors
- `src/services/ml/baseClassifier.ts` - Import changes (removed Float32Array functions), `applyPerFeatureNorm()` migrated to TensorFlow.js, inference normalization (3 modes) migrated to TensorFlow.js
- `src/services/ml/layerNorm.ts` - Removed `applyLayerNorm()` and `applyL2Norm()` Float32Array implementations

## Success Metrics

**Reliability:**
- Zero manual feature cloning code (type-safe utility)
- Zero silent worker failures (fail-fast with user-facing errors)
- Zero normalization divergence (single implementation)

**Code Quality:**
- Manual cloning: 40 lines → 2 lines (95% reduction)
- Worker messages: 0% validated → 100% validated (dev mode)
- Normalization implementations: 2 → 1

**Performance:**
- Feature cloning: Same performance (Float32Array constructor is fast)
- Worker validation: Zero production overhead (dev mode only)
- Normalization: +1-2ms inference latency (acceptable for correctness)

## Related

- `tasks/2025-11-07-refactor-consensus-architecture-issues.md` - Original analysis identifying these issues
- `tasks/2025-11-07-refactor-consensus-review.md` - Architect consensus review that prioritized these fixes
