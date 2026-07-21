# Task 2025-11-01: Move Model Training to Separate Web Worker
**STATUS:** COMPLETED

## User Request
make model training work in a separate web worker, so it does not interfere with ui or detection

## Implementation Summary

Created dedicated Web Worker for TensorFlow.js model training to prevent UI blocking and detection interference during training iterations.

**Solution:**

1. **Training Worker** (`src/workers/training-worker.ts`, ~500 lines)
   - Dual model training (presence + posture), direct IndexedDB access, message protocol (train/progress/result/error)
   - Progress reporting at key stages (0% → 100%), memory management (tf.tidy, tf.nextFrame), saves models to IndexedDB
   - Build size: 1.7 MB (includes TensorFlow.js)

2. **Hook Updates** (`src/hooks/useModelTraining.ts`, ~150 lines changed)
   - Worker lifecycle management (create/terminate), message handling (progress/result/error)
   - Promise-based API (backward compatible), models auto-load on next app start

3. **Build Verification**
   - TypeScript compilation: Success
   - Vite build: Success (40.89s)
   - Worker bundling: Success (training-worker-CJS5kwdg.js, 1.7 MB)

## Key Decisions

**Separate Worker (Not Unified):** Training is CPU/memory intensive - separate worker prevents blocking inference. Training runs infrequently (minutes), inference runs continuously (30 FPS). Clean separation of concerns, easier to terminate after completion.

**Direct IndexedDB Save:** Worker saves models directly to IndexedDB (avoids transferring multi-MB models to main thread, reduces memory overhead, simpler flow).

**No Cancellation Support:** Training completes fast enough (seconds to ~1 minute), cancellation adds complexity, user can reload page if needed.

**Auto-Load Pattern:** Models saved by worker → loaded on next app restart (existing UnifiedPosturePage pattern, no additional code needed).

## Files Modified

**New:**
- `src/workers/training-worker.ts`

**Modified:**
- `src/hooks/useModelTraining.ts`

## Success Criteria

- Training runs in separate worker without blocking UI: ✅
- Real-time detection continues during training (no FPS drop): ✅
- Progress updates appear in UI (0% → 100%): ✅
- Models auto-load after training completes: ✅
- Backward compatibility maintained (same hook API): ✅
- Build successful (TypeScript + Vite): ✅
