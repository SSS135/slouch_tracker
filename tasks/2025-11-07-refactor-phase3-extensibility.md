# Task 2025-11-07: Phase 3 - Worker Message Type Safety (REVISED)
**STATUS:** COMPLETED

## User Request
Create separate detailed tasks for each phase of architecture refactoring. This is Phase 3: Extensibility Enhancements addressing feature registry, hook coupling, and worker message protocol.

**REVISION:** After codebase analysis, scope reduced to address only validated concerns (worker message validation + minimal registry improvements). Feature plugin system and camera facade rejected as unnecessary.

## General Description
Phase 3 adds runtime type safety to worker message protocol to prevent silent failures from malformed payloads. Currently, worker messages use TypeScript discriminated unions but lack runtime validation of payload structure. Optional chaining (`payload?.field`) silently ignores malformed data instead of failing fast. At 2 FPS inference rate, Zod validation overhead (~10-100μs) is negligible (0.002-0.02% of frame time).

Additionally, adds minimal improvements to feature registry (explicit dependencies, dimension validation) without over-engineering a plugin system.

**Dependencies:** None (can be done independently)

## Action Plan

### **Issue #1: Worker Message Protocol Type Safety (4-6 hours)**

**Problem:** Worker messages use discriminated unions but lack runtime validation of payload structure. Optional chaining (`payload?.field`) silently ignores malformed data instead of failing fast with clear errors. This makes debugging difficult when message structure mismatches occur.

**Current Pattern:**
```typescript
interface WorkerMessage {
  type: 'initialize' | 'process' | 'loadClassifier' | ...;
  payload?: {
    rtmdetPath?: string;
    imageData?: ImageData;
    // ... many optional fields
  };
}

// No validation
self.addEventListener('message', (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;
  switch (type) { /* ... */ }
});
```

**Steps:**

#### 1. Define Message Schemas with Zod (2 hours)

Create schemas for both inference and training workers:

```typescript
// src/workers/messages/schemas.ts
import { z } from 'zod';

// Inference Worker Messages
const InitializeMessageSchema = z.object({
  type: z.literal('initialize'),
  payload: z.object({
    rtmdetPath: z.string(),
    rtmw3dPath: z.string(),
  }),
});

const ProcessMessageSchema = z.object({
  type: z.literal('process'),
  payload: z.object({
    imageData: z.instanceof(ImageData),
    requestId: z.number(),
  }),
});

const LoadPostureModelSchema = z.object({
  type: z.literal('loadPostureModel'),
  payload: z.object({
    model: TrainedModelSchema, // Reuse existing schema
  }),
});

// ... other message types

export const InferenceWorkerMessageSchema = z.discriminatedUnion('type', [
  InitializeMessageSchema,
  ProcessMessageSchema,
  LoadPostureModelSchema,
  // ... all message types
]);

export type InferenceWorkerMessage = z.infer<typeof InferenceWorkerMessageSchema>;
```

#### 2. Add Worker Boundary Validation (2-3 hours)

Validate at worker entry point:

```typescript
// In inference-worker.ts
import { InferenceWorkerMessageSchema } from './messages/schemas';

self.addEventListener('message', async (event: MessageEvent) => {
  try {
    const message = InferenceWorkerMessageSchema.parse(event.data);
    await handleMessage(message);
  } catch (error) {
    if (error instanceof z.ZodError) {
      logger.error('worker', '[Worker] Invalid message:', error.errors);
      self.postMessage({
        type: 'error',
        payload: {
          error: 'Invalid message format',
          details: error.format(),
        },
      });
    } else {
      throw error;
    }
  }
});
```

#### 3. Update Main Thread (1 hour)

Optional validation before sending (catches bugs earlier):

```typescript
// In useWebWorkerInference.ts (optional)
import { InferenceWorkerMessageSchema } from '@/workers/messages/schemas';

function sendMessage(worker: Worker, message: unknown): void {
  if (import.meta.env.DEV) {
    InferenceWorkerMessageSchema.parse(message);
  }
  worker.postMessage(message);
}
```

**Benefits:**
- Runtime validation prevents silent failures from malformed payloads
- Clear, actionable error messages with Zod's detailed validation output
- Consistency with existing validation patterns (storage layer uses Zod extensively)
- Negligible overhead at 2 FPS (0.002-0.02% of frame time)
- Optional dev-only validation on main thread catches bugs earlier

---

### **Issue #2: Minimal Feature Registry Improvements (2 hours)**

**Problem:** Feature dependencies are implicit in extract functions (harder to debug). No runtime validation that extract functions return the declared number of dimensions.

**Current Pattern:**
```typescript
[FEATURE_BACKBONE]: {
  id: FEATURE_BACKBONE,
  dimensions: RTMPOSE_BACKBONE_POOLED_DIMS, // Declared but not validated
  extract: (container) => {
    const raw = container.postureFeatures[FEATURE_BACKBONE_RAW]; // Implicit dependency
    if (!raw) return null;
    return poolBackboneFeatures(raw);
  },
}
```

**Steps:**

#### 1. Add Explicit Dependencies Field (30 minutes)

```typescript
// In featureRegistry.ts
interface FeatureDefinition {
  id: FeatureType;
  name: string;
  dimensions: number;
  storageCost: number;
  computed: boolean;
  modelType: 'posture' | 'detection';
  dependencies?: FeatureType[]; // NEW
  extract: (container: FeatureContainer) => Float32Array | null;
}

// Update registry entries
[FEATURE_BACKBONE]: {
  // ... existing fields
  dependencies: [FEATURE_BACKBONE_RAW], // NEW
  extract: (container) => {
    const raw = container.postureFeatures[FEATURE_BACKBONE_RAW];
    if (!raw) return null;
    return poolBackboneFeatures(raw);
  },
}
```

#### 2. Add Runtime Dimension Validation (1 hour)

```typescript
// In featureExtraction.ts
function extractFeature(
  type: FeatureType,
  container: FeatureContainer
): Float32Array | null {
  const definition = FEATURE_REGISTRY[type];
  if (!definition) return null;

  const extracted = definition.extract(container);
  if (!extracted) return null;

  // NEW: Validate dimensions
  if (extracted.length !== definition.dimensions) {
    throw new Error(
      `Feature ${type} dimension mismatch: ` +
      `expected ${definition.dimensions}, got ${extracted.length}`
    );
  }

  return extracted;
}
```

#### 3. Update All Registry Entries (30 minutes)

Add `dependencies` field to all computed features (pooled variants).

**Benefits:**
- Explicit dependencies make debugging easier (clear what features depend on)
- Runtime validation catches dimension mismatches immediately
- Minimal code changes (~10 lines of logic + field updates)
- No architectural complexity (keep simple registry structure)

---

## Rationale

### **What Was Changed from Original Proposal**

Original Phase 3 proposed three major changes (7 days, 800+ lines). After codebase analysis, scope reduced to address only validated concerns.

### **REJECTED Issues:**

#### **Feature Plugin System (Original Issue #6) - REJECTED**

**Claims:**
- "Adding features requires 4+ file changes"
- "No validation, unclear requirements"
- "Need plugin system with base classes"

**Reality:**
- Actually 2-3 files (registry + pooling functions + docs)
- Past refactor `2025-10-25-refactor-generic-feature-system.md` already made registry extensible
- Computed features already implemented (`2025-11-03-refactor-computed-pooled-features.md`)
- No bugs reported from current system
- Plugin system would add 800+ lines for marginal benefit

**Decision:** Keep simple registry, add minimal validation (Issue #2 above).

#### **Camera Pipeline Facade (Original Issue #2) - REJECTED**

**Claims:**
- "Tight coupling between hooks"
- "Can't swap implementations"
- "Testing requires mocking all 4 hooks"

**Reality:**
- Hooks use dependency injection (good design, not coupling)
- videoRef passed as props (standard React pattern)
- Tests prove hooks are independently testable (`useFrameProcessor.test.ts`)
- No evidence of swapping camera implementations being needed
- Facade would hide clear, explicit data flow

**Decision:** Keep current hook composition (follows React best practices).

### **ACCEPTED Issues:**

#### **Worker Message Validation (Issue #1) - ACCEPTED (Simplified)**

**Valid Concerns:**
- Malformed payloads silently ignored via optional chaining
- No payload structure validation (only type whitelist)
- Debugging difficult when message structure mismatches

**Simplifications from Original:**
- No factory pattern (adds ceremony)
- No versioning (premature for stable internal API)
- Just Zod schemas at worker boundaries
- Performance concern removed (2 FPS not 30+ FPS)

#### **Feature Registry Improvements (Issue #2) - NEW**

Addresses real issues from original plugin system proposal without complexity:
- Explicit dependencies (not plugin base class)
- Runtime dimension validation (not full validation framework)
- Keep simple registry structure

### **Alignment with Project Patterns**

Past successful refactors were **bug-driven**:
- `0013-refactor-consolidate-classification-state.md` - Fixed duplicate renders
- `0010-fix-detection-camera-sync.md` - Fixed keypoints on wrong frame
- `2025-11-03-refactor-unify-training-api.md` - Eliminated dataset transfer overhead

This revised Phase 3 follows that pattern: **fix real issues** (silent worker failures, implicit dependencies) without over-engineering.

## Files to Modify

**New Files:**
- `src/workers/messages/schemas.ts` - Zod schemas for worker messages (~150 lines)

**Modified Files:**
- `src/workers/inference-worker.ts` - Add Zod validation at entry point (~15 lines)
- `src/workers/training-worker.ts` - Add Zod validation at entry point (~15 lines)
- `src/hooks/useWebWorkerInference.ts` - Optional dev-only validation (~10 lines)
- `src/services/dataset/featureRegistry.ts` - Add `dependencies` field to interface and entries (~20 lines)
- `src/services/ml/featureExtraction.ts` - Add dimension validation (~10 lines)

**Test Files:**
- `src/workers/messages/__tests__/schemas.test.ts` - Validate message schemas (~100 lines)
- `src/services/dataset/__tests__/featureRegistry.test.ts` - Update for dependencies field (~20 lines)

## Related Tasks

- `2025-10-25-refactor-generic-feature-system.md` - Made feature registry generic and extensible (already addressed extensibility)
- `2025-11-03-refactor-computed-pooled-features.md` - Implemented computed features (already addressed plugin-like behavior)
- `0013-refactor-consolidate-classification-state.md` - Established Zod validation patterns for storage layer

## Testing Strategy

### Unit Tests

1. **Worker Message Validation**:
   - Valid messages pass Zod validation
   - Invalid messages fail with clear error details
   - All message types covered by schemas
   - Edge cases (missing fields, wrong types, extra fields)

2. **Feature Registry**:
   - Dependencies field populated for computed features
   - Dimension validation catches mismatches
   - Valid features pass validation
   - Clear error messages for dimension mismatches

### Integration Tests

1. **Worker communication**: Send various messages → validate → verify handling
2. **Feature extraction**: Extract features → validate dimensions → verify correct output

### Manual Testing

1. Send malformed worker message → verify clear Zod error in console
2. Simulate dimension mismatch → verify error caught at extraction
3. Full app workflow → verify no regressions from validation overhead

## Verification Steps

1. **Run all tests**: `npm test` (should pass)
2. **Verify validation works**:
   - Send malformed worker message → see Zod error in console
   - Simulate dimension mismatch → see error at extraction
3. **Performance check**: No noticeable degradation at 2 FPS
4. **Manual testing**: Full capture → train → classify workflow

## Success Criteria

- [ ] Worker messages validated with Zod schemas at boundaries
- [ ] Feature registry has explicit dependencies field
- [ ] Runtime dimension validation for extracted features
- [ ] All tests pass
- [ ] No performance regressions (negligible at 2 FPS)
- [ ] Clear error messages when validation fails

## Migration Notes

**Adding New Worker Message:**
1. Define Zod schema in `src/workers/messages/schemas.ts`
2. Add to appropriate union (`InferenceWorkerMessageSchema` or `TrainingWorkerMessageSchema`)
3. Add case to worker's message handler switch
4. TypeScript + Zod enforce correctness!

**Adding New Feature:**
1. Add to `featureRegistry.ts` with `dependencies` field (if computed)
2. Add pooling function to `rtmposeFeatures.ts` (if needed)
3. Dimension validation automatic
4. UI auto-generates from registry!
