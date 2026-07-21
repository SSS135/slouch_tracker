# Task 2025-11-07: PostureTrackerApp Cleanup & Capture Workflow Extraction
**STATUS:** NOT STARTED

## User Request
Create simpler refactoring task based on Phase 2 rejection. Focus on proven wins with low risk: remove redundant state, extract capture workflow, improve testability through focused changes.

## General Description
PostureTrackerApp (738 lines) has code quality issues that can be addressed through targeted cleanup:
1. **Redundant dataset stats sync** - Local state duplicates React Query cache
2. **Scattered capture logic** - Manual capture, auto-capture, timer fallback across 150+ lines
3. **Missing tests** - No tests exist for main orchestration component

This task makes **evidence-based improvements** with minimal risk, following patterns from successful past refactors.

## Action Plan

### **Step 1: Remove Redundant Dataset Stats Sync** (2 hours)

**Problem**: Lines 104-120 in PostureTrackerApp.tsx
- `datasetStats` local state duplicates `datasetOps.stats.data` from React Query
- Manual sync logic in `refreshDatasetStats` and `useEffect`
- Unnecessary complexity and potential stale state

**Solution**: Use React Query data directly
```typescript
// BEFORE
const [datasetStats, setDatasetStats] = useState<DatasetStats>(...);
const refreshDatasetStats = useCallback(() => {
  setDatasetStats(datasetOps.stats.data || ...);
}, [datasetOps.stats.data]);

// AFTER
const datasetStats = datasetOps.stats.data || { /* defaults */ };
```

**Files to modify**:
- `src/pages/PostureTrackerApp.tsx` (remove state + sync logic)

**Expected impact**: -15 lines, eliminate sync bugs

---

### **Step 2: Extract Capture Workflow Hook** (6 hours)

**Problem**: Manual capture logic scattered across PostureTrackerApp
- `handleCaptureWithLabel` (96 lines, lines 285-381)
- Duplicates persistence logic
- Hard to test in isolation

**Solution**: Create `useCaptureWorkflow` hook following existing patterns

**Hook Interface**:
```typescript
interface CaptureWorkflowOptions {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  inferenceResult: InferenceResult | null;
  onSuccess?: (label: FrameLabel) => void;
  onError?: (error: Error) => void;
}

interface CaptureWorkflowReturn {
  captureWithLabel: (label: FrameLabel) => Promise<void>;
  isCapturing: boolean;
}

function useCaptureWorkflow(options: CaptureWorkflowOptions): CaptureWorkflowReturn
```

**Implementation steps**:
1. Create `src/hooks/useCaptureWorkflow.ts`
2. Extract capture logic from `handleCaptureWithLabel`
3. Integrate with existing hooks:
   - `useDatasetOperations` (persist frame)
   - `useAutoTraining` (trigger training)
   - `useActionHistory` (push undo action)
   - `useNotification` (show feedback)
4. Update PostureTrackerApp to use new hook

**Files to modify**:
- `src/hooks/useCaptureWorkflow.ts` (new, ~150 lines)
- `src/pages/PostureTrackerApp.tsx` (replace inline logic)

**Expected impact**: -96 lines from PostureTrackerApp, +150 lines in focused hook

---

### **Step 3: Extract Keyboard Shortcuts Hook** (3 hours)

**Problem**: Keyboard event handling mixed with component logic (lines 173-179)

**Solution**: Create reusable `useGlobalKeyboardShortcuts` hook

**Hook Interface**:
```typescript
interface KeyboardShortcutsConfig {
  onGoodPosture: () => void;
  onBadPosture: () => void;
  onUnused: () => void;
  onClear: () => void;
  enabled: boolean;
}

function useGlobalKeyboardShortcuts(config: KeyboardShortcutsConfig): void
```

**Files to modify**:
- `src/hooks/useGlobalKeyboardShortcuts.ts` (new, ~50 lines)
- `src/pages/PostureTrackerApp.tsx` (use hook)

**Expected impact**: -20 lines from PostureTrackerApp, reusable hook

---

### **Step 4: Write Tests** (4 hours)

**Test coverage**:
1. **useCaptureWorkflow.test.ts** (new)
   - Capture with valid inference → saves to IndexedDB
   - Capture without inference → shows error
   - Capture triggers auto-training at threshold
   - Capture pushes undo action
   - Concurrent captures handled correctly

2. **useGlobalKeyboardShortcuts.test.ts** (new)
   - G key → calls onGoodPosture
   - B key → calls onBadPosture
   - U key → calls onUnused
   - C key → calls onClear
   - Shortcuts disabled → no callbacks called
   - Non-shortcut keys → no callbacks called

3. **PostureTrackerApp.test.tsx** (new, basic)
   - Renders without crashing
   - Context providers work correctly
   - Tab switching works
   - Full integration tests (optional, lower priority)

**Files to create**:
- `src/hooks/__tests__/useCaptureWorkflow.test.ts`
- `src/hooks/__tests__/useGlobalKeyboardShortcuts.test.ts`
- `src/pages/__tests__/PostureTrackerApp.test.tsx` (optional)

---

## Rationale

### **Why These Changes**

**Evidence-based approach**:
1. **Redundant state sync**: Clear code smell, no benefit to duplication
2. **Capture workflow**: Follows existing patterns (`useFrameSampler`, `useAutoCapture`, `usePostureChangeDetector`)
3. **Keyboard shortcuts**: Reusable, focused concern, easy to test

**Patterns from past tasks**:
- Task 2025-11-06: Extract shared logic aggressively when duplicated
- Task 0013: Remove redundant state that duplicates single source
- Task 2025-11-06: Inline over abstraction (no class, just hooks)

### **Why NOT Context Merge**

Analysis proved context separation provides:
- **Performance**: Config changes don't trigger execution consumers to re-render
- **Testability**: Test storage logic separately from worker logic
- **Clarity**: Persistent config vs ephemeral execution are different concerns

Only 2 components need both contexts - acceptable coupling.

### **Why NOT More Extractions**

- `useAppLayout` (80 lines): Too small, well-encapsulated already
- `FrameManager` class: Past refactors removed classes in favor of simpler hooks
- Frame buffer logic: Already well-abstracted in `useFrameSampler` (220 lines)

### **Why This Scope**

**Risk assessment**:
- Step 1: LOW risk (simple state removal)
- Step 2: MEDIUM risk (touches capture flow, but well-tested)
- Step 3: LOW risk (pure extraction, no logic changes)
- Step 4: NO risk (tests only)

**Impact**:
- PostureTrackerApp: 738 → ~610 lines (17% reduction)
- Clear responsibility boundaries
- Testable capture logic
- Reusable keyboard shortcuts
- No performance regressions

## Files to Modify

**New Files**:
- `src/hooks/useCaptureWorkflow.ts` (~150 lines)
- `src/hooks/useGlobalKeyboardShortcuts.ts` (~50 lines)
- `src/hooks/__tests__/useCaptureWorkflow.test.ts` (~200 lines)
- `src/hooks/__tests__/useGlobalKeyboardShortcuts.test.ts` (~100 lines)
- `src/pages/__tests__/PostureTrackerApp.test.tsx` (~100 lines, optional)

**Modified Files**:
- `src/pages/PostureTrackerApp.tsx` (reduce by ~130 lines)

**No files deleted**

## Testing Strategy

### Unit Tests (New)

**useCaptureWorkflow**:
- Mock: datasetOperations, autoTraining, actionHistory, notification
- Test: capture flow, error handling, auto-training trigger, undo integration
- Expected coverage: >90%

**useGlobalKeyboardShortcuts**:
- Mock: None needed (pure DOM event handling)
- Test: keyboard events, enabled/disabled states, key filtering
- Expected coverage: 100%

### Integration Tests (Optional)

**PostureTrackerApp**:
- Render with all providers
- Test tab switching
- Test keyboard shortcuts integration
- Expected coverage: >60% (basic smoke tests)

### Manual Testing

1. Capture 10 frames (G key) → verify stats update
2. Capture triggers auto-training → verify training starts
3. Undo last capture → verify frame removed
4. Switch tabs → verify no state loss
5. Keyboard shortcuts work correctly (G/B/U/C)

## Verification Steps

1. **Run tests**: `npm test` (all pass)
2. **Check line counts**:
   - PostureTrackerApp < 620 lines
   - useCaptureWorkflow < 200 lines
   - useGlobalKeyboardShortcuts < 60 lines
3. **Manual testing**: Full capture → train → classify workflow
4. **Performance check**: No increase in re-renders (React DevTools)
5. **Code review**: Clear hook boundaries, no circular dependencies

## Success Criteria

- [ ] Redundant dataset stats sync removed
- [ ] useCaptureWorkflow hook extracted and tested (>90% coverage)
- [ ] useGlobalKeyboardShortcuts hook extracted and tested (100% coverage)
- [ ] PostureTrackerApp reduced to ~610 lines (17% reduction)
- [ ] All tests pass (no regressions)
- [ ] Manual testing passes (capture, train, undo workflows)
- [ ] No performance regressions (re-render count stable)

## Related Tasks

- `2025-11-07-refactor-phase2-testability.md` - Rejected (over-abstraction, wrong patterns)
- `2025-11-06-refactor-classifier-hierarchy.md` - Extract shared logic pattern (950 lines → BaseClassifier)
- `0013-refactor-consolidate-classification-state.md` - Remove redundant state pattern
- `2025-11-03-refactor-unify-training-api.md` - Simplification over complexity

## Migration Notes

**For Future Developers**:
- When adding capture features → modify `useCaptureWorkflow`
- When adding keyboard shortcuts → modify `useGlobalKeyboardShortcuts`
- When changing dataset operations → update `useDatasetOperations` (no PostureTrackerApp changes needed)

**Breaking Changes**: None (internal refactor only, no API changes)
