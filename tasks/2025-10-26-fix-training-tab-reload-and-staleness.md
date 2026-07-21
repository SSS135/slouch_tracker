# Task 2025-10-26: Fix Training Tab Reload and Staleness Detection
**STATUS:** COMPLETED

## User Request
"when I click on a frame in training tab to remove it or change type, the whole tab gets reloaded to do it. Make it work without reloading. Also popup about outdated model does not appear when I do that, fix it."

## Critical Discoveries

**1. Optimistic updates require error rollback:**
Frame operations now update local state immediately, then persist async. On persistence failure, state rolls back to prevent UI/storage desync.

**2. Callback timing matters:**
`onFramesChanged` callback must fire AFTER both optimistic update AND persistence complete to ensure staleness detection uses current data.

**3. Multiple operations need callback:**
Not just delete/label change - cleanup unused, reset dataset, and import dataset also modify frames and need to trigger staleness detection.

## Solution

**Implemented optimistic UI pattern** for responsive frame operations:

1. **TrainingTab.tsx** - Added optimistic updates with error rollback:
   - `handleFrameClick`: Update local state → persist → callback → rollback on error
   - `handleDeleteFrame`: Update local state → persist → callback → rollback on error
   - Added `onFramesChanged?: () => void` prop
   - Wired callback to cleanup, reset, and import operations

2. **app/index.tsx** - Connected staleness detection:
   - Passed `reloadFrames` to TrainingTab as `onFramesChanged` prop

3. **TrainingTab.test.tsx** - Added 9 comprehensive tests:
   - Immediate state updates before persistence
   - Callback invocation after operations
   - Error rollback on persistence failure

## Files Modified
- `src/components/unified/TrainingTab.tsx`
- `app/index.tsx`
- `src/components/unified/__tests__/TrainingTab.test.tsx`

## Impact

**UX Improvements:**
- Frame operations instant (no reload flash)
- Scroll position preserved during operations
- Model staleness popup now appears correctly

**Tests:** 9/9 new tests passing, 45/54 total (9 pre-existing failures unrelated to this task)
