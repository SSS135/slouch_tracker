# Task 2025-11-07: Fix Reset Buttons

**STATUS:** COMPLETED

## User Request

reset all data button in runtime settings does not do anything, fix it. It should do complete hard wipe of all app data. Reset dataset button does nothing, even not closes when I press delete, only prints this to logs: [DatasetOperations] Resetting dataset (keeping trained model). fix

**Update:** Reset dataset does work but takes ~1 minute with no feedback, making users think it's broken.

## Critical Discoveries

**Reset All Data Regression:** Previous fix (2025-10-28) was correct but got regressed. Current code only cleared camera settings via `resetSettings()` instead of `localStorage.clear()`, missed `clearTrainingSettings()`, and only invalidated stats instead of clearing all React Query cache.

**Async Modal UX Pattern:** Long-running operations (>1 minute) in modals need loading state. Without it, users think operation failed. Must also close modal on error, not just success, to avoid modal being stuck open when operation fails.

**ConfirmationModal Enhancement:** Extended to support loading state with disabled buttons, spinner on confirm button, and disabled close actions (backdrop click, escape key).

## Solution

**Reset All Data Fix (`PostureTrackerApp.tsx:480-489`):**
- Added `clearTrainingSettings()` to clear training defaults from IndexedDB
- Replaced `resetSettings()` with `localStorage.clear()` for complete wipe
- Replaced `invalidateStats()` with `queryClient.clear()` for full cache wipe
- Used `window.location.reload()` for clean state (simplest, most reliable)

**Reset Dataset Modal Fix (`TrainingTab.tsx:159-171`, `TrainingTab.tsx:504-514`):**
- Added `loading={datasetOps.resetDataset.isPending}` prop to ConfirmationModal
- Added `setResetModalOpen(false)` to `onError` callback (not just `onSuccess`)
- Also applied same pattern to Cleanup modal for consistency

**ConfirmationModal Component (`ConfirmationModal.tsx`):**
- Added optional `loading?: boolean` prop
- Disabled both buttons when loading
- Added loading spinner to confirm button
- Disabled backdrop click and escape key when loading
- Prevents accidental interruption of long operations

## Files Modified

- `src/pages/PostureTrackerApp.tsx` - Fixed Reset All Data handler
- `src/components/unified/TrainingTab.tsx` - Added loading state to modals, close on error
- `src/components/dataset/ConfirmationModal.tsx` - Added loading state support

## Related

- `tasks/2025-10-28-fix-reset-all-data-button.md` - Previous correct fix that was regressed
