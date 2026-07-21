# Task 2025-10-28: Fix Reset All Data Button
**STATUS:** COMPLETED

## User Request
"reset all settings button does not do anything, fix it. Should perform complete factory reset wiping ALL app data."

## Critical Discoveries

**Button wiring issue:**
"Reset All Settings to Defaults" button only called `onResetSettings()` from `useCameraSettings`, which only reset camera settings to defaults. Button was completely disconnected from IndexedDB clearing logic - did NOT clear trained models, datasets, training defaults, or other localStorage keys.

## Solution

Implemented complete factory reset with confirmation dialog:
- Created `handleFactoryReset` in `app/index.tsx` using `showConfirm` hook
- Clears ALL IndexedDB: dataset, posture model, presence model, training defaults
- Clears ALL localStorage
- Reloads page for fresh start
- Renamed button "Reset All Settings to Defaults" → "Reset All Data"
- Updated button styling to destructive red (#d32f2f with border)
- Added warning help text below button

**Implementation:**
```typescript
// app/index.tsx (lines 643-676)
const handleFactoryReset = useCallback(() => {
  showConfirm(
    'Reset All Data?',
    'This will permanently delete all settings, datasets, and trained models...',
    {
      confirmText: 'Delete All Data',
      onConfirm: async () => {
        await datasetStorage.clearDataset();
        await datasetStorage.clearPostureModel();
        await datasetStorage.clearPresenceModel();
        await datasetStorage.clearTrainingDefaults();
        localStorage.clear();
        window.location.reload();
      }
    }
  );
}, [showConfirm, showError]);
```

## Lessons

1. **Button placement matters**: Factory reset in Developer Settings section is appropriate for destructive operations
2. **Confirmation dialogs are essential**: Prevents accidental data loss for irreversible actions
3. **Page reload ensures clean state**: Simpler than manually resetting all React contexts/hooks
4. **Clear button text**: "Reset All Data" is unambiguous, unlike "Reset All Settings to Defaults"

## Files Modified

- `app/index.tsx` - Added factory reset handler (lines 122, 643-676, 726)
- `src/components/unified/RuntimeTab.tsx` - Updated button text, styling, help text (lines 41-42, 285-290, 335-348)
- `src/components/unified/__tests__/RuntimeTab.test.tsx` - Updated button text assertion (line 245)

## Test Results

All 22 tests pass in RuntimeTab.test.tsx after updating button text expectation from "Reset All Settings to Defaults" → "Reset All Data".

## Impact

Users can now perform complete factory reset from UI. Button properly wired to clear all data (IndexedDB + localStorage) with confirmation dialog, preventing accidental data loss.
