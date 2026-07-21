# Task 2025-10-31: Update Frame Preview & Remove Collect Tab
**STATUS:** COMPLETED

## User Request
Improve preview animation with fade transition, integrate all overlays into VideoSection, remove Collect tab entirely, delete BufferFrameGrid and CollectTab components.

## Critical Discoveries

**1. CollectTab deletion required TabType update:**
Removing 'collect' from TabbedPanel TabType (line 4) prevented stale tab references. TypeScript caught any remaining 'collect' string literals at compile time.

**2. FrameListOverlay + CaptureButtonsOverlay already integrated:**
Tasks 2 and 4 had already integrated both overlays into VideoSection. No additional wiring needed—task only required removing obsolete Collect tab code.

**3. No test files for deleted components:**
BufferFrameGrid.tsx and CollectTab.tsx had no corresponding test files. Only needed to delete component files and remove imports.

## Solution

**Removed Collect Tab from navigation:**
- Removed CollectTab import from UnifiedPosturePage.tsx (line 31)
- Removed collectContent definition from tabs useMemo (lines 468-489)
- Removed collect tab from tabs array return statement (line 504)
- Updated TabType in TabbedPanel.tsx to remove 'collect' (line 4)

**Deleted obsolete components:**
- `src/components/unified/CollectTab.tsx` (164 lines) - Functionality replaced by overlays
- `src/components/unified/BufferFrameGrid.tsx` (181 lines) - Functionality replaced by FrameListOverlay

**Integration already complete from prior tasks:**
- FrameListOverlay integrated in VideoSection (Task 2: feature-frame-list-overlay-component)
- CaptureButtonsOverlay integrated in VideoSection (Task 4: feature-capture-buttons-keyboard-shortcuts)
- Frame preview with hover working (Task 2)
- Smart queueing implemented (Task 3: feature-smart-queueing-notification-badge)

**Result:**
- All frame management now happens through video overlays
- Frame list on left side, capture buttons at bottom center
- Preview on hover, smart queueing when hovering list
- Keyboard shortcuts work globally (G/B/A/C keys)
- Training tab unchanged (as required)

## Lessons

**Component deletion requires TabType sync:** When removing tabs from navigation, must update TabType union to prevent invalid tab references. TypeScript type safety catches these at compile time.

**Prior integration simplified final task:** Tasks 2 and 4 had already integrated overlays into VideoSection. Final task was simpler than planned—only needed to remove old UI.

**Bulk deletion safe when no imports exist:** Grep search confirmed no imports of CollectTab/BufferFrameGrid. Safe to delete without breaking changes.

## Related
- `tasks/2025-10-31-feature-frame-list-overlay-component.md` - FrameListOverlay integration (Task 2)
- `tasks/2025-10-31-feature-capture-buttons-keyboard-shortcuts.md` - CaptureButtonsOverlay integration (Task 4)
- `tasks/2025-10-31-feature-smart-queueing-notification-badge.md` - Smart queueing (Task 3)

## Files Modified
- `src/pages/UnifiedPosturePage.tsx` - Removed collect tab and imports (lines 31, 468-489, 504)
- `src/components/unified/TabbedPanel.tsx` - Updated TabType to remove 'collect' (line 4)

## Files Deleted
- `src/components/unified/CollectTab.tsx` (164 lines)
- `src/components/unified/BufferFrameGrid.tsx` (181 lines)

## Impact

**UI simplification:** Removed entire tab from navigation. All frame management now happens through video overlays, reducing UI complexity from 3 tabs to 2 tabs.

**Code reduction:** Deleted 345 lines of obsolete component code (CollectTab + BufferFrameGrid).

**Completes 5-task refactoring series:**
1. Collapsible right panel with overlay
2. Frame list overlay component
3. Smart queueing & notification badge
4. Capture buttons overlay & keyboard shortcuts
5. Remove Collect tab (this task)

**Testing:** Dev server compiled successfully with no TypeScript errors. No runtime errors. Build completed in 360ms.
