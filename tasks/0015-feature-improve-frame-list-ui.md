# Task 0015: Improve Frame List UI in Collection Tab
**STATUS:** COMPLETED

## User Request
Improve frame list in collection tab so new frames are added at top and remove empty pre-filled frames from it. Also make images 1.5x smaller. Increase auto-save size to 30.

## Critical Discoveries

**1. Frame selection independence:**
Selection uses `frameId` from PostureFrame object, not array index. Reversing display order had no impact on selection logic.

**2. Placeholder removal scope:**
Entire PlaceholderCell component deleted (80 lines) including styles, rendering logic, and calculation. Grid naturally grows without empty slot display.

**3. Named constant extraction:**
`MAX_BUFFER_SIZE = 30` used consistently across UnifiedFrameGrid and CollectTab. Single source of truth prevents future drift.

## Solution

**Reverse ordering:** Added `.reverse()` to `frames.slice(-maxFrames)` in SimpleLayout function. Newest frames now render at top.

**Removed placeholders:** Deleted PlaceholderCell component and all calculation/rendering logic. Grid only shows captured frames.

**Reduced sizes:** Scaled all dimensions by 1.5x: 240x180 → 160x120 (main), 60x40 (badges), 48x32 (multi-dataset), 36x24 (badges on small). Maintained 4:3 aspect ratio throughout.

**Increased buffer:** Extracted constant `MAX_BUFFER_SIZE = 30` used in CollectTab (`maxFrames` prop, buffer slicing, auto-save condition) and UnifiedFrameGrid (SimpleLayout slicing). Updated buffer stats display to show "/30".

## Lessons

Named constants for configuration values prevent hardcoded duplication. Frame ordering change trivial when selection uses stable IDs instead of indices. Removing UI elements (placeholders) simpler than anticipated when grid layout handles dynamic content naturally.

## Files Modified

- `src/components/dataset/UnifiedFrameGrid.tsx` - Reversed frame order, deleted PlaceholderCell, MAX_BUFFER_SIZE constant
- `src/components/dataset/FrameThumbnail.tsx` - Reduced dimensions by 1.5x across all layouts
- `src/components/unified/CollectTab.tsx` - MAX_BUFFER_SIZE constant replaces hardcoded 20
- `src/components/unified/__tests__/CollectTab.test.tsx` - Updated assertions for /30 buffer size

## Impact

More frames visible without scrolling (smaller thumbnails). Reduced IndexedDB write frequency (30-frame buffer vs 20). Cleaner UI without placeholder clutter. Latest captures immediately visible (newest-first ordering). All 45 tests passed.
