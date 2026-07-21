# Task 2025-10-26: Drag & Drop Frame Labeling with Delete Button
**STATUS:** COMPLETED

## User Request
"remove class switch button, add delete button instead. implement moving between different classes using drag & drop"

**User Clarifications:**
- Drop zones: Section headers (Good/Bad/Away/Unused)
- Delete button: On hover (overlay in corner)
- Implementation: Everything at once (not phased)

## Critical Discoveries

**1. HTML5 Drag & Drop API works seamlessly with React Native Web**
- Zero dependencies needed, native browser API sufficient for drag-move-drop workflow
- Platform.OS checks enable clean web-only features without mobile compatibility issues

**2. Section headers as drop zones provide superior UX**
- Large target areas easier to hit during drag (vs individual frame drop zones)
- Clear visual affordance (section header = label category)
- Highlight feedback on dragover indicates valid drop target

**3. Optimistic update pattern reused successfully**
- Pattern from hover-pause task (2025-10-26-feature-hover-pause-auto-collection.md) proven reliable
- Instant UI updates with background persistence and error rollback prevents data loss
- Store previous state → update local → persist → rollback on error

## Solution

Replaced click-to-cycle frame labeling with drag & drop UX and hover-triggered delete buttons.

**FrameThumbnail.tsx**: Added delete button (hover overlay, web-only) and drag handlers. Delete button positioned absolute (top-right corner), visible on hover. Made frames draggable with HTML5 API - onDragStart stores frame ID in dataTransfer, onDragEnd provides cleanup.

**UnifiedFrameGrid.tsx**: Section headers become drop zones with visual feedback. Added onDragOver/onDragEnter/onDragLeave/onDrop handlers. Dragover state tracks highlighting per section. Drop handler extracts frame ID and triggers label update. Passed frameId and onDelete props to FrameThumbnail.

**TrainingTab.tsx**: Removed handleFrameClick (click-to-cycle behavior). Added handleFrameDrag with optimistic update pattern - immediate UI updates, background persistence, error rollback. Maintains onFramesChanged callback for parent notification.

**Tests**: Updated TrainingTab.test.tsx - removed click cycling tests, added 5 drag & drop tests (label update, stats update, callback, error rollback, same-label skip). Created FrameThumbnail.drag.test.tsx with 11 comprehensive drag & drop tests. All 1133 tests pass.

## Bug Fixes

**Bug Fix #1: Drag & Drop Event Propagation**
- **Symptom**: Visual feedback worked but frames stayed in same category after drop
- **Root cause**: Missing `e.stopPropagation()` + TouchableOpacity intercepting events on web
- **Solution**: Added `stopPropagation()` to dragover/drop handlers in UnifiedFrameGrid.tsx. Added `onStartShouldSetResponder={() => Platform.OS !== 'web'}` to TouchableOpacity in FrameThumbnail.tsx

**Bug Fix #2: Delete Button Icon Centering**
- **Root cause**: Missing explicit line-height and text-align on icon text
- **Solution**: Added `lineHeight: 24`, `textAlign: 'center'` to deleteIcon, `display: 'flex'` to deleteButton in FrameThumbnail.tsx

**Bug Fix #3: Blob URL Transfer Prevention (FINAL SOLUTION - 3 iterations)**
- **Symptom**: Blob URL transferred instead of frame ID
- **Attempt 1**: Added `draggable={false}` to img → **FAILED**: Broke drag entirely (img fills container, prevents all drag)
- **Attempt 2**: Removed `draggable={false}`, added `clearData()` in parent TouchableOpacity handler → **FAILED**: Ran too late, blob URL still transferred
- **Root cause**: Browser's default `<img>` drag behavior runs FIRST, sets dataTransfer to blob URL before parent handlers execute
- **Final solution**: Moved drag handlers from TouchableOpacity to `<img>` element directly
  - Made img explicitly `draggable={Platform.OS === 'web' && !!frameId}`
  - Handlers on img run at correct time: `clearData()` prevents blob URL, then `setData()` sets frame ID
  - Drag attributes removed from parent TouchableOpacity

**Bug Fix #4: Frames Not Appearing at End After Drag (Regression)**
- **Symptom**: Dragged frames stayed in original position within target category instead of appearing at end (regression of Enhancement feature)
- **Root cause**: `handleFrameDrag` in TrainingTab.tsx updated frame timestamp during optimistic update but didn't re-sort the frames array, causing stale display order until page reload
- **Solution**: Added sorting step after optimistic update in TrainingTab.tsx (line 188): `const sortedFrames = updatedFrames.sort((a, b) => a.timestamp - b.timestamp);`. Added defensive sorting in UnifiedFrameGrid.tsx after grouping frames (lines 216-219) to ensure consistent display order
- **Test coverage**: Added test case `'should sort frames by timestamp after drag, placing dragged frame at end'` to TrainingTab.test.tsx verifying frames are sorted correctly after drag operations
- **Files modified**: TrainingTab.tsx, UnifiedFrameGrid.tsx, TrainingTab.test.tsx

## Enhancement

**Frame Ordering on Drag**: User requested "make the newly dragged frame appear at back of existing". Previously, frames maintained original capture timestamp when dragged, appearing in middle of new category. Solution: Update frame timestamp to `Date.now()` when dragged, making it sort to end. Added timestamp update in TrainingTab.tsx optimistic update, persisted in storage.ts `updateFrameLabel`, and explicit sorting by timestamp after loading frames. Dragged frames now appear at end of target category (most intuitive behavior).

## Files Modified
- src/components/dataset/FrameThumbnail.tsx - Delete button overlay, drag handlers moved to img element (final fix for blob URL)
- src/components/dataset/UnifiedFrameGrid.tsx - Drop zones on section headers, visual feedback
- src/components/unified/TrainingTab.tsx - Removed click cycling, added drag handler, timestamp update + sorting
- src/services/dataset/storage.ts - Persist timestamp on label update
- src/components/unified/__tests__/TrainingTab.test.tsx - Updated all tests for drag interaction, test timestamp update
- src/services/dataset/__tests__/storage.test.ts - Test timestamp persistence
- src/components/dataset/__tests__/FrameThumbnail.drag.test.tsx - NEW FILE: 11 drag tests (all verify handlers on img element)

## Impact

**UX**: More intuitive labeling (drag to target vs multi-click cycling), faster relabeling workflow, direct delete button access, visual feedback during drag.

**Code Quality**: Zero dependencies (native HTML5 API), reused proven optimistic update pattern, web-only platform checks prevent mobile issues.

**Performance**: Instant feedback via optimistic updates, no loading states or UI blocking.
