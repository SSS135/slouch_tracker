# Task 2025-10-31: Create Frame List Overlay Component
**STATUS:** COMPLETED

## Completion Summary

Successfully created a scrollable frame list overlay on the left side of the video feed, replacing the previous Collect tab UI.

### Implemented Changes:

1. **New Components Created**
   - `src/hooks/useThumbnailUrl.ts` - Shared hook for thumbnail URL management
   - `src/components/unified/FrameListOverlay.tsx` - Frame list overlay component

2. **Frame List Overlay Features**
   - **Fully transparent background** - No dark panel, only frames visible
   - **160px width** - Compact vertical list on left side
   - **4px colored borders** - Green (good), Red (bad), Gray (away), Dark (unused)
   - **Original aspect ratio** - Images use `fit="contain"`
   - **No scrollbar** - `scrollbarSize={0}` for clean appearance
   - **Reverse chronological order** - Newest frames at top
   - **Slide-in animation** - 0.2s fade + transform on appearance

3. **Frame Card Design**
   - Thumbnail displayed with original aspect ratio
   - Thin colored border (4px) instead of thick gray card
   - No timestamp display
   - Label badges removed (border color indicates label)

4. **Hover Actions**
   - Action buttons overlay on image bottom center
   - Don't push other frames down
   - Medium-sized buttons (size="md") with 20px icons
   - Semantic icons: IconCheck (good), IconX (bad), IconUserOff (away)

5. **Frame Preview Updates**
   - Centered between frame list and settings panel
   - Fixed max-width: 800px (maintains consistent size)
   - Height: 50% of viewport
   - Accounts for panel state (collapsed/expanded)
   - Dark overlay background (75% opacity)
   - 0.2s fade transition

6. **FPS Indicator Relocation**
   - Removed from video overlay
   - Added to Camera Settings section in RuntimeTab
   - Clean cyan-colored display with description

### Files Created:
- `src/hooks/useThumbnailUrl.ts`
- `src/components/unified/FrameListOverlay.tsx`

### Files Modified:
- `src/components/unified/BufferFrameGrid.tsx` - Uses shared useThumbnailUrl hook
- `src/components/unified/VideoSection.tsx` - Integrates frame list overlay, preview centering, removed FPS badge
- `src/pages/UnifiedPosturePage.tsx` - Passes frame data and handlers to VideoSection
- `src/components/unified/RuntimeTab.tsx` - Added FPS display to Camera Settings

### Result:
Clean, minimalist frame list that doesn't obstruct the video. Frames are clearly labeled by colored borders, with intuitive hover actions. Preview properly centered regardless of panel state.

## User Request
Build scrollable frame list overlay for left side of video showing captured frames with thumbnails, label badges, and hover action buttons.

## General Description
Create a new component that displays a vertical scrollable list of captured frames positioned on the left side of the video feed. Frames appear in reverse chronological order (newest at top) with always-visible label badges. Hovering individual frames reveals action buttons for saving with different labels. The component slides in with animation when frames are added.

## Action Plan

1. **Create FrameListOverlay.tsx component:**
   - Location: `src/components/unified/FrameListOverlay.tsx`
   - Accept props:
     - `frames: CapturedFrame[]` - Array of captured frames
     - `onSaveAsGood: (id: string) => Promise<void>`
     - `onSaveAsBad: (id: string) => Promise<void>`
     - `onSaveAsAway: (id: string) => Promise<void>`
     - `onFramePreview?: (blobUrl: string, label: FrameLabel) => void`
     - `onFramePreviewClear?: () => void`

2. **Implement container structure:**
   - Use Mantine `Stack` with fixed width 200px
   - Position absolutely: `position: 'absolute', left: 16, top: 16, bottom: 16`
   - Background: `rgba(0, 0, 0, 0.85)` with `backdropFilter: 'blur(6px)'`
   - Border radius: 12px
   - z-index: 60 (above video, below preview overlay which is z-index 50 in VideoSection)
   - Add `boxShadow: '0 4px 20px rgba(0, 0, 0, 0.6)'`

3. **Make list scrollable:**
   - Wrap frame items in Mantine `ScrollArea` component
   - `scrollbarSize={4}` for slim scrollbar
   - `offsetScrollbars` to overlay scrollbar on content
   - `style={{ flex: 1, minHeight: 0 }}` to enable scrolling within absolute positioned parent

4. **Display frames in reverse order:**
   - Reverse array before mapping: `[...frames].reverse().map(...)`
   - Ensures newest frames appear at top
   - Maintains stable keys using `frame.id`

5. **Create FrameListItem component (internal):**
   - Structure:
     - Mantine `Card` with `padding="xs"` and `radius="sm"`
     - Image thumbnail (80x60px, maintains 4:3 aspect ratio)
     - Badge positioned absolutely top-right (always visible)
     - Action buttons group (only visible on hover)
   - Hover state management: `const [isHovered, setIsHovered] = useState(false);`
   - Wire `onMouseEnter` and `onMouseLeave` to toggle hover state

6. **Implement label badge (always visible):**
   - Use Mantine `Badge` with `size="sm"`
   - Position absolutely: `top: 4, right: 4`
   - Color mapping:
     - `FrameLabel.GOOD` → `color="green"`
     - `FrameLabel.BAD` → `color="red"`
     - `FrameLabel.AWAY` → `color="gray"`
     - `FrameLabel.UNUSED` → `color="dark"`
   - Text: label name in uppercase (e.g., "GOOD", "BAD", "AWAY")
   - Badge should have semi-transparent background: `variant="filled"` with 90% opacity

7. **Implement hover action buttons:**
   - Use Mantine `ActionIcon.Group` with 3 buttons
   - Show only when `isHovered === true`
   - Conditional rendering: `{isHovered && <ActionIcon.Group>...</ActionIcon.Group>}`
   - Buttons:
     - IconDeviceFloppy with green color → Save as Good
     - IconDeviceFloppy with red color → Save as Bad
     - IconDeviceFloppy with gray color → Save as Away
   - Use @tabler/icons-react for icons
   - Button size: "xs"
   - Position below thumbnail with `mt="xs"`

8. **Add slide-in animation:**
   - Use CSS transition on entire overlay container
   - When `frames.length > 0`: `opacity: 1, transform: 'translateX(0)'`
   - When `frames.length === 0`: `opacity: 0, transform: 'translateX(-20px)'`
   - Transition duration: 0.2s ease-out
   - Component mounts/unmounts based on frames array length

9. **Integrate frame preview trigger:**
   - Wire `onMouseEnter` on FrameListItem to call `onFramePreview?.(thumbnailUrl, frame.label)`
   - Wire `onMouseLeave` to call `onFramePreviewClear?.()`
   - Reuse existing thumbnail URL generation logic from BufferFrameGrid (useThumbnailUrl hook)

10. **Handle empty state:**
    - If `frames.length === 0`, render nothing (no overlay)
    - Overlay only appears when frames exist

## Rationale

**Component Reuse:** Extract `useThumbnailUrl` hook from BufferFrameGrid.tsx into shared utility `src/hooks/useThumbnailUrl.ts`. Both components need identical blob URL generation and cleanup logic. Avoids duplication and ensures consistent memory management.

**Overlay Positioning:** Past task `0009-feature-hover-preview-frames.md` demonstrated successful absolute positioning patterns. Frame list overlay uses same z-index hierarchy (60 for list, 50 for preview, 100 for right panel).

**Reverse Order:** Task `0015-feature-improve-frame-list-ui.md` implemented `.reverse()` pattern for newest-first ordering. Proven approach that doesn't affect selection logic (uses stable `frame.id` keys).

**Hover Actions:** Following Mantine ActionIcon patterns from existing codebase (RuntimeTab uses ActionIcon for dev settings). Hover-to-reveal pattern is standard UX for non-primary actions.

**Always-Visible Badges:** Unlike BufferFrameGrid where badges are secondary to timestamp, in overlay list the label is primary identifier. Users need to see label at a glance without interaction.

**Fixed Width (200px):** Task `2025-10-29-react-dom-vite-mantine-migration.md` showed Mantine components work best with fixed widths. 200px provides enough space for 80px thumbnail + padding + scrollbar without feeling cramped.

## Files to Create

- `src/components/unified/FrameListOverlay.tsx` - New overlay component
- `src/hooks/useThumbnailUrl.ts` - Shared hook for thumbnail URL management

## Files to Modify

- `src/components/unified/BufferFrameGrid.tsx` - Refactor to use shared useThumbnailUrl hook

## Related Tasks

- `tasks/0015-feature-improve-frame-list-ui.md` - Reverse ordering pattern
- `tasks/0009-feature-hover-preview-frames.md` - Overlay positioning and preview integration
- `tasks/2025-10-29-react-dom-vite-mantine-migration.md` - Mantine component patterns
