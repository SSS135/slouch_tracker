<script lang="ts">
  import type { FrameLabel } from '@/services/dataset/types';
  import { useThumbnailUrl } from '@/hooks/useThumbnailUrl';
  import DatasetFrameThumbnail from './DatasetFrameThumbnail.svelte';
  import { slide } from 'svelte/transition';
  import { tick } from 'svelte';

  type ThumbnailMimeType = 'image/jpeg' | 'image/png' | 'image/webp';

  export interface DatasetFrameGridFrame {
    id: string;
    label: FrameLabel;
    thumbnail?: Blob;
    thumbnailMimeType?: string;
    timestamp?: number;
  }

  export interface DatasetFrameGridProps {
    frames: DatasetFrameGridFrame[];
    onFrameClick: (id: string) => void;
    onDeleteFrame?: (id: string) => void;
    onFramePreview?: (thumbnailUrl: string, label: FrameLabel) => void;
    onFramePreviewClear?: () => void;
    onFrameDrag?: (frameId: string, newLabel: FrameLabel) => void;
  }

  interface GroupedFrames {
    good: DatasetFrameGridFrame[];
    bad: DatasetFrameGridFrame[];
    away: DatasetFrameGridFrame[];
    unused: DatasetFrameGridFrame[];
  }

  type SectionKey = keyof GroupedFrames;

  interface ContextMenuState {
    frameId: string;
    x: number;
    y: number;
  }

  const sectionDefinitions: ReadonlyArray<{
    key: SectionKey;
    title: string;
    color: string;
    label: FrameLabel;
    emptyMessage: string;
  }> = [
    {
      key: 'good',
      title: 'Good Frames',
      color: '#28a745',
      label: 'good' as FrameLabel,
      emptyMessage: 'No good posture frames yet',
    },
    {
      key: 'bad',
      title: 'Bad Frames',
      color: '#dc3545',
      label: 'bad' as FrameLabel,
      emptyMessage: 'No bad posture frames yet',
    },
    {
      key: 'away',
      title: 'Away Frames',
      color: '#3b82f6',
      label: 'away' as FrameLabel,
      emptyMessage: 'No away frames yet',
    },
    {
      key: 'unused',
      title: 'Unused Frames',
      color: '#6c757d',
      label: 'unused' as FrameLabel,
      emptyMessage: 'No unused frames',
    },
  ];

  let {
    frames,
    onFrameClick,
    onDeleteFrame,
    onFramePreview,
    onFramePreviewClear,
    onFrameDrag,
  }: DatasetFrameGridProps = $props();

  let expandedSections = $state<Record<SectionKey, boolean>>({
    good: true,
    bad: true,
    away: true,
    unused: false,
  });
  let contextMenu = $state<ContextMenuState | null>(null);
  let activeId = $state<string | null>(null);
  let overLabel = $state<FrameLabel | null>(null);
  let contextMenuElement = $state<HTMLDivElement | null>(null);
  let contextMenuReturnFocus: HTMLElement | null = null;

  const groupedFrames = $derived.by((): GroupedFrames => {
    const grouped: GroupedFrames = {
      good: [],
      bad: [],
      away: [],
      unused: [],
    };

    for (const frame of frames) {
      grouped[frame.label].push(frame);
    }

    for (const group of Object.values(grouped) as DatasetFrameGridFrame[][]) {
      // Newest first so freshly captured frames surface at the top of each section.
      group.sort((a: DatasetFrameGridFrame, b: DatasetFrameGridFrame) => (b.timestamp || 0) - (a.timestamp || 0));
    }

    return grouped;
  });

  $effect(() => {
    if (!contextMenu) {
      return;
    }

    const handleClick = (): void => closeContextMenu(true);
    void tick().then(() => contextMenuElement?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus());

    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  });

  function toggleSection(section: SectionKey): void {
    expandedSections[section] = !expandedSections[section];
  }

  function closeContextMenu(restoreFocus = false): void {
    contextMenu = null;
    if (restoreFocus) contextMenuReturnFocus?.focus();
    contextMenuReturnFocus = null;
  }

  function handleContextMenu(frameId: string, event: MouseEvent): void {
    event.preventDefault();
    contextMenuReturnFocus = event.currentTarget as HTMLElement;
    contextMenu = {
      frameId,
      x: event.clientX,
      y: event.clientY,
    };
  }

  // The dragged frame id arrives through a component callback from the thumbnail,
  // not from dataTransfer.getData() (which Chromium blanks outside the drop event),
  // so `activeId` is the single source of truth for the whole drag gesture.
  function handleFrameDragStart(frameId: string): void {
    if (!frames.some((candidate) => candidate.id === frameId)) {
      return;
    }
    activeId = frameId;
  }

  function handleDragOver(event: DragEvent, label: FrameLabel): void {
    if (!activeId) {
      return;
    }
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'move';
    }
    overLabel = label;
  }

  function handleDragLeave(event: DragEvent): void {
    const currentTarget = event.currentTarget as HTMLElement;
    const relatedTarget = event.relatedTarget as Node | null;
    if (!relatedTarget || !currentTarget.contains(relatedTarget)) {
      overLabel = null;
    }
  }

  function handleDrop(event: DragEvent, label: FrameLabel): void {
    event.preventDefault();
    const frameId = activeId;
    if (frameId && frames.some((frame) => frame.id === frameId) && onFrameDrag) {
      onFrameDrag(frameId, label);
    }
    activeId = null;
    overLabel = null;
  }

  function handleFrameDragEnd(): void {
    activeId = null;
    overLabel = null;
  }

  function handleDeleteFrame(frameId: string): void {
    closeContextMenu(true);
    onDeleteFrame?.(frameId);
  }

  function handleMenuKeydown(event: KeyboardEvent): void {
    const items = Array.from(contextMenuElement?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? []);
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === 'Escape') {
      event.preventDefault();
      closeContextMenu(true);
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      items[(index + direction + items.length) % items.length]?.focus();
    } else if (event.key === 'Home') {
      event.preventDefault();
      items[0]?.focus();
    } else if (event.key === 'End') {
      event.preventDefault();
      items.at(-1)?.focus();
    }
  }

  function validatedThumbnailMimeType(value: string | undefined): ThumbnailMimeType | undefined {
    if (value === 'image/jpeg' || value === 'image/png' || value === 'image/webp') return value;
    return undefined;
  }
</script>

<div
  class="grid-root"
  role="region"
  aria-label="Dataset frames"
>
  <div class="scroll-area">
    <div class="sections">
      {#each sectionDefinitions as section (section.key)}
        {@const sectionFrames = groupedFrames[section.key]}
        {@const expanded = expandedSections[section.key]}
        {@const isOver = overLabel === section.label}

        <section
          class:dragging={activeId !== null}
          class="frame-section"
          role="group"
          aria-label={section.title}
          ondragover={(event) => handleDragOver(event, section.label)}
          ondragleave={handleDragLeave}
          ondrop={(event) => handleDrop(event, section.label)}
        >
          <button
            type="button"
            class="section-header"
            class:over={isOver}
            aria-expanded={expanded}
            aria-controls={`dataset-section-${section.key}`}
            onclick={() => toggleSection(section.key)}
          >
            <span class="section-title">
              <span class="section-dot" style={`background-color: ${section.color};`}></span>
              <span>{section.title} ({sectionFrames.length})</span>
            </span>
            <span class="chevron" aria-hidden="true">{expanded ? '⌄' : '›'}</span>
          </button>

          <div
            id={`dataset-section-${section.key}`}
            class:over-content={isOver}
            class="section-content"
            hidden={!expanded}
            transition:slide
          >
            {#if sectionFrames.length === 0}
              <p class="empty-message">{section.emptyMessage}</p>
            {:else}
              <div class="frame-list">
                {#each sectionFrames as frame (frame.id)}
                  {@render FrameThumbnail({
                    frame,
                    color: section.color,
                    onFrameClick,
                    onFrameContextMenu: handleContextMenu,
                    onFrameDelete: onDeleteFrame ? handleDeleteFrame : undefined,
                    onFramePreview,
                    onFramePreviewClear,
                  })}
                {/each}
              </div>
            {/if}
          </div>
        </section>
      {/each}
    </div>
  </div>
</div>

{#if contextMenu}
  <div bind:this={contextMenuElement} class="context-menu" role="menu" tabindex="-1" aria-label="Frame actions" style={`left: ${contextMenu.x}px; top: ${contextMenu.y}px;`} onkeydown={handleMenuKeydown}>
    {#if onFrameDrag}
      {#each sectionDefinitions as section (section.key)}
        <button type="button" role="menuitem" onclick={() => { onFrameDrag?.(contextMenu!.frameId, section.label); closeContextMenu(true); }}>
          Move to {section.title.replace(' Frames', '')}
        </button>
      {/each}
    {/if}
    {#if onDeleteFrame}
      <button type="button" role="menuitem" class="delete-menu-item" onclick={() => handleDeleteFrame(contextMenu!.frameId)}>Delete frame</button>
    {/if}
  </div>
{/if}

{#snippet FrameThumbnail({
  frame,
  color,
  onFrameClick: frameClick,
  onFrameContextMenu: frameContextMenu,
  onFrameDelete: frameDelete,
  onFramePreview: framePreview,
  onFramePreviewClear: framePreviewClear,
}: {
  frame: DatasetFrameGridFrame;
  color: string;
  onFrameClick: (id: string) => void;
  onFrameContextMenu: (id: string, event: MouseEvent) => void;
  onFrameDelete?: (id: string) => void;
  onFramePreview?: (thumbnailUrl: string, label: FrameLabel) => void;
  onFramePreviewClear?: () => void;
})}
  {@const thumbnail = useThumbnailUrl(() => frame.thumbnail ?? null)}
  <div class="frame-list-item">
    <DatasetFrameThumbnail
      thumbnailUrl={thumbnail.url ?? undefined}
      nativeThumbnailId={frame.thumbnail ? undefined : frame.id}
      thumbnailMimeType={validatedThumbnailMimeType(frame.thumbnailMimeType)}
      label={frame.label}
      borderColor={color}
      frameId={frame.id}
      testID={`frame-${frame.id}`}
      onPress={() => frameClick(frame.id)}
      onContextMenu={(event) => frameContextMenu(frame.id, event)}
      onDelete={frameDelete ? () => frameDelete(frame.id) : undefined}
      onMouseEnter={framePreview}
      onMouseLeave={framePreviewClear}
      onDragStart={onFrameDrag ? handleFrameDragStart : undefined}
      onDragEnd={onFrameDrag ? handleFrameDragEnd : undefined}
    />
  </div>
{/snippet}

<style>
  .grid-root {
    position: relative;
    width: 100%;
  }

  .scroll-area {
    box-sizing: border-box;
    width: 100%;
    padding: 8px;
    /* The grid flows inside the control panel's own scroll area and must not be
       its own scrollport. A nested `overflow: auto` here combined with
       `overscroll-behavior: contain` trapped wheel events, so scrolling while
       the pointer was over the grid never chained to the panel. Every frame
       renders on one page and scrolls with the panel. */
  }

  .sections {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .frame-section {
    overflow: hidden;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    box-shadow: none;
  }

  .frame-section.dragging {
    box-shadow: 0 0 0 2px rgb(0 123 255 / 40%);
  }

  .section-header {
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: space-between;
    padding: 12px;
    border: 0;
    border-bottom: 1px solid #333;
    background: #0a0a0a;
    color: white;
    cursor: pointer;
    text-align: left;
  }

  .section-header.over {
    border-bottom-color: #007bff;
    background: #1a3a5a;
    box-shadow: 0 0 12px rgb(0 123 255 / 60%);
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 600;
  }

  .section-dot {
    width: 12px;
    height: 12px;
    flex: 0 0 auto;
    border-radius: 50%;
  }

  .chevron {
    color: #999;
    font-size: 24px;
    line-height: 16px;
  }

  .section-content {
    padding: 12px;
    border-radius: 4px;
  }

  .section-content.over-content {
    background: #0f2537;
    box-shadow: 0 0 0 2px rgb(0 123 255 / 60%);
  }

  .empty-message {
    margin: 0;
    padding: 20px 0;
    color: #909296;
    font-size: 14px;
    font-style: italic;
    text-align: center;
  }

  .frame-list {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 8px;
    transition: all 0.2s ease-out;
  }

  .context-menu {
    position: fixed;
    z-index: 10001;
    display: flex;
    min-width: 10rem;
    flex-direction: column;
    padding: 0.25rem;
    border: 1px solid #495057;
    border-radius: 0.375rem;
    color: white;
    background: #212529;
    box-shadow: 0 0.5rem 1.5rem rgb(0 0 0 / 40%);
  }

  .context-menu button {
    border: 0;
    padding: 0.5rem 0.625rem;
    color: inherit;
    background: transparent;
    text-align: left;
    cursor: pointer;
  }

  .context-menu button:hover,
  .context-menu button:focus-visible {
    background: #343a40;
  }

  .context-menu .delete-menu-item {
    color: #ff8787;
  }

  .frame-list-item {
    display: block;
    width: 80px;
  }
</style>
