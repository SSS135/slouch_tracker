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
  let isDragging = $state(false);
  let activeId = $state<string | null>(null);
  let activeFrame = $state<DatasetFrameGridFrame | null>(null);
  let overLabel = $state<FrameLabel | null>(null);
  let dragX = $state(0);
  let dragY = $state(0);
  let contextMenuElement = $state<HTMLDivElement | null>(null);
  let contextMenuReturnFocus: HTMLElement | null = null;

  const activeThumbnail = useThumbnailUrl(() => activeFrame?.thumbnail ?? null);

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
      group.sort((a: DatasetFrameGridFrame, b: DatasetFrameGridFrame) => (a.timestamp || 0) - (b.timestamp || 0));
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

  function handleDragStart(event: DragEvent): void {
    const frameId = event.dataTransfer?.getData('application/x-slouch-frame-id');
    const frame = frames.find((candidate) => candidate.id === frameId);

    if (!frame) {
      return;
    }

    activeFrame = frame;
    activeId = frame.id;
    dragX = event.clientX + 12;
    dragY = event.clientY + 12;
    isDragging = true;
  }

  function handleDragOver(event: DragEvent, label: FrameLabel): void {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'move';
    }
    dragX = event.clientX + 12;
    dragY = event.clientY + 12;
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
    const frameId = event.dataTransfer?.getData('application/x-slouch-frame-id');
    const isKnownActiveFrame = Boolean(
      frameId && activeId === frameId && frames.some((frame) => frame.id === frameId),
    );

    if (isKnownActiveFrame && frameId && onFrameDrag) {
      onFrameDrag(frameId, label);
    }

    overLabel = null;
  }

  function handleDragEnd(): void {
    activeId = null;
    activeFrame = null;
    isDragging = false;
    overLabel = null;
    dragX = 0;
    dragY = 0;
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

  function getLabelColor(label: FrameLabel): string {
    switch (label) {
      case 'good':
        return '#28a745';
      case 'bad':
        return '#dc3545';
      case 'away':
        return '#3b82f6';
      case 'unused':
        return '#6c757d';
      default:
        return '#6c757d';
    }
  }
</script>

<div
  class="grid-root"
  role="region"
  aria-label="Dataset frames"
  ondragstart={handleDragStart}
  ondragend={handleDragEnd}
>
  <div class="scroll-area">
    <div class="sections">
      {#each sectionDefinitions as section (section.key)}
        {@const sectionFrames = groupedFrames[section.key]}
        {@const expanded = expandedSections[section.key]}
        {@const isOver = overLabel === section.label}

        <section
          class:dragging={isDragging}
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
                    onFrameRelabel: onFrameDrag,
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

  {#if activeFrame}
    <div class="drag-overlay" aria-hidden="true" style={`left: ${dragX}px; top: ${dragY}px;`}>
      <DatasetFrameThumbnail
        thumbnailUrl={activeThumbnail.url ?? undefined}
        nativeThumbnailId={activeFrame.thumbnail ? undefined : activeFrame.id}
        thumbnailMimeType={validatedThumbnailMimeType(activeFrame.thumbnailMimeType)}
        label={activeFrame.label}
        borderColor={getLabelColor(activeFrame.label)}
        onPress={() => undefined}
      />
    </div>
  {/if}
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
  onFrameRelabel: frameRelabel,
  onFramePreview: framePreview,
  onFramePreviewClear: framePreviewClear,
}: {
  frame: DatasetFrameGridFrame;
  color: string;
  onFrameClick: (id: string) => void;
  onFrameContextMenu: (id: string, event: MouseEvent) => void;
  onFrameDelete?: (id: string) => void;
  onFrameRelabel?: (id: string, label: FrameLabel) => void;
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
    />
    {#if frameRelabel}
      <label class="label-select">
        <span class="visually-hidden">Change label for frame {frame.id}</span>
        <select
          value={frame.label}
          aria-label={`Change label for frame ${frame.id}`}
          onchange={(event) => frameRelabel(frame.id, (event.currentTarget as HTMLSelectElement).value as FrameLabel)}
        >
          {#each sectionDefinitions as section (section.key)}
            <option value={section.label}>{section.title.replace(' Frames', '')}</option>
          {/each}
        </select>
      </label>
    {/if}
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
       the pointer was over the grid never chained to the panel. The dataset is
       paged, so no independent scroll is needed. */
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

  .drag-overlay {
    position: fixed;
    z-index: 10000;
    pointer-events: none;
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

  .frame-list-item,
  .label-select {
    display: block;
    width: 80px;
  }

  .label-select {
    margin-top: 0.25rem;
  }

  .label-select select {
    width: 100%;
    min-height: 1.75rem;
    border: 1px solid #495057;
    border-radius: 0.25rem;
    color: white;
    background: #212529;
    font-size: 0.7rem;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
