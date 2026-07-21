<script lang="ts">
  import type { FrameLabel } from '@/services/dataset/types';
  import { nativeClient } from '@/lib/native/client';

  export type ThumbnailMimeType = 'image/jpeg' | 'image/png' | 'image/webp';

  export interface DatasetFrameThumbnailProps {
    thumbnailUrl?: string;
    nativeThumbnailId?: string;
    thumbnailMimeType?: ThumbnailMimeType;
    label: FrameLabel;
    borderColor: string;
    onPress: () => void;
    onContextMenu?: (event: MouseEvent) => void;
    testID?: string;
    onMouseEnter?: (thumbnailUrl: string, label: FrameLabel) => void;
    onMouseLeave?: () => void;
    onDelete?: () => void;
    frameId?: string;
  }

  let {
    thumbnailUrl,
    nativeThumbnailId,
    thumbnailMimeType,
    label,
    borderColor,
    onPress,
    onContextMenu,
    testID,
    onMouseEnter,
    onMouseLeave,
    onDelete,
    frameId,
  }: DatasetFrameThumbnailProps = $props();

  let isHovered = $state(false);
  let isDragging = $state(false);
  let loadedThumbnailUrl = $state<string | null>(null);
  let thumbnailError = $state<string | null>(null);
  let thumbnailLoading = $state(false);
  let thumbnailRetry = $state(0);
  let dragOffsetX = $state(0);
  let dragOffsetY = $state(0);
  let previewedUrl = $state<string | null>(null);
  let dragOriginX = 0;
  let dragOriginY = 0;

  const transform = $derived(
    isDragging ? `translate3d(${dragOffsetX}px, ${dragOffsetY}px, 0)` : undefined,
  );
  const resolvedThumbnailUrl = $derived(thumbnailUrl ?? loadedThumbnailUrl);

  function isAllowedMimeType(value: string | undefined): value is ThumbnailMimeType {
    return value === 'image/jpeg' || value === 'image/png' || value === 'image/webp';
  }

  function errorMessage(cause: unknown): string {
    if (cause instanceof Error) return cause.message;
    if (
      typeof cause === 'object' && cause !== null &&
      'message' in cause && typeof cause.message === 'string'
    ) {
      return cause.message;
    }
    return String(cause);
  }

  $effect(() => {
    thumbnailRetry;
    if (thumbnailUrl) {
      thumbnailError = null;
      thumbnailLoading = false;
      loadedThumbnailUrl = null;
      return;
    }
    if (!nativeThumbnailId) return;
    if (!isAllowedMimeType(thumbnailMimeType)) {
      thumbnailError = 'Thumbnail MIME type is missing or invalid';
      thumbnailLoading = false;
      return;
    }

    let disposed = false;
    let objectUrl: string | null = null;
    thumbnailLoading = true;
    thumbnailError = null;
    void nativeClient.getThumbnail(nativeThumbnailId).then((bytes) => {
      if (disposed) return;
      objectUrl = URL.createObjectURL(new Blob([bytes], { type: thumbnailMimeType }));
      loadedThumbnailUrl = objectUrl;
    }).catch((cause: unknown) => {
      if (!disposed) thumbnailError = errorMessage(cause);
    }).finally(() => {
      if (!disposed) thumbnailLoading = false;
    });
    return () => {
      disposed = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      if (loadedThumbnailUrl === objectUrl) loadedThumbnailUrl = null;
    };
  });

  $effect(() => {
    if (!isHovered) {
      previewedUrl = null;
      return;
    }
    if (resolvedThumbnailUrl && previewedUrl !== resolvedThumbnailUrl) {
      previewedUrl = resolvedThumbnailUrl;
      onMouseEnter?.(resolvedThumbnailUrl, label);
    }
  });

  function setBodyCursor(cursor: string): void {
    if (typeof document !== 'undefined') {
      document.body.style.cursor = cursor;
    }
  }

  function handleDragStart(event: DragEvent): void {
    if (!frameId) {
      event.preventDefault();
      return;
    }

    isDragging = true;
    dragOriginX = event.clientX;
    dragOriginY = event.clientY;
    dragOffsetX = 0;
    dragOffsetY = 0;
    setBodyCursor('grabbing');

    event.dataTransfer?.setData('application/x-slouch-frame-id', frameId);
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move';
    }
  }

  function handleDrag(event: DragEvent): void {
    if (!isDragging || (event.clientX === 0 && event.clientY === 0)) return;
    dragOffsetX = event.clientX - dragOriginX;
    dragOffsetY = event.clientY - dragOriginY;
  }

  function handleDragEnd(): void {
    isDragging = false;
    dragOffsetX = 0;
    dragOffsetY = 0;
    setBodyCursor('');
  }

  function handleMouseEnter(): void {
    isHovered = true;
  }

  function handleMouseLeave(): void {
    isHovered = false;
    onMouseLeave?.();
  }

  function handleMouseDown(): void {
    if (frameId) {
      setBodyCursor('grabbing');
    }
  }

  function handleMouseUp(): void {
    setBodyCursor('');
  }

  function stopDeletePointer(event: MouseEvent): void {
    event.stopPropagation();
    event.preventDefault();
  }

  function handleDeleteClick(event: MouseEvent): void {
    stopDeletePointer(event);
    onDelete?.();
  }

  $effect(() => () => setBodyCursor(''));

</script>

<div
  class="thumbnail-stack"
  role="group"
  aria-label={`Frame ${frameId ?? label} thumbnail actions`}
  onmouseenter={handleMouseEnter}
  onmouseleave={handleMouseLeave}
>
  <button
    type="button"
    class="thumbnail"
    aria-label={`Preview frame ${frameId ?? ''} labeled ${label}`.replace('  ', ' ')}
    draggable={Boolean(frameId)}
    data-testid={testID}
    data-frame-id={frameId}
    style={`transform: ${transform ?? 'none'}; opacity: ${isDragging ? 0.5 : 1}; border-color: ${borderColor}; cursor: ${frameId ? 'grab' : 'pointer'};`}
    onclick={onPress}
    oncontextmenu={onContextMenu}
    onmousedown={handleMouseDown}
    onmouseup={handleMouseUp}
    ondragstart={handleDragStart}
    ondrag={handleDrag}
    ondragend={handleDragEnd}
  >
    {#if resolvedThumbnailUrl}
      <img src={resolvedThumbnailUrl} alt="" />
    {:else if thumbnailLoading}
      <span class="thumbnail-status" role="status">Loading thumbnail…</span>
    {:else}
      <span class="thumbnail-status">No thumbnail</span>
    {/if}
  </button>

  {#if onDelete}
    <button
      type="button"
      class="delete-button"
      aria-label={`Delete frame ${frameId ?? ''} labeled ${label}`.replace('  ', ' ')}
      data-testid={testID ? `${testID}-delete` : undefined}
      style:opacity={isHovered ? 1 : undefined}
      onmousedown={stopDeletePointer}
      onclick={handleDeleteClick}
    >
      <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path d="M6 6l12 12M18 6L6 18" />
      </svg>
    </button>
  {/if}

  {#if thumbnailError}
    <div class="thumbnail-error" role="alert">Thumbnail failed: {thumbnailError}</div>
    <button type="button" class="thumbnail-retry" aria-label={`Retry thumbnail for frame ${frameId ?? label}`} onclick={() => { thumbnailRetry += 1; }}>
      Retry thumbnail
    </button>
  {/if}
</div>

<style>
  .thumbnail-stack {
    position: relative;
    width: 80px;
  }

  .thumbnail {
    position: relative;
    width: 80px;
    height: 60px;
    overflow: hidden;
    padding: 0;
    border: 3px solid;
    border-radius: 6px;
    background: #0a0a0a;
    user-select: none;
  }

  .thumbnail img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    pointer-events: none;
  }

  .thumbnail-status {
    display: grid;
    height: 100%;
    padding: 0.25rem;
    place-items: center;
    color: #ced4da;
    font-size: 0.625rem;
    line-height: 1.2;
  }

  .thumbnail-error {
    width: 80px;
    margin-top: 0.25rem;
    color: #ff8787;
    font-size: 0.625rem;
    overflow-wrap: anywhere;
  }

  .thumbnail-retry {
    width: 80px;
    min-height: 1.75rem;
    margin-top: 0.25rem;
    border: 1px solid #ff8787;
    border-radius: 0.25rem;
    color: #fff;
    background: #c92a2a;
    font-size: 0.625rem;
    cursor: pointer;
  }

  .delete-button {
    position: absolute;
    top: 4px;
    right: 4px;
    z-index: 10;
    display: grid;
    width: 24px;
    height: 24px;
    padding: 0;
    place-items: center;
    border: 0;
    border-radius: 50%;
    background: rgb(0 0 0 / 70%);
    color: white;
    cursor: pointer;
    opacity: 0;
    pointer-events: auto;
    transition: opacity 0.2s ease-in-out;
  }

  .thumbnail-stack:hover .delete-button,
  .thumbnail-stack:focus-within .delete-button {
    opacity: 1;
  }

  .delete-button svg {
    display: block;
    fill: none;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-width: 2;
  }
</style>
