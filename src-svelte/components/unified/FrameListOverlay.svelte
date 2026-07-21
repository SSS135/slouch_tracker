<script lang="ts">
  import type { CapturedFrame } from '@/hooks/useFrameSampler';
  import { useThumbnailUrl } from '@/hooks/useThumbnailUrl';
  import type { FrameLabel } from '@/services/dataset/types';
  import { flip } from 'svelte/animate';
  import { tick } from 'svelte';
  import { fly, type TransitionConfig } from 'svelte/transition';

  export interface FrameListOverlayProps {
    frames: CapturedFrame[];
    onSaveAsGood: (id: string) => Promise<void>;
    onSaveAsBad: (id: string) => Promise<void>;
    onSaveAsAway: (id: string) => Promise<void>;
    onFramePreview?: (blobUrl: string, label: FrameLabel) => void;
    onFramePreviewClear?: () => void;
    queuedFrameCount?: number;
    onHoverStart?: () => void;
    onHoverEnd?: () => void;
  }

  let {
    frames,
    onSaveAsGood,
    onSaveAsBad,
    onSaveAsAway,
    onFramePreview,
    onFramePreviewClear,
    queuedFrameCount = 0,
    onHoverStart,
    onHoverEnd,
  }: FrameListOverlayProps = $props();

  let scrollAreaElement = $state<HTMLDivElement | null>(null);
  let previousFrameCount = $state<number | null>(null);
  let itemIntrosEnabled = $state(false);

  const reversedFrames = $derived([...frames].reverse());

  $effect(() => {
    const currentFrameCount = frames.length;

    if (previousFrameCount !== null && currentFrameCount > previousFrameCount) {
      scrollAreaElement?.scrollTo({ top: 0, behavior: 'smooth' });
    }

    previousFrameCount = currentFrameCount;
  });

  $effect(() => {
    void tick().then(() => {
      itemIntrosEnabled = true;
    });
  });

  function popLayoutExit(_node: Element): TransitionConfig {
    return {
      duration: 200,
      css: (t) => `position: absolute; width: 100%; opacity: ${t}; transform: scale(${0.8 + (0.2 * t)});`,
    };
  }

  function getBorderColor(label: FrameLabel): string {
    switch (label) {
      case 'good':
        return '4px solid #40c057';
      case 'bad':
        return '4px solid #fa5252';
      case 'away':
        return '4px solid #868e96';
      case 'unused':
      default:
        return '4px solid #373a40';
    }
  }

  function handleFrameMouseEnter(frame: CapturedFrame, thumbnailUrl: string | undefined): void {
    if (thumbnailUrl && onFramePreview) {
      onFramePreview(thumbnailUrl, frame.label);
    }
  }

  function handleFrameMouseLeave(): void {
    onFramePreviewClear?.();
  }

  function handleSave(
    event: MouseEvent,
    id: string,
    save: (frameId: string) => Promise<void>,
  ): void {
    event.stopPropagation();
    void save(id);
  }
</script>

{#if frames.length > 0}
  <div
    class="overlay"
    role="region"
    aria-label="Captured frame buffer"
    onmouseenter={() => onHoverStart?.()}
    onmouseleave={() => onHoverEnd?.()}
  >
    {#if queuedFrameCount > 0}
      <span class="queued-badge">+{queuedFrameCount}</span>
    {/if}

    <div class="scroll-area" bind:this={scrollAreaElement}>
      <div class="frame-stack">
        {#each reversedFrames as frame (frame.id)}
          <div
            class="frame-transition"
            animate:flip={{ duration: 200 }}
            in:fly={{ y: itemIntrosEnabled ? -10 : 0, duration: itemIntrosEnabled ? 200 : 0 }}
            out:popLayoutExit
          >
            {@render FrameListItem({ frame })}
          </div>
        {/each}
      </div>
    </div>
  </div>
{/if}

{#snippet FrameListItem({ frame }: { frame: CapturedFrame })}
  {@const thumbnail = useThumbnailUrl(() => frame.thumbnail)}
  <div
    class="frame-item"
    role="group"
    aria-label="Buffered capture"
    onmouseenter={() => handleFrameMouseEnter(frame, thumbnail.url)}
    onmouseleave={handleFrameMouseLeave}
  >
    {#if thumbnail.url}
      <img
        class="frame-image"
        src={thumbnail.url}
        alt="Frame"
        style={`border: ${getBorderColor(frame.label)};`}
      />
    {:else}
      <div
        class="skeleton"
        aria-label="Loading frame"
        style={`border: ${getBorderColor(frame.label)};`}
      ></div>
    {/if}

    <div class="action-group">
        <button
          type="button"
          class="action-button good"
          aria-label="Save frame as good"
          onclick={(event) => handleSave(event, frame.id, onSaveAsGood)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m5 12 4 4L19 6"></path>
          </svg>
        </button>
        <button
          type="button"
          class="action-button bad"
          aria-label="Save frame as bad"
          onclick={(event) => handleSave(event, frame.id, onSaveAsBad)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 6 18 18M18 6 6 18"></path>
          </svg>
        </button>
        <button
          type="button"
          class="action-button away"
          aria-label="Save frame as away"
          onclick={(event) => handleSave(event, frame.id, onSaveAsAway)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M15 19a6 6 0 0 0-12 0M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8m7-5 5 5m0-5-5 5"></path>
          </svg>
        </button>
    </div>
  </div>
{/snippet}

<style>
  .overlay {
    position: absolute;
    top: 16px;
    bottom: 16px;
    left: 16px;
    z-index: 60;
    display: flex;
    width: 160px;
    flex-direction: column;
    opacity: 1;
    transform: translateX(0);
    transition: opacity 0.2s ease-out, transform 0.2s ease-out;
  }

  .queued-badge {
    position: absolute;
    top: 8px;
    left: 50%;
    z-index: 70;
    padding: 4px 10px;
    border-radius: 999px;
    background: #fd7e14;
    color: white;
    font-size: 14px;
    font-weight: 600;
    line-height: 1.25;
    transform: translateX(-50%);
    transition: opacity 0.2s ease-in-out;
  }

  .scroll-area {
    box-sizing: border-box;
    min-height: 0;
    height: 100%;
    flex: 1;
    overflow-y: auto;
    /* Isolate scroll repaints to this container so they do not invalidate the
       live camera canvas painting behind the overlay. */
    contain: paint;
    overscroll-behavior: contain;
    scrollbar-width: none;
  }

  .scroll-area::-webkit-scrollbar {
    display: none;
  }

  .frame-stack {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-sm, 12px);
    padding: 8px;
  }

  .frame-transition {
    position: relative;
    width: 100%;
  }

  .frame-item {
    position: relative;
    cursor: pointer;
  }

  .frame-image,
  .skeleton {
    box-sizing: border-box;
    display: block;
    width: 100%;
    height: auto;
    min-height: 90px;
    border-radius: var(--mantine-radius-sm, 4px);
  }

  .frame-image {
    object-fit: contain;
  }

  .skeleton {
    background: #373a40;
    animation: pulse 1.5s ease-in-out infinite;
  }

  .action-group {
    position: absolute;
    bottom: 4px;
    left: 50%;
    z-index: 10;
    display: inline-flex;
    opacity: 0;
    pointer-events: none;
    transform: translateX(-50%);
  }

  .action-button {
    display: grid;
    width: 28px;
    height: 28px;
    padding: 0;
    place-items: center;
    border: 0;
    color: white;
    cursor: pointer;
  }

  .action-button:first-child {
    border-radius: var(--mantine-radius-md, 8px) 0 0 var(--mantine-radius-md, 8px);
  }

  .action-button:last-child {
    border-radius: 0 var(--mantine-radius-md, 8px) var(--mantine-radius-md, 8px) 0;
  }

  .action-button.good {
    background: #40c057;
  }

  .action-button.bad {
    background: #fa5252;
  }

  .action-button.away {
    background: #868e96;
  }

  .frame-item:hover .action-group,
  .frame-item:focus-within .action-group {
    opacity: 1;
    pointer-events: auto;
  }

  .action-button:hover {
    filter: brightness(1.1);
  }

  .action-button svg {
    width: 20px;
    height: 20px;
    fill: none;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 2;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }

    50% {
      opacity: 0.55;
    }
  }
</style>
