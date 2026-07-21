<script module lang="ts">
  let nextUndoButtonId = 0;
</script>

<script lang="ts">
  import type { CaptureAction } from '@/services/dataset/types';

  export interface UndoButtonProps {
    onUndo: () => void;
    canUndo: boolean;
    lastAction: CaptureAction | null;
  }

  let { onUndo, canUndo, lastAction }: UndoButtonProps = $props();
  let opened = $state(false);
  let popoverRoot = $state<HTMLDivElement | null>(null);
  const tooltipId = `undo-action-description-${++nextUndoButtonId}`;

  function setOpened(next: boolean): void {
    opened = next;
  }

  function handleUndo(): void {
    if (canUndo) onUndo();
  }

  function getLabelColor(label: string): string {
    switch (label) {
      case 'good':
        return '#40c057';
      case 'bad':
        return '#fa5252';
      case 'away':
        return '#868e96';
      default:
        return '#868e96';
    }
  }

  $effect(() => {
    if (!opened) return;

    const handlePointerDown = (event: PointerEvent): void => {
      if (!popoverRoot?.contains(event.target as Node)) {
        opened = false;
      }
    };
    const handleKeydown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault();
        opened = false;
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleKeydown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleKeydown);
    };
  });
</script>

<div
  bind:this={popoverRoot}
  class="popover-root"
>
  <button
    type="button"
    class="undo-button"
    aria-label="Undo"
    aria-expanded={opened}
    aria-describedby={opened ? tooltipId : undefined}
    onclick={handleUndo}
    onmouseenter={() => setOpened(true)}
    onmouseleave={() => setOpened(false)}
    onfocus={() => setOpened(true)}
    onblur={() => setOpened(false)}
    disabled={!canUndo}
    style:opacity={canUndo ? 1 : 0.3}
  >
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path d="M9 14 5 10l4-4" />
      <path d="M5 10h7a4 4 0 1 1 0 8h-1" />
    </svg>
  </button>

  {#if opened}
    <div id={tooltipId} class="popover-dropdown" role="tooltip">
      {#if lastAction}
        <div class="action-stack">
          <div
            class="thumbnail-box"
            style:border-color={getLabelColor(lastAction.label)}
          >
            <img src={lastAction.thumbnailUrl} alt="Undo preview" />
          </div>

          <div class="label-info">
            <span class="undo-label">
              Undo: Remove
              <span style:color={getLabelColor(lastAction.label)}>{lastAction.label}</span>
              frame
            </span>
          </div>

          <span class="keyboard-hint">Press U key to undo</span>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .popover-root {
    position: relative;
    display: inline-block;
  }

  .undo-button {
    display: inline-grid;
    width: 36px;
    height: 36px;
    padding: 0;
    place-items: center;
    border: 0;
    border-radius: var(--mantine-radius-default, 4px);
    background: var(--mantine-color-gray-7, #495057);
    color: var(--mantine-color-white, #fff);
    cursor: pointer;
    transition: background-color 100ms ease;
  }

  .undo-button:hover:not(:disabled) {
    background: var(--mantine-color-gray-6, #5c636a);
  }

  .undo-button:disabled {
    cursor: not-allowed;
  }

  .undo-button svg {
    display: block;
    fill: none;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 2;
  }

  .popover-dropdown {
    position: absolute;
    top: calc(100% + 8px);
    left: 50%;
    z-index: 100;
    min-width: 180px;
    min-height: 1px;
    padding: 12px;
    transform: translateX(-50%);
    border: 1px solid var(--mantine-color-default-border, #373a40);
    border-radius: var(--mantine-radius-default, 4px);
    background: var(--mantine-color-body, #25262b);
    box-shadow: var(--mantine-shadow-md, 0 4px 16px rgb(0 0 0 / 25%));
  }

  .popover-dropdown::before {
    position: absolute;
    top: -5px;
    left: 50%;
    width: 8px;
    height: 8px;
    border-top: 1px solid var(--mantine-color-default-border, #373a40);
    border-left: 1px solid var(--mantine-color-default-border, #373a40);
    background: var(--mantine-color-body, #25262b);
    content: '';
    transform: translateX(-50%) rotate(45deg);
  }

  .action-stack {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-xs, 8px);
    min-width: 180px;
  }

  .thumbnail-box {
    width: 100%;
    overflow: hidden;
    border: 3px solid;
    border-radius: 4px;
  }

  .thumbnail-box img {
    display: block;
    width: 100%;
    height: 120px;
    object-fit: contain;
  }

  .label-info {
    margin: 0;
  }

  .undo-label {
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: var(--mantine-line-height-sm, 1.4);
    font-weight: 500;
  }

  .keyboard-hint {
    color: var(--mantine-color-dimmed, #909296);
    font-size: var(--mantine-font-size-xs, 0.75rem);
    line-height: var(--mantine-line-height-xs, 1.4);
    font-style: italic;
  }
</style>
