<script lang="ts">
  export interface TrackingToggleButtonProps {
    /** True when tracking is paused (camera + detection stopped). */
    paused: boolean;
    /** Disabled while a start/stop is in flight (prevents overlapping toggles). */
    disabled?: boolean;
    onToggle: () => void;
  }

  let { paused, disabled = false, onToggle }: TrackingToggleButtonProps = $props();
</script>

<button
  type="button"
  class="tracking-toggle"
  class:paused
  {disabled}
  aria-pressed={paused}
  aria-label={paused ? 'Resume tracking' : 'Pause tracking'}
  onclick={() => {
    if (!disabled) onToggle();
  }}
>
  <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
    {#if paused}
      <path d="M8 5v14l11-7z" fill="currentColor" />
    {:else}
      <rect x="7" y="5" width="4" height="14" rx="1" fill="currentColor" />
      <rect x="13" y="5" width="4" height="14" rx="1" fill="currentColor" />
    {/if}
  </svg>
  <span>{paused ? 'Resume' : 'Pause'}</span>
</button>

<style>
  .tracking-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: 36px;
    padding: 0 16px;
    border: 1px solid transparent;
    border-radius: 12px;
    color: var(--mantine-color-white, #fff);
    background: rgb(0 0 0 / 75%);
    box-shadow: 0 4px 16px rgb(0 0 0 / 50%);
    backdrop-filter: blur(6px);
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
    line-height: 1;
    cursor: pointer;
    transition:
      background-color 150ms ease,
      opacity 150ms ease;
  }

  .tracking-toggle:hover:not(:disabled) {
    background: rgb(20 20 20 / 82%);
  }

  /* Paused reads as a distinct warm state (tracking stopped, action = resume). */
  .tracking-toggle.paused {
    color: var(--mantine-color-white, #fff);
    background: var(--mantine-color-orange-6, #fd7e14);
    box-shadow: 0 4px 16px rgb(253 126 20 / 35%);
  }

  .tracking-toggle.paused:hover:not(:disabled) {
    background: var(--mantine-color-orange-7, #e8590c);
  }

  .tracking-toggle:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .tracking-toggle svg {
    display: block;
  }
</style>
