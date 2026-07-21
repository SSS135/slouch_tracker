<script lang="ts">
  import { logger } from '@/services/logging';

  export interface AnimatedCaptureButtonProps {
    label: string;
    color: string;
    onPress: () => Promise<void>;
    disabled?: boolean;
  }

  type ButtonState = 'idle' | 'loading' | 'success' | 'error';

  let { label, color, onPress, disabled = false }: AnimatedCaptureButtonProps = $props();
  let state = $state<ButtonState>('idle');

  function paletteColor(value: string): string {
    const colors: Record<string, string> = {
      blue: 'var(--mantine-color-blue-6, #228be6)',
      green: 'var(--mantine-color-green-6, #40c057)',
      red: 'var(--mantine-color-red-6, #fa5252)',
    };
    return colors[value] ?? value;
  }

  async function handleClick(): Promise<void> {
    if (disabled || state === 'loading') {
      return;
    }

    try {
      state = 'loading';
      await onPress();
      state = 'success';
      setTimeout(() => {
        state = 'idle';
      }, 1200);
    } catch (error) {
      logger.error('detection', '[AnimatedCaptureButton] Capture failed:', error);
      state = 'error';
      setTimeout(() => {
        state = 'idle';
      }, 1200);
      // Re-throw to allow parent handlers to show user notifications.
      throw error;
    }
  }
</script>

<button
  type="button"
  aria-label={state === 'error' ? 'Failed' : label}
  aria-busy={state === 'loading'}
  onclick={handleClick}
  disabled={disabled || state === 'loading'}
  style:background-color={paletteColor(state === 'error' ? 'red' : color)}
  style:opacity={state === 'success' ? 0.7 : disabled || state === 'loading' ? 0.6 : 1}
>
  {#if state === 'loading'}
    <span class="loader" aria-hidden="true"></span>
  {:else if state === 'error'}
    Failed
  {:else}
    {label}
  {/if}
</button>

<style>
  button {
    box-sizing: border-box;
    width: 90px;
    min-width: 90px;
    height: 36px;
    min-height: 36px;
    padding: 0 18px;
    border: 1px solid transparent;
    border-radius: var(--mantine-radius-md, 8px);
    color: white;
    font-family: inherit;
    font-size: 14px;
    font-weight: 600;
    line-height: 1;
    cursor: pointer;
    transition: all 200ms ease-in-out;
  }

  button:disabled {
    border-color: var(--mantine-color-dark-4, #373a40);
    color: var(--mantine-color-dark-2, #909296);
    background: var(--mantine-color-dark-6, #25262b) !important;
    cursor: not-allowed;
    opacity: 0.6;
  }

  .loader {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgb(255 255 255 / 45%);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 700ms linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
