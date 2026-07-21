<script lang="ts">
  import type { Snippet } from 'svelte';
  import { logger } from '../services/logging/logger';

  export interface SvelteErrorContext {
    source: 'svelte-boundary';
    componentStack: null;
  }

  interface ErrorBoundaryProps {
    children?: Snippet;
    fallback?: Snippet;
    onError?: (error: Error, context: SvelteErrorContext) => void;
  }

  let { children, fallback, onError }: ErrorBoundaryProps = $props();
  let capturedError = $state<Error | null>(null);
  let showDetails = $state(false);
  let retryBoundary = $state<(() => void) | undefined>(undefined);

  function toError(value: unknown): Error {
    return value instanceof Error ? value : new Error(String(value));
  }

  function handleError(value: unknown, resetBoundary: () => void): void {
    const error = toError(value);
    capturedError = error;
    retryBoundary = resetBoundary;
    showDetails = false;

    const context: SvelteErrorContext = {
      source: 'svelte-boundary',
      componentStack: null,
    };
    logger.error('debug', '[ErrorBoundary] Caught error:', error, context);
    onError?.(error, context);
  }

  function reset(): void {
    capturedError = null;
    showDetails = false;
    const retry = retryBoundary;
    retryBoundary = undefined;
    retry?.();
  }

  function reload(): void {
    window.location.reload();
  }

  function toggleDetails(): void {
    showDetails = !showDetails;
  }
</script>

<svelte:boundary onerror={handleError}>
  {#if children}
    {@render children()}
  {/if}

  {#snippet failed(boundaryError, _resetBoundary)}
    {@const error = capturedError ?? toError(boundaryError)}

    {#if fallback}
      {@render fallback()}
    {:else}
      <div class="boundary" role="alert">
        <div class="panel">
          <div class="content">
            <h2>Something went wrong</h2>

            {#if error}
              <p class="message">{error.message}</p>
            {/if}

            <div class="actions">
              <button type="button" class="primary" onclick={reset}>Try Again</button>
              <button type="button" class="secondary" onclick={reload}>Reload App</button>

              {#if error.stack}
                <button type="button" class="details-toggle" onclick={toggleDetails}>
                  {showDetails ? 'Hide' : 'Show'} Technical Details
                </button>
              {/if}
            </div>

            {#if showDetails}
              <div class="details">
                <h3>Technical Details:</h3>
                <pre>{error.stack}</pre>
              </div>
            {/if}
          </div>
        </div>
      </div>
    {/if}
  {/snippet}
</svelte:boundary>

<style>
  .boundary {
    align-items: center;
    display: flex;
    height: 100%;
    justify-content: center;
    min-height: 400px;
    padding: 32px;
  }

  .panel {
    border: 1px solid var(--mantine-color-default-border, #373a40);
    border-radius: 8px;
    background: var(--mantine-color-body, #1a1b1e);
    box-shadow: 0 4px 12px rgb(0 0 0 / 35%);
    max-width: 600px;
    padding: 32px;
    width: 100%;
  }

  .content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  h2 {
    color: var(--mantine-color-red-6, #fa5252);
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    text-align: center;
  }

  .message {
    color: #868e96;
    font-size: 1rem;
    margin: 0;
    text-align: center;
  }

  .actions {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  button {
    border: 1px solid transparent;
    border-radius: 4px;
    cursor: pointer;
    font: inherit;
    min-height: 36px;
    padding: 8px 16px;
    width: 100%;
  }

  .primary {
    background: #228be6;
    color: #fff;
  }

  .secondary {
    background: var(--mantine-color-default, #25262b);
    border-color: var(--mantine-color-default-border, #373a40);
    color: var(--mantine-color-white, #fff);
  }

  .details-toggle {
    background: transparent;
    color: #228be6;
    font-size: 0.75rem;
  }

  .details {
    background: #f1f3f5;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    color: #212529;
    padding: 16px;
  }

  h3 {
    font-size: 0.875rem;
    font-weight: 600;
    margin: 0 0 8px;
  }

  pre {
    font-size: 11px;
    margin: 0;
    max-height: 200px;
    overflow: auto;
    white-space: pre-wrap;
  }
</style>
