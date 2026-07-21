<script module lang="ts">
  let nextModalId = 0;
</script>

<script lang="ts">
  export interface ConfirmationModalProps {
    visible: boolean;
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    confirmButtonColor?: string;
    loading?: boolean;
    onConfirm: () => void;
    onCancel: () => void;
  }

  let {
    visible,
    title,
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    confirmButtonColor = 'red',
    loading = false,
    onConfirm,
    onCancel,
  }: ConfirmationModalProps = $props();

  const modalId = `confirmation-modal-${++nextModalId}`;
  const titleId = `${modalId}-title`;
  const messageId = `${modalId}-message`;
  let dialog: HTMLDialogElement;
  const interactionLocked = $derived(loading);

  $effect(() => {
    if (!dialog) {
      return;
    }

    if (visible && !dialog.open) {
      dialog.showModal();
    } else if (!visible && dialog.open) {
      dialog.close();
    }
  });

  function handleDialogClick(event: MouseEvent): void {
    if (event.target === event.currentTarget && !interactionLocked) {
      onCancel();
    }
  }

  function handleDialogCancel(event: Event): void {
    event.preventDefault();
    if (!interactionLocked) {
      onCancel();
    }
  }

  function handleCancel(): void {
    if (!interactionLocked) onCancel();
  }

  function handleConfirm(): void {
    if (interactionLocked) return;
    onConfirm();
  }

  function confirmColor(value: string): string {
    const semanticColors: Record<string, string> = {
      orange: 'var(--mantine-color-orange-6, #fd7e14)',
      red: 'var(--mantine-color-red-6, #fa5252)',
    };
    return semanticColors[value] ?? value;
  }
</script>

<dialog
  bind:this={dialog}
  aria-labelledby={titleId}
  aria-describedby={messageId}
  aria-busy={interactionLocked}
  onclick={handleDialogClick}
  oncancel={handleDialogCancel}
>
  <div class="modal-content">
    <div class="modal-header">
      <h2 id={titleId}>{title}</h2>
      <button type="button" class="close-button" aria-label="Close modal" onclick={onCancel}>×</button>
    </div>
    <p id={messageId}>{message}</p>

    <div class="modal-actions">
      <button type="button" class="cancel-button" onclick={handleCancel} disabled={interactionLocked}>
        {cancelText}
      </button>

      <button
        type="button"
        class="confirm-button"
        style:background-color={confirmColor(confirmButtonColor)}
        data-semantic-color={confirmButtonColor}
        onclick={handleConfirm}
        disabled={interactionLocked}
        aria-busy={interactionLocked}
      >
        {#if interactionLocked}
          <span class="loading-indicator" aria-hidden="true"></span>
        {/if}
        {confirmText}
      </button>
    </div>
  </div>
</dialog>

<style>
  dialog {
    width: min(440px, calc(100vw - 2rem));
    max-width: calc(100vw - 2rem);
    padding: 0;
    border: 0;
    border-radius: var(--mantine-radius-md, 8px);
    background: var(--mantine-color-body, #1a1b1e);
    color: var(--mantine-color-text, #c1c2c5);
    box-shadow: var(--mantine-shadow-xl, 0 1.5rem 4rem rgb(0 0 0 / 40%));
  }

  dialog::backdrop {
    background: rgb(0 0 0 / 70%);
    backdrop-filter: blur(3px);
  }

  .modal-content {
    padding: var(--mantine-spacing-md, 16px);
  }

  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--mantine-spacing-md, 16px);
    margin-bottom: var(--mantine-spacing-md, 16px);
  }

  h2 {
    margin: 0;
    font-size: 1rem;
    line-height: 1.55;
    font-weight: 400;
  }

  p {
    margin: 0;
    color: var(--mantine-color-dimmed, #909296);
    font-size: 0.875rem;
    line-height: 1.45;
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--mantine-spacing-sm, 12px);
    margin-top: var(--mantine-spacing-md, 16px);
  }

  button {
    min-height: 2.25rem;
    padding: 0.5rem 1rem;
    border: 0;
    border-radius: 0.25rem;
    font: inherit;
    font-size: 0.875rem;
    cursor: pointer;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.65;
  }

  .cancel-button {
    border: 1px solid var(--mantine-color-default-border, #373a40);
    background: var(--mantine-color-default, #25262b);
    color: var(--mantine-color-text, #c1c2c5);
  }

  .close-button {
    width: 28px;
    min-height: 28px;
    padding: 0;
    color: var(--mantine-color-dimmed, #909296);
    background: transparent;
    font-size: 1.25rem;
    line-height: 1;
  }

  .close-button:hover {
    background: var(--mantine-color-dark-5, #2c2e33);
  }

  .confirm-button {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: white;
  }

  .confirm-button[data-semantic-color='red']:hover:not(:disabled) {
    background-color: var(--mantine-color-red-7, #f03e3e) !important;
  }

  .confirm-button[data-semantic-color='orange']:hover:not(:disabled) {
    background-color: var(--mantine-color-orange-7, #f76707) !important;
  }

  .loading-indicator {
    width: 0.875rem;
    height: 0.875rem;
    border: 2px solid rgb(255 255 255 / 45%);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
