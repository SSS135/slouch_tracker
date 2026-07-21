import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import ConfirmationModal from '../ConfirmationModal.svelte';

beforeAll(() => {
  // jsdom does not implement the modal dialog methods used by the component.
  if (!HTMLDialogElement.prototype.showModal) {
    HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
      this.open = true;
    };
  }
  if (!HTMLDialogElement.prototype.close) {
    HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
      this.open = false;
    };
  }
});

afterEach(cleanup);

describe('ConfirmationModal parity with controlled oracle', () => {
  it('keeps controls interactive after a confirm that neither closes nor sets loading', async () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    render(ConfirmationModal, {
      props: {
        visible: true,
        title: 'Reset',
        message: 'Are you sure?',
        onConfirm,
        onCancel,
      },
    });

    const confirmButton = screen.getByRole('button', { name: 'Confirm' });
    const cancelButton = screen.getByRole('button', { name: 'Cancel' });
    const closeButton = screen.getByRole('button', { name: 'Close modal' });

    await fireEvent.click(confirmButton);
    expect(onConfirm).toHaveBeenCalledTimes(1);

    // Oracle contract: disabling is driven solely by the `loading` prop, so a
    // failed confirm that keeps the modal open must leave everything usable.
    expect(cancelButton).not.toBeDisabled();
    expect(closeButton).not.toBeDisabled();
    expect(confirmButton).not.toBeDisabled();

    await fireEvent.click(cancelButton);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('disables the action buttons when loading but keeps the close (X) button dismissable', async () => {
    const onCancel = vi.fn();

    render(ConfirmationModal, {
      props: {
        visible: true,
        title: 'Reset',
        message: 'Are you sure?',
        loading: true,
        onConfirm: vi.fn(),
        onCancel,
      },
    });

    expect(screen.getByRole('button', { name: 'Confirm' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();

    // Oracle contract: Mantine's default close (X) button gates only
    // closeOnClickOutside/closeOnEscape on loading; its onClose=onCancel is
    // never disabled, so the X button must still dismiss while loading.
    const closeButton = screen.getByRole('button', { name: 'Close modal' });
    expect(closeButton).not.toBeDisabled();

    await fireEvent.click(closeButton);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
