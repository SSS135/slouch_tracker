import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import DatasetFrameGrid from '../DatasetFrameGrid.svelte';

const frame = {
  id: 'frame-a',
  label: 'good' as const,
  thumbnail: new Blob(['image'], { type: 'image/webp' }),
  thumbnailMimeType: 'image/webp',
  timestamp: 1,
};

let createObjectUrl: ReturnType<typeof vi.spyOn>;
let revokeObjectUrl: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  createObjectUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:dataset-frame');
  revokeObjectUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('DatasetFrameGrid accessibility', () => {
  it('exposes disclosure state and controls its labelled region', async () => {
    render(DatasetFrameGrid, { props: { frames: [frame], onFrameClick: vi.fn() } });
    const disclosure = screen.getByRole('button', { name: 'Good Frames (1)' });
    expect(disclosure).toHaveAttribute('aria-expanded', 'true');
    expect(disclosure).toHaveAttribute('aria-controls', 'dataset-section-good');
    await fireEvent.click(disclosure);
    expect(disclosure).toHaveAttribute('aria-expanded', 'false');
    expect(document.getElementById('dataset-section-good')).toHaveAttribute('hidden');
  });

  it('keeps preview and delete as sibling controls with independent keyboard actions', async () => {
    const preview = vi.fn();
    const remove = vi.fn();
    render(DatasetFrameGrid, { props: { frames: [frame], onFrameClick: preview, onDeleteFrame: remove } });
    const deleteButton = screen.getByRole('button', { name: 'Delete frame frame-a labeled good' });
    deleteButton.focus();
    await userEvent.keyboard('{Enter}');
    expect(remove).toHaveBeenCalledOnce();
    expect(remove).toHaveBeenCalledWith('frame-a');
    expect(preview).not.toHaveBeenCalled();
    expect(deleteButton.parentElement?.querySelector('button.thumbnail')).not.toContainElement(deleteButton);
  });

  it('creates and revokes thumbnail object URLs on unmount', async () => {
    const mounted = render(DatasetFrameGrid, { props: { frames: [frame], onFrameClick: vi.fn() } });
    await waitFor(() => expect(createObjectUrl).toHaveBeenCalledOnce());
    mounted.unmount();
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:dataset-frame');
  });

  it('focuses the menu, supports arrow navigation and restores focus on Escape', async () => {
    render(DatasetFrameGrid, {
      props: { frames: [frame], onFrameClick: vi.fn(), onDeleteFrame: vi.fn(), onFrameDrag: vi.fn() },
    });
    const preview = screen.getByRole('button', { name: 'Preview frame frame-a labeled good' });
    preview.focus();
    await fireEvent.contextMenu(preview, { clientX: 10, clientY: 10 });
    const items = await screen.findAllByRole('menuitem');
    await waitFor(() => expect(items[0]).toHaveFocus());
    await fireEvent.keyDown(items[0], { key: 'ArrowDown' });
    expect(items[1]).toHaveFocus();
    await fireEvent.keyDown(items[1], { key: 'Escape' });
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();
    expect(preview).toHaveFocus();
  });
});
