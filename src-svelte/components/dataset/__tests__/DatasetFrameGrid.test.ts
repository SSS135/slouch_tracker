import '@testing-library/jest-dom/vitest';
import { cleanup, createEvent, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
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

function makeDataTransfer() {
  const store = new Map<string, string>();
  return {
    effectAllowed: 'none' as string,
    dropEffect: 'none' as string,
    setData(type: string, value: string) {
      store.set(type, value);
    },
    getData(type: string) {
      return store.get(type) ?? '';
    },
  };
}

async function fireDrag(
  node: Element,
  type: 'dragStart' | 'dragOver' | 'drop',
  dataTransfer: ReturnType<typeof makeDataTransfer>,
): Promise<Event> {
  const event = createEvent[type](node, { bubbles: true, cancelable: true });
  Object.defineProperty(event, 'dataTransfer', { value: dataTransfer, configurable: true });
  await fireEvent(node, event);
  return event;
}

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

describe('DatasetFrameGrid bug regressions', () => {
  const good = (id: string, timestamp: number) => ({
    id,
    label: 'good' as const,
    thumbnail: new Blob(['image'], { type: 'image/webp' }),
    thumbnailMimeType: 'image/webp',
    timestamp,
  });

  // BUG 1: newly captured frames (largest timestamp) must surface first in the
  // section so a capture is visible immediately instead of buried at the bottom.
  it('orders frames newest-first within a section', () => {
    render(DatasetFrameGrid, {
      props: { frames: [good('older', 100), good('newer', 200)], onFrameClick: vi.fn() },
    });
    const newer = screen.getByTestId('frame-newer');
    const older = screen.getByTestId('frame-older');
    expect(newer.compareDocumentPosition(older) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  // BUG 2: the X button must delete the frame, never relabel/move it to another section.
  it('deletes via the X button without relabeling the frame', async () => {
    const remove = vi.fn();
    const relabel = vi.fn();
    render(DatasetFrameGrid, {
      props: { frames: [frame], onFrameClick: vi.fn(), onDeleteFrame: remove, onFrameDrag: relabel },
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Delete frame frame-a labeled good' }));
    expect(remove).toHaveBeenCalledOnce();
    expect(remove).toHaveBeenCalledWith('frame-a');
    expect(relabel).not.toHaveBeenCalled();
  });

  // BUG 3: dragging over a section must accept the drop (preventDefault) and dropping
  // must relabel the frame to that section's label.
  it('accepts a drop on another section and relabels the dragged frame', async () => {
    const relabel = vi.fn();
    const frames = [good('g1', 1), { ...good('placeholder', 2), id: 'b1', label: 'bad' as const }];
    render(DatasetFrameGrid, {
      props: { frames, onFrameClick: vi.fn(), onDeleteFrame: vi.fn(), onFrameDrag: relabel },
    });
    const source = screen.getByTestId('frame-g1');
    const badSection = screen.getByRole('group', { name: 'Bad Frames' });
    const dataTransfer = makeDataTransfer();

    await fireDrag(source, 'dragStart', dataTransfer);
    const overEvent = await fireDrag(badSection, 'dragOver', dataTransfer);
    expect(overEvent.defaultPrevented).toBe(true);
    await fireDrag(badSection, 'drop', dataTransfer);

    expect(relabel).toHaveBeenCalledOnce();
    expect(relabel).toHaveBeenCalledWith('g1', 'bad');
  });
});

describe('DatasetFrameGrid drag relabeling', () => {
  const good = (id: string, timestamp: number) => ({
    id,
    label: 'good' as const,
    thumbnail: new Blob(['image'], { type: 'image/webp' }),
    thumbnailMimeType: 'image/webp',
    timestamp,
  });

  // The dragstart callback from the thumbnail is the only thing that arms a drop;
  // a bare drop with no active drag (e.g. a foreign file) must be a no-op.
  it('ignores a drop when no drag is active', async () => {
    const relabel = vi.fn();
    render(DatasetFrameGrid, {
      props: { frames: [good('g1', 1)], onFrameClick: vi.fn(), onFrameDrag: relabel },
    });
    const goodSection = screen.getByRole('group', { name: 'Good Frames' });
    await fireDrag(goodSection, 'drop', makeDataTransfer());
    expect(relabel).not.toHaveBeenCalled();
  });

  it('highlights the hovered section while a drag is active and clears it on dragend', async () => {
    render(DatasetFrameGrid, {
      props: { frames: [good('g1', 1)], onFrameClick: vi.fn(), onFrameDrag: vi.fn() },
    });
    const source = screen.getByTestId('frame-g1');
    const badSection = screen.getByRole('group', { name: 'Bad Frames' });
    const header = badSection.querySelector('.section-header') as HTMLElement;
    const dataTransfer = makeDataTransfer();

    await fireDrag(source, 'dragStart', dataTransfer);
    await fireDrag(badSection, 'dragOver', dataTransfer);
    expect(header).toHaveClass('over');

    await fireEvent.dragEnd(source);
    expect(header).not.toHaveClass('over');
  });

  // dragend fires after a cancelled drag (dropped outside any section); it must wipe
  // the active id so the next stray drop cannot relabel a stale frame.
  it('disarms the drop after a cancelled drag', async () => {
    const relabel = vi.fn();
    render(DatasetFrameGrid, {
      props: { frames: [good('g1', 1)], onFrameClick: vi.fn(), onFrameDrag: relabel },
    });
    const source = screen.getByTestId('frame-g1');
    const badSection = screen.getByRole('group', { name: 'Bad Frames' });
    const dataTransfer = makeDataTransfer();

    await fireDrag(source, 'dragStart', dataTransfer);
    await fireEvent.dragEnd(source);
    await fireDrag(badSection, 'drop', dataTransfer);

    expect(relabel).not.toHaveBeenCalled();
  });
});

describe('DatasetFrameGrid relabel affordances', () => {
  // The per-thumbnail label dropdown was removed; relabeling is drag + context menu only.
  it('renders no per-thumbnail label dropdown', () => {
    render(DatasetFrameGrid, {
      props: { frames: [frame], onFrameClick: vi.fn(), onDeleteFrame: vi.fn(), onFrameDrag: vi.fn() },
    });
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument();
    expect(document.querySelector('select')).toBeNull();
  });

  it('relabels a frame through the right-click context menu', async () => {
    const relabel = vi.fn();
    render(DatasetFrameGrid, {
      props: { frames: [frame], onFrameClick: vi.fn(), onDeleteFrame: vi.fn(), onFrameDrag: relabel },
    });
    const preview = screen.getByRole('button', { name: 'Preview frame frame-a labeled good' });
    await fireEvent.contextMenu(preview, { clientX: 10, clientY: 10 });
    await fireEvent.click(await screen.findByRole('menuitem', { name: 'Move to Bad' }));
    expect(relabel).toHaveBeenCalledOnce();
    expect(relabel).toHaveBeenCalledWith('frame-a', 'bad');
  });
});
