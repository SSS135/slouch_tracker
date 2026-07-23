import '@testing-library/jest-dom/vitest';
import { cleanup, createEvent, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

const native = vi.hoisted(() => ({ getThumbnail: vi.fn() }));
vi.mock('../../../lib/native/client', () => ({ nativeClient: native }));

import DatasetFrameThumbnail from '../DatasetFrameThumbnail.svelte';

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('DatasetFrameThumbnail native loading state', () => {
  it('renders the exact native error and retries thumbnail loading', async () => {
    native.getThumbnail.mockRejectedValueOnce(new Error('thumbnail bytes unavailable'))
      .mockResolvedValueOnce(new Uint8Array([1, 2, 3]));
    render(DatasetFrameThumbnail, {
      props: {
        nativeThumbnailId: 'frame-a',
        thumbnailMimeType: 'image/webp',
        label: 'good',
        borderColor: 'green',
        onPress: vi.fn(),
        frameId: 'frame-a',
      },
    });
    expect(await screen.findByRole('alert')).toHaveTextContent('Thumbnail failed: thumbnail bytes unavailable');
    await fireEvent.click(screen.getByRole('button', { name: 'Retry thumbnail for frame frame-a' }));
    await waitFor(() => expect(native.getThumbnail).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  });

  // Drag start must hand the frame id to the grid through a callback and seed the
  // dataTransfer store, without ever mutating the source's pointer-events - setting
  // pointer-events:none (or translating the element) during dragstart aborts the
  // native drag in Chromium, which is exactly what broke real-browser relabeling.
  it('reports the frame id through drag callbacks and seeds dataTransfer without disabling pointer events', async () => {
    const onDragStart = vi.fn();
    const onDragEnd = vi.fn();
    const setData = vi.fn();
    render(DatasetFrameThumbnail, {
      props: {
        thumbnailUrl: 'blob:preview',
        label: 'good',
        borderColor: 'green',
        onPress: vi.fn(),
        frameId: 'frame-a',
        onDragStart,
        onDragEnd,
      },
    });
    const button = screen.getByRole('button', { name: 'Preview frame frame-a labeled good' });
    const dataTransfer = { setData, effectAllowed: '' };
    const startEvent = createEvent.dragStart(button, { bubbles: true });
    Object.defineProperty(startEvent, 'dataTransfer', { value: dataTransfer, configurable: true });
    await fireEvent(button, startEvent);

    expect(onDragStart).toHaveBeenCalledOnce();
    expect(onDragStart).toHaveBeenCalledWith('frame-a');
    expect(setData).toHaveBeenCalledWith('application/x-slouch-frame-id', 'frame-a');
    expect(dataTransfer.effectAllowed).toBe('move');

    const style = button.getAttribute('style') ?? '';
    expect(style).not.toContain('translate3d');
    expect(style).not.toContain('pointer-events: none');
    expect(style).toContain('opacity: 0.5');

    await fireEvent.dragEnd(button);
    expect(onDragEnd).toHaveBeenCalledOnce();
  });

  it('cancels a drag that has no frame id and never reports one', async () => {
    const onDragStart = vi.fn();
    render(DatasetFrameThumbnail, {
      props: { thumbnailUrl: 'blob:preview', label: 'good', borderColor: 'green', onPress: vi.fn(), onDragStart },
    });
    const button = screen.getByRole('button', { name: 'Preview frame labeled good' });
    const startEvent = createEvent.dragStart(button, { bubbles: true, cancelable: true });
    Object.defineProperty(startEvent, 'dataTransfer', { value: { setData: vi.fn(), effectAllowed: '' }, configurable: true });
    await fireEvent(button, startEvent);

    expect(startEvent.defaultPrevented).toBe(true);
    expect(onDragStart).not.toHaveBeenCalled();
  });
});
