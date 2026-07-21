import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
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
});
