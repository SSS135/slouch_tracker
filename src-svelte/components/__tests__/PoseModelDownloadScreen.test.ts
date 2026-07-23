import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import PoseModelDownloadScreen from '../PoseModelDownloadScreen.svelte';
import type { PoseModelPhase } from '@/hooks/usePoseModelDownload.svelte';

afterEach(() => {
  cleanup();
});

const renderScreen = (state: PoseModelPhase, handlers: { onCancel?: () => void; onRetry?: () => void } = {}) =>
  render(PoseModelDownloadScreen, {
    props: {
      state,
      onCancel: handlers.onCancel ?? vi.fn(),
      onRetry: handlers.onRetry ?? vi.fn(),
    },
  });

describe('PoseModelDownloadScreen', () => {
  it('frames the download as a one-time setup step and always credits NLF', () => {
    renderScreen({ kind: 'downloading', received: 0, total: 0 });
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/one-time download of the pose-detection model/i)).toBeInTheDocument();
    // Non-error tone: no "failed"/"error" heading in the happy path.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText(/NLF by István Sárándi/)).toBeInTheDocument();
    expect(screen.getByText(/Non-commercial use only/)).toBeInTheDocument();
    expect(screen.getByText(/github.com\/isarandi\/nlf/)).toBeInTheDocument();
  });

  it('shows a starting indeterminate bar before a total is known', () => {
    renderScreen({ kind: 'downloading', received: 0, total: 0 });
    expect(screen.getByText('Starting download…')).toBeInTheDocument();
    const bar = screen.getByRole('progressbar', { name: 'Model download progress' });
    expect(bar).not.toHaveAttribute('aria-valuenow');
  });

  it('renders byte totals and a percentage once the download reports a total', () => {
    renderScreen({ kind: 'downloading', received: 122 * 1024 * 1024, total: 245 * 1024 * 1024 });
    const bar = screen.getByRole('progressbar', { name: 'Model download progress' });
    expect(bar).toHaveAttribute('aria-valuenow', '49');
    expect(screen.getByText(/122 MB of 245 MB \(49%\)/)).toBeInTheDocument();
  });

  it('exposes a Cancel action while downloading', () => {
    const onCancel = vi.fn();
    renderScreen({ kind: 'downloading', received: 10, total: 100 }, { onCancel });
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it('shows a verifying state with no cancel control', () => {
    renderScreen({ kind: 'verifying' });
    expect(screen.getByText('Verifying the downloaded model…')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
  });

  it('surfaces the failure reason and a retry action', () => {
    const onRetry = vi.fn();
    renderScreen({ kind: 'failed', reason: 'server returned 503', offline: false }, { onRetry });
    expect(screen.getByRole('alert')).toHaveTextContent('server returned 503');
    // A non-offline failure must not push the offline README hint.
    expect(screen.queryByText(/Fully offline installation/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry download' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('points offline failures at the README offline-install instructions', () => {
    renderScreen({ kind: 'failed', reason: 'network connection failed', offline: true });
    expect(screen.getByText(/Fully offline installation/i)).toBeInTheDocument();
    expect(screen.getByText(/README/i)).toBeInTheDocument();
  });

  it('offers a resume action after a cancellation', () => {
    const onRetry = vi.fn();
    renderScreen({ kind: 'cancelled' }, { onRetry });
    expect(screen.getByText('Download paused.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Resume download' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
