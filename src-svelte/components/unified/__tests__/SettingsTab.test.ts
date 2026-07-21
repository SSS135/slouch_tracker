import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import SettingsTab from '../SettingsTab.svelte';

const baseSettings = {
  cameraWidth: 800,
  cameraHeight: 600,
  captureIntervalSeconds: 1,
  alertVolume: 0.5,
  autoCaptureEnabled: false,
  autoCaptureIntervalSeconds: 5,
  alertDelaySeconds: 5,
  privacyMode: false,
  claheStrength: 0,
  gaussianBlurKernel: 0,
  smoothingFrames: 1,
};

function renderTab(
  overrides: Partial<{
    settings: typeof baseSettings;
    processedView: boolean;
    onProcessedViewChange: (value: boolean) => void;
  }> = {},
) {
  return render(SettingsTab, {
    props: {
      settings: overrides.settings ?? baseSettings,
      onUpdateSettings: vi.fn(),
      onResetSettings: vi.fn(),
      isModelLoaded: false,
      processedView: overrides.processedView ?? false,
      onProcessedViewChange: overrides.onProcessedViewChange,
    },
  });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('SettingsTab processed view toggle', () => {
  it('renders the toggle with the detector-view hint', () => {
    renderTab();
    const toggle = screen.getByRole('checkbox', { name: /show processed view/i });
    expect(toggle).toBeEnabled();
    expect(toggle).not.toBeChecked();
    expect(screen.getByText(/shows the detector's preprocessing live/i)).toBeInTheDocument();
  });

  it('reports toggle changes through onProcessedViewChange', async () => {
    const onProcessedViewChange = vi.fn();
    renderTab({ onProcessedViewChange });
    await fireEvent.click(screen.getByRole('checkbox', { name: /show processed view/i }));
    expect(onProcessedViewChange).toHaveBeenCalledWith(true);
  });

  it('reflects an enabled processed view', () => {
    renderTab({ processedView: true });
    expect(screen.getByRole('checkbox', { name: /show processed view/i })).toBeChecked();
  });

  it('is disabled and unchecked in privacy mode with an explanatory hint', () => {
    renderTab({
      settings: { ...baseSettings, privacyMode: true },
      processedView: true,
    });
    const toggle = screen.getByRole('checkbox', { name: /show processed view/i });
    expect(toggle).toBeDisabled();
    expect(toggle).not.toBeChecked();
    expect(screen.getByText(/unavailable in privacy mode/i)).toBeInTheDocument();
  });
});
