import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const nativeMock = vi.hoisted(() => ({
  getAutostartEnabled: vi.fn(),
  setAutostartEnabled: vi.fn(),
}));

vi.mock('@/lib/native/client', () => ({ nativeClient: nativeMock }));

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
  claheStrength: 3.5,
  smoothingFrames: 1,
  tileMotionThreshold: 3,
  claheTemporalAlpha: 0.15,
  preprocessingDebugView: false,
  showDetectionOverlay: false,
  minimizeToTrayOnClose: true,
  startHiddenOnLogin: true,
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

beforeEach(() => {
  nativeMock.getAutostartEnabled.mockResolvedValue(false);
  nativeMock.setAutostartEnabled.mockResolvedValue(undefined);
});

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

describe('SettingsTab preprocessing tuning controls', () => {
  it('renders the motion threshold and CLAHE smoothing sliders with their help text', () => {
    renderTab();
    expect(screen.getByRole('slider', { name: /motion threshold/i })).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: /clahe smoothing/i })).toBeInTheDocument();
    expect(screen.getByText(/lower = stricter/i)).toBeInTheDocument();
    expect(screen.getByText(/steadier contrast/i)).toBeInTheDocument();
  });

  it('reports motion threshold changes through onUpdateSettings', async () => {
    const onUpdateSettings = vi.fn();
    render(SettingsTab, {
      props: { settings: baseSettings, onUpdateSettings, onResetSettings: vi.fn(), isModelLoaded: false },
    });
    await fireEvent.input(screen.getByRole('slider', { name: /motion threshold/i }), {
      target: { value: '10' },
    });
    expect(onUpdateSettings).toHaveBeenCalledWith({ tileMotionThreshold: 10 });
  });

  it('reports CLAHE smoothing changes through onUpdateSettings', async () => {
    const onUpdateSettings = vi.fn();
    render(SettingsTab, {
      props: { settings: baseSettings, onUpdateSettings, onResetSettings: vi.fn(), isModelLoaded: false },
    });
    await fireEvent.input(screen.getByRole('slider', { name: /clahe smoothing/i }), {
      target: { value: '0.5' },
    });
    expect(onUpdateSettings).toHaveBeenCalledWith({ claheTemporalAlpha: 0.5 });
  });

  it('reflects the tuning slider values from settings', () => {
    renderTab({ settings: { ...baseSettings, tileMotionThreshold: 7.5, claheTemporalAlpha: 0.3 } });
    expect(screen.getByText('7.5')).toBeInTheDocument();
    expect(screen.getByText('0.30')).toBeInTheDocument();
  });
});

describe('SettingsTab CLAHE smoothing gating', () => {
  it('disables the CLAHE smoothing slider and explains why when CLAHE is off', () => {
    renderTab({ settings: { ...baseSettings, claheStrength: 0 } });
    expect(screen.getByRole('slider', { name: /clahe smoothing/i })).toBeDisabled();
    expect(screen.getByText(/clahe is off \(strength 0\), so this has no effect/i)).toBeInTheDocument();
    // The motion-threshold slider is independent of CLAHE and stays enabled.
    expect(screen.getByRole('slider', { name: /motion threshold/i })).toBeEnabled();
  });

  it('enables the CLAHE smoothing slider and explains the flicker-only effect when CLAHE is on', () => {
    renderTab({ settings: { ...baseSettings, claheStrength: 3.5 } });
    expect(screen.getByRole('slider', { name: /clahe smoothing/i })).toBeEnabled();
    expect(screen.getByText(/no visible change on a static, evenly-lit scene/i)).toBeInTheDocument();
  });
});

describe('SettingsTab preprocessing debug view toggle', () => {
  it('renders the debug view toggle off by default with its help text', () => {
    renderTab();
    const toggle = screen.getByRole('checkbox', { name: /preprocessing debug view/i });
    expect(toggle).not.toBeChecked();
    expect(screen.getByText(/tint tiles by accumulation depth/i)).toBeInTheDocument();
  });

  it('reports debug view changes through onUpdateSettings', async () => {
    const onUpdateSettings = vi.fn();
    render(SettingsTab, {
      props: { settings: baseSettings, onUpdateSettings, onResetSettings: vi.fn(), isModelLoaded: false },
    });
    await fireEvent.click(screen.getByRole('checkbox', { name: /preprocessing debug view/i }));
    expect(onUpdateSettings).toHaveBeenCalledWith({ preprocessingDebugView: true });
  });

  it('reflects an enabled debug view', () => {
    renderTab({ settings: { ...baseSettings, preprocessingDebugView: true } });
    expect(screen.getByRole('checkbox', { name: /preprocessing debug view/i })).toBeChecked();
  });
});

describe('SettingsTab detection overlay toggle', () => {
  it('renders the detection overlay toggle off by default with its diagnostic hint', () => {
    renderTab();
    const toggle = screen.getByRole('checkbox', { name: /detection overlay/i });
    expect(toggle).not.toBeChecked();
    expect(
      screen.getByText(/draw skeleton and detection box with confidence/i),
    ).toBeInTheDocument();
  });

  it('reports toggle changes through onUpdateSettings', async () => {
    const onUpdateSettings = vi.fn();
    render(SettingsTab, {
      props: {
        settings: baseSettings,
        onUpdateSettings,
        onResetSettings: vi.fn(),
        isModelLoaded: false,
      },
    });
    await fireEvent.click(screen.getByRole('checkbox', { name: /detection overlay/i }));
    expect(onUpdateSettings).toHaveBeenCalledWith({ showDetectionOverlay: true });
  });

  it('reflects an enabled detection overlay', () => {
    renderTab({ settings: { ...baseSettings, showDetectionOverlay: true } });
    expect(screen.getByRole('checkbox', { name: /detection overlay/i })).toBeChecked();
  });
});

describe('SettingsTab start-on-login toggle', () => {
  it('reads the live autostart state on mount and reflects it', async () => {
    nativeMock.getAutostartEnabled.mockResolvedValue(true);
    renderTab();
    const toggle = await screen.findByRole('checkbox', { name: /start on login/i });
    await waitFor(() => expect(toggle).toBeChecked());
    expect(nativeMock.getAutostartEnabled).toHaveBeenCalled();
    expect(
      screen.getByText(/launch slouch tracker automatically when you log into windows/i),
    ).toBeInTheDocument();
  });

  it('enabling calls the setter then re-reads the registry as the source of truth', async () => {
    let enabled = false;
    nativeMock.getAutostartEnabled.mockImplementation(async () => enabled);
    nativeMock.setAutostartEnabled.mockImplementation(async (value: boolean) => {
      enabled = value;
    });
    renderTab();
    const toggle = await screen.findByRole('checkbox', { name: /start on login/i });
    await waitFor(() => expect(toggle).not.toBeChecked());

    await fireEvent.click(toggle);

    expect(nativeMock.setAutostartEnabled).toHaveBeenCalledWith(true);
    await waitFor(() => expect(toggle).toBeChecked());
    // Once on mount, once after the toggle: the displayed state is the re-read
    // registry value, not the optimistic click.
    expect(nativeMock.getAutostartEnabled.mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it('surfaces a set failure and leaves the checkbox at the true registry state', async () => {
    nativeMock.getAutostartEnabled.mockResolvedValue(false);
    nativeMock.setAutostartEnabled.mockRejectedValue(new Error('Access is denied. (os error 5)'));
    renderTab();
    const toggle = await screen.findByRole('checkbox', { name: /start on login/i });
    await waitFor(() => expect(toggle).not.toBeChecked());

    await fireEvent.click(toggle);

    await waitFor(() => expect(screen.getByText(/access is denied/i)).toBeInTheDocument());
    expect(toggle).not.toBeChecked();
  });
});

describe('SettingsTab tray startup toggles', () => {
  it('renders both tray toggles checked by default with their descriptions', () => {
    renderTab();
    expect(screen.getByRole('checkbox', { name: /minimize to tray on close/i })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: /start hidden at login/i })).toBeChecked();
    expect(
      screen.getByText(/closing the window keeps tracking running in the system tray/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/open in the tray instead of showing the window/i),
    ).toBeInTheDocument();
  });

  it('reports a minimize-to-tray toggle through onUpdateSettings', async () => {
    const onUpdateSettings = vi.fn();
    render(SettingsTab, {
      props: { settings: baseSettings, onUpdateSettings, onResetSettings: vi.fn(), isModelLoaded: false },
    });
    await fireEvent.click(screen.getByRole('checkbox', { name: /minimize to tray on close/i }));
    expect(onUpdateSettings).toHaveBeenCalledWith({ minimizeToTrayOnClose: false });
  });

  it('reports a start-hidden toggle through onUpdateSettings', async () => {
    const onUpdateSettings = vi.fn();
    render(SettingsTab, {
      props: { settings: baseSettings, onUpdateSettings, onResetSettings: vi.fn(), isModelLoaded: false },
    });
    await fireEvent.click(screen.getByRole('checkbox', { name: /start hidden at login/i }));
    expect(onUpdateSettings).toHaveBeenCalledWith({ startHiddenOnLogin: false });
  });

  it('reflects disabled tray toggles when the settings are false', () => {
    renderTab({ settings: { ...baseSettings, minimizeToTrayOnClose: false, startHiddenOnLogin: false } });
    expect(screen.getByRole('checkbox', { name: /minimize to tray on close/i })).not.toBeChecked();
    expect(screen.getByRole('checkbox', { name: /start hidden at login/i })).not.toBeChecked();
  });
});
