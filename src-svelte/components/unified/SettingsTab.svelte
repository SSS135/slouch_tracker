<script lang="ts">
  import { nativeClient } from '@/lib/native/client';
  import Slider from '../ui/Slider.svelte';
  import LoggerSettings from './LoggerSettings.svelte';

  type RuntimeSettings = {
    cameraWidth: number;
    cameraHeight: number;
    captureIntervalSeconds: number;
    alertVolume: number;
    autoCaptureEnabled: boolean;
    autoCaptureIntervalSeconds: number;
    alertDelaySeconds: number;
    privacyMode: boolean;
    claheStrength: number;
    smoothingFrames: number;
    tileMotionThreshold: number;
    claheTemporalAlpha: number;
    preprocessingDebugView: boolean;
    showDetectionOverlay: boolean;
    minimizeToTrayOnClose: boolean;
    startHiddenOnLogin: boolean;
  };

  export interface SettingsTabProps {
    settings: RuntimeSettings;
    onUpdateSettings: (updates: Partial<RuntimeSettings>) => void;
    onResetSettings: () => void;
    onRunSetupAgain?: () => void;
    isModelLoaded: boolean;
    // Ephemeral (not persisted): preview the detector-input feed in the viewport.
    processedView?: boolean;
    onProcessedViewChange?: (value: boolean) => void;
    fps?: number;
    modelInfo?: {
      featureType: string;
      accuracy?: number;
      lastTrained?: number;
      // Presence model feature types, or null when no presence model is loaded
      // (posture-only generation) and runtime falls back to detector confidence.
      presenceFeatureType?: string | null;
    } | null;
  }

  let {
    settings,
    onUpdateSettings,
    onResetSettings,
    onRunSetupAgain,
    isModelLoaded,
    processedView = false,
    onProcessedViewChange,
    fps,
    modelInfo,
  }: SettingsTabProps = $props();

  // Camera + privacy + developer tooling now live behind a disclosure that stays
  // collapsed until opened; defaults are tuned, so a normal user never opens it.
  // Session-only UI state (deliberately not persisted): a fresh mount starts closed.
  let developerExpanded = $state(false);

  // CLAHE temporal smoothing only affects the contrast-curve EMA, which is a no-op
  // when CLAHE itself is off (Strength 0) and produces no visible change on a
  // static, evenly-lit scene — it only damps flicker when the lighting shifts.
  const claheSmoothingHelp = $derived(
    settings.claheStrength === 0
      ? 'CLAHE is off (Strength 0), so this has no effect. Raise CLAHE Strength to use it.'
      : 'Damps frame-to-frame flicker in the contrast curve when the lighting shifts — no visible change on a static, evenly-lit scene. Lower = steadier contrast (less flicker); 1.0 = off (per-frame CLAHE).',
  );

  function handlePrivacyChange(event: Event): void {
    onUpdateSettings({ privacyMode: (event.currentTarget as HTMLInputElement).checked });
  }

  function handleProcessedViewChange(event: Event): void {
    onProcessedViewChange?.((event.currentTarget as HTMLInputElement).checked);
  }

  function handleDetectionOverlayChange(event: Event): void {
    onUpdateSettings({ showDetectionOverlay: (event.currentTarget as HTMLInputElement).checked });
  }

  function handleDebugViewChange(event: Event): void {
    onUpdateSettings({ preprocessingDebugView: (event.currentTarget as HTMLInputElement).checked });
  }

  function handleMinimizeToTrayChange(event: Event): void {
    onUpdateSettings({ minimizeToTrayOnClose: (event.currentTarget as HTMLInputElement).checked });
  }

  function handleStartHiddenChange(event: Event): void {
    onUpdateSettings({ startHiddenOnLogin: (event.currentTarget as HTMLInputElement).checked });
  }

  // Autostart lives in the Windows registry (HKCU Run + Explorer\StartupApproved),
  // NOT in SQLite settings: Task Manager can flip it behind our back, so a mirrored
  // copy would desync. We always read the live state. This component is remounted
  // whenever the Runtime Settings tab is (re)activated, so this on-mount $effect
  // doubles as the on-activation refresh.
  // bind:checked drives the DOM box; because a failed toggle leaves the value
  // unchanged (false→false), a one-way `checked={}` binding would never snap the
  // box back — binding a $state boolean makes the reset a real change.
  let autostartOn = $state(false);
  let autostartBusy = $state(false);
  let autostartError = $state<string | null>(null);
  let autostartGeneration = 0;

  function toMessage(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
  }

  async function refreshAutostart(): Promise<void> {
    const token = ++autostartGeneration;
    try {
      const enabled = await nativeClient.getAutostartEnabled();
      if (token === autostartGeneration) {
        autostartOn = enabled;
        autostartError = null;
      }
    } catch (cause) {
      if (token === autostartGeneration) {
        autostartOn = false;
        autostartError = toMessage(cause);
      }
    }
  }

  async function handleAutostartChange(event: Event): Promise<void> {
    const next = (event.currentTarget as HTMLInputElement).checked;
    autostartBusy = true;
    autostartError = null;
    let setError: string | null = null;
    try {
      await nativeClient.setAutostartEnabled(next);
    } catch (cause) {
      setError = toMessage(cause);
    }
    // Re-read the registry (source of truth) so the box reflects reality even if
    // enable/disable partially failed or Task Manager disagrees. A set failure
    // message takes precedence over the re-read's cleared-error state.
    await refreshAutostart();
    if (setError) {
      autostartError = setError;
    }
    autostartBusy = false;
  }

  $effect(() => {
    void refreshAutostart();
  });
</script>

<div class="settings-stack">
  <section class="settings-paper">
    <div class="section-stack large-gap">
      <h2>Alert Settings</h2>

      <Slider
        label="Audio Volume"
        value={settings.alertVolume * 100}
        minimumValue={0}
        maximumValue={100}
        step={1}
        formatValue={(value) => `${Math.round(value)}%`}
        onValueChange={(value) => onUpdateSettings({ alertVolume: value / 100 })}
        helpText="Volume for posture alert sound (0 = off, 30 = recommended)."
        showTooltip
      />

      <Slider
        label="Alert Delay"
        value={settings.alertDelaySeconds}
        minimumValue={1}
        maximumValue={15}
        step={1}
        fixedValues={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
        formatValue={(value) => `${value}s`}
        onValueChange={(value) => onUpdateSettings({ alertDelaySeconds: Math.round(value) })}
        helpText="Seconds of continued bad posture required before each alert (reduces false positives)."
        showTooltip
        showMinMax
      />
    </div>
  </section>

  <section class="settings-paper">
    <div class="section-stack medium-gap">
      <h2>Startup</h2>
      <label class="checkbox-row" class:checkbox-disabled={autostartBusy}>
        <input
          type="checkbox"
          bind:checked={autostartOn}
          disabled={autostartBusy}
          onchange={handleAutostartChange}
        />
        <span>
          <span class="checkbox-label">Start on login</span>
          <span class="checkbox-description">
            Launch Slouch Tracker automatically when you log into Windows. Also manageable in Task Manager → Startup apps.
          </span>
        </span>
      </label>
      {#if autostartError}
        <div class="help-text small-text">Couldn't read the startup setting: {autostartError}</div>
      {/if}

      <label class="checkbox-row">
        <input
          type="checkbox"
          checked={settings.minimizeToTrayOnClose}
          onchange={handleMinimizeToTrayChange}
        />
        <span>
          <span class="checkbox-label">Minimize to tray on close</span>
          <span class="checkbox-description">
            Closing the window keeps tracking running in the system tray.
          </span>
        </span>
      </label>

      <label class="checkbox-row">
        <input
          type="checkbox"
          checked={settings.startHiddenOnLogin}
          onchange={handleStartHiddenChange}
        />
        <span>
          <span class="checkbox-label">Start hidden at login</span>
          <span class="checkbox-description">
            When started at login, open in the tray instead of showing the window.
          </span>
        </span>
      </label>
    </div>
  </section>

  <section class="settings-paper">
    <h2 class="collapsible-heading">
      <button
        type="button"
        class="collapsible-toggle"
        aria-expanded={developerExpanded}
        aria-controls="developer-settings-content"
        onclick={() => { developerExpanded = !developerExpanded; }}
      >
        <span>Developer settings</span>
        <span class="chevron" aria-hidden="true">{developerExpanded ? '⌄' : '›'}</span>
      </button>
    </h2>
    <div id="developer-settings-content" class="developer-content" hidden={!developerExpanded}>
      <div class="settings-paper nested-paper">
        <div class="section-stack medium-gap">
          <h3>Camera Settings</h3>

          {#if fps !== undefined}
            <div class="fps-stack">
              <div class="small-label">Detection FPS</div>
              <div class="fps-value">{fps.toFixed(1)}</div>
              <div class="help-text small-text">
                Native detection cadence (typically 1-2 fps; the preview renders smoothly on top)
              </div>
            </div>
          {/if}

          <hr />

          <Slider
            label="Capture Interval"
            value={settings.captureIntervalSeconds}
            minimumValue={0.1}
            maximumValue={10}
            step={0.1}
            fixedValues={[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
            formatValue={(value) => `${value.toFixed(1)}s`}
            onValueChange={(value) => onUpdateSettings({ captureIntervalSeconds: value })}
            helpText="Used while collecting training data."
            showTooltip
            showMinMax
          />

          <hr />
          <h4 class="image-heading">Image Preprocessing</h4>

          <Slider
            label="CLAHE Strength"
            value={settings.claheStrength}
            minimumValue={0}
            maximumValue={10}
            step={0.1}
            formatValue={(value) => (value === 0 ? 'Off' : value.toFixed(1))}
            onValueChange={(value) => onUpdateSettings({ claheStrength: value })}
            helpText="Contrast enhancement. 0 = off, higher = stronger enhancement."
            showTooltip
          />

          <Slider
            label="Temporal Smoothing"
            value={settings.smoothingFrames}
            minimumValue={1}
            maximumValue={10}
            step={1}
            formatValue={(value) => (value === 1 ? 'Off' : `${value} frames`)}
            onValueChange={(value) => onUpdateSettings({ smoothingFrames: value })}
            helpText="Max frames accumulated while a region stays static. Motion-gated per tile, so higher values add no motion blur or ghosting. 1 = off."
            showTooltip
          />

          <Slider
            label="Motion threshold"
            value={settings.tileMotionThreshold}
            minimumValue={0.5}
            maximumValue={20}
            step={0.5}
            formatValue={(value) => value.toFixed(1)}
            onValueChange={(value) => onUpdateSettings({ tileMotionThreshold: value })}
            helpText="How much a tile must change to reset its accumulation to the live frame. Lower = stricter (any change resets); higher = more averaging under small movements."
            showTooltip
            showMinMax
          />

          <Slider
            label="CLAHE smoothing"
            value={settings.claheTemporalAlpha}
            minimumValue={0.05}
            maximumValue={1}
            step={0.05}
            disabled={settings.claheStrength === 0}
            formatValue={(value) => (value >= 1 ? 'Off' : value.toFixed(2))}
            onValueChange={(value) => onUpdateSettings({ claheTemporalAlpha: value })}
            helpText={claheSmoothingHelp}
            showTooltip
            showMinMax
          />

          <label class="checkbox-row" class:checkbox-disabled={settings.privacyMode}>
            <input
              type="checkbox"
              checked={processedView && !settings.privacyMode}
              disabled={settings.privacyMode}
              onchange={handleProcessedViewChange}
            />
            <span>
              <span class="checkbox-label">Show processed view</span>
              <span class="checkbox-description">
                {settings.privacyMode
                  ? 'Unavailable in privacy mode - the feed stays obscured.'
                  : "Shows the detector's preprocessing live (CLAHE, blur, smoothing). Note: temporal smoothing appears time-compressed at preview rate."}
              </span>
            </span>
          </label>

          <label class="checkbox-row">
            <input
              type="checkbox"
              checked={settings.preprocessingDebugView}
              onchange={handleDebugViewChange}
            />
            <span>
              <span class="checkbox-label">Preprocessing debug view</span>
              <span class="checkbox-description">
                Tint tiles by accumulation depth (green = averaging, red = live) in the processed view.
              </span>
            </span>
          </label>

          <label class="checkbox-row">
            <input
              type="checkbox"
              checked={settings.showDetectionOverlay}
              onchange={handleDetectionOverlayChange}
            />
            <span>
              <span class="checkbox-label">Detection Overlay</span>
              <span class="checkbox-description">
                Draw skeleton and detection box with confidence over the video (diagnostic)
              </span>
            </span>
          </label>
        </div>
      </div>

      <div class="settings-paper nested-paper">
        <div class="section-stack medium-gap">
          <h3>Privacy Settings</h3>
          <label class="checkbox-row">
            <input type="checkbox" checked={settings.privacyMode} onchange={handlePrivacyChange} />
            <span>
              <span class="checkbox-label">Privacy Mode</span>
              <span class="checkbox-description">Hide video feed and save only skeleton visualizations</span>
            </span>
          </label>
          <div class="help-text small-text">
            When enabled, no real camera images are saved anywhere - only human-like skeleton representations. ML models still work normally.
          </div>
        </div>
      </div>

      <div class="section-stack large-gap">
        <Slider
          label="Auto-Capture Interval"
          value={settings.autoCaptureIntervalSeconds}
          minimumValue={1}
          maximumValue={15}
          step={1}
          formatValue={(value) => `${Math.round(value)}s`}
          onValueChange={(value) => onUpdateSettings({ autoCaptureIntervalSeconds: Math.round(value) })}
          helpText="Time between auto-captures when no model is trained (used for posture-change backup timer when model exists)."
          showTooltip
        />

        <div class="logger-stack">
          <h3 class="logger-heading">Console Logging</h3>
          <LoggerSettings />
        </div>

        {#if isModelLoaded && modelInfo}
          <div class="labeled-divider"><span>Model Status</span></div>
          <div class="model-status-stack">
            <div class="status-row">
              <span class="status-label">Posture</span>
              <span class="status-value">{modelInfo.featureType}</span>
            </div>

            <div class="status-row">
              <span class="status-label">Presence</span>
              <span class="status-value">{modelInfo.presenceFeatureType ?? 'Using detector fallback'}</span>
            </div>

            {#if typeof modelInfo.accuracy === 'number'}
              <div class="status-row">
                <span class="status-label">Accuracy</span>
                <span class="status-value">{(modelInfo.accuracy * 100).toFixed(1)}%</span>
              </div>
            {/if}

            {#if modelInfo.lastTrained}
              <div class="status-row">
                <span class="status-label">Last Trained</span>
                <span class="status-value">{new Date(modelInfo.lastTrained).toLocaleString()}</span>
              </div>
            {/if}
          </div>
        {/if}

        <hr />

        <div class="reset-stack">
          <button type="button" class="setup-again-button" onclick={() => onRunSetupAgain?.()}>Run Setup Again</button>
          <div class="help-text">Reopens the first-run setup wizard. Your dataset and settings are kept.</div>
        </div>

        <div class="reset-stack">
          <button type="button" class="reset-button" onclick={onResetSettings}>Reset All Data</button>
          <div class="help-text">Deletes settings, trained models, and collected dataset frames. This action cannot be undone.</div>
        </div>
      </div>
    </div>
  </section>
</div>

<style>
  .settings-stack {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-lg, 20px);
  }

  .settings-paper {
    box-sizing: border-box;
    padding: var(--mantine-spacing-lg, 20px);
    border: 1px solid var(--mantine-color-default-border, #495057);
    border-radius: var(--mantine-radius-lg, 16px);
    background: rgb(0 0 0 / 30%);
  }

  .nested-paper {
    padding: var(--mantine-spacing-md, 16px);
    background: rgb(0 0 0 / 20%);
  }

  .section-stack {
    display: flex;
    flex-direction: column;
  }

  .medium-gap {
    gap: var(--mantine-spacing-md, 16px);
  }

  .large-gap {
    gap: var(--mantine-spacing-lg, 20px);
  }

  h2,
  h3,
  h4 {
    margin: 0;
    color: inherit;
  }

  h2 {
    font-size: var(--mantine-font-size-md, 1rem);
    line-height: 1.55;
    font-weight: 700;
  }

  h3,
  h4,
  .small-label,
  .checkbox-label {
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: 1.45;
    font-weight: 600;
  }

  .logger-heading {
    font-size: var(--mantine-font-size-md, 1rem);
    line-height: 1.55;
  }

  .collapsible-heading {
    margin: 0;
  }

  /* Disclosure header styled like the section h2, full-width with a trailing chevron. */
  .collapsible-toggle {
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0;
    border: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: var(--mantine-font-size-md, 1rem);
    line-height: 1.55;
    font-weight: 700;
    text-align: left;
    cursor: pointer;
  }

  .collapsible-toggle .chevron {
    color: var(--mantine-color-dimmed, #909296);
    font-size: 1.25rem;
    line-height: 1;
  }

  .developer-content {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-lg, 20px);
    margin-top: var(--mantine-spacing-md, 16px);
  }

  /* Setting display on the content defeats the `hidden` attribute's default
     display:none, so restore it when collapsed. */
  .developer-content[hidden] {
    display: none;
  }

  hr {
    width: 100%;
    margin: 0;
    border: 0;
    border-top: 1px solid var(--mantine-color-default-border, #495057);
  }

  .fps-stack {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .fps-value {
    color: var(--mantine-color-cyan-5, #22b8cf);
    font-size: var(--mantine-font-size-lg, 1.125rem);
    line-height: 1.6;
    font-weight: 700;
  }

  .help-text,
  .status-label {
    color: var(--mantine-color-dimmed, #909296);
  }

  .small-text {
    font-size: var(--mantine-font-size-xs, 0.75rem);
    line-height: 1.4;
  }

  .checkbox-row {
    display: flex;
    align-items: flex-start;
    gap: 0.625rem;
    cursor: pointer;
  }

  .checkbox-row input {
    width: 1rem;
    height: 1rem;
    margin-top: 0.25rem;
    accent-color: var(--mantine-color-blue-6, #228be6);
  }

  .checkbox-row.checkbox-disabled {
    cursor: default;
    opacity: 0.55;
  }

  .checkbox-row > span {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
  }

  .checkbox-description {
    color: var(--mantine-color-dimmed, #909296);
    font-size: var(--mantine-font-size-sm, 0.875rem);
  }

  .logger-stack {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-sm, 12px);
  }

  .labeled-divider {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--mantine-color-dimmed, #909296);
    font-size: var(--mantine-font-size-xs, 0.75rem);
    line-height: 1.4;
  }

  .labeled-divider::before,
  .labeled-divider::after {
    flex: 1;
    border-top: 1px solid var(--mantine-color-default-border, #495057);
    content: '';
  }

  .model-status-stack {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .status-label,
  .status-value {
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: 1.45;
  }

  .status-value {
    font-weight: 600;
    text-align: right;
  }

  .reset-stack {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-xs, 10px);
  }

  .setup-again-button {
    align-self: flex-start;
    padding: 0.5rem 0.875rem;
    border: 1px solid var(--mantine-color-default-border, #495057);
    border-radius: var(--mantine-radius-sm, 4px);
    color: inherit;
    background: var(--mantine-color-dark-5, #373a40);
    font: inherit;
    cursor: pointer;
  }

  .setup-again-button:hover,
  .setup-again-button:focus-visible {
    filter: brightness(1.15);
  }

  .reset-button {
    align-self: flex-start;
    padding: 0.5rem 0.875rem;
    border: 1px solid rgb(250 82 82 / 45%);
    border-radius: var(--mantine-radius-sm, 4px);
    color: var(--mantine-color-red-4, #ff8787);
    background: rgb(250 82 82 / 15%);
    font: inherit;
    cursor: pointer;
  }

  .reset-button:hover,
  .reset-button:focus-visible {
    background: rgb(250 82 82 / 25%);
  }
</style>
