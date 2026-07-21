<script lang="ts">
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
    gaussianBlurKernel: number;
    smoothingFrames: number;
  };

  export interface SettingsTabProps {
    settings: RuntimeSettings;
    onUpdateSettings: (updates: Partial<RuntimeSettings>) => void;
    onResetSettings: () => void;
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
    isModelLoaded,
    processedView = false,
    onProcessedViewChange,
    fps,
    modelInfo,
  }: SettingsTabProps = $props();

  function handlePrivacyChange(event: Event): void {
    onUpdateSettings({ privacyMode: (event.currentTarget as HTMLInputElement).checked });
  }

  function handleProcessedViewChange(event: Event): void {
    onProcessedViewChange?.((event.currentTarget as HTMLInputElement).checked);
  }
</script>

<div class="settings-stack">
  <section class="settings-paper">
    <div class="section-stack medium-gap">
      <h2>Camera Settings</h2>

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
      <h3 class="image-heading">Image Preprocessing</h3>

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
        label="Gaussian Blur"
        value={settings.gaussianBlurKernel}
        minimumValue={0}
        maximumValue={15}
        fixedValues={[0, 3, 5, 7, 9, 11, 13, 15]}
        formatValue={(value) => (value === 0 ? 'Off' : `${value}`)}
        onValueChange={(value) => onUpdateSettings({ gaussianBlurKernel: value })}
        helpText="Noise reduction (kernel size). 0 = off, larger = more smoothing. Kernel must be odd."
        showTooltip
        showMinMax
      />

      <Slider
        label="Temporal Smoothing"
        value={settings.smoothingFrames}
        minimumValue={1}
        maximumValue={10}
        step={1}
        formatValue={(value) => (value === 1 ? 'Off' : `${value} frames`)}
        onValueChange={(value) => onUpdateSettings({ smoothingFrames: value })}
        helpText="Number of frames to average. 1 = off, higher = smoother but more motion blur."
        showTooltip
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
    </div>
  </section>

  <section class="settings-paper">
    <div class="section-stack medium-gap">
      <h2>Privacy Settings</h2>
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
  </section>

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
        helpText="Alert triggers after bad posture is detected for this duration (reduces false positives)."
        showTooltip
        showMinMax
      />
    </div>
  </section>

  <section class="settings-paper">
    <div class="section-stack large-gap">
      <h2>Developer Settings</h2>

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
        <button type="button" class="reset-button" onclick={onResetSettings}>Reset All Data</button>
        <div class="help-text">Deletes settings, trained models, and collected dataset frames. This action cannot be undone.</div>
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
  h3 {
    margin: 0;
    color: inherit;
  }

  h2 {
    font-size: var(--mantine-font-size-md, 1rem);
    line-height: 1.55;
    font-weight: 700;
  }

  h3,
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
