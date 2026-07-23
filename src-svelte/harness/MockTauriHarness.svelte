<script lang="ts">
  import type { DatasetPage, InferenceUiResult, TrainingEvent_Deserialize } from '@generated/bindings';
  import { Channel } from '@tauri-apps/api/core';
  import {
    createTrainingChannel,
    nativeClient,
  } from '../lib/native/client';
  import { getHarnessMetrics } from './mockTauri';
  import { useCameraSettings } from '../hooks/useCameraSettings';

  let readiness = $state('not initialized');
  let captureStatus = $state('idle');
  let dataset = $state<DatasetPage | null>(null);
  let trainingStatus = $state('idle');
  let trainingEvents = $state<string[]>([]);
  let settingsStatus = $state('not loaded');
  let error = $state('');

  // Exercises the real settings pipeline (combine/split, native round-trip) so the
  // tray startup toggles can be asserted end-to-end against the mocked backend.
  const startupSettings = useCameraSettings();

  function message(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
  }

  function setMinimizeToTray(event: Event): void {
    startupSettings.updateSettings({ minimizeToTrayOnClose: (event.currentTarget as HTMLInputElement).checked });
  }

  function setStartHidden(event: Event): void {
    startupSettings.updateSettings({ startHiddenOnLogin: (event.currentTarget as HTMLInputElement).checked });
  }

  async function reloadStartupSettings(): Promise<void> {
    await startupSettings.reload();
  }

  async function initialize(): Promise<void> {
    error = '';
    try {
      const before = await nativeClient.appStatus();
      if (!before.inferenceReady) {
        await nativeClient.initializeInference();
      }
      const after = await nativeClient.appStatus();
      readiness = after.inferenceReady ? 'ready' : 'not ready';
    } catch (cause) {
      error = message(cause);
    }
  }

  async function capture(): Promise<void> {
    error = '';
    try {
      const received: InferenceUiResult[] = [];
      const channel = new Channel<InferenceUiResult>();
      channel.onmessage = (result) => { received.push(result); };
      await nativeClient.startCamera(channel);
      const captured = received[0];
      if (!captured) throw new Error('No inference result was pushed by the camera.');
      await nativeClient.saveCapture(new Uint8Array([8, 9, 10]), {
        requestId: captured.requestId,
        token: captured.token,
        frameId: 'captured-frame',
        timestamp: 2,
        label: 'good',
        mimeType: 'image/webp',
      });
      await nativeClient.stopCamera();
      const metrics = getHarnessMetrics();
      captureStatus = `saved ${metrics.captureBytes} bytes`;
      await refreshDataset();
    } catch (cause) {
      error = message(cause);
    }
  }

  async function refreshDataset(): Promise<void> {
    dataset = await nativeClient.getDatasetPage(0, 100);
  }

  async function relabel(): Promise<void> {
    await nativeClient.updateFrameLabel('frame-1', 'bad');
    await refreshDataset();
  }

  async function remove(): Promise<void> {
    await nativeClient.deleteFrame('frame-1');
    await refreshDataset();
  }

  async function undo(): Promise<void> {
    await nativeClient.undoLastDatasetChange();
    await refreshDataset();
  }

  function handleTrainingEvent(event: TrainingEvent_Deserialize): void {
    trainingStatus = event.type;
    trainingEvents = [...trainingEvents, event.type];
  }

  async function train(doCv: boolean | null): Promise<void> {
    error = '';
    trainingStatus = 'starting';
    trainingEvents = [];
    try {
      await nativeClient.trainModels(doCv, createTrainingChannel(handleTrainingEvent));
    } catch (cause) {
      error = message(cause);
    }
  }

  async function cancel(): Promise<void> {
    await nativeClient.cancelTraining();
  }

  async function saveAndReloadSettings(): Promise<void> {
    const camera = await nativeClient.getCameraSettings();
    const ui = await nativeClient.getUiSettings();
    await nativeClient.saveCameraSettings({ ...camera, cameraWidth: 1280 });
    await nativeClient.saveUiSettings({ ...ui, alertVolume: 0.75 });
    const [savedCamera, savedUi] = await Promise.all([
      nativeClient.getCameraSettings(),
      nativeClient.getUiSettings(),
    ]);
    settingsStatus = `${savedCamera.cameraWidth}/${savedUi.alertVolume}`;
  }

  async function resetSettings(): Promise<void> {
    const [camera, ui] = await Promise.all([
      nativeClient.resetCameraSettings(),
      nativeClient.resetUiSettings(),
    ]);
    settingsStatus = `${camera.cameraWidth}/${ui.alertVolume}`;
  }
</script>

<main>
  <h1>Mocked Tauri plumbing harness</h1>

  <section aria-labelledby="readiness-heading">
    <h2 id="readiness-heading">Readiness</h2>
    <button type="button" onclick={initialize}>Initialize</button>
    <output data-testid="readiness">{readiness}</output>
  </section>

  <section aria-labelledby="capture-heading">
    <h2 id="capture-heading">Capture</h2>
    <button type="button" onclick={capture}>Capture frame</button>
    <output data-testid="capture-status">{captureStatus}</output>
  </section>

  <section aria-labelledby="dataset-heading">
    <h2 id="dataset-heading">Dataset</h2>
    <div class="actions">
      <button type="button" onclick={refreshDataset}>Refresh dataset</button>
      <button type="button" onclick={relabel}>Relabel frame</button>
      <button type="button" onclick={remove}>Delete frame</button>
      <button type="button" onclick={undo}>Undo dataset change</button>
    </div>
    <output data-testid="dataset">
      {dataset ? dataset.frames.map((frame) => `${frame.id}:${frame.label}`).join(',') : 'not loaded'}
    </output>
  </section>

  <section aria-labelledby="settings-heading">
    <h2 id="settings-heading">Settings</h2>
    <div class="actions">
      <button type="button" onclick={saveAndReloadSettings}>Save settings</button>
      <button type="button" onclick={resetSettings}>Reset settings</button>
    </div>
    <output data-testid="settings-status">{settingsStatus}</output>
  </section>

  <section aria-labelledby="startup-heading">
    <h2 id="startup-heading">Startup</h2>
    <label>
      <input
        type="checkbox"
        checked={startupSettings.settings.minimizeToTrayOnClose}
        onchange={setMinimizeToTray}
      />
      Minimize to tray on close
    </label>
    <label>
      <input
        type="checkbox"
        checked={startupSettings.settings.startHiddenOnLogin}
        onchange={setStartHidden}
      />
      Start hidden at login
    </label>
    <button type="button" onclick={reloadStartupSettings}>Reload startup settings</button>
    <output data-testid="startup-status">
      {`${startupSettings.settings.minimizeToTrayOnClose}/${startupSettings.settings.startHiddenOnLogin}`}
    </output>
  </section>

  <section aria-labelledby="training-heading">
    <h2 id="training-heading">Training</h2>
    <div class="actions">
      <button type="button" onclick={() => train(null)}>Train success</button>
      <button type="button" onclick={() => void train(true)}>Train cancellable</button>
      <button type="button" onclick={cancel}>Cancel training</button>
      <button type="button" onclick={() => train(false)}>Train failure</button>
    </div>
    <output data-testid="training-status">{trainingStatus}</output>
    <output data-testid="training-events">{trainingEvents.join(',')}</output>
  </section>

  {#if error}
    <p role="alert">{error}</p>
  {/if}
</main>

<style>
  main {
    display: grid;
    width: min(52rem, calc(100% - 2rem));
    margin: 2rem auto;
    gap: 1rem;
  }

  section {
    display: grid;
    gap: 0.75rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1rem;
    background: var(--color-surface);
  }

  h1,
  h2 {
    margin: 0;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  button {
    border: 1px solid var(--color-primary);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.75rem;
    background: var(--color-surface-raised);
    cursor: pointer;
  }

  output {
    min-height: 1.5rem;
    color: var(--color-text-muted);
  }
</style>
