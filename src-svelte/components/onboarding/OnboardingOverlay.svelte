<script lang="ts">
  import type { CameraDeviceInfo } from '@generated/bindings';
  import { nativeClient } from '@/lib/native/client';
  import { FrameLabel } from '@/services/dataset/types';
  import {
    ONBOARDING_TARGETS,
    type OnboardingState,
    type OnboardingStep,
  } from '@/hooks/useOnboarding.svelte';

  export interface OnboardingOverlayProps {
    onboarding: OnboardingState;
    cameraOk: boolean;
    personFound: boolean;
    captureReady: boolean;
    cameraError: string | null;
    selectedCameraIndex: number;
    onCapture: (label: FrameLabel) => void;
  }

  let {
    onboarding,
    cameraOk,
    personFound,
    captureReady,
    cameraError,
    selectedCameraIndex,
    onCapture,
  }: OnboardingOverlayProps = $props();

  const STEPS: OnboardingStep[] = ['camera', 'good', 'bad', 'away'];
  const HEADINGS: Record<OnboardingStep, string> = {
    camera: 'Select your camera',
    good: 'Capture good posture',
    bad: 'Capture bad posture',
    away: 'Capture away frames',
  };
  const stepIndex = $derived(STEPS.indexOf(onboarding.step));

  let devices = $state<Array<{ index: number; name: string }>>([]);
  let devicesLoaded = $state(false);
  let devicesError = $state<string | null>(null);

  $effect(() => {
    let disposed = false;
    void nativeClient
      .listCameras()
      .then((list: CameraDeviceInfo[]) => {
        if (disposed) return;
        // Native device indexes arrive as strings from nokhwa's enumeration.
        devices = list
          .map((device) => ({ index: Number.parseInt(device.index, 10), name: device.name }))
          .filter((device) => Number.isFinite(device.index));
        devicesLoaded = true;
      })
      .catch((cause: unknown) => {
        if (disposed) return;
        devicesError = cause instanceof Error ? cause.message : String(cause);
        devicesLoaded = true;
      });
    return () => {
      disposed = true;
    };
  });

  const selectedValue = $derived(
    devices.some((device) => device.index === selectedCameraIndex)
      ? selectedCameraIndex
      : devices[0]?.index,
  );

  const captureLabel = $derived.by(() => {
    const step = onboarding.step;
    return step === 'bad' ? FrameLabel.BAD : step === 'away' ? FrameLabel.AWAY : FrameLabel.GOOD;
  });
  const captured = $derived.by(() => {
    const step = onboarding.step;
    if (step === 'good') return onboarding.capturedGood;
    if (step === 'bad') return onboarding.capturedBad;
    return onboarding.capturedAway;
  });
  const target = $derived.by(() => {
    const step = onboarding.step;
    return step === 'camera' ? 0 : ONBOARDING_TARGETS[step];
  });

  function handleCameraChange(event: Event & { currentTarget: HTMLSelectElement }): void {
    const parsed = Number.parseInt(event.currentTarget.value, 10);
    if (Number.isFinite(parsed)) void onboarding.selectCamera(parsed);
  }
</script>

<!-- Transparent center: only the chrome captures pointer events, so the live
     camera layer underneath stays visible and reachable. -->
<div class="onboarding" data-testid="onboarding-overlay" role="dialog" aria-label="First-run setup">
  <div class="top">
    <div class="header-card">
      <div class="dots" aria-hidden="true">
        {#each STEPS as markerStep, index (markerStep)}
          <span class="dot" class:done={index < stepIndex} class:current={index === stepIndex}></span>
        {/each}
      </div>
      <h2 class="heading">{HEADINGS[onboarding.step]}</h2>
    </div>
    <button type="button" class="ghost skip-setup" onclick={() => onboarding.skip()}>Skip setup</button>
  </div>

  <div class="bottom">
    <div class="step-card">
      {#if onboarding.step === 'camera'}
        <p class="copy">
          Pick the webcam that should watch your posture. The live preview behind this panel shows
          what it sees.
        </p>
        {#if devicesError}
          <p class="error-line">Could not list cameras: {devicesError}</p>
        {:else if devicesLoaded && devices.length === 0}
          <p class="muted">No cameras found.</p>
        {:else if devices.length > 0}
          <label class="camera-select">
            <span>Camera</span>
            <select onchange={handleCameraChange}>
              {#each devices as device (device.index)}
                <option value={device.index} selected={device.index === selectedValue}>
                  {device.name}
                </option>
              {/each}
            </select>
          </label>
        {/if}
        <div class="status">
          <span class="status-line" class:ok={cameraOk}>
            {cameraOk ? 'Camera OK' : 'Waiting for camera…'}
          </span>
          <span class="status-line" class:ok={personFound}>
            {personFound ? 'Person detected' : 'No person detected'}
          </span>
        </div>
        {#if cameraError}
          <p class="error-line">{cameraError}</p>
        {/if}
        <button type="button" class="primary" onclick={() => onboarding.next()}>Continue</button>
      {:else}
        {#if onboarding.step === 'good'}
          <p class="copy">Sit upright the way you normally would when working comfortably.</p>
        {:else if onboarding.step === 'bad'}
          <p class="copy">Slouch the way you want to be warned about.</p>
        {:else}
          <p class="copy">
            Lean out of frame so only part of you is visible. Capturing still needs a person
            detection, so stay partially in view.
          </p>
        {/if}
        <p class="progress">{captured} / {target}</p>
        <div class="actions">
          <button
            type="button"
            class="primary"
            disabled={!captureReady}
            onclick={() => onCapture(captureLabel)}
          >
            Capture frame
          </button>
          {#if onboarding.step === 'away'}
            <button type="button" class="ghost" onclick={() => onboarding.skipAwayStep()}>
              Skip this step
            </button>
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .onboarding {
    position: absolute;
    inset: 0;
    z-index: 200;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 1.25rem;
    color: #f1f3f5;
    pointer-events: none;
  }

  .top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
  }

  .header-card {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.875rem 1.25rem;
    border-radius: 0.75rem;
    background: rgb(12 17 22 / 85%);
    box-shadow: 0 4px 24px rgb(0 0 0 / 45%);
    pointer-events: auto;
  }

  .dots {
    display: flex;
    gap: 0.5rem;
  }

  .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: rgb(255 255 255 / 25%);
  }

  .dot.done {
    background: #4dabf7;
  }

  .dot.current {
    background: #f1f3f5;
    box-shadow: 0 0 0 2px rgb(77 171 247 / 55%);
  }

  .heading {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 700;
  }

  .skip-setup {
    pointer-events: auto;
  }

  .bottom {
    display: flex;
    justify-content: center;
  }

  .step-card {
    display: flex;
    max-width: 28rem;
    width: 100%;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    background: rgb(12 17 22 / 85%);
    box-shadow: 0 4px 24px rgb(0 0 0 / 45%);
    text-align: center;
    pointer-events: auto;
  }

  .copy {
    margin: 0;
    color: rgb(241 243 245 / 82%);
    font-size: 0.95rem;
    line-height: 1.45;
  }

  .muted {
    margin: 0;
    color: rgb(241 243 245 / 60%);
    font-size: 0.875rem;
  }

  .camera-select {
    display: flex;
    width: 100%;
    flex-direction: column;
    gap: 0.375rem;
    align-items: stretch;
    font-size: 0.875rem;
    color: rgb(241 243 245 / 78%);
    text-align: left;
  }

  .camera-select select {
    min-height: 2.25rem;
    padding: 0.375rem 0.5rem;
    border: 1px solid rgb(255 255 255 / 25%);
    border-radius: 0.5rem;
    color: #f1f3f5;
    background: rgb(20 32 43 / 95%);
    font-family: inherit;
    font-size: 0.9rem;
  }

  .status {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .status-line {
    color: rgb(241 243 245 / 60%);
    font-size: 0.875rem;
  }

  .status-line.ok {
    color: #69db7c;
  }

  .error-line {
    margin: 0;
    color: rgb(255 212 212 / 85%);
    font-size: 0.85rem;
    word-break: break-word;
  }

  .progress {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
  }

  button {
    min-height: 2.25rem;
    padding: 0.5rem 1.25rem;
    border-radius: 0.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
  }

  button.primary {
    border: 1px solid #74c0fc;
    color: #fff;
    background: #1971c2;
  }

  button.primary:disabled {
    border-color: rgb(116 192 252 / 40%);
    background: rgb(25 113 194 / 45%);
    color: rgb(255 255 255 / 65%);
    cursor: default;
  }

  button.ghost {
    border: 1px solid rgb(255 255 255 / 35%);
    color: rgb(241 243 245 / 88%);
    background: rgb(12 17 22 / 65%);
  }
</style>
