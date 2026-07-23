<script lang="ts">
  import type { PoseModelPhase } from '@/hooks/usePoseModelDownload.svelte';

  export interface PoseModelDownloadScreenProps {
    state: PoseModelPhase;
    onCancel: () => void;
    onRetry: () => void;
  }

  let { state, onCancel, onRetry }: PoseModelDownloadScreenProps = $props();

  const MEGABYTE = 1024 * 1024;
  const formatMb = (bytes: number): string => (bytes / MEGABYTE).toFixed(bytes >= 100 * MEGABYTE ? 0 : 1);

  const downloading = $derived(state.kind === 'downloading' ? state : null);
  // Determinate only once the server reports a total (after the `started` event).
  const hasTotal = $derived(downloading !== null && downloading.total > 0);
  const percent = $derived(
    downloading && downloading.total > 0
      ? Math.min(100, Math.floor((downloading.received / downloading.total) * 100))
      : 0,
  );
</script>

<div
  class="pose-download"
  role="dialog"
  aria-modal="true"
  aria-labelledby="pose-download-title"
>
  <div class="card">
    <h1 id="pose-download-title" class="title">Setting up posture detection</h1>
    <p class="explainer">
      This is a one-time download of the pose-detection model (about 245&nbsp;MB). It runs once on
      first launch, then the app is ready.
    </p>

    {#if downloading}
      <div class="progress-block">
        <div
          class="track"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={hasTotal ? percent : undefined}
          aria-label="Model download progress"
        >
          <div class="fill" class:indeterminate={!hasTotal} style:width={hasTotal ? `${percent}%` : undefined}></div>
        </div>
        <p class="progress-text">
          {#if hasTotal}
            {formatMb(downloading.received)} MB of {formatMb(downloading.total)} MB ({percent}%)
          {:else}
            Starting download…
          {/if}
        </p>
      </div>
      <button type="button" class="ghost" onclick={onCancel}>Cancel</button>
    {:else if state.kind === 'verifying'}
      <div class="progress-block">
        <div class="track" role="progressbar" aria-label="Verifying model">
          <div class="fill indeterminate"></div>
        </div>
        <p class="progress-text">Verifying the downloaded model…</p>
      </div>
    {:else if state.kind === 'failed'}
      <div class="failure" role="alert">
        <p class="failure-line">The pose-model download did not complete.</p>
        <p class="failure-reason">{state.reason}</p>
        {#if state.offline}
          <p class="offline-hint">
            You may be offline. To install the model without a network connection, follow the
            “Fully offline installation” instructions in the project README.
          </p>
        {/if}
      </div>
      <button type="button" class="primary" onclick={onRetry}>Retry download</button>
    {:else if state.kind === 'cancelled'}
      <p class="progress-text">Download paused.</p>
      <button type="button" class="primary" onclick={onRetry}>Resume download</button>
    {/if}

    <p class="attribution">
      Pose model: NLF by István Sárándi (nlf paper + github.com/isarandi/nlf).<br />
      Non-commercial use only: scientific research, education, or artistic projects.
    </p>
  </div>
</div>

<style>
  .pose-download {
    position: fixed;
    inset: 0;
    z-index: 250;
    display: grid;
    place-items: center;
    padding: 2rem;
    color: #f1f3f5;
    /* Calm setup tone (blue-slate), deliberately not the error-red palette. */
    background: radial-gradient(circle at 50% 30%, #14202b 0%, #0c1116 70%);
  }

  .card {
    display: flex;
    max-width: 32rem;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    text-align: center;
  }

  .title {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
  }

  .explainer {
    margin: 0;
    color: rgb(241 243 245 / 78%);
    font-size: 0.95rem;
    line-height: 1.45;
  }

  .progress-block {
    display: flex;
    width: 100%;
    flex-direction: column;
    gap: 0.5rem;
    align-items: center;
  }

  .track {
    position: relative;
    width: 100%;
    height: 10px;
    border-radius: 999px;
    background: rgb(255 255 255 / 12%);
    overflow: hidden;
  }

  .fill {
    height: 100%;
    border-radius: 999px;
    background: #4dabf7;
    transition: width 200ms ease;
  }

  .fill.indeterminate {
    width: 35%;
    animation: slide 1.2s ease-in-out infinite;
  }

  .progress-text {
    margin: 0;
    color: rgb(241 243 245 / 82%);
    font-size: 0.875rem;
    font-variant-numeric: tabular-nums;
  }

  .failure {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .failure-line {
    margin: 0;
    font-weight: 600;
  }

  .failure-reason {
    margin: 0;
    color: rgb(255 212 212 / 85%);
    font-size: 0.85rem;
    word-break: break-word;
  }

  .offline-hint {
    margin: 0;
    color: rgb(241 243 245 / 72%);
    font-size: 0.85rem;
    line-height: 1.4;
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

  button.ghost {
    border: 1px solid rgb(255 255 255 / 35%);
    color: rgb(241 243 245 / 88%);
    background: transparent;
  }

  .attribution {
    margin: 0.5rem 0 0;
    color: rgb(241 243 245 / 55%);
    font-size: 0.75rem;
    line-height: 1.4;
  }

  @keyframes slide {
    0% {
      transform: translateX(-120%);
    }
    100% {
      transform: translateX(340%);
    }
  }
</style>
