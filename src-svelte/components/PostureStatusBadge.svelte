<script lang="ts">
  import { Colors } from '@/constants/Colors';
  import type { InferenceUiResult } from '@generated/bindings';

  export interface PostureStatusBadgeProps {
    data?: InferenceUiResult['classification'];
    hasModel: boolean;
    presenceThreshold?: number;
  }

  let {
    data,
    hasModel,
    presenceThreshold = 0.5,
  }: PostureStatusBadgeProps = $props();

  const confidenceThreshold = 0.5;

  function percentage(value: number | undefined | null): number {
    if (typeof value !== 'number' || Number.isNaN(value)) {
      return 0;
    }

    return Math.max(0, Math.min(1, value)) * 100;
  }

  function colorValue(color: string): string {
    const colors: Record<string, string> = {
      'blue.4': '#74c0fc',
      'blue.7': '#1971c2',
      'gray.7': '#495057',
      'green.4': '#69db7c',
      'green.7': '#2f9e44',
      'red.4': '#ff8787',
      'red.7': '#c92a2a',
    };

    return colors[color] ?? color;
  }

  const presentProbability = $derived(data?.presentProbability ?? null);
  const personIsAway = $derived(
    !data ||
      (presentProbability !== null && presentProbability < presenceThreshold),
  );
  const goodProbability = $derived(data?.goodProbability ?? null);
  const isGoodPosture = $derived(
    typeof goodProbability === 'number' &&
      goodProbability >= confidenceThreshold,
  );
  const statusColor = $derived(
    isGoodPosture ? Colors.postureBadge.good : Colors.postureBadge.bad,
  );
  const progressColor = $derived(isGoodPosture ? 'green.4' : 'red.4');
  const statusTitle = $derived(isGoodPosture ? 'Good Posture' : 'Bad Posture');

  const noModelColor = colorValue(Colors.postureBadge.noModel);
  const awayColor = colorValue(Colors.postureBadge.personAway);
</script>

{#if !hasModel}
  <section
    class="paper"
    aria-label="Posture model status"
    style={`background-color: ${noModelColor};`}
  >
    <div class="stack">
      <strong class="title">No Model Trained</strong>
      <p class="description">Train a classifier to enable posture scoring.</p>
    </div>
  </section>
{:else if personIsAway}
  <section
    class="paper"
    aria-label="Posture presence status"
    style={`background-color: ${awayColor};`}
  >
    <div class="stack">
      <strong class="title">Person Away</strong>
      {#if presentProbability !== null}
        <div class="probability-bar">
          <div class="probability-header">
            <span class="label">Present</span>
            <strong class="value">{Math.round(percentage(presentProbability))}%</strong>
          </div>
          <div
            class="progress"
            role="progressbar"
            aria-label="Present probability"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={percentage(presentProbability)}
            aria-valuetext={`${Math.round(percentage(presentProbability))}%`}
          >
            <div
              class="progress-value"
              style={`width: ${percentage(presentProbability)}%; background-color: ${colorValue('blue.4')};`}
            ></div>
          </div>
        </div>
      {/if}
    </div>
  </section>
{:else}
  <section
    class="paper"
    aria-label="Posture classification status"
    style={`background-color: ${colorValue(statusColor)};`}
  >
    <div class="stack">
      <strong class="title">{statusTitle}</strong>
      <div class="probability-bar">
        <div class="probability-header">
          <span class="label">Good</span>
          <strong class="value">{Math.round(percentage(goodProbability))}%</strong>
        </div>
        <div
          class="progress"
          role="progressbar"
          aria-label="Good posture probability"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={percentage(goodProbability)}
          aria-valuetext={`${Math.round(percentage(goodProbability))}%`}
        >
          <div
            class="progress-value"
            style={`width: ${percentage(goodProbability)}%; background-color: ${colorValue(progressColor)};`}
          ></div>
        </div>
      </div>
      {#if presentProbability !== null}
        <div class="probability-bar">
          <div class="probability-header">
            <span class="label">Present</span>
            <strong class="value">{Math.round(percentage(presentProbability))}%</strong>
          </div>
          <div
            class="progress"
            role="progressbar"
            aria-label="Present probability"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={percentage(presentProbability)}
            aria-valuetext={`${Math.round(percentage(presentProbability))}%`}
          >
            <div
              class="progress-value"
              style={`width: ${percentage(presentProbability)}%; background-color: ${colorValue('blue.4')};`}
            ></div>
          </div>
        </div>
      {/if}
    </div>
  </section>
{/if}

<style>
  .paper {
    box-sizing: border-box;
    width: 100%;
    min-height: 100px;
    padding: 16px;
    border-radius: 8px;
    color: #fff;
  }

  .stack {
    display: flex;
    width: 100%;
    flex-direction: column;
    gap: var(--mantine-spacing-sm, 12px);
  }

  .title,
  .value {
    color: #fff;
  }

  .title {
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.55;
  }

  .description,
  .label,
  .value {
    margin: 0;
    font-size: 0.875rem;
    line-height: 1.45;
  }

  .description,
  .label {
    color: #fff;
    opacity: 0.8;
  }

  .probability-bar {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .probability-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .value {
    font-weight: 600;
  }

  .progress {
    width: 100%;
    height: 5px;
    overflow: hidden;
    border-radius: 4px;
    background-color: var(--mantine-color-dark-4, #373a40);
  }

  .progress-value {
    height: 100%;
    border-radius: inherit;
    transition: width 100ms ease;
  }
</style>
