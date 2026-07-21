<script lang="ts">
  import type { DatasetStats } from '@/services/dataset/types';

  let { stats }: { stats: DatasetStats } = $props();

  const totalLabeled = $derived(stats.good + stats.bad + stats.away);
  const goodPercent = $derived(totalLabeled > 0 ? (stats.good / totalLabeled) * 100 : 0);
  const badPercent = $derived(totalLabeled > 0 ? (stats.bad / totalLabeled) * 100 : 0);
  const awayPercent = $derived(totalLabeled > 0 ? (stats.away / totalLabeled) * 100 : 0);

  const imbalancePercent = $derived((stats.imbalanceRatio ?? 0) * 100);
  const hasImbalanceWarning = $derived((stats.imbalanceRatio ?? 0) > 0.35);
  const hasInsufficientDataWarning = $derived(
    !stats.hasMinimumFrames && (stats.good > 0 || stats.bad > 0),
  );
  const hasNoDataWarning = $derived(stats.good === 0 && stats.bad === 0);
  const hasNoAwayWarning = $derived(!stats.hasAwayFrames && totalLabeled > 0);
</script>

<article class="card" aria-label="Dataset statistics">
  <div class="stack gap-md">
    <div class="stats-grid">
      <div class="stat">
        <strong class="value total">{stats.total}</strong>
        <span class="label">Total</span>
      </div>

      <div class="stat">
        <strong class="value good">{stats.good}</strong>
        <span class="label">Good</span>
      </div>

      <div class="stat">
        <strong class="value bad">{stats.bad}</strong>
        <span class="label">Bad</span>
      </div>

      <div class="stat">
        <strong class="value away">{stats.away}</strong>
        <span class="label">Away</span>
      </div>

      <div class="stat">
        <strong class="value unused">{stats.unused}</strong>
        <span class="label">Unused</span>
      </div>
    </div>

    {#if totalLabeled > 0}
      <section class="distribution" aria-label="Class distribution">
        <span class="label">Class Distribution:</span>
        <div class="progress" role="img" aria-label={`Good ${goodPercent.toFixed(1)}%, Bad ${badPercent.toFixed(1)}%, Away ${awayPercent.toFixed(1)}%`}>
          {#if stats.good > 0}
            <span class="progress-section good-fill" style={`width: ${goodPercent}%;`}></span>
          {/if}
          {#if stats.bad > 0}
            <span class="progress-section bad-fill" style={`width: ${badPercent}%;`}></span>
          {/if}
          {#if stats.away > 0}
            <span class="progress-section away-fill" style={`width: ${awayPercent}%;`}></span>
          {/if}
        </div>
        <div class="distribution-labels">
          <span class="label">Good: {stats.good} ({goodPercent.toFixed(1)}%)</span>
          <span class="label">Bad: {stats.bad} ({badPercent.toFixed(1)}%)</span>
          <span class="label">Away: {stats.away} ({awayPercent.toFixed(1)}%)</span>
        </div>
      </section>
    {/if}

    {#if hasNoDataWarning}
      <div class="alert red" role="alert">
        <span class="icon" aria-hidden="true">⚠️</span>
        <span>No labeled data. Collect and label frames to begin.</span>
      </div>
    {/if}

    {#if !hasNoDataWarning && hasInsufficientDataWarning}
      <div class="alert red" role="alert">
        <span class="icon" aria-hidden="true">⚠️</span>
        <span>Need at least 5 frames per class to train (Good: {stats.good}, Bad: {stats.bad})</span>
      </div>
    {/if}

    {#if !hasNoDataWarning && hasImbalanceWarning}
      <div class="alert yellow" role="alert">
        <span class="icon" aria-hidden="true">⚠️</span>
        <span>Class imbalance: {imbalancePercent.toFixed(0)}% (recommended &lt; 35%)</span>
      </div>
    {/if}

    {#if !hasNoDataWarning && hasNoAwayWarning}
      <div class="alert yellow" role="alert">
        <span class="icon" aria-hidden="true">ℹ️</span>
        <span>No away frames collected. Presence detection will use RTMDet fallback.</span>
      </div>
    {/if}

    {#if !hasNoDataWarning && stats.hasMinimumFrames && !hasImbalanceWarning}
      <div class="alert green" role="alert">
        <span class="icon" aria-hidden="true">✅</span>
        <span>
          Dataset ready for training! {stats.hasAwayFrames
            ? 'Full dual-model training available.'
            : 'Posture model only (no away frames).'}
        </span>
      </div>
    {/if}
  </div>
</article>

<style>
  .card {
    box-sizing: border-box;
    width: 100%;
    padding: 16px;
    color: #f8f9fa;
    background: var(--mantine-color-dark-8, #141517);
    border: 1px solid var(--mantine-color-default-border, #373a40);
    border-radius: 8px;
  }

  .stack {
    display: flex;
    flex-direction: column;
  }

  .gap-md {
    gap: 16px;
  }

  .stats-grid {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-around;
    gap: 16px;
    padding: 12px 0;
  }

  .stat {
    display: flex;
    min-width: 0;
    flex: 1 1 64px;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .value {
    font-size: 20px;
    font-weight: 700;
    line-height: 1.2;
  }

  .total {
    color: #fff;
  }

  .good {
    color: var(--mantine-color-green-5, #40c057);
  }

  .bad {
    color: var(--mantine-color-red-5, #fa5252);
  }

  .away {
    color: var(--mantine-color-blue-5, #339af0);
  }

  .unused {
    color: #adb5bd;
  }

  .label {
    color: #909296;
    font-size: 12px;
    line-height: 1.4;
  }

  .distribution {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .progress {
    display: flex;
    width: 100%;
    height: 16px;
    overflow: hidden;
    background: #373a40;
    border-radius: 4px;
  }

  .progress-section {
    display: block;
    height: 100%;
    flex: 0 0 auto;
  }

  .good-fill {
    background: #2f9e44;
  }

  .bad-fill {
    background: #c92a2a;
  }

  .away-fill {
    background: #1971c2;
  }

  .distribution-labels {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 12px;
  }

  .alert {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 12px;
    border: 1px solid;
    border-radius: 4px;
    font-size: 14px;
    line-height: 1.45;
  }

  .icon {
    flex: 0 0 auto;
  }

  .red {
    color: var(--mantine-color-red-4, #ff8787);
    background: rgb(250 82 82 / 10%);
    border-color: rgb(250 82 82 / 20%);
  }

  .yellow {
    color: var(--mantine-color-yellow-4, #ffd43b);
    background: rgb(250 176 5 / 10%);
    border-color: rgb(250 176 5 / 20%);
  }

  .green {
    color: var(--mantine-color-green-4, #69db7c);
    background: rgb(64 192 87 / 10%);
    border-color: rgb(64 192 87 / 20%);
  }
</style>
