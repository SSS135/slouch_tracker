<script lang="ts">
  import type { FeatureId } from '@generated/bindings';
  import { useTrainingConfig } from '@/contexts/TrainingConfigContext';

  export interface FeatureMultiSelectorProps {
    postureSelected: FeatureId[];
    presenceSelected: FeatureId[];
    onPostureChange: (features: FeatureId[]) => void;
    onPresenceChange: (features: FeatureId[]) => void;
    disabled?: boolean;
  }

  let {
    postureSelected,
    presenceSelected,
    onPostureChange,
    onPresenceChange,
    disabled = false,
  }: FeatureMultiSelectorProps = $props();

  const trainingConfig = useTrainingConfig();
  const availableFeatures = $derived(trainingConfig.features.filter((feature) => feature.userSelectable));
  const missingSelectedFeatures = $derived(
    [...new Set([...postureSelected, ...presenceSelected])].filter(
      (featureType) => !trainingConfig.features.some((feature) => feature.id === featureType),
    ),
  );

  function totalDimensions(selected: FeatureId[]): number | null {
    let total = 0;
    for (const featureType of selected) {
      const definition = trainingConfig.features.find((feature) => feature.id === featureType);
      if (!definition) return null;
      total += definition.dimensions;
    }
    return total;
  }

  const postureTotalDimensions = $derived(totalDimensions(postureSelected));
  const presenceTotalDimensions = $derived(totalDimensions(presenceSelected));

  function formatDimensions(dims: number): string {
    if (dims >= 1000) {
      return `${(dims / 1000).toFixed(1)}K`;
    }
    return dims.toString();
  }

  function handlePostureToggle(featureType: FeatureId): void {
    if (disabled) return;

    const isSelected = postureSelected.includes(featureType);
    if (isSelected) {
      if (postureSelected.length === 1) return;
      onPostureChange(postureSelected.filter((feature) => feature !== featureType));
    } else {
      onPostureChange([...postureSelected, featureType]);
    }
  }

  function handlePresenceToggle(featureType: FeatureId): void {
    if (disabled) return;

    const isSelected = presenceSelected.includes(featureType);
    if (isSelected) {
      if (presenceSelected.length === 1) return;
      onPresenceChange(presenceSelected.filter((feature) => feature !== featureType));
    } else {
      onPresenceChange([...presenceSelected, featureType]);
    }
  }
</script>

<div class="selector-stack">
  {#if trainingConfig.error}
    <p class="registry-error" role="alert">Feature registry unavailable: {trainingConfig.error}</p>
  {:else if !trainingConfig.ready}
    <p class="registry-status" role="status">Loading feature registry…</p>
  {:else if missingSelectedFeatures.length > 0}
    <p class="registry-error" role="alert">
      Selected feature metadata is unavailable: {missingSelectedFeatures.join(', ')}
    </p>
  {/if}

  {#if trainingConfig.ready && postureTotalDimensions !== null && presenceTotalDimensions !== null}
    <section class="summary-card" aria-label="Selected feature dimensions">
      <div class="summary-group">
        <div class="summary-item">
          <span class="summary-label">Posture Model:</span>
          <strong class="summary-value posture-value">
            {formatDimensions(postureTotalDimensions)} ({postureTotalDimensions.toLocaleString()})
          </strong>
        </div>

        <div class="summary-item">
          <span class="summary-label">Presence Model:</span>
          <strong class="summary-value presence-value">
            {formatDimensions(presenceTotalDimensions)} ({presenceTotalDimensions.toLocaleString()})
          </strong>
        </div>
      </div>
    </section>
  {/if}

  <div class="feature-stack">
    {#each availableFeatures as definition (definition.id)}
      {@const featureType = definition.id}
      {@const isPostureSelected = postureSelected.includes(featureType)}
      {@const isPresenceSelected = presenceSelected.includes(featureType)}
      {@const isLastPosture = isPostureSelected && postureSelected.length === 1}
      {@const isLastPresence = isPresenceSelected && presenceSelected.length === 1}
      {@const postureDisabled = disabled}
      {@const presenceDisabled = disabled}

      <article class:disabled={disabled} class="feature-card">
        <div class="feature-content">
          <div class="feature-controls">
            <label
              class:disabled={postureDisabled}
              class:locked={isLastPosture}
              class="feature-checkbox"
            >
              <input
                type="checkbox"
                aria-label={`${definition.name} for posture model`}
                checked={isPostureSelected}
                onchange={() => handlePostureToggle(featureType)}
                disabled={postureDisabled || isLastPosture}
              />
              <span>Posture</span>
            </label>

            <label
              class:disabled={presenceDisabled}
              class:locked={isLastPresence}
              class="feature-checkbox"
            >
              <input
                type="checkbox"
                aria-label={`${definition.name} for presence model`}
                checked={isPresenceSelected}
                onchange={() => handlePresenceToggle(featureType)}
                disabled={presenceDisabled || isLastPresence}
              />
              <span>Presence</span>
            </label>

            <strong class="feature-name">{definition.name}</strong>
          </div>

          <p class="feature-description">{definition.description}</p>
        </div>
      </article>
    {/each}
  </div>

  <p class="help-text">
    Select features for each model type. At least 1 feature must be selected per model.
  </p>
</div>

<style>
  .selector-stack,
  .feature-stack {
    display: flex;
    flex-direction: column;
  }

  .selector-stack {
    gap: var(--mantine-spacing-sm, 8px);
  }

  .summary-card,
  .feature-card {
    box-sizing: border-box;
    width: 100%;
    border: 1px solid var(--mantine-color-default-border, #373a40);
    border-radius: var(--mantine-radius-sm, 4px);
    background: var(--mantine-color-dark-7, #2c2e33);
  }

  .summary-card {
    padding: var(--mantine-spacing-sm, 8px);
  }

  .summary-group {
    display: flex;
    align-items: flex-start;
    gap: var(--mantine-spacing-xl, 32px);
  }

  .summary-item {
    display: flex;
    flex-direction: column;
  }

  .summary-label,
  .help-text,
  .feature-description,
  .registry-status {
    color: var(--mantine-color-dimmed, #909296);
  }

  .summary-label {
    margin-bottom: 4px;
    font-size: var(--mantine-font-size-xs, 0.75rem);
    line-height: 1.5;
  }

  .summary-value {
    font-size: var(--mantine-font-size-md, 1rem);
    line-height: 1.55;
  }

  .posture-value {
    color: var(--mantine-color-blue-4, #74c0fc);
  }

  .presence-value {
    color: var(--mantine-color-teal-4, #63e6be);
  }

  .feature-stack {
    gap: var(--mantine-spacing-xs, 4px);
  }

  .feature-card {
    padding: var(--mantine-spacing-sm, 8px);
    color: var(--mantine-color-text, #c1c2c5);
  }

  .feature-card.disabled {
    opacity: 0.5;
  }

  .feature-content {
    display: flex;
    flex-direction: column;
    gap: var(--mantine-spacing-xs, 4px);
  }

  .feature-controls {
    display: flex;
    align-items: center;
    gap: var(--mantine-spacing-md, 16px);
    flex-wrap: nowrap;
  }

  .feature-checkbox {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--mantine-color-text, #c1c2c5);
    font-size: var(--mantine-font-size-sm, 0.875rem);
    cursor: pointer;
    white-space: nowrap;
  }

  .feature-checkbox.disabled,
  .feature-checkbox.locked {
    cursor: not-allowed;
  }

  .feature-checkbox.disabled {
    color: var(--mantine-color-dimmed, #909296);
  }

  .feature-checkbox input {
    width: 16px;
    height: 16px;
    margin: 0;
    accent-color: var(--mantine-primary-color-filled, #228be6);
    cursor: inherit;
  }

  .feature-name {
    min-width: 0;
    flex: 1;
    font-size: var(--mantine-font-size-sm, 0.875rem);
    line-height: 1.55;
  }

  .feature-description {
    margin: 0 0 0 170px;
    font-size: var(--mantine-font-size-xs, 0.75rem);
    line-height: 1.5;
  }

  .help-text,
  .registry-status,
  .registry-error {
    margin: 0;
    font-size: var(--mantine-font-size-xs, 0.75rem);
    line-height: 1.5;
  }

  .registry-error {
    padding: 0.5rem;
    border: 1px solid #fa5252;
    border-radius: 4px;
    color: #ffc9c9;
    background: rgb(201 42 42 / 20%);
  }

  @media (max-width: 560px) {
    .summary-group {
      flex-wrap: wrap;
      gap: var(--mantine-spacing-md, 16px);
    }

    .feature-controls {
      flex-wrap: wrap;
    }

    .feature-description {
      margin-left: 0;
    }
  }
</style>
