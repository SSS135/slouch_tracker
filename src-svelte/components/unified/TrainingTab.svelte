<script lang="ts">
  import { useTrainingConfig } from '@/contexts/TrainingConfigContext';
  import { useTraining } from '@/contexts/TrainingContext';
  import { useDatasetOperations } from '@/hooks/useDatasetOperations';
  import { useNotification } from '@/hooks/useNotification';
  import type { DatasetStats, FrameLabel } from '@/services/dataset/types';
  import type { NativeStateSnapshot_Serialize, NormalizationMode } from '@generated/bindings';
  import ClassifierSelector from '../dataset/ClassifierSelector.svelte';
  import ConfirmationModal from '../dataset/ConfirmationModal.svelte';
  import DatasetStatsCard from '../dataset/DatasetStatsCard.svelte';
  import FeatureMultiSelector from '../dataset/FeatureMultiSelector.svelte';
  import DatasetFrameGrid from '../dataset/DatasetFrameGrid.svelte';
  import HelpText from '../ui/HelpText.svelte';
  import RadioGroup from '../ui/RadioGroup.svelte';
  import Section from '../ui/Section.svelte';
  import Slider from '../ui/Slider.svelte';

  export interface TrainingTabProps {
    onTrainingComplete: () => Promise<void>;
    onFramesChanged?: () => void;
    onBeforeNativeReplace?: () => Promise<void>;
    onNativeStateChanged?: (state: NativeStateSnapshot_Serialize) => Promise<void>;
    onFramePreview?: (url: string, label: FrameLabel) => void;
    onFramePreviewClear?: () => void;
  }

  const DEFAULT_STATS: DatasetStats = {
    total: 0,
    good: 0,
    bad: 0,
    away: 0,
    unused: 0,
    imbalanceRatio: 0,
    hasMinimumFrames: false,
    hasAwayFrames: false,
  };

  let {
    onTrainingComplete,
    onFramesChanged,
    onBeforeNativeReplace,
    onNativeStateChanged,
    onFramePreview,
    onFramePreviewClear,
  }: TrainingTabProps = $props();

  const datasetOps = useDatasetOperations();
  const trainingConfig = useTrainingConfig();
  const training = useTraining();
  const config = $derived(trainingConfig.config);
  const trainingState = $derived(training.state);
  const updatePostureFeatureTypes = trainingConfig.updatePostureFeatureTypes;
  const updatePresenceFeatureTypes = trainingConfig.updatePresenceFeatureTypes;
  const updateClassifierConfig = trainingConfig.updateClassifierConfig;
  const updateDimReductionConfig = trainingConfig.updateDimReductionConfig;
  const updateNormalizationMode = trainingConfig.updateNormalizationMode;
  const { showSuccess, showInfo, showError } = useNotification();

  let resetModalOpen = $state(false);
  let cleanupModalOpen = $state(false);

  const stats = $derived(datasetOps.stats.data ?? DEFAULT_STATS);
  const reservoir = $derived(datasetOps.reservoir.data ?? { count: 0, totalSeen: 0, maxSamples: 1000 });
  const pcaReady = $derived(reservoir.count >= 100);
  const statsLoading = $derived(datasetOps.stats.isLoading);

  $effect(() => {
    if (datasetOps.reservoir.data && !pcaReady && config.dimReductionConfig.method === 'pca') {
      updateDimReductionConfig({ method: 'random_projection', components: 32 });
    }
  });
  const statsError = $derived(datasetOps.stats.error);
  const frames = $derived(datasetOps.frames.data ?? []);
  const framesLoading = $derived(datasetOps.frames.isLoading);
  const framesError = $derived(datasetOps.frames.error);
  const postureResult = $derived(trainingState.postureResult);
  const presenceResult = $derived(trainingState.presenceResult);
  const readyToTrain = $derived(
    stats.hasMinimumFrames &&
      config.postureFeatureTypes.length > 0 &&
      config.presenceFeatureTypes.length > 0 &&
      !trainingState.isTraining &&
      trainingConfig.ready,
  );
  const displayFrames = $derived(
    frames.map((frame) => ({
      id: frame.id,
      label: frame.label,
      thumbnail: frame.thumbnail,
      thumbnailMimeType: frame.thumbnailMimeType,
      timestamp: frame.timestamp,
    })),
  );

  async function handleTrain(): Promise<void> {
    try {
      const completed = await training.train();
      if (!completed) {
        showInfo('Training cancelled.');
        return;
      }
      await onTrainingComplete();
      showSuccess('Training complete. Models updated.');
    } catch (error) {
      showError(error instanceof Error ? error.message : 'Training failed');
    }
  }

  function handleExport(): void {
    if (datasetOps.exportDataset.isPending) return;
    datasetOps.exportDataset.mutate(undefined, {
      onSuccess: (summary) => summary
        ? showSuccess(`Exported ${summary.frameCount} frame(s).`)
        : showInfo('Export cancelled.'),
      onError: (error: Error) => showError(error.message || 'Export failed'),
    });
  }

  async function handleImport(): Promise<void> {
    if (datasetOps.importDataset.isPending) return;
    try {
      await onBeforeNativeReplace?.();
    } catch (error) {
      showError(error instanceof Error ? error.message : 'Failed to flush settings before import');
      return;
    }
    datasetOps.importDataset.mutate(undefined, {
      onSuccess: async (summary) => {
        if (!summary) {
          showInfo('Import cancelled.');
          return;
        }
        onFramesChanged?.();
        await onNativeStateChanged?.(summary.state);
        showSuccess(`Imported ${summary.frameCount} frame(s).`);
      },
      onError: (error: Error) => showError(error.message || 'Import failed'),
    });
  }

  function handleCleanupUnused(): void {
    if (datasetOps.cleanupUnused.isPending) return;
    datasetOps.cleanupUnused.mutate(undefined, {
      onSuccess: (removed) => {
        cleanupModalOpen = false;
        onFramesChanged?.();
        showSuccess(`Removed ${removed} unused frame(s).`);
      },
      onError: (error: Error) => {
        cleanupModalOpen = false;
        showError(error.message || 'Cleanup failed');
      },
    });
  }

  function handleReset(): void {
    if (datasetOps.resetDataset.isPending) return;
    datasetOps.resetDataset.mutate(undefined, {
      onSuccess: async (state) => {
        resetModalOpen = false;
        onFramesChanged?.();
        await onNativeStateChanged?.(state);
        showSuccess('Dataset reset complete. Settings preserved; model deactivated.');
      },
      onError: (error: Error) => {
        resetModalOpen = false;
        showError(error.message || 'Reset failed');
      },
    });
  }

  function handleDeleteFrame(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      datasetOps.deleteFrame.mutate(id, {
        onSuccess: async () => {
          onFramePreviewClear?.();
          resolve();
        },
        onError: (error: Error) => reject(error),
      });
    });
  }

  function handleFrameDrag(frameId: string, newLabel: FrameLabel): void {
    datasetOps.updateLabel.mutate(
      { id: frameId, label: newLabel },
      {
        onError: () => {
          console.error('Failed to update frame label');
          showError('Failed to update frame label');
        },
      },
    );
  }

  function handleDimReductionMethodChange(value: string): void {
    const method = value as 'random_projection' | 'pca' | 'none';
    const components = method === 'none' ? 64 : 32;
    updateDimReductionConfig({ method, components });
  }

  function handleDimReductionComponentsChange(value: string): void {
    const components = Number.parseInt(value, 10);
    if (config.dimReductionConfig.method === 'none') return;

    const validValues = [1, 2, 4, 8, 16, 32, 64, 128, 256];
    if (validValues.includes(components)) {
      updateDimReductionConfig({
        ...config.dimReductionConfig,
        components,
      });
    }
  }
</script>

<div class="training-stack">
  {#if trainingConfig.error}
    <div class="error-card" role="alert">
      <span>Failed to load training configuration: {trainingConfig.error}</span>
      <button type="button" onclick={() => { void trainingConfig.reload().catch(() => undefined); }}>Retry training configuration</button>
    </div>
  {/if}

  <Section title="Overview">
    {#if statsLoading}
      <div class="inline-row">
        <span class="spinner" aria-hidden="true"></span>
        <span class="small-text">Loading dataset…</span>
      </div>
    {:else if statsError}
      <div class="error-card" role="alert">
        <span>Failed to load dataset statistics: {statsError.message}</span>
        <button type="button" onclick={() => { void datasetOps.stats.refetch(); }}>Retry statistics</button>
      </div>
    {:else}
      <DatasetStatsCard stats={stats} />
    {/if}
    {#if datasetOps.reservoir.error}
      <p class="extra-small-text warning-text" role="alert">Feature reservoir metadata unavailable.</p>
    {:else}
      <p class="extra-small-text muted">Feature reservoir: {reservoir.count}/{reservoir.maxSamples} samples ({reservoir.totalSeen} observed).</p>
    {/if}
  </Section>

  <Section title="Feature Types" subtitle="Select which feature vectors feed each classifier">
    <FeatureMultiSelector
      postureSelected={config.postureFeatureTypes}
      presenceSelected={config.presenceFeatureTypes}
      onPostureChange={updatePostureFeatureTypes}
      onPresenceChange={updatePresenceFeatureTypes}
      disabled={trainingState.isTraining || !trainingConfig.ready}
    />
  </Section>

  <Section title="Preprocessing" subtitle="Feature normalization and dimensionality reduction">
    <div class="section-stack">
      <div class="control-stack">
        <strong class="small-label">Feature Normalization</strong>
        <RadioGroup
          name="normalization-mode"
          value={config.normalizationMode ?? 'layer'}
          onChange={(value) => updateNormalizationMode(value as NormalizationMode)}
          disabled={trainingState.isTraining || !trainingConfig.ready}
          options={[
            { value: 'none', label: 'None', description: 'No feature normalization' },
            { value: 'layer', label: 'Layer Normalization', description: 'Normalize each sample independently (mean=0, std=1 per sample)' },
            { value: 'z_score', label: 'Z-Score Normalization', description: 'Standardize each feature to mean=0, std=1', badge: 'Recommended', badgeColor: 'green' },
            { value: 'calibrated', label: 'Calibrated (relative to good)', description: 'Center features on the good/present class so slouch reads as a consistent deviation' },
          ]}
        />
        <HelpText text="Normalization improves model training stability. Z-score normalization is recommended for most use cases." />
      </div>

      <div class="control-stack">
        <strong class="small-label">Method:</strong>
        <RadioGroup
          options={[
            { value: 'random_projection', label: 'Random Projection', description: 'Unsupervised, fast, preserves distances' },
            { value: 'pca', label: pcaReady ? `PCA (${reservoir.count} samples available)` : `PCA (needs 100 samples, have ${reservoir.count})`, description: 'Unsupervised native dimensionality reduction', disabled: !pcaReady },
            { value: 'none', label: 'None', description: 'Use all features' },
          ]}
          value={config.dimReductionConfig.method}
          onChange={handleDimReductionMethodChange}
          name="dimensionality-reduction-method"
          disabled={trainingState.isTraining || !trainingConfig.ready}
        />
      </div>

      {#if config.dimReductionConfig.method === 'random_projection'}
        <Slider
          label="Dimensions"
          value={config.dimReductionConfig.components}
          minimumValue={1}
          maximumValue={256}
          fixedValues={[1, 2, 4, 8, 16, 32, 64, 128, 256]}
          formatValue={(value) => String(value)}
          onValueChange={(value) => handleDimReductionComponentsChange(String(value))}
          helpText="Number of dimensions after reduction (power-of-2 steps)"
          showTooltip
          showMinMax
          disabled={trainingState.isTraining || !trainingConfig.ready}
        />
      {/if}

      {#if config.dimReductionConfig.method === 'pca'}
        <Slider
          label="Components"
          value={config.dimReductionConfig.components}
          minimumValue={1}
          maximumValue={256}
          fixedValues={[1, 2, 4, 8, 16, 32, 64, 128, 256]}
          formatValue={(value) => String(value)}
          onValueChange={(value) => handleDimReductionComponentsChange(String(value))}
          helpText="Number of principal components (power-of-2 steps)"
          showTooltip
          showMinMax
          disabled={trainingState.isTraining || !trainingConfig.ready}
        />
      {/if}

      <HelpText text="Changing reduction settings requires retraining the model" />
    </div>
  </Section>

  <Section title="Classifier" subtitle="Select algorithm and parameters">
    <ClassifierSelector
      config={config.classifierConfig}
      onChange={updateClassifierConfig}
      disabled={trainingState.isTraining || !trainingConfig.ready}
    />
  </Section>

  <Section title="Train Model">
    <div class="control-stack">
      <button
        type="button"
        class="primary-button"
        onclick={handleTrain}
        disabled={!readyToTrain}
        aria-busy={trainingState.isTraining}
      >
        {#if trainingState.isTraining}<span class="spinner" aria-hidden="true"></span>{/if}
        {trainingState.isTraining ? 'Training…' : 'Train'}
      </button>
      {#if trainingState.isTraining}
        <button
          type="button"
          class="secondary-button warning-button"
          onclick={() => { void training.cancel().catch((error: unknown) => showError(error instanceof Error ? error.message : 'Cancellation failed')); }}
        >
          Cancel training
        </button>
      {/if}

      {#if !stats.hasMinimumFrames}<HelpText text="Collect at least 1 frame per class to train." />{/if}
      {#if config.postureFeatureTypes.length === 0}<HelpText text="Select at least one feature type for Posture model." />{/if}
      {#if config.presenceFeatureTypes.length === 0}<HelpText text="Select at least one feature type for Presence model." />{/if}

      {#if trainingState.error}
        <div class="error-card preserve-lines" role="alert">{trainingState.error}</div>
      {/if}

      {#if trainingState.warnings.length > 0}
        <div class="warning-card" role="status">
          {#each trainingState.warnings as warning (warning)}
            <div>{warning}</div>
          {/each}
        </div>
      {/if}

      {#if trainingState.isTraining}
        <div class="training-progress" role="status" aria-live="polite">
          <div class="inline-row">
            <span class="spinner" aria-hidden="true"></span>
            <span class="small-text">Training {trainingState.stage.replace('_', ' ')}: {Math.round(trainingState.progress)}%</span>
          </div>
          <progress aria-label="Training progress" max="100" value={trainingState.progress}></progress>
        </div>
      {/if}

      {#if postureResult}
        <div class="result-card green-card">
          <strong>Posture Model Results</strong>
          {#if postureResult.metrics.foldAccuracies.length > 0}
            <span>CV Accuracy: {((postureResult.metrics.cvAccuracy ?? 0) * 100).toFixed(0)}% [{((postureResult.metrics.accuracyCiLow ?? 0) * 100).toFixed(0)}-{((postureResult.metrics.accuracyCiHigh ?? 0) * 100).toFixed(0)}%]</span>
            <span>Balanced Accuracy: {((postureResult.metrics.balancedAccuracy ?? 0) * 100).toFixed(1)}%</span>
            <span>Worst Fold: {((postureResult.metrics.worstFoldAccuracy ?? 0) * 100).toFixed(1)}%</span>
            <span>MCC: {((postureResult.metrics.mcc ?? 0) * 100).toFixed(1)}%</span>
            <span>F1 Score: {((postureResult.metrics.f1Score ?? 0) * 100).toFixed(1)}%</span>
            {#if postureResult.metrics.cvType}<span class="muted">CV method: {postureResult.metrics.cvType}</span>{/if}
          {:else}
            <span class="warning-text">CV skipped - metrics unavailable</span>
          {/if}
        </div>
      {/if}

      {#if presenceResult}
        <div class="result-card blue-card">
          <strong>Presence Model Results</strong>
          {#if presenceResult.metrics.foldAccuracies.length > 0}
            <span>CV Accuracy: {((presenceResult.metrics.cvAccuracy ?? 0) * 100).toFixed(0)}% [{((presenceResult.metrics.accuracyCiLow ?? 0) * 100).toFixed(0)}-{((presenceResult.metrics.accuracyCiHigh ?? 0) * 100).toFixed(0)}%]</span>
            <span>Balanced Accuracy: {((presenceResult.metrics.balancedAccuracy ?? 0) * 100).toFixed(1)}%</span>
            <span>Worst Fold: {((presenceResult.metrics.worstFoldAccuracy ?? 0) * 100).toFixed(1)}%</span>
            <span>MCC: {((presenceResult.metrics.mcc ?? 0) * 100).toFixed(1)}%</span>
            <span>F1 Score: {((presenceResult.metrics.f1Score ?? 0) * 100).toFixed(1)}%</span>
            {#if presenceResult.metrics.cvType}<span class="muted">CV method: {presenceResult.metrics.cvType}</span>{/if}
          {:else}
            <span class="warning-text">CV skipped - metrics unavailable</span>
          {/if}
        </div>
      {/if}
    </div>
  </Section>

  <Section title="Dataset">
    <div class="button-row">
      <button type="button" class="secondary-button" onclick={handleExport} disabled={stats.total === 0 || datasetOps.exportDataset.isPending} aria-busy={datasetOps.exportDataset.isPending}>Export</button>
      <button type="button" class="secondary-button" onclick={() => { void handleImport(); }} disabled={datasetOps.importDataset.isPending} aria-busy={datasetOps.importDataset.isPending}>Import</button>
      <button type="button" class="secondary-button warning-button" onclick={() => (cleanupModalOpen = true)} disabled={stats.unused === 0 || datasetOps.cleanupUnused.isPending}>Cleanup Unused</button>
      <button type="button" class="secondary-button orange-button" onclick={() => (resetModalOpen = true)} disabled={(stats.total === 0 && reservoir.count === 0) || datasetOps.resetDataset.isPending}>Reset Dataset</button>
    </div>

    {#if framesLoading}
      <div class="inline-row">
        <span class="spinner" aria-hidden="true"></span>
        <span class="small-text">Loading frames…</span>
      </div>
    {:else if framesError}
      <div class="error-card" role="alert">
        <span>Failed to load dataset frames: {framesError.message}</span>
        <button type="button" onclick={() => { void datasetOps.frames.refetch(); }}>Retry frames</button>
      </div>
    {:else}
      <DatasetFrameGrid
        frames={displayFrames}
        onFrameClick={() => undefined}
        onDeleteFrame={handleDeleteFrame}
        onFramePreview={onFramePreview}
        onFramePreviewClear={onFramePreviewClear}
        onFrameDrag={handleFrameDrag}
      />
      {#if datasetOps.pageTotal > datasetOps.pageSize}
        <nav class="page-controls" aria-label="Dataset pages">
          <button type="button" onclick={datasetOps.previousPage} disabled={datasetOps.pageOffset === 0}>Previous page</button>
          <span>Frames {datasetOps.pageOffset + 1}–{Math.min(datasetOps.pageOffset + displayFrames.length, datasetOps.pageTotal)} of {datasetOps.pageTotal}</span>
          <button type="button" onclick={datasetOps.nextPage} disabled={datasetOps.pageOffset + datasetOps.pageSize >= datasetOps.pageTotal}>Next page</button>
        </nav>
      {/if}
    {/if}
  </Section>
</div>

<ConfirmationModal
  visible={cleanupModalOpen}
  title="Cleanup Unused Frames"
  message={`Delete ${stats.unused} unused frame(s)? This cannot be undone.`}
  confirmText="Cleanup Unused"
  cancelText="Cancel"
  confirmButtonColor="red"
  loading={datasetOps.cleanupUnused.isPending}
  onConfirm={handleCleanupUnused}
  onCancel={() => (cleanupModalOpen = false)}
/>

<ConfirmationModal
  visible={resetModalOpen}
  title="Reset Dataset"
  message="Delete all frames (labeled + unlabeled). Settings are preserved and the stale model is deactivated."
  confirmText="Reset Dataset"
  cancelText="Cancel"
  confirmButtonColor="orange"
  loading={datasetOps.resetDataset.isPending}
  onConfirm={handleReset}
  onCancel={() => (resetModalOpen = false)}
/>

<style>
  .training-stack,
  .section-stack,
  .control-stack,
  .result-card,
  .training-progress {
    display: flex;
    flex-direction: column;
  }

  .training-stack { gap: var(--mantine-spacing-lg, 20px); }
  .section-stack { gap: var(--mantine-spacing-md, 16px); }
  .control-stack,
  .training-progress { gap: var(--mantine-spacing-sm, 8px); }
  .training-progress progress { width: 100%; }
  .inline-row,
  .button-row {
    display: flex;
    align-items: center;
    gap: var(--mantine-spacing-sm, 8px);
  }
  .button-row { flex-wrap: wrap; margin-bottom: var(--mantine-spacing-md, 16px); }
  .page-controls { display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-top: 0.75rem; font-size: 0.8rem; }
  .small-label,
  .small-text { font-size: var(--mantine-font-size-sm, 0.875rem); }
  .extra-small-text { margin: 0; font-size: var(--mantine-font-size-xs, 0.75rem); }
  .muted { color: var(--mantine-color-dimmed, #909296); }
  .spinner {
    width: 1rem;
    height: 1rem;
    flex: 0 0 auto;
    border: 2px solid rgb(255 255 255 / 30%);
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  button {
    min-height: 2.25rem;
    padding: 0.5rem 0.875rem;
    border: 1px solid transparent;
    border-radius: var(--mantine-radius-sm, 4px);
    color: inherit;
    background: var(--mantine-color-dark-5, #373a40);
    font: inherit;
    cursor: pointer;
  }
  button:hover:not(:disabled),
  button:focus-visible { filter: brightness(1.15); }
  button:disabled { cursor: not-allowed; opacity: 0.55; }
  .primary-button { display: inline-flex; align-items: center; gap: 0.5rem; align-self: flex-start; color: white; background: var(--mantine-color-blue-6, #228be6); }
  .secondary-button { color: white; background: var(--mantine-color-dark-5, #373a40); }
  .warning-button { color: #ffec99; background: rgb(245 159 0 / 18%); }
  .orange-button { color: #ffd8a8; background: rgb(230 119 0 / 18%); }
  .error-card,
  .warning-card,
  .result-card { padding: 0.75rem; border: 1px solid; border-radius: var(--mantine-radius-md, 8px); }
  .error-card { color: white; background: rgb(201 42 42 / 70%); border-color: rgb(255 135 135 / 45%); }
  .warning-card { display: flex; flex-direction: column; gap: 0.25rem; color: #ffec99; background: rgb(245 159 0 / 18%); border-color: rgb(245 159 0 / 45%); }
  .preserve-lines { white-space: pre-wrap; }
  .result-card { gap: 0.25rem; color: white; }
  .green-card { background: rgb(47 158 68 / 70%); border-color: rgb(105 219 124 / 45%); }
  .blue-card { background: rgb(25 113 194 / 70%); border-color: rgb(116 192 252 / 45%); }
  .warning-text { color: #ffec99; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
